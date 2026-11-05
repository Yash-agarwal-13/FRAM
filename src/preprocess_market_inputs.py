from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

EQUITY_FILE = RAW / "equity" / "Biocon_data.xlsx"
RATE_91_FILE = RAW / "rates" / "Auctions of 91-Day Government of India Treasury Bills.xlsx"
RATE_364_FILE = RAW / "rates" / "Auctions of 364-Day Government of India Treasury Bills.xlsx"

VALUATION_DATES = pd.to_datetime([
    "2025-10-31", "2025-11-28", "2025-12-31", "2026-01-30",
    "2026-02-27", "2026-03-30", "2026-04-30", "2026-05-29",
    "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30",
])

YEAR_ENDS = pd.to_datetime([
    "2022-03-31", "2023-03-31", "2024-03-31",
    "2025-03-31", "2026-03-31",
])


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def latest_on_or_before(data, dates, columns):
    targets = pd.DataFrame({"target_date": dates.astype("datetime64[ns]")})
    source = data[["date", *columns]].dropna(subset=["date"]).copy()
    source["date"] = pd.to_datetime(source["date"]).astype("datetime64[ns]")

    result = pd.merge_asof(
        targets.sort_values("target_date"),
        source.sort_values("date"),
        left_on="target_date",
        right_on="date",
        direction="backward",
    )
    result["lag_calendar_days"] = (result["target_date"] - result["date"]).dt.days
    return result


def load_equity():
    data = pd.read_excel(EQUITY_FILE, sheet_name="Sheet1")
    data.columns = data.columns.str.strip()
    data = data.rename(columns={
        "Symbol": "symbol",
        "Series": "series",
        "Date": "date",
        "Prev Close": "previous_close",
        "Open Price": "open_price",
        "High Price": "high_price",
        "Low Price": "low_price",
        "Last Price": "last_price",
        "Close Price": "close_price",
        "Average Price": "average_price",
        "Total Traded Quantity": "volume",
        "Turnover ₹": "turnover_inr",
        "No. of Trades": "number_of_trades",
        "Deliverable Qty": "deliverable_quantity",
        "% Dly Qt to Traded Qty": "deliverable_percentage",
        "Adjusted Prices": "adjusted_price",
        "Adjusted Returns": "adjusted_return",
    })

    data["symbol"] = data["symbol"].astype(str).str.strip()
    data["series"] = data["series"].astype(str).str.strip()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data[(data["symbol"] == "BIOCON") & (data["series"] == "EQ")].copy()

    numeric_columns = [
        "previous_close", "open_price", "high_price", "low_price",
        "last_price", "close_price", "average_price", "volume",
        "turnover_inr", "number_of_trades", "deliverable_quantity",
        "deliverable_percentage", "adjusted_price", "adjusted_return",
    ]
    for column in numeric_columns:
        data[column] = to_number(data[column])

    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    data["calculated_adjusted_return"] = np.log(
        data["adjusted_price"] / data["adjusted_price"].shift(1)
    )
    data["return_difference"] = (
        data["adjusted_return"] - data["calculated_adjusted_return"]
    )
    data["model_return"] = data["adjusted_return"]
    return data


def load_rate(file, maturity):
    raw = pd.read_excel(file, header=None)
    rates = raw.iloc[:, [1, 13]].copy()
    rates.columns = ["date", "yield_percent"]
    rates["date"] = pd.to_datetime(rates["date"], format="mixed", errors="coerce")
    rates["yield_percent"] = pd.to_numeric(rates["yield_percent"], errors="coerce")
    rates = rates.dropna().sort_values("date").drop_duplicates("date", keep="last")
    rates["maturity_days"] = maturity
    rates["source_file"] = file.name
    return rates.reset_index(drop=True)


def main():
    equity = load_equity()

    valuation_prices = latest_on_or_before(
        equity, VALUATION_DATES, ["close_price", "adjusted_price"]
    )
    year_end_prices = latest_on_or_before(
        equity, YEAR_ENDS, ["close_price", "adjusted_price"]
    )
    year_end_prices.insert(0, "fiscal_year", [2022, 2023, 2024, 2025, 2026])

    rate_91 = load_rate(RATE_91_FILE, 91)
    rate_364 = load_rate(RATE_364_FILE, 364)
    all_rates = pd.concat([rate_91, rate_364], ignore_index=True)

    selected_rates = []
    for maturity, group in all_rates.groupby("maturity_days"):
        selected = latest_on_or_before(group, VALUATION_DATES, ["yield_percent"])
        selected["maturity_days"] = maturity
        selected["simple_rate_decimal"] = selected["yield_percent"] / 100
        selected["continuous_rate"] = np.log1p(selected["simple_rate_decimal"])
        selected_rates.append(selected)

    valuation_rates = pd.concat(selected_rates, ignore_index=True)
    valuation_rates = valuation_rates.sort_values(["target_date", "maturity_days"])

    equity.to_csv(OUT / "biocon_equity_adjusted.csv", index=False, date_format="%Y-%m-%d")
    valuation_prices.to_csv(OUT / "valuation_date_equity_prices.csv", index=False, date_format="%Y-%m-%d")
    year_end_prices.to_csv(OUT / "fiscal_year_end_equity_prices.csv", index=False, date_format="%Y-%m-%d")
    all_rates.to_csv(OUT / "rbi_tbill_auction_yields.csv", index=False, date_format="%Y-%m-%d")
    valuation_rates.to_csv(OUT / "valuation_date_tbill_rates.csv", index=False, date_format="%Y-%m-%d")

    print(f"Equity observations processed: {len(equity)}")

    print(f"Duplicate dates: {equity['date'].duplicated().sum()}")
    print(f"91-day rate observations: {len(rate_91)}")
    print(f"364-day rate observations: {len(rate_364)}")
    print(f"Output saved to: {OUT}")
    
    print("\nFiscal year-end prices")
    print(year_end_prices.to_string(index=False))

    print("\nValuation-date equity prices")
    print(valuation_prices.to_string(index=False))

    print("\nValuation-date T-bill rates")
    print(valuation_rates.to_string(index=False))


if __name__ == "__main__":
    main()
