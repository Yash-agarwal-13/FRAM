from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import input_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
FIGURE_OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

MARKET_FILE = input_data.file(input_data.MARKET_FILE)
RATE_91_FILE = input_data.file(input_data.RATE_91_FILE)
RATE_364_FILE = input_data.file(input_data.RATE_364_FILE)
OPTIONS_FILE = input_data.file(input_data.OPTIONS_FILE)

DIVIDEND_EVENTS = pd.read_excel(MARKET_FILE, sheet_name="Dividends").rename(
    columns={"dividend_per_share_inr": "dividend_per_share"}
)
DIVIDEND_EVENTS["ex_date"] = pd.to_datetime(DIVIDEND_EVENTS["ex_date"])

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
    data = pd.read_excel(MARKET_FILE, sheet_name="Biocon")
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
    header_ok = raw.iloc[:10, 13].astype(str).str.contains(
        "Implicit Yield", case=False, regex=False
    ).any()
    assert header_ok, "RBI implicit-yield column layout changed"
    rates = raw.iloc[:, [1, 13]].copy()
    rates.columns = ["date", "yield_percent"]
    rates["date"] = pd.to_datetime(rates["date"], format="mixed", errors="coerce")
    rates["yield_percent"] = pd.to_numeric(rates["yield_percent"], errors="coerce")
    rates = rates.dropna()
    rates = rates[rates["yield_percent"] > 0]
    rates = rates.sort_values("date").drop_duplicates("date", keep="last")
    rates["maturity_days"] = maturity
    rates["source_file"] = file.name
    return rates.reset_index(drop=True)


# -----------------------------------------------------------------------------
# OPTIONS PREPROCESSING
# -----------------------------------------------------------------------------
def trailing_dividend_per_share(valuation_date):
    """Return cash dividends with ex-dates in the trailing 12 months."""
    window_start = valuation_date - pd.DateOffset(years=1)
    in_window = DIVIDEND_EVENTS["ex_date"].gt(window_start) & DIVIDEND_EVENTS["ex_date"].le(valuation_date)
    return DIVIDEND_EVENTS.loc[in_window, "dividend_per_share"].sum()


def preprocess_options(valuation_rates, valuation_prices):
    """Apply the assignment's option-chain cleaning rules for BSM."""
    workbook = pd.ExcelFile(OPTIONS_FILE)
    frames = []

    for sheet in workbook.sheet_names:
        frame = pd.read_excel(OPTIONS_FILE, sheet_name=sheet)
        frame.columns = frame.columns.str.strip()
        frame.insert(0, "source_sheet", sheet)
        frame.insert(1, "source_row", np.arange(2, len(frame) + 2))
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={
        "TradDt": "valuation_date",
        "FinInstrmTp": "instrument_type",
        "FinInstrmId": "instrument_id",
        "TckrSymb": "ticker",
        "XpryDt": "expiry_date",
        "StrkPric": "strike_price",
        "OptnTp": "option_type",
        "FinInstrmNm": "contract_name",
        "ClsPric": "market_price",
        "UndrlygPric": "spot_price",
        "SttlmPric": "settlement_price",
        "OpnIntrst": "open_interest",
        "TtlTradgVol": "traded_volume",
        "NewBrdLotQty": "lot_size",
    })

    raw["valuation_date"] = pd.to_datetime(raw["valuation_date"], errors="coerce")
    raw["expiry_date"] = pd.to_datetime(raw["expiry_date"], errors="coerce")
    numeric_columns = [
        "strike_price", "market_price", "spot_price", "settlement_price",
        "open_interest", "traded_volume", "lot_size",
    ]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["days_to_expiry"] = (raw["expiry_date"] - raw["valuation_date"]).dt.days
    # Section 2.2 requires settlement prices for IV.
    raw["option_close_price"] = raw["market_price"]
    raw["market_price"] = raw["settlement_price"]
    raw["time_to_expiry_years"] = raw["days_to_expiry"] / 365
    raw["moneyness_strike_to_spot"] = raw["strike_price"] / raw["spot_price"]

    # Use the near month when it has more than seven days left; otherwise use
    # the next listed monthly expiry, as required by the assignment.
    is_option = raw["instrument_type"].eq("STO")
    selected_expiries = {}
    for valuation_date, group in raw.loc[is_option].groupby("valuation_date"):
        expiries = sorted(group.loc[group["expiry_date"] > valuation_date, "expiry_date"].dropna().unique())
        if not expiries:
            selected_expiries[valuation_date] = pd.NaT
            continue
        near_month = pd.Timestamp(expiries[0])
        selected_expiries[valuation_date] = (
            near_month
            if (near_month - valuation_date).days > 7
            else (pd.Timestamp(expiries[1]) if len(expiries) > 1 else pd.NaT)
        )
    raw["selected_expiry_date"] = raw["valuation_date"].map(selected_expiries)

    # Attach the assignment risk-free rate and trailing dividend yield.
    bsm_rates = valuation_rates.loc[
        valuation_rates["maturity_days"].eq(91),
        ["target_date", "continuous_rate"],
    ].rename(columns={
        "target_date": "valuation_date",
        "continuous_rate": "risk_free_rate",
    })
    raw = raw.merge(bsm_rates, on="valuation_date", how="left")
    raw["ttm_dividend_per_share"] = raw["valuation_date"].map(trailing_dividend_per_share)
    raw["dividend_yield"] = raw["ttm_dividend_per_share"] / raw["spot_price"]

    discounted_spot = raw["spot_price"] * np.exp(
        -raw["dividend_yield"] * raw["time_to_expiry_years"]
    )
    discounted_strike = raw["strike_price"] * np.exp(
        -raw["risk_free_rate"] * raw["time_to_expiry_years"]
    )
    raw["no_arbitrage_lower_bound"] = np.where(
        raw["option_type"].eq("CE"),
        np.maximum(discounted_spot - discounted_strike, 0),
        np.where(
            raw["option_type"].eq("PE"),
            np.maximum(discounted_strike - discounted_spot, 0),
            np.nan,
        ),
    )
    raw["no_arbitrage_upper_bound"] = np.where(
        raw["option_type"].eq("CE"),
        discounted_spot,
        np.where(raw["option_type"].eq("PE"), discounted_strike, np.nan),
    )
    raw["exclusion_reason"] = ""

    def add_reason(mask, reason):
        separator = np.where(raw.loc[mask, "exclusion_reason"].eq(""), "", "; ")
        raw.loc[mask, "exclusion_reason"] += separator + reason

    add_reason(~is_option, "futures contract")
    add_reason(is_option & ~raw["option_type"].isin(["CE", "PE"]), "invalid option type")
    add_reason(is_option & raw["strike_price"].le(0), "missing or non-positive strike")
    add_reason(is_option & raw["spot_price"].le(0), "missing or non-positive spot")
    add_reason(is_option & raw["market_price"].le(0), "missing or non-positive settlement price")
    add_reason(
        is_option & raw["expiry_date"].ne(raw["selected_expiry_date"]),
        "not assignment-selected expiry",
    )
    add_reason(is_option & raw["traded_volume"].fillna(0).le(0), "no trading volume")
    add_reason(
        is_option & ~raw["moneyness_strike_to_spot"].between(0.90, 1.10, inclusive="both"),
        "outside 0.90 to 1.10 moneyness",
    )
    add_reason(
        is_option & raw["market_price"].lt(raw["no_arbitrage_lower_bound"] - 1e-9),
        "below no-arbitrage lower bound",
    )
    # The upper bound is retained as an additional diagnostic. Section 2.2 only
    # mandates deletion for lower-bound violations.
    raw["upper_bound_violation"] = (
        is_option & raw["market_price"].gt(raw["no_arbitrage_upper_bound"] + 1e-9)
    )

    raw["included_for_bsm"] = raw["exclusion_reason"].eq("")
    clean_columns = [
        "source_sheet", "source_row", "valuation_date", "expiry_date",
        "instrument_id", "contract_name", "option_type", "strike_price",
        "spot_price", "market_price", "option_close_price", "settlement_price", "traded_volume",
        "open_interest", "lot_size", "days_to_expiry", "time_to_expiry_years",
        "moneyness_strike_to_spot", "risk_free_rate", "ttm_dividend_per_share", "dividend_yield",
        "no_arbitrage_lower_bound", "no_arbitrage_upper_bound",
        "upper_bound_violation",
    ]
    clean = raw.loc[raw["included_for_bsm"], clean_columns].copy()
    clean = clean.sort_values(
        ["valuation_date", "expiry_date", "option_type", "strike_price"]
    ).reset_index(drop=True)

    exclusions = raw.loc[~raw["included_for_bsm"], [
        "source_sheet", "source_row", "valuation_date", "instrument_type",
        "instrument_id", "contract_name", "option_type", "expiry_date",
        "selected_expiry_date", "strike_price", "spot_price", "market_price",
        "traded_volume", "moneyness_strike_to_spot", "risk_free_rate",
        "ttm_dividend_per_share", "dividend_yield", "no_arbitrage_lower_bound", "no_arbitrage_upper_bound",
        "upper_bound_violation",
        "exclusion_reason",
    ]].copy()

    summary = raw.groupby("valuation_date").agg(
        source_rows=("instrument_id", "size"),
        futures_rows=("instrument_type", lambda x: x.eq("STF").sum()),
        option_rows=("instrument_type", lambda x: x.eq("STO").sum()),
        selected_expiry_options=("expiry_date", lambda x: (x == raw.loc[x.index, "selected_expiry_date"]).sum()),
        zero_volume_options=("traded_volume", lambda x: ((x.fillna(0) <= 0) & is_option.loc[x.index]).sum()),
        within_moneyness_options=("moneyness_strike_to_spot", lambda x: (x.between(0.90, 1.10, inclusive="both") & is_option.loc[x.index]).sum()),
        bsm_ready_options=("included_for_bsm", "sum"),
    ).reset_index()

    # Sequential attrition required by the assignment data appendix.
    audit_rows = []
    for valuation_date, date_rows in raw.groupby("valuation_date"):
        current = date_rows["instrument_type"].eq("STO")
        start = int(current.sum())
        audit_rows.append([valuation_date, "Raw stock options", 0, start])
        rules = [
            ("Wrong expiry", date_rows["expiry_date"].eq(date_rows["selected_expiry_date"])),
            ("Zero volume", date_rows["traded_volume"].fillna(0).gt(0)),
            ("Outside 0.90 to 1.10 moneyness", date_rows["moneyness_strike_to_spot"].between(0.90, 1.10, inclusive="both")),
            ("Below no-arbitrage lower bound", date_rows["market_price"].ge(date_rows["no_arbitrage_lower_bound"] - 1e-9)),
        ]
        for step, passes in rules:
            dropped = int((current & ~passes).sum())
            current &= passes
            audit_rows.append([valuation_date, step, dropped, int(current.sum())])
        audit_rows.append([
            valuation_date,
            "Above upper bound (diagnostic only)",
            int((current & date_rows["upper_bound_violation"]).sum()),
            int(current.sum()),
        ])
    cleaning_audit = pd.DataFrame(
        audit_rows, columns=["valuation_date", "step", "dropped_at_step", "remaining"]
    )
    total_rows = []
    for step, group in cleaning_audit.groupby("step", sort=False):
        total_rows.append(["TOTAL", step, int(group["dropped_at_step"].sum()), int(group["remaining"].sum())])
    cleaning_audit = pd.concat([
        cleaning_audit,
        pd.DataFrame(total_rows, columns=cleaning_audit.columns),
    ], ignore_index=True)

    option_spot = raw.loc[is_option].groupby("valuation_date", as_index=False).agg(
        option_spot=("spot_price", "first"),
        option_spot_min=("spot_price", "min"),
        option_spot_max=("spot_price", "max"),
    )
    spot_reconciliation = option_spot.merge(
        valuation_prices[["target_date", "close_price"]].rename(
            columns={"target_date": "valuation_date", "close_price": "equity_close"}
        ),
        on="valuation_date",
        how="left",
    )
    spot_reconciliation["option_minus_equity"] = (
        spot_reconciliation["option_spot"] - spot_reconciliation["equity_close"]
    )

    assert len(raw) == 1541, "Raw F&O row count changed"
    assert is_option.sum() == 1505 and (~is_option).sum() == 36, "Instrument counts changed"
    assert raw["valuation_date"].nunique() == 12, "Expected all 12 valuation dates"
    assert not raw.duplicated(["valuation_date", "instrument_id"]).any(), "Duplicate contracts found"
    assert clean["risk_free_rate"].notna().all(), "Missing BSM rate"
    assert clean["ttm_dividend_per_share"].eq(0.50).all(), "Unexpected TTM dividend"
    assert not clean.duplicated(["valuation_date", "instrument_id"]).any(), "Duplicate clean contracts"
    total_audit = cleaning_audit.loc[cleaning_audit["valuation_date"].eq("TOTAL")]
    assert int(total_audit.loc[total_audit["step"].eq("Below no-arbitrage lower bound"), "remaining"].iloc[0]) == len(clean)

    return raw, clean, exclusions, summary, cleaning_audit, spot_reconciliation


def save_line_chart(data, x, y, title, y_label, file_name, color, note=None):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(data[x], data[y], color=color, linewidth=1.25)
    if data[y].min() < 0 < data[y].max():
        ax.axhline(0, color="#64748b", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    if note:
        fig.text(0.01, 0.01, note, fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT / file_name, dpi=200, bbox_inches="tight")
    # Keep the figure open so VS Code displays it after processing.


def create_charts(equity, all_rates):
    save_line_chart(
        equity,
        "date",
        "adjusted_price",
        "Biocon Corporate-Action-Adjusted Share Price (10 Years)",
        "Adjusted price (INR)",
        "biocon_adjusted_price_10_years.png",
        "#7c3aed",
        "Source: user-supplied Adjusted Prices; adjusted for applicable corporate actions.",
    )

    returns = equity[["date", "model_return"]].dropna().copy()
    returns["return_percent"] = returns["model_return"] * 100
    save_line_chart(
        returns,
        "date",
        "return_percent",
        "Biocon Adjusted Daily Returns (10 Years)",
        "Adjusted return (%)",
        "biocon_adjusted_returns_10_years.png",
        "#2563eb",
    )

    required_period = all_rates[
        all_rates["date"].between(VALUATION_DATES.min(), VALUATION_DATES.max())
    ]
    for maturity, color in [(91, "#0f766e"), (364, "#b45309")]:
        rates = required_period[required_period["maturity_days"] == maturity]
        save_line_chart(
            rates,
            "date",
            "yield_percent",
            f"RBI {maturity}-Day Treasury Bill Yield",
            "Implicit yield at cut-off price (%)",
            f"rbi_{maturity}_day_tbill_rates_required_period.png",
            color,
        )


def main():
    # Load and validate the user-supplied equity history.
    equity = load_equity()

    valuation_prices = latest_on_or_before(
        equity, VALUATION_DATES, ["close_price", "adjusted_price"]
    )
    year_end_prices = latest_on_or_before(
        equity, YEAR_ENDS, ["close_price", "adjusted_price"]
    )
    year_end_prices.insert(0, "fiscal_year", [2022, 2023, 2024, 2025, 2026])

    # Load the 91-day and 364-day RBI auction series.
    rate_91 = load_rate(RATE_91_FILE, 91)
    rate_364 = load_rate(RATE_364_FILE, 364)
    all_rates = pd.concat([rate_91, rate_364], ignore_index=True)

    # Match each valuation date to the latest available auction rate.
    selected_rates = []
    for maturity, group in all_rates.groupby("maturity_days"):
        selected = latest_on_or_before(group, VALUATION_DATES, ["yield_percent"])
        selected["maturity_days"] = maturity
        selected["simple_rate_decimal"] = selected["yield_percent"] / 100
        selected["continuous_rate"] = np.log1p(selected["simple_rate_decimal"])
        selected_rates.append(selected)

    valuation_rates = pd.concat(selected_rates, ignore_index=True)
    valuation_rates = valuation_rates.sort_values(["target_date", "maturity_days"])

    # -------------------------------------------------------------------------
    # OPTIONS PREPROCESSING
    # -------------------------------------------------------------------------
    (
        options_raw,
        options_clean,
        option_exclusions,
        options_summary,
        options_cleaning_audit,
        option_spot_reconciliation,
    ) = preprocess_options(
        valuation_rates, valuation_prices
    )

    # Save all four charts and display them together on screen.
    create_charts(equity, all_rates)
    plt.show()

    # Save the cleaned datasets used by the models.
    equity.to_csv(OUT / "biocon_equity_adjusted.csv", index=False, date_format="%Y-%m-%d")
    valuation_prices.to_csv(OUT / "valuation_date_equity_prices.csv", index=False, date_format="%Y-%m-%d")
    year_end_prices.to_csv(OUT / "fiscal_year_end_equity_prices.csv", index=False, date_format="%Y-%m-%d")
    all_rates.to_csv(OUT / "rbi_tbill_auction_yields.csv", index=False, date_format="%Y-%m-%d")
    valuation_rates.to_csv(OUT / "valuation_date_tbill_rates.csv", index=False, date_format="%Y-%m-%d")
    options_raw.to_csv(OUT / "options_all_contracts.csv", index=False, date_format="%Y-%m-%d")
    options_clean.to_csv(OUT / "options_bsm_ready.csv", index=False, date_format="%Y-%m-%d")
    option_exclusions.to_csv(OUT / "options_exclusion_log.csv", index=False, date_format="%Y-%m-%d")
    options_summary.to_csv(OUT / "options_preprocessing_summary.csv", index=False, date_format="%Y-%m-%d")
    options_cleaning_audit.to_csv(OUT / "options_sequential_cleaning_audit.csv", index=False, date_format="%Y-%m-%d")
    option_spot_reconciliation.to_csv(OUT / "option_spot_reconciliation.csv", index=False, date_format="%Y-%m-%d")

    print(f"Equity observations processed: {len(equity)}")

    print(f"Duplicate dates: {equity['date'].duplicated().sum()}")
    print(f"91-day rate observations: {len(rate_91)}")
    print(f"364-day rate observations: {len(rate_364)}")
    print(f"Raw option-chain rows: {len(options_raw)}")
    print(f"BSM-ready option rows: {len(options_clean)}")
    print(f"Excluded option/futures rows: {len(option_exclusions)}")
    print(f"Processed data saved to: {OUT}")
    print(f"Charts saved to: {FIGURE_OUT}")

    print("\nFiscal year-end prices")
    print(year_end_prices.to_string(index=False))

    print("\nValuation-date equity prices")
    print(valuation_prices.to_string(index=False))

    print("\nValuation-date T-bill rates")
    print(valuation_rates.to_string(index=False))

    print("\nOptions preprocessing summary")
    print(options_summary.to_string(index=False))

    print("\nSequential options cleaning audit (total)")
    print(options_cleaning_audit.loc[
        options_cleaning_audit["valuation_date"].eq("TOTAL")
    ].to_string(index=False))

    print("\nOption spot reconciliation")
    print(option_spot_reconciliation.to_string(index=False))


if __name__ == "__main__":
    main()
