# FRAM Biocon

Financial Risk Analytics and Management project for Biocon Limited. The project prepares market and financial inputs for GARCH, Black-Scholes, Merton structural credit-risk, Altman Z-score, and machine-learning analysis.

## Current status

Market-input preprocessing is complete for the supplied Biocon equity history, RBI Treasury-bill auction files, and the 12-sheet NSE options workbook. The script validates the data, selects the required dates and rates, prepares actively traded option contracts for BSM, records every contract exclusion, saves clean CSV files, and prints compact review tables.

## Project structure

```text
data/
  raw/
    equity/       Biocon adjusted daily prices
    financials/   FY2022-FY2026 financial statements
    options/      NSE option-chain data
    rates/        RBI 91-day and 364-day T-bill auctions
  processed/      Generated clean CSV files
src/
  preprocess_market_inputs.py
notebooks/
outputs/
final/
```

Raw workbooks and generated data are kept locally and are not committed to GitHub.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
pip install -r requirements.txt
```

## Run market-input preprocessing

From the project root:

```powershell
python src/preprocess_market_inputs.py
```

The script reads:

- `data/raw/equity/Biocon_data.xlsx`
- `data/raw/rates/Auctions of 91-Day Government of India Treasury Bills.xlsx`
- `data/raw/rates/Auctions of 364-Day Government of India Treasury Bills.xlsx`
- `data/raw/options/Biocon_options-_data.xlsx`

It creates:

- `data/processed/biocon_equity_adjusted.csv`
- `data/processed/fiscal_year_end_equity_prices.csv`
- `data/processed/valuation_date_equity_prices.csv`
- `data/processed/rbi_tbill_auction_yields.csv`
- `data/processed/valuation_date_tbill_rates.csv`
- `data/processed/options_all_contracts.csv`
- `data/processed/options_bsm_ready.csv`
- `data/processed/options_exclusion_log.csv`
- `data/processed/options_preprocessing_summary.csv`
- `data/processed/options_sequential_cleaning_audit.csv`
- `data/processed/option_spot_reconciliation.csv`

The terminal also displays fiscal-year-end equity prices, the 12 assignment valuation-date prices, matching T-bill rates, and a date-by-date options preprocessing summary. Four charts are saved under `outputs/figures/` and displayed on screen.

## Modeling conventions

- Preserve the user-supplied adjusted equity prices and returns.
- Use the latest RBI auction yield available by each valuation cutoff.
- Convert the published percentage yield with `R / 100`, then calculate the continuous rate as `ln(1 + R)`.
- Use fiscal years ending 31 March for the Altman Z-score inputs.
- Keep futures and rejected option rows in the audit files. For BSM, apply the assignment rules: near-month expiry when more than seven calendar days remain, otherwise the next expiry; positive volume; `0.90 <= K/S <= 1.10`; and the applicable call or put no-arbitrage bounds. The lower bound follows the assignment formula, while the standard theoretical upper bound is retained as an additional validation.

## Next steps

Part A IV is implemented in `src/implied_volatility.py`. Run it after preprocessing. Market option prices use settlement prices as required by the assignment. The own Newton-Raphson solver starts at 30%, falls back to bisection on [1%, 300%], and checks an absolute price error below INR 0.0001. It exports contract IVs, monthly ATM IVs and put-call parity diagnostics to `data/processed/`, plus the September smile and monthly ATM chart to `outputs/figures/`. ATM IV interpolates calls and puts separately and averages them without extrapolation.

- Verify the August 2026 option settlement-price anomaly against the archived NSE source and discuss it in the report.
- Collect dated shares outstanding and borrowings, then calibrate the Merton model.
- Build GARCH, Altman Z-score and ML modules.
- Assemble the final Google Colab notebook after the model modules stabilize.
- Produce the final report and supporting outputs.
