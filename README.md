# FRAM Biocon

Financial Risk Analytics and Management project for Biocon Limited. The project prepares market and financial inputs for GARCH, Black-Scholes, Merton structural credit-risk, Altman Z-score, and machine-learning analysis.

## Current status

Market-input preprocessing is complete for the supplied Biocon equity history and RBI Treasury-bill auction files. The script validates the data, selects fiscal-year-end and assignment valuation-date observations, converts published yields to model rates, saves clean CSV files, and prints compact review tables.

## Project structure

```text
FRAM_BIOCON/
├── data/               # Input datasets
├── src/                # Python source code
├── notebooks/          # Independent Jupyter notebooks
├── outputs/            # Generated figures and results
├── final/              # Final consolidated notebook
├── .gitignore          # Git exclusions
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

### Development Workflow

The project follows a modular development approach:

1. Develop and test individual models independently.
2. Track changes using Git feature branches.
3. Merge validated work into the `develop` branch.
4. Integrate and test the complete workflow.
5. Merge the validated version into `main`.
6. Prepare the final Jupyter notebook for Google Colab.

### Environment Setup

Create a Python virtual environment:

```powershell
py -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
python src/preprocess_market_inputs.py
```

The script reads:

- `data/raw/equity/Biocon_data.xlsx`
- `data/raw/rates/Auctions of 91-Day Government of India Treasury Bills.xlsx`
- `data/raw/rates/Auctions of 364-Day Government of India Treasury Bills.xlsx`

It creates:

- `data/processed/biocon_equity_adjusted.csv`
- `data/processed/fiscal_year_end_equity_prices.csv`
- `data/processed/valuation_date_equity_prices.csv`
- `data/processed/rbi_tbill_auction_yields.csv`
- `data/processed/valuation_date_tbill_rates.csv`

The terminal also displays fiscal-year-end equity prices, the 12 assignment valuation-date prices, and the matching 91-day and 364-day T-bill rates.

## Modeling conventions

- Preserve the user-supplied adjusted equity prices and returns.
- Use the latest RBI auction yield available by each valuation cutoff.
- Convert the published percentage yield with `R / 100`, then calculate the continuous rate as `ln(1 + R)`.
- Use fiscal years ending 31 March for the Altman Z-score inputs.

## Next steps

- Preprocess NSE option contracts and document exclusions.
- Collect dated shares outstanding and borrowings for Merton analysis.
- Build GARCH, BSM, Merton, Altman Z-score, and ML notebooks.
- Produce the final report and supporting outputs.
