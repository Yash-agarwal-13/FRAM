# FRAM Biocon

Financial Risk Analytics and Management project for Biocon Limited. The project prepares market and financial inputs for GARCH, Black-Scholes, Merton structural credit-risk, Altman Z-score, and machine-learning analysis.

## Current status

Market preprocessing and Parts A-E are implemented, including implied volatility, Merton calibration, GARCH, Random Forest, rating-implied PD, Altman accounting scores and the final route comparison. The code preserves valuation-date cutoffs, exports audit tables, and rebuilds the figures from five consolidated input workbooks.

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
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/preprocess_market_inputs.py
python src/implied_volatility.py
python src/merton_model.py
python src/garch_model.py
python src/ml_volatility.py
python src/rating_model.py
python src/altman_model.py
python src/part_e_comparison.py
```

The complete workflow reads only these five files from `data/raw/input_files` (or from the current Colab upload folder):

- `Biocon_data_updated.xlsx`
- `Financial_Statements (2) (1).xlsx`
- `Biocon_options-_data.xlsx`
- `Auctions of 91-Day Government of India Treasury Bills.xlsx`
- `Auctions of 364-Day Government of India Treasury Bills.xlsx`

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

- Add filing-level citations for the dated Merton debt and share inputs.
- Assemble the final Colab notebook from the audited modules.
- Produce the final report and supporting outputs.

### Implied-volatility validation

Part A keeps the assignment-required Newton-Raphson solver with bisection fallback as the authoritative calculation. `py_vollib` is used only as the brief's independent check. Running `src/implied_volatility.py` writes the contract-level comparison to `data/processed/iv_solver_validation.csv`, produces `outputs/figures/atm_iv_own_vs_py_vollib.png`, and stops if any absolute IV difference exceeds `0.00001`.

### Merton model

Run `python src/merton_model.py` after preprocessing and IV calculation. It combines ATM IV with dated shares, closing equity prices, latest-public consolidated borrowings and 364-day RBI rates. SciPy `fsolve` remains the authoritative assignment solver; `ItsSypher/merton==1.0.2` is used only to check V, sigma_V, DD and raw risk-neutral PD. The comparison is written to `merton_solver_validation.csv`; the remaining Merton outputs and figures are also rebuilt by the script.

### GARCH(1,1)

Run `python src/garch_model.py` after the three scripts above. It fits normal and Student-t GARCH(1,1) models on the previous 1,000 adjusted daily log returns available through each valuation date. The Student-t specification is used for the Part B Merton route because it has lower BIC on all 12 dates. Parameters, standard errors, persistence, long-run variance, half-life, information criteria and diagnostics are exported to `garch_model_comparison.csv`; dated volatility forecasts and GARCH-based Merton results are exported separately. Annualisation follows the assignment's 252-day average-variance formula.

### Machine-learning volatility

Run `python src/ml_volatility.py` for the final Random Forest route. It builds the assignment features from Biocon returns, India VIX, Nifty 50, the as-of-date GARCH volatility and Biocon result months. Realised volatility follows the brief exactly as the square root of 252 times mean squared returns; the model forecasts its logarithm for 21 and 252 days. Hyperparameters are chosen on the chronological validation period, labels crossing a split boundary are purged, and the model is refitted using only fully observable targets at each valuation date. Out-of-sample R-squared is calculated relative to the matching GARCH forecast. The 252-day forecast is used in Merton because it matches the one-year structural horizon.

Exponential recency weighting is retained as a sensitivity test with 252, 504, 756 and 1,260-trading-day half-lives. It is not used in the locked final specification; equal weights preserve the pre-specified model while the decay results remain available in the validation-tuning output.

### Rating-implied PD

Run `python src/rating_model.py` after the Merton and GARCH scripts. It uses Biocon's CRISIL AA+/Stable rating and the FY2026 CRISIL one-year default study. The published AA default rate of 0.02% and the adjacent AAA rate of 0.00% are both raised to the assignment's 0.03% floor; the adjacent A rate is 0.07%. This gives a 3.00 bp point/lower-edge PD and a 4.5826 bp upper edge. The script reverse-solves the Merton equity and DD equations under risk-neutral and physical drift, independently checks every solution with bounded least squares, and compares the resulting equity-volatility band with ATM IV and Student-t GARCH.

### Altman accounting route

Run `python src/altman_model.py` after market preprocessing and the Merton scripts. It calculates the original Z, Z-double-prime and EM scores for FY2022-FY2026 from the consolidated statements. In accordance with the brief, other equity is used as retained earnings under Ind AS, and EBIT is profit before tax plus finance costs. Fiscal year-end market equity uses the dated closing price and exact shares outstanding. The FY2026 EM score is mapped through Altman's published rating table and then to the matching CRISIL one-year default rate; the script reverse-solves the Merton equations under risk-neutral and physical drift and verifies each solution independently.

### Part E comparison

Run `python src/part_e_comparison.py` last. It fills `data/processed/part_e_summary_table.csv` with all five routes, including equity volatility, asset volatility, distance to default, reported PD, measure basis and risk perspective. It also exports route-gap calculations and separate equity-volatility and PD comparison figures.
