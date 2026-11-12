"""Build the self-contained Google Colab submission notebook."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'final' / 'FRAM_Biocon_Final.ipynb'
MODULES = [
    'input_data.py',
    'preprocess_market_inputs.py',
    'implied_volatility.py',
    'merton_model.py',
    'garch_model.py',
    'ml_volatility.py',
    'rating_model.py',
    'altman_model.py',
    'part_e_comparison.py',
]


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def source_cell(filename):
    source = (ROOT / 'src' / filename).read_text(encoding='utf-8')
    return code(f'%%writefile src/{filename}\n{source}')


cells = [
    markdown(r"""
# Four Routes to Default Probability: Biocon Limited

**Financial Risk Analytics and Management**  
**Company:** Biocon Limited (NSE: BIOCON)  
**Main valuation date:** 30 September 2026

This notebook is the complete reproducible submission. It starts with the five consolidated raw workbooks, preserves valuation-date information cutoffs, calculates every required model, runs the independent checks, and ends with the Part E comparison.

## Research design

The assignment compares four conceptually different routes to default risk:

1. **Part A, structural market route:** option-implied equity volatility is converted into firm asset value and asset volatility using the Merton model.
2. **Part B, forecast volatility routes:** Student_t GARCH and Random Forest forecasts replace option-implied volatility in the same structural framework.
3. **Part C, rating route:** the CRISIL one-year default rate is converted to distance to default and reverse-solved into the volatility required by Merton.
4. **Part D, accounting route:** Altman Z, Z_double_prime and emerging-market scores are calculated from consolidated financial statements; the EM rating equivalent is mapped to a PD and reverse-solved through Merton.

Part E compares the routes without treating them as interchangeable. Market measures are point_in_time, ratings are primarily through_the_cycle, and Altman is an accounting point_in_time measure.
"""),
    markdown(r"""
## Inputs and reproducibility

Upload exactly these five files when prompted:

1. `Biocon_data_updated.xlsx`
2. `Financial_Statements (2) (1).xlsx`
3. `Biocon_options-_data.xlsx`
4. `Auctions of 91-Day Government of India Treasury Bills.xlsx`
5. `Auctions of 364-Day Government of India Treasury Bills.xlsx`

The market workbook contains Biocon prices and adjusted returns, Nifty 50, India VIX, result months, dated shares outstanding and dividends. The financial workbook contains annual statements, quarterly balance sheets, rating inputs and the Altman rating map. Options and both exact-tenor RBI auction series remain separate raw workbooks.

All monetary statement values are in **INR million** unless stated otherwise. Published T_bill yields remain in percent in the raw tables and are converted using `r = ln(1 + R/100)`.
"""),
    code(r"""
%pip install -q arch==8.0.0 statsmodels==0.15.0 scikit-learn==1.9.1 py-vollib==1.0.12 merton==1.0.2 openpyxl==3.1.5
"""),
    code(r"""
from pathlib import Path
import sys

try:
    from google.colab import files
except ImportError as exc:
    raise RuntimeError('Run this upload cell in Google Colab.') from exc

expected_files = {
    'Biocon_data_updated.xlsx',
    'Financial_Statements (2) (1).xlsx',
    'Biocon_options-_data.xlsx',
    'Auctions of 91-Day Government of India Treasury Bills.xlsx',
    'Auctions of 364-Day Government of India Treasury Bills.xlsx',
}
uploaded = files.upload()
missing = expected_files.difference(uploaded)
assert not missing, f'Missing required files: {sorted(missing)}'

Path('src').mkdir(exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path.cwd() / 'src'))
print('All five input files are available.')
"""),
    markdown(r"""
## Embedded model code

The next cells write the audited modules bundled inside this notebook. This keeps the submission self-contained while retaining short, separately testable model sections. The modules do not download or silently replace data.
"""),
]

cells.extend(source_cell(name) for name in MODULES)

cells.extend([
    markdown(r"""
## Data preparation and option-chain cleaning

Biocon's supplied adjusted prices and adjusted returns are used directly; they are not adjusted a second time. Market capitalisation uses the contemporaneous published close. The single missing India VIX value on the 2 March 2024 special Saturday session is carried forward from the latest prior observation.

For each option valuation date, the near-month expiry is retained when more than seven calendar days remain; otherwise the next expiry is used. Zero-volume contracts are removed, moneyness is restricted to `0.90 <= K/S <= 1.10`, and contracts violating the BSM no-arbitrage lower bound are excluded. The upper bound remains a reported diagnostic. Every exclusion is traceable.
"""),
    code("%run src/preprocess_market_inputs.py"),
    markdown(r"""
## Part A: implied volatility and Merton structural model

### Black_Scholes_Merton implied volatility

For calls and puts with continuous dividend yield `q`:

\[
c=S e^{-qT}N(d_1)-K e^{-rT}N(d_2), \qquad
p=K e^{-rT}N(-d_2)-S e^{-qT}N(-d_1)
\]

\[
d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T}.
\]

The authoritative solver is Newton_Raphson starting from 30%, with bisection on `[1%, 300%]` as fallback. Each solution must reprice within INR 0.0001. `py_vollib` is check_only, as required by the assignment. Call and put ATM IVs are interpolated separately between strikes bracketing `K/S=1` and then averaged.

### Merton equations

Equity is treated as a European call on firm assets:

\[
E=VN(d_1)-De^{-rT}N(d_2), \qquad
\sigma_E E=N(d_1)\sigma_V V.
\]

The default point is `D = short_term_debt + 0.5 x long_term_debt`, `T=1`, and equity value is close price times dated shares outstanding. SciPy `fsolve` jointly estimates `V` and `sigma_V`, starting at `V_0=E+D` and `sigma_V0=sigma_E E/(E+D)`. Every solution is substituted back into both equations and checked independently with the assignment-listed `ItsSypher/merton` package.

Risk_neutral distance to default uses `mu=r`; the physical case uses the annualised historical mean adjusted log return. Reported PD is subject to the mandatory 3 bp floor.
"""),
    code("%run src/implied_volatility.py"),
    code("%run src/merton_model.py"),
    markdown(r"""
### Part A interpretation

On 30 September 2026, ATM IV is approximately **24.915%**, solved asset volatility is **21.653%**, and risk_neutral DD is **9.281**. Raw PD is below the reporting threshold, so PD is shown at the assignment floor of **3 bp**. This high DD does not mean Biocon has no borrowings; it reflects market-implied asset value being large relative to the KMV default point. Observed `D/V` is about 13.9% at the main date.

August 2026 remains a disclosed data-quality concern because call and put ATM IV differ materially. No unassigned discretionary filter is applied to conceal that observation.
"""),
    markdown(r"""
## Part B: volatility forecasting

### GARCH(1,1)

Daily adjusted log returns follow:

\[
\sigma_t^2=\omega+\alpha u_{t-1}^2+\beta\sigma_{t-1}^2.
\]

Normal and Student_t errors are estimated separately on the previous 1,000 trading days at each valuation date. Student_t is selected because it has lower BIC on all 12 dates. Reported diagnostics include standard errors, persistence, long_run variance, half_life, likelihood, AIC, BIC, ARCH_LM, and Ljung_Box tests on standardised and squared standardised residuals.

The assignment annualisation uses expected average variance over 252 days:

\[
\bar{\sigma}^2(T)=V_L+\frac{1-e^{-aT}}{aT}[V(0)-V_L],
\qquad \sigma_E^{GARCH}=\sqrt{252\bar{\sigma}^2(T)}.
\]
"""),
    code("%run src/garch_model.py"),
    markdown(r"""
### Random Forest

The target is log annualised realised volatility over both 21 and 252 future trading days:

\[
RV_{t,h}=\sqrt{\frac{252}{h}\sum_{i=1}^{h}u_{t+i}^2}.
\]

Features known at time `t` are realised volatility over 5, 21 and 63 days; returns over 1, 5 and 21 days; GARCH volatility; India VIX; Nifty 50 21-day return; and a results-month dummy. Training, validation and test periods are chronological 60%/20%/20%, and overlapping forward labels are purged at split boundaries. Hyperparameters are selected on validation data only. Equal weighting is the locked specification; recency weighting is retained only as sensitivity analysis.

Random Forest is used because the sample is modest, nonlinear interactions are plausible, ensemble averaging is robust, and feature importance is directly reportable. The final test result is reported honestly even when GARCH performs better.
"""),
    code("%run src/ml_volatility.py"),
    markdown(r"""
### Part B interpretation

At the main date, Student_t GARCH gives `sigma_E` about **32.690%**, `sigma_V` **28.410%**, and DD **7.014**. Random Forest gives `sigma_E` about **31.480%**, `sigma_V` **27.359%**, and DD **7.295**. Both raw PDs remain below 3 bp.

The 252-day Random Forest does **not** beat GARCH on the untouched test sample: its GARCH-relative out_of_sample R_squared is negative. This is a substantive result, not a reason to revise the model after observing the test period.
"""),
    markdown(r"""
## Part C: rating-implied PD

Biocon's CRISIL AA+ Stable rating maps to the CRISIL study's AA category. The published one-year rates are AAA 0.00%, AA 0.02%, and A 0.07%. The assignment floor raises the point and better category to 0.03%. Geometric adjacent-category boundaries produce a point/lower edge of 3 bp and an upper edge of `sqrt(3 x 7) = 4.5826` bp.

For each target PD, `DD=-N^{-1}(PD)`. Holding 30 September 2026 equity, debt, rate and horizon fixed, the equity equation and target-DD equation are jointly reverse-solved for asset value and asset volatility. The resulting equity volatility is then derived from the Merton volatility relation.
"""),
    code("%run src/rating_model.py"),
    markdown(r"""
### Part C interpretation

The risk_neutral CRISIL-consistent equity-volatility band is approximately **63.124% to 65.024%**, well above both observed ATM IV and GARCH. This does not prove that one route is wrong. The structural market routes are point_in_time, while agency ratings and historical rating transition/default studies are more through_the_cycle and incorporate qualitative and prospective considerations.
"""),
    markdown(r"""
## Part D: Altman accounting models

The original listed-manufacturer score is:

\[
Z=1.2X_1+1.4X_2+3.3X_3+0.6X_4+X_5.
\]

For the non_US/emerging-market comparison:

\[
Z''=6.56X_1+3.26X_2+6.72X_3+1.05X_4, \qquad EM=Z''+3.25.
\]

`X1` is working capital/total assets; `X2` is reserves and surplus/total assets, implemented as Other equity under the assignment's Ind AS instruction; `X3` is EBIT/total assets with EBIT defined consistently as PBT plus finance cost; `X4` uses market equity/total liabilities for original Z and book equity/total liabilities for Z_double_prime; `X5` is sales/total assets.

EM is the preferred accounting model for Biocon because it is the emerging-market specification. The unrounded EM score is mapped through Altman's rating-equivalent intervals and then to the same CRISIL default study used in Part C.
"""),
    code("%run src/altman_model.py"),
    markdown(r"""
### Part D interpretation

For FY2026, original Z is approximately **2.428, grey**, Z_double_prime is **3.597, safe**, and EM is **6.847, safe**. EM maps to an A equivalent, four notches below the observed AA+ agency rating. The corresponding 7 bp PD gives reverse-solved `sigma_E` about **67.098%**, `sigma_V` **58.319%**, and DD **3.195** under risk_neutral drift.

The difference is economically plausible: Altman reacts mechanically to accounting ratios and weak earnings, whereas the agency assessment also includes liquidity, prospective business conditions, qualitative factors and possible group support.
"""),
    markdown(r"""
## Part E: synthesis

The final table places every route on the same 30 September 2026 page while preserving its interpretation. For the structural routes, PD is the reported floored value. Rating and accounting volatilities are reverse-solved values required to make Merton consistent with their target PDs.
"""),
    code("%run src/part_e_comparison.py"),
    code(r"""
import pandas as pd
from IPython.display import display, Image

summary = pd.read_csv('data/processed/part_e_summary_table.csv')
formatted = summary.copy()
formatted['sigma_e'] = (100 * formatted.sigma_e).map(lambda x: f'{x:.3f}%')
formatted['sigma_v'] = (100 * formatted.sigma_v).map(lambda x: f'{x:.3f}%')
formatted['distance_to_default'] = formatted.distance_to_default.map(lambda x: f'{x:.4f}')
formatted['pd_bp'] = formatted.pd_bp.map(lambda x: f'{x:.3f}')
display(formatted)
display(Image('outputs/figures/part_e_volatility_comparison.png'))
display(Image('outputs/figures/part_e_pd_comparison.png'))
"""),
    markdown(r"""
## Overall conclusion

The models agree directionally that Biocon's one-year default risk is low at the valuation date, but they disagree substantially about the volatility consistent with that conclusion.

- ATM IV is the lowest volatility route and produces the highest DD.
- GARCH and Random Forest are higher than ATM IV but still produce raw PD below the reporting floor.
- The CRISIL historical rating calibration requires much higher volatility to reach even its floored rating PD.
- Altman EM is the most conservative route, mapping FY2026 accounting ratios to A and 7 bp.

The ATM_IV minus GARCH volatility gap is negative, approximately **minus 7.78 percentage points**. It may be discussed as a snapshot volatility-risk-premium proxy, but only cautiously because option IV and the one-year GARCH forecast use different data, market microstructure and expectations.

### Limitations

1. The 3 bp floor masks differences among very small structural PD estimates.
2. Merton assumes one zero-coupon debt issue, constant asset volatility and default only at maturity.
3. Option settlement prices and sparse contracts can distort the smile, especially August 2026.
4. GARCH leaves some short-lag mean autocorrelation even though squared-residual diagnostics are satisfactory.
5. Random Forest underperforms GARCH out of sample and should not be presented as superior.
6. Rating and accounting routes are calibrated to different concepts and horizons from market-implied structural PD.

Detailed company filing, RBI, NSE, CRISIL and repository citations should be listed in the accompanying technical report appendix.
"""),
])

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python', 'version': '3'},
        'colab': {'name': OUTPUT.name, 'provenance': []},
    },
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f'Created {OUTPUT} with {len(cells)} cells')
