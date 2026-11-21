"""Build a continuous, self-contained Colab notebook."""
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'final' / 'FRAM_Biocon_Final.ipynb'
MAIN_GUARD = re.compile(
    r'\nif\s+__name__\s*==\s*([\'\"])__main__\1\s*:\s*\n\s+main\(\)\s*\n?'
)

def md(text):
    return nbf.v4.new_markdown_cell(text.strip())

def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata['cellView'] = 'form'
        cell.metadata['jupyter'] = {'source_hidden': True}
    return cell

def source_text(name):
    text = (ROOT / 'src' / name).read_text(encoding='utf-8')
    text = text.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
    return MAIN_GUARD.sub('\n', text)

def model_cell(name, register=None):
    text = source_text(name)
    if register:
        text += f"""

# Make these definitions available to the later model sections.
import sys as _sys
import types as _types
_{register}_module = _types.ModuleType('{register}')
_{register}_module.__dict__.update({{k: v for k, v in globals().copy().items()
                                    if not k.startswith('_')}})
_sys.modules['{register}'] = _{register}_module
"""
    return code(text + '\nmain()')

cells = [
md(r"""
# Four Routes to Default Probability: Biocon Limited

**Financial Risk Analytics and Management**  
**Company:** Biocon Limited, NSE: BIOCON  
**Main valuation date:** 30 September 2026

The notebook follows the assignment in sequence. Theory and assumptions appear immediately before each calculation, and each calculation prints its own tables and figures directly beneath the same cell.
"""),
md(r"""
## Data and assumptions

### Files used

The notebook uses five files already placed in Colab's **Files** panel:

1. `Biocon_data_updated.xlsx`: Biocon prices and returns, Nifty 50, India VIX, result months, dated shares and dividends.
2. `Financial_Statements (2) (1).xlsx`: annual statements, quarterly debt, rating data and Altman mapping.
3. `Biocon_options-_data.xlsx`: NSE option and futures observations for the 12 dates.
4. `Auctions of 91-Day Government of India Treasury Bills.xlsx`: BSM rates.
5. `Auctions of 364-Day Government of India Treasury Bills.xlsx`: one-year Merton rates.

### Corporate-action treatment

- `Adjusted Prices` and `Adjusted Returns` are the authoritative corporate-action-adjusted series. They already remove discontinuities created by splits, bonus issues and similar actions, so they are not adjusted again.
- The contemporaneous unadjusted `Close Price` is used for market capitalisation because equity value equals actual shares outstanding multiplied by the traded price.
- Actual share counts change only on verified allotment dates and are carried forward until the next allotment. They are not inferred from rounded share capital.
- Cash dividends are handled separately in BSM. Trailing-twelve-month dividend per share is divided by spot. The verified events are INR 0.50 per share on 4 July 2025 and 3 July 2026.

### Common assumptions

- Only information publicly available by each valuation date is used.
- Latest applicable debt and shares are carried forward as step functions; no growth forecast is applied.
- Consolidated statements are used and financial values are in INR million.
- RBI yields are selected without using a future auction and converted as $r=\ln(1+R/100)$.
- Merton uses $T=1$, $E=\text{price}\times\text{shares}$ and $D=\text{short-term debt}+0.5\times\text{long-term debt}$.
- Risk-neutral PD, obtained with $\mu=r$, is the primary cross-route result. Wherever the assignment invokes the Merton or inverse-Merton system, the physical specification using the annualised historical mean adjusted return is also calculated and reported.
- CRISIL and Altman supply empirical PD inputs. They are not themselves risk-neutral probabilities; only the associated inverse-Merton calibration is labelled risk-neutral or physical according to the drift used.
- The minimum reported PD is 0.03%, equal to 3 basis points.
"""),
md(r"""
## Model assumptions and applicability to Biocon

The assumptions below are modelling benchmarks rather than literal descriptions of Biocon or its markets. Their limitations determine how the resulting IV, DD and PD estimates should be interpreted.

### Black-Scholes-Merton assumptions

| Assumption | Assessment for Biocon | What may fail if the assumption does not hold |
|---|---|---|
| The share price follows geometric Brownian motion with normally distributed log returns and constant drift and volatility. | Approximate only. Biocon returns can be fat-tailed and volatility changes through time. | A single constant-volatility BSM model cannot price all strikes consistently; the observed volatility smile is evidence of this limitation. |
| Volatility and the risk-free rate remain constant over the option's life. | Constant volatility is restrictive; the latest 91-day RBI rate is a reasonable short-horizon rate approximation. | Changing volatility biases IV interpretation, while an inappropriate rate changes discounting and the recovered IV. |
| The stock pays a known continuous dividend yield. | Approximate. Verified discrete cash dividends are represented by trailing-twelve-month dividend per share divided by spot. | Incorrect dividend treatment changes the forward value, no-arbitrage bound and implied volatility. |
| Markets are frictionless, trading is continuous, short selling is permitted and investors can borrow or lend at the risk-free rate. | Does not hold literally because of bid-ask spreads, taxes, margin requirements, discrete trading and funding constraints. | Exact replication and no-arbitrage pricing may not be attainable, especially for less-liquid contracts. |
| There are no persistent arbitrage opportunities. | Reasonable as a pricing benchmark. | If violated, BSM identities and theoretical price bounds need not hold. |
| The options are European. | Appropriate for the NSE stock-option contracts used in the analysis. | An early-exercise feature would require a different pricing model. |
| Settlement prices and the synchronous underlying price are fair observations. | Approximate; liquidity and timing effects can affect settlement prices. Spot is reconciled with the equity close. | IV may capture stale pricing or market microstructure effects rather than expected volatility alone. |

### Merton structural-model assumptions

| Assumption | Assessment for Biocon | What may fail if the assumption does not hold |
|---|---|---|
| Firm asset value follows geometric Brownian motion with constant asset volatility. | Approximate because firm value is unobservable and business risk changes through time. | Estimated asset value, asset volatility, DD and PD may be biased. |
| The firm has one zero-coupon debt claim with face value $D$ maturing at $T$. | Does not hold literally. Biocon has several liabilities and maturities, represented by the KMV default point. | The simplified default boundary may not capture the actual maturity structure or refinancing risk. |
| Default can occur only at $T$, and only when $V<D$. | Restrictive; liquidity stress, covenants or missed payments can cause earlier distress. | The model can understate default paths that arise before the one-year horizon. |
| Equity is a European call option on firm assets with strike $D$. | A useful structural approximation rather than an exact legal description of the capital structure. | Multiple claims, guarantees and stakeholder priorities weaken the exact option analogy. |
| There are no bankruptcy costs, taxes, dividends or new debt before $T$. | Does not hold literally for a continuing listed company. | Payouts, financing transactions, taxes and recovery costs change firm and equity values. |
| The risk-free rate and one-year horizon are fixed, and market equity and equity volatility contain sufficient information to infer $V$ and $\sigma_V$. | Reasonable for the assignment's one-year point-in-time calibration, but model-dependent. | Solutions become sensitive to the debt proxy, observed equity value, volatility input and rate convention. |

Consequently, the calculated PDs are model-based risk indicators rather than literal forecasts of the exact probability that Biocon will default.
"""),
md("""## Setup

This cell installs only the external model libraries required by the assignment and its independent checks."""),
code(r"""%pip install -q arch==8.0.0 statsmodels==0.15.0 scikit-learn==1.9.1 py-vollib==1.0.12 merton==1.0.2 openpyxl==3.1.5

# Print the versions actually loaded by this runtime.
from importlib.metadata import version

for package in ['arch', 'statsmodels', 'scikit-learn', 'py-vollib', 'merton', 'openpyxl']:
    print(f'{package}: {version(package)}')"""),
md("""This cell finds the five files already present in Colab's Files area. It does not create another upload button."""),
code(r"""
from pathlib import Path
import sys

expected_files = {
    'Biocon_data_updated.xlsx',
    'Financial_Statements (2) (1).xlsx',
    'Biocon_options-_data.xlsx',
    'Auctions of 91-Day Government of India Treasury Bills.xlsx',
    'Auctions of 364-Day Government of India Treasury Bills.xlsx',
}
search_root = Path('/content') if Path('/content').exists() else Path.cwd()
located_files = {}
for name in sorted(expected_files):
    direct = search_root / name
    matches = [direct] if direct.exists() else list(search_root.rglob(name))
    if not matches:
        raise FileNotFoundError(f'{name} was not found in the Colab Files area.')
    located_files[name] = min(matches, key=lambda p: (len(p.parts), str(p)))

Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
print('Input files found:')
for name, path in located_files.items():
    print(f'  {name}: {path}')
"""),
md("""The following collapsed support cell only makes the common file locator available in memory. It does not write a Python file, run a model, or produce a figure."""),
]

# Keep the small common locator in memory. This avoids %%writefile and %run cells.
locator = source_text('input_data.py')
cells.append(code(f"""
import sys as _sys
import types as _types
input_data = _types.ModuleType('input_data')
exec({locator!r}, input_data.__dict__)
_sys.modules['input_data'] = input_data
""", hidden=True))

cells += [
md(r"""
## Data preparation

Dates and numeric fields are standardised first. Adjusted prices and returns are preserved as supplied. Fiscal-year-end prices use the last trading day on or before 31 March.

For options, the near expiry is used when more than seven calendar days remain; otherwise the next expiry is selected. Zero-volume contracts are removed, moneyness is restricted to $0.90\leq K/S\leq1.10$, and lower-bound violations are excluded. The theoretical upper bound remains a diagnostic. RBI rates are matched without using a future auction.

The calculation below performs all validation and preprocessing. Its audit tables and four input graphs appear once directly below the cell.
"""),
model_cell('preprocess_market_inputs.py'),
md(r"""
## Part A: Option-implied volatility

For a dividend-paying stock,

$$c=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2),\qquad p=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1),$$

$$d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad d_2=d_1-\sigma\sqrt{T}.$$

Implied volatility is obtained as the root of $F(\sigma)=C_{BSM}(\sigma)-C_{mkt}$, or the corresponding put equation. Newton_Raphson starts from $\sigma_0=30\%$ and uses BSM vega as the derivative. If the Newton step leaves the admissible region or fails to converge, bisection is applied on $[1\%,300\%]$. A solution is accepted only when the absolute repricing error is below INR 0.0001. The assignment-listed `py_vollib` implementation is used strictly as an independent numerical check. Call and put ATM IVs are linearly interpolated separately across strikes bracketing $K/S=1$ and then averaged; no extrapolation is permitted.

The calculation below solves and validates every retained contract, prints the ATM table, and displays the smile and solver-comparison figures once.
"""),
model_cell('implied_volatility.py'),
md(r"""
## Part A: Merton structural model

Equity is treated as a European call on firm assets:

$$E=VN(d_1)-De^{-rT}N(d_2),\qquad \sigma_EE=N(d_1)\sigma_VV.$$

Here $d_1=[\ln(V/D)+(r+\tfrac12\sigma_V^2)T]/(\sigma_V\sqrt{T})$ and $d_2=d_1-\sigma_V\sqrt{T}$. The nonlinear system is solved jointly for $V$ and $\sigma_V$ by `scipy.optimize.fsolve`, starting at $V_0=E+D$ and $\sigma_{V,0}=\sigma_EE/(E+D)$. Distance to default is

$$DD=\frac{\ln(V/D)+(\mu-\tfrac12\sigma_V^2)T}{\sigma_V\sqrt{T}},\qquad PD=N(-DD).$$

The risk-neutral specification sets $\mu=r$ and is the primary result. The physical specification replaces $r$ with the annualised historical mean adjusted log return. Both DD and PD series are reported for all 12 dates. Solver convergence is verified by substituting $(V,\sigma_V)$ back into both equations, and DD/PD is independently checked with the assignment-listed Merton package. The cell also reports the equity-volatility sensitivity analysis required for 30 September 2026.
"""),
model_cell('merton_model.py', register='merton_model'),
md(r"""**Part A interpretation.** On 30 September 2026, $\sigma_E=24.915\%$, $\sigma_V=21.653\%$ and risk-neutral $DD=9.281$. The corresponding raw normal-tail PD is below the assignment threshold and is therefore reported at the mandatory 3 bp floor. The high DD is driven by the estimated asset value relative to the KMV default point; it is not evidence that Biocon has no borrowings."""),
md(r"""
## Part B: GARCH(1,1)

$$\sigma_t^2=\omega+\alpha u_{t-1}^2+\beta\sigma_{t-1}^2.$$

Daily returns are $u_t=\ln(S_t/S_{t-1})$. Normal and Student_t conditional likelihoods are estimated separately on the previous 1,000 trading days at each valuation date. Admissibility requires $\omega>0$, $\alpha,\beta\geq0$ and $\alpha+\beta<1$. Persistence is $\alpha+\beta$, long-run variance is $V_L=\omega/(1-\alpha-\beta)$, and volatility half-life is $\ln(0.5)/\ln(\alpha+\beta)$. Distribution selection is based on BIC. ARCH_LM tests the raw returns for conditional heteroskedasticity, while Ljung_Box tests standardised and squared standardised residuals for remaining serial dependence.

$$\bar\sigma^2(T)=V_L+\frac{1-e^{-aT}}{aT}[V(0)-V_L],\qquad \sigma_E^{GARCH}=\sqrt{252\bar\sigma^2(T)}.$$

The calculation below compares both distributions, selects Student_t by BIC, reports parameters with standard errors and every required diagnostic for all 12 dates, and displays the GARCH volatility and DD figures once. Merton results are shown under both risk-neutral and physical drift.
"""),
model_cell('garch_model.py'),
md(r"""
## Part B: Random Forest

For $h=21$ and $252$,

$$RV_{t,h}=\sqrt{\frac{252}{h}\sum_{i=1}^{h}u_{t+i}^2}.$$

The dependent variable is $\log(RV_{t,h})$. Features known at $t$ are realised volatility over 5, 21 and 63 days; returns over 1, 5 and 21 days; filtered GARCH volatility; India VIX; Nifty 50 21-day return; and a results-month indicator. Observations are split chronologically into 60% training, 20% validation and 20% test samples. Any forward target whose horizon crosses a split boundary is purged to prevent label leakage. Hyperparameters are selected only on validation QLIKE, after which the untouched test set is evaluated using RMSE, MAE, QLIKE and $R^2_{OOS}=1-MSE_{RF}/MSE_{GARCH}$. The 252-day forecast enters Merton because it matches the one-year structural horizon.

The cell prints test accuracy and monthly forecasts, then displays feature importance and the ML-versus-GARCH figures once. It also creates the assignment-required 30 September 2026 Part B table, placing GARCH and Random Forest $\sigma_E$, $\sigma_V$, risk-neutral DD/PD and physical DD/PD side by side.
"""),
model_cell('ml_volatility.py'),
md(r"""**Part B interpretation.** At the main date, Student_t GARCH gives $\sigma_E=32.690\%$ and risk-neutral $DD=7.014$, while Random Forest gives $\sigma_E=31.480\%$ and risk-neutral $DD=7.295$. The corresponding physical DD and PD estimates appear in the dedicated comparison table above. Both raw structural PDs remain below the 3 bp reporting floor. The 252-day Random Forest has negative GARCH-relative out-of-sample $R^2$, so the ML specification does not improve on the volatility benchmark and is retained as a separate required route rather than claimed as the preferred forecast."""),
md(r"""
## Part C: Rating-implied PD

Biocon's CRISIL AA+ Stable rating maps to AA. Published one-year default rates are 0.00% for AAA, 0.02% for AA and 0.07% for A. After the 3 bp floor, the point and lower edge are 3 bp, while the upper geometric edge is

$$\sqrt{3\times7}=4.5826\text{ bp}.$$

Adjacent-category bucket edges are constructed geometrically after applying the reporting floor. Each target is converted using $DD=-N^{-1}(PD)$. The CRISIL category PD is an empirical through-the-cycle input, not a risk-neutral market probability. Holding 30 September 2026 equity value, default point, rate and horizon fixed, the Merton equity-pricing equation and target-DD equation are solved jointly for $(V,\sigma_V)$. The corresponding $\sigma_E$ follows from the Merton volatility identity. The calculation is repeated with $\mu=r$ and with the historical physical drift; these labels describe the inverse calibration, not the original CRISIL PD.
"""),
model_cell('rating_model.py', register='rating_model'),
md("""**Part C interpretation.** The risk-neutral CRISIL-consistent equity-volatility interval is 63.124% to 65.024%, materially above both ATM IV and GARCH. The difference is economically interpretable because structural market estimates respond to the current equity and option state, whereas the rating default study is a historical, through-the-cycle calibration supplemented by qualitative credit assessment."""),
md(r"""
## Part D: Altman accounting models

$$Z=1.2X_1+1.4X_2+3.3X_3+0.6X_4+X_5,$$

$$Z''=6.56X_1+3.26X_2+6.72X_3+1.05X_4,\qquad EM=Z''+3.25.$$

$X_1$ measures short-term liquidity through working capital/total assets. $X_2$ measures cumulative profitability through Other equity/total assets under the assignment's reserves-and-surplus instruction. $X_3$ is operating profitability, with EBIT defined consistently as PBT plus finance cost. $X_4$ measures solvency using market equity/total liabilities for original Z and book equity/total liabilities for $Z''$. $X_5$ is asset turnover, sales/total assets. EM is the preferred accounting specification because Biocon is an emerging-market issuer. The unrounded FY2026 EM score is mapped to Altman's bond-rating interval and then to the corresponding empirical CRISIL category PD. The implied volatilities are reverse-calibrated under both risk-neutral and physical drift; the empirical PD input itself is not reclassified as risk-neutral.

The cell prints FY2022 to FY2026 scores and the reverse-solved result, then displays the three separate score graphs once.
"""),
model_cell('altman_model.py'),
md("""**Part D interpretation.** FY2026 original $Z=2.428$ lies in the grey zone, $Z''=3.597$ lies in the safe zone, and $EM=6.847$ lies in the safe zone but maps to an A bond-rating equivalent. The associated 7 bp PD implies $DD=3.195$. This accounting calibration is more conservative than Biocon's observed AA+ agency rating because it is a mechanical function of reported balance-sheet and income-statement ratios rather than a forward-looking committee assessment."""),
md(r"""
## Part E: Comparison

The assignment requires a common comparison of $\sigma_E$, $\sigma_V$, DD and PD. Structural market routes estimate PD from observed or forecast volatility and are point-in-time. The CRISIL route starts from an empirical through-the-cycle category PD. The Altman route starts from an accounting score mapped to an empirical rating-category PD. For CRISIL and Altman, volatility is reverse-engineered to make the Merton system consistent with the external PD. The headline Part E table uses the assignment's risk-neutral calibration for comparability; detailed Part A to D outputs retain the corresponding physical results.

The calculation prints the completed table and displays separate volatility and PD comparison figures once.
"""),
model_cell('part_e_comparison.py'),
md(r"""
The comparison table displayed by the Part E calculation is generated directly from the computed outputs of Parts A to D. The first three routes map estimated $\sigma_E$ into structural DD and PD. Parts C and D reverse the direction: an external PD is mapped into the $\sigma_V$ and $\sigma_E$ required for consistency with the Merton equity equation.
"""),
md("""
## Conclusion and limitations

- ATM IV is the lowest volatility route and produces the highest DD.
- GARCH and Random Forest produce higher volatility, but raw PD remains below the mandatory floor.
- CRISIL requires much higher volatility to reach its rating-implied PD.
- Altman EM is the most conservative route, mapping FY2026 ratios to A and 7 bp.
- ATM IV minus GARCH is about minus 7.78 percentage points. It is only a cautious snapshot volatility-risk-premium proxy because the measures use different information.

Limitations include the 3 bp floor, Merton's simplified debt structure, sparse option contracts, remaining short-lag mean autocorrelation in GARCH, Random Forest underperformance, and conceptual differences across market, rating and accounting routes.
"""),
]

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
    'language_info': {'name':'python','version':'3'},
    'colab': {'name':OUTPUT.name,'provenance':[]},
})
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f'Created {OUTPUT} with {len(cells)} cells')
