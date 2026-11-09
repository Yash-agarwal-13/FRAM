"""Part C: CRISIL rating PD and reverse-engineered Merton volatility."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares
from scipy.stats import norm

import merton_model
import input_data


ROOT = Path(__file__).resolve().parents[1]
PD_FLOOR = 0.0003
FINANCIAL_FILE = input_data.file(input_data.FINANCIAL_FILE)


def rating_inputs():
    source = pd.read_excel(
        FINANCIAL_FILE, sheet_name='Rating Data', header=3, nrows=1
    ).iloc[0]
    defaults = pd.read_excel(
        FINANCIAL_FILE, sheet_name='Rating Data', header=7, nrows=3,
        usecols='A:C',
    ).set_index('rating_category')
    main = merton_model.inputs().iloc[-1]

    rating_pd_raw = defaults.loc['AA', 'one_year_default_rate']
    better_pd_raw = defaults.loc['AAA', 'one_year_default_rate']
    worse_pd_raw = defaults.loc['A', 'one_year_default_rate']
    rating_pd = max(rating_pd_raw, PD_FLOOR)
    better_pd = max(better_pd_raw, PD_FLOOR)
    worse_pd = max(worse_pd_raw, PD_FLOOR)

    cases = pd.DataFrame({
        'case': ['lower_edge', 'rating_point', 'upper_edge'],
        'target_pd': [
            np.sqrt(better_pd * rating_pd),
            rating_pd,
            np.sqrt(rating_pd * worse_pd),
        ],
    })
    cases['target_pd_bp'] = cases.target_pd * 10000
    cases['target_dd'] = -norm.ppf(cases.target_pd)
    cases['company'] = 'Biocon Limited'
    cases['valuation_date'] = source['valuation_date']
    cases['agency'] = source['agency']
    cases['long_term_rating'] = source['long_term_rating']
    cases['outlook'] = source['outlook']
    cases['rationale_date'] = source['rationale_date']
    cases['published_pd_category'] = source['published_pd_category']
    cases['published_pd_percent'] = rating_pd_raw * 100
    cases['better_category'] = 'AAA'
    cases['better_pd_percent'] = better_pd_raw * 100
    cases['worse_category'] = 'A'
    cases['worse_pd_percent'] = worse_pd_raw * 100
    cases['rating_pd_raw'] = rating_pd_raw
    cases['rating_pd_after_floor'] = rating_pd
    cases['better_pd_after_floor'] = better_pd
    cases['worse_pd_after_floor'] = worse_pd
    cases['equity_value'] = main.equity_value
    cases['debt'] = main.debt
    cases['risk_free_rate'] = main.risk_free_rate
    cases['physical_drift'] = main.physical_drift
    return cases


def equations(x, equity, debt, rate, drift, target_dd):
    value, sigma_v = x
    if value <= 0 or sigma_v <= 0:
        return [1e6, 1e6]
    d1 = (np.log(value / debt) + rate + 0.5 * sigma_v**2) / sigma_v
    d2 = d1 - sigma_v
    model_equity = value * norm.cdf(d1) - debt * np.exp(-rate) * norm.cdf(d2)
    model_dd = (
        np.log(value / debt) + drift - 0.5 * sigma_v**2
    ) / sigma_v
    return [(model_equity - equity) / equity, model_dd - target_dd]


def solve_case(row, drift_name, drift):
    args = (row.equity_value, row.debt, row.risk_free_rate,
            drift, row.target_dd)
    start = [row.equity_value + row.debt, 0.50]
    solution, _, ier, message = fsolve(
        equations, start, args=args, full_output=True, xtol=1e-11
    )
    residual = np.max(np.abs(equations(solution, *args)))

    check = least_squares(
        lambda x: equations(x, *args), start,
        bounds=([1e-8, 1e-8], [np.inf, 3]),
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    value, sigma_v = solution
    d1 = (
        np.log(value / row.debt) + row.risk_free_rate + 0.5 * sigma_v**2
    ) / sigma_v
    sigma_e = norm.cdf(d1) * sigma_v * value / row.equity_value
    return {
        'drift_case': drift_name, 'drift': drift,
        'asset_value': value, 'sigma_v': sigma_v, 'sigma_e': sigma_e,
        'fsolve_ier': ier, 'fsolve_message': message,
        'max_scaled_residual': residual,
        'check_success': check.success,
        'check_asset_value': check.x[0], 'check_sigma_v': check.x[1],
        'asset_value_check_difference': value - check.x[0],
        'sigma_v_check_difference': sigma_v - check.x[1],
    }


def calculate(cases):
    rows = []
    for case in cases.itertuples(index=False):
        for name, drift in [
            ('risk_neutral', case.risk_free_rate),
            ('physical', case.physical_drift),
        ]:
            rows.append({**case._asdict(), **solve_case(case, name, drift)})
    return pd.DataFrame(rows)


def market_comparison(results):
    band = results.query("drift_case == 'risk_neutral'").set_index('case')
    lower = band.loc['lower_edge', 'sigma_e']
    upper = band.loc['upper_edge', 'sigma_e']
    iv = pd.read_csv(ROOT / 'data/processed/atm_implied_volatility.csv').iloc[-1].atm_iv
    garch = pd.read_csv(
        ROOT / 'data/processed/garch_volatility_forecasts.csv'
    ).iloc[-1].sigma_e
    comparison = pd.DataFrame({
        'measure': ['ATM implied volatility', 'Student-t GARCH'],
        'sigma_e': [iv, garch],
    })
    comparison['rating_band_lower'] = lower
    comparison['rating_band_upper'] = upper
    comparison['inside_rating_band'] = comparison.sigma_e.between(lower, upper)
    return comparison


def figure(comparison):
    folder = ROOT / 'outputs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    lower = comparison.rating_band_lower.iloc[0] * 100
    upper = comparison.rating_band_upper.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    band_colour = '#dc2626'
    ax.axvspan(lower, upper, color=band_colour, alpha=.24,
               label='CRISIL-consistent volatility band')
    ax.axvline(lower, color=band_colour, linewidth=2)
    ax.axvline(upper, color=band_colour, linewidth=2)
    for row in comparison.itertuples():
        ax.scatter(row.sigma_e * 100, 0, s=90, label=row.measure)
    ax.annotate(
        '63.12% to 65.02% volatility is required\n'
        'for Merton PD to reach the CRISIL-implied\n'
        'range of 3.00 to 4.58 bp',
        xy=((lower + upper) / 2, .05),
        xytext=(47, .52),
        ha='center', va='center', fontsize=9.5, color='#7f1d1d',
        bbox=dict(boxstyle='round,pad=.45', facecolor='#fef2f2',
                  edgecolor=band_colour, linewidth=1.4),
        arrowprops=dict(arrowstyle='-|>', color=band_colour, linewidth=1.6),
    )
    ax.set(xlabel='Annualised equity volatility (%)', yticks=[],
           title='Biocon market volatility versus CRISIL-implied band')
    ax.set_ylim(-.28, .72)
    ax.grid(axis='x', alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'rating_implied_volatility_band.png', dpi=180)
    plt.show()


def main():
    cases = rating_inputs()
    results = calculate(cases)
    comparison = market_comparison(results)

    assert len(results) == 6
    assert results.fsolve_ier.eq(1).all()
    assert results.max_scaled_residual.max() < 1e-8
    assert results.check_success.all()
    assert results.asset_value_check_difference.abs().max() < 0.001
    assert results.sigma_v_check_difference.abs().max() < 1e-8
    assert cases.rating_pd_after_floor.eq(PD_FLOOR).all()
    assert cases.target_pd.between(PD_FLOOR, 0.001).all()

    out = ROOT / 'data/processed'
    cases.to_csv(out / 'rating_pd_bucket.csv', index=False)
    results.to_csv(out / 'rating_reverse_engineered_volatility.csv', index=False)
    comparison.to_csv(out / 'rating_market_comparison.csv', index=False)

    print('\nCRISIL RATING AND PD BUCKET')
    print(cases[[
        'agency', 'long_term_rating', 'outlook', 'rationale_date',
        'case', 'target_pd_bp', 'target_dd',
    ]].to_string(index=False))
    print('\nRATING-IMPLIED VOLATILITY')
    print(results[[
        'case', 'drift_case', 'target_pd_bp', 'target_dd',
        'asset_value', 'sigma_v', 'sigma_e',
    ]].to_string(index=False))
    print('\nMARKET COMPARISON')
    print(comparison.to_string(index=False))
    figure(comparison)


if __name__ == '__main__':
    main()
