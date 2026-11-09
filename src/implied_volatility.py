"""Part A: own Newton-Raphson solver and assignment ATM interpolation."""
from pathlib import Path
from math import erf, exp, log, sqrt, pi, isfinite
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The assignment names py_vollib as a check-only reference implementation.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='py_vollib is deprecated.*')
    from py_vollib.black_scholes_merton.implied_volatility import (
        implied_volatility as py_vollib_iv,
    )

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.0001
REFERENCE_IV_TOL = 0.00001


def bsm(s, k, t, r, q, sigma, kind):
    if kind not in {'CE', 'PE'}:
        raise ValueError(f'Unsupported option type: {kind}')
    d1 = (log(s / k) + (r - q + sigma * sigma / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    cdf = lambda x: (1 + erf(x / sqrt(2))) / 2
    a, b = s * exp(-q * t), k * exp(-r * t)
    price = a * cdf(d1) - b * cdf(d2) if kind == 'CE' else b * cdf(-d2) - a * cdf(-d1)
    vega = a * sqrt(t) * exp(-d1 * d1 / 2) / sqrt(2 * pi)
    return price, vega


def solve_iv(s, k, t, r, q, price, kind):
    if kind not in {'CE', 'PE'}:
        raise ValueError(f'Unsupported option type: {kind}')
    if not all(isfinite(x) for x in [s, k, t, r, q, price]) or min(s, k, t, price) <= 0:
        return np.nan, 'invalid_input', 0, np.nan
    sigma = 0.3
    for n in range(1, 101):
        fitted, vega = bsm(s, k, t, r, q, sigma, kind)
        error = fitted - price
        if abs(error) < TOL:
            return sigma, 'newton', n, error
        if vega < 1e-10:
            break
        candidate = sigma - error / vega
        if not 0.01 < candidate < 3:
            break
        sigma = candidate
    # Own bisection fallback, without a library IV solver.
    lo, hi = 0.01, 3.0
    flo = bsm(s, k, t, r, q, lo, kind)[0] - price
    fhi = bsm(s, k, t, r, q, hi, kind)[0] - price
    for edge, error in [(lo, flo), (hi, fhi)]:
        if abs(error) < TOL:
            return edge, 'bisection', n, error
    if flo * fhi > 0:
        return np.nan, 'no_solution_in_bracket', n, np.nan
    for j in range(1, 201):
        sigma = (lo + hi) / 2
        error = bsm(s, k, t, r, q, sigma, kind)[0] - price
        if abs(error) < TOL:
            return sigma, 'bisection', n + j, error
        if error > 0:
            hi = sigma
        else:
            lo = sigma
    return np.nan, 'not_converged', n + 200, error


def reference_iv(s, k, t, r, q, price, kind):
    """Return py_vollib IV for validation only; never use it as model input."""
    flag = {'CE': 'c', 'PE': 'p'}[kind]
    try:
        return py_vollib_iv(price, s, k, t, r, q, flag)
    except Exception:
        return np.nan


def atm(group, iv_column='iv'):
    group = group.dropna(subset=[iv_column]).sort_values('moneyness_strike_to_spot')
    x = group.moneyness_strike_to_spot
    exact = group.loc[x.eq(1)]
    if len(exact):
        return float(exact[iv_column].iloc[0])
    below, above = group.loc[x < 1], group.loc[x > 1]
    if below.empty or above.empty:
        return np.nan  # Never extrapolate a missing side.
    a, b = below.iloc[-1], above.iloc[0]
    return a[iv_column] + (b[iv_column] - a[iv_column]) * (1 - a.moneyness_strike_to_spot) / (b.moneyness_strike_to_spot - a.moneyness_strike_to_spot)


def main():
    data = pd.read_csv(ROOT / 'data/processed/options_bsm_ready.csv')
    results = [solve_iv(row.spot_price, row.strike_price, row.time_to_expiry_years,
                        row.risk_free_rate, row.dividend_yield, row.market_price,
                        row.option_type) for row in data.itertuples()]
    data[['iv', 'solver_status', 'iterations', 'price_error']] = pd.DataFrame(results, index=data.index)
    data['py_vollib_iv_check'] = [
        reference_iv(
            row.spot_price, row.strike_price, row.time_to_expiry_years,
            row.risk_free_rate, row.dividend_yield, row.market_price,
            row.option_type,
        )
        for row in data.itertuples()
    ]
    data['iv_check_difference'] = data.iv - data.py_vollib_iv_check
    data['iv_check_passed'] = data.iv_check_difference.abs() <= REFERENCE_IV_TOL
    records = []
    for date, group in data.groupby('valuation_date'):
        call, put = [atm(group.loc[group.option_type.eq(kind)]) for kind in ['CE', 'PE']]
        reference_call, reference_put = [
            atm(group.loc[group.option_type.eq(kind)], 'py_vollib_iv_check')
            for kind in ['CE', 'PE']
        ]
        records.append({'valuation_date': date, 'call_atm_iv': call, 'put_atm_iv': put,
                        'atm_iv': (call + put) / 2, 'contracts': len(group),
                        'failed_solutions': int(group.iv.isna().sum()),
                        'py_vollib_call_atm_iv': reference_call,
                        'py_vollib_put_atm_iv': reference_put,
                        'py_vollib_atm_iv': (reference_call + reference_put) / 2})
    summary = pd.DataFrame(records)
    parity = data.pivot_table(
        index=['valuation_date', 'expiry_date', 'strike_price', 'spot_price',
               'time_to_expiry_years', 'risk_free_rate', 'dividend_yield'],
        columns='option_type', values='market_price'
    ).dropna().reset_index()
    parity['parity_error'] = parity['CE'] - parity['PE'] - (
        parity.spot_price * np.exp(-parity.dividend_yield * parity.time_to_expiry_years)
        - parity.strike_price * np.exp(-parity.risk_free_rate * parity.time_to_expiry_years)
    )
    parity_stats = parity.groupby('valuation_date', as_index=False).agg(
        paired_strikes=('strike_price', 'size'),
        mean_parity_error=('parity_error', 'mean'),
        max_abs_parity_error=('parity_error', lambda x: x.abs().max()),
    )
    summary['atm_call_put_gap'] = (summary.call_atm_iv - summary.put_atm_iv).abs()
    summary['atm_iv_check_difference'] = (
        summary.atm_iv - summary.py_vollib_atm_iv
    )
    summary['contracts_volume_below_10'] = summary.valuation_date.map(
        data.loc[data.traded_volume < 10].groupby('valuation_date').size()
    ).fillna(0).astype(int)
    summary = summary.merge(parity_stats, on='valuation_date', how='left')
    summary['atm_gap_over_10pp'] = summary['atm_call_put_gap'] > 0.10

    assert len(data) == 302, 'Unexpected BSM-ready contract count'
    assert data.valuation_date.nunique() == 12, 'Expected all 12 valuation dates'
    assert not data.duplicated(['valuation_date', 'instrument_id']).any(), 'Duplicate contracts found'
    assert data.iv.notna().all(), 'One or more IV solutions failed'
    assert data.price_error.abs().max() < TOL, 'Repricing tolerance failed'
    assert data.py_vollib_iv_check.notna().all(), 'One or more py_vollib checks failed'
    assert data.iv_check_passed.all(), 'Own IV differs unexpectedly from py_vollib'
    assert summary[['call_atm_iv', 'put_atm_iv', 'atm_iv']].notna().all().all(), 'Missing ATM bracket'
    assert summary.atm_iv_check_difference.abs().max() <= REFERENCE_IV_TOL, 'ATM IV check failed'
    out = ROOT / 'data/processed'
    data.to_csv(out / 'option_implied_volatility.csv', index=False)
    validation_columns = [
        'valuation_date', 'instrument_id', 'option_type', 'expiry_date',
        'strike_price', 'spot_price', 'market_price', 'iv',
        'py_vollib_iv_check', 'iv_check_difference', 'iv_check_passed',
    ]
    data[validation_columns].to_csv(out / 'iv_solver_validation.csv', index=False)
    summary.to_csv(out / 'atm_implied_volatility.csv', index=False)
    parity.to_csv(out / 'option_put_call_parity_diagnostics.csv', index=False)
    print(summary.to_string(index=False))
    print('Solver status:', data.solver_status.value_counts().to_dict())
    print('Maximum absolute repricing error:', data.price_error.abs().max())
    print('Maximum absolute IV difference vs py_vollib:',
          data.iv_check_difference.abs().max())
    print('py_vollib validation:', f'{int(data.iv_check_passed.sum())}/{len(data)} passed')
    print('\nDates requiring ATM data-quality review')
    print(summary.loc[summary.atm_gap_over_10pp, [
        'valuation_date', 'call_atm_iv', 'put_atm_iv', 'atm_call_put_gap',
        'max_abs_parity_error'
    ]].to_string(index=False))
    figures = ROOT / 'outputs/figures'
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {'CE': '#1f77b4', 'PE': '#ff7f0e'}
    for kind, label in [('CE', 'Calls'), ('PE', 'Puts')]:
        smile = data.loc[data.valuation_date.eq('2026-09-30') & data.option_type.eq(kind)].sort_values('moneyness_strike_to_spot')
        sizes = 35 + 18 * np.log1p(smile.traded_volume)
        ax.scatter(smile.moneyness_strike_to_spot, smile.iv * 100, s=sizes,
                   alpha=.72, edgecolor='white', linewidth=.7,
                   color=colors[kind], label=f'{label} observations')
        # A liquidity-weighted quadratic shows the broad smile/skew while the
        # scatter points retain every assignment-valid contract.
        weights = np.sqrt(smile.traded_volume)
        coefficients = np.polyfit(
            smile.moneyness_strike_to_spot, smile.iv * 100, 2, w=weights
        )
        curve_x = np.linspace(
            smile.moneyness_strike_to_spot.min(),
            smile.moneyness_strike_to_spot.max(),
            200,
        )
        ax.plot(curve_x, np.polyval(coefficients, curve_x),
                color=colors[kind], linewidth=2.1,
                label=f'{label} weighted curve')
        thin = smile.loc[smile.traded_volume < 10]
        for row in thin.itertuples():
            ax.annotate(f'Vol {int(row.traded_volume)}',
                        (row.moneyness_strike_to_spot, row.iv * 100),
                        xytext=(5, 6), textcoords='offset points', fontsize=8)
    sep = summary.loc[summary.valuation_date.eq('2026-09-30')].iloc[0]
    ax.scatter([1, 1], [sep.call_atm_iv * 100, sep.put_atm_iv * 100],
               marker='X', s=90, color=['#1f77b4', '#ff7f0e'], zorder=4)
    ax.axvline(1, color='#64748b', linewidth=.8, linestyle='--')
    ax.set(xlabel='Moneyness K/S', ylabel='Annualised IV (%)', title='Biocon IV smile_30 September 2026')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.text(.01, .01, 'Settlement prices. Marker size reflects volume; curves are liquidity-weighted quadratic fits.', fontsize=9, color='#475569')
    fig.tight_layout()
    fig.savefig(figures / 'iv_smile_2026_09_30.png', dpi=180)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(summary.valuation_date), summary.atm_iv * 100, 'o-')
    ax.set(xlabel='Valuation date', ylabel='Annualised ATM IV (%)', title='Biocon monthly ATM implied volatility')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / 'atm_iv_monthly.png', dpi=180)

    # Compare the assignment IV and the repository check on one simple chart.
    comparison_dates = pd.to_datetime(summary.valuation_date)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        comparison_dates, summary.py_vollib_atm_iv * 100,
        's--', color='#f59e0b', linewidth=2.2, markersize=6,
        label='py_vollib ATM IV (check only)',
    )
    ax.plot(
        comparison_dates, summary.atm_iv * 100,
        'o-', color='#2563eb', linewidth=1.7, markersize=4,
        label='Own Newton-Raphson ATM IV',
    )
    ax.set(
        xlabel='Valuation date',
        ylabel='Annualised ATM IV (%)',
        title='Biocon ATM IV validation: own solver vs py_vollib',
    )
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / 'atm_iv_own_vs_py_vollib.png', dpi=180)
    plt.show()


if __name__ == '__main__':
    main()
