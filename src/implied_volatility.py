"""Part A: own Newton-Raphson solver and assignment ATM interpolation."""
from pathlib import Path
from math import erf, exp, log, sqrt, pi, isfinite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.0001


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


def atm(group):
    group = group.dropna(subset=['iv']).sort_values('moneyness_strike_to_spot')
    x = group.moneyness_strike_to_spot
    exact = group.loc[x.eq(1)]
    if len(exact):
        return float(exact.iv.iloc[0])
    below, above = group.loc[x < 1], group.loc[x > 1]
    if below.empty or above.empty:
        return np.nan  # Never extrapolate a missing side.
    a, b = below.iloc[-1], above.iloc[0]
    return a.iv + (b.iv - a.iv) * (1 - a.moneyness_strike_to_spot) / (b.moneyness_strike_to_spot - a.moneyness_strike_to_spot)


def main():
    data = pd.read_csv(ROOT / 'data/processed/options_bsm_ready.csv')
    results = [solve_iv(row.spot_price, row.strike_price, row.time_to_expiry_years,
                        row.risk_free_rate, row.dividend_yield, row.market_price,
                        row.option_type) for row in data.itertuples()]
    data[['iv', 'solver_status', 'iterations', 'price_error']] = pd.DataFrame(results, index=data.index)
    records = []
    for date, group in data.groupby('valuation_date'):
        call, put = [atm(group.loc[group.option_type.eq(kind)]) for kind in ['CE', 'PE']]
        records.append({'valuation_date': date, 'call_atm_iv': call, 'put_atm_iv': put,
                        'atm_iv': (call + put) / 2, 'contracts': len(group),
                        'failed_solutions': int(group.iv.isna().sum())})
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
    assert summary[['call_atm_iv', 'put_atm_iv', 'atm_iv']].notna().all().all(), 'Missing ATM bracket'
    out = ROOT / 'data/processed'
    data.to_csv(out / 'option_implied_volatility.csv', index=False)
    summary.to_csv(out / 'atm_implied_volatility.csv', index=False)
    parity.to_csv(out / 'option_put_call_parity_diagnostics.csv', index=False)
    print(summary.to_string(index=False))
    print('Solver status:', data.solver_status.value_counts().to_dict())
    print('Maximum absolute repricing error:', data.price_error.abs().max())
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
    ax.set(xlabel='Moneyness K/S', ylabel='Annualised IV (%)', title='Biocon IV smile — 30 September 2026')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.text(.01, .01, 'Settlement prices. Marker size reflects volume; curves are liquidity-weighted quadratic fits.', fontsize=9, color='#475569')
    fig.tight_layout()
    fig.savefig(figures / 'iv_smile_2026_09_30.png', dpi=180)

    # Separate presentation-only combined smile. At matched strikes, average
    # call and put IVs before fitting one liquidity-weighted quadratic curve.
    september = data.loc[data.valuation_date.eq('2026-09-30')].copy()
    combined = september.pivot_table(
        index='moneyness_strike_to_spot',
        columns='option_type',
        values=['iv', 'traded_volume'],
        aggfunc='first',
    ).dropna().reset_index()
    combined['combined_iv'] = (combined[('iv', 'CE')] + combined[('iv', 'PE')]) / 2
    combined['combined_volume'] = (
        combined[('traded_volume', 'CE')] + combined[('traded_volume', 'PE')]
    )
    combined_x = combined['moneyness_strike_to_spot'].to_numpy()
    combined_y = combined['combined_iv'].to_numpy() * 100
    combined_weights = np.sqrt(combined['combined_volume'].to_numpy())
    combined_coefficients = np.polyfit(
        combined_x, combined_y, 2, w=combined_weights
    )
    combined_curve_x = np.linspace(combined_x.min(), combined_x.max(), 200)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(
        combined_x,
        combined_y,
        s=40 + 16 * np.log1p(combined['combined_volume']),
        color='#7c3aed',
        alpha=.68,
        edgecolor='white',
        linewidth=.7,
        label='Matched-strike call/put averages',
    )
    ax.plot(
        combined_curve_x,
        np.polyval(combined_coefficients, combined_curve_x),
        color='#6d28d9',
        linewidth=2.4,
        label='Hypothetical combined curve',
    )
    ax.scatter(
        [1], [sep.atm_iv * 100], marker='X', s=110,
        color='#111827', zorder=4, label='Assignment ATM IV',
    )
    ax.axvline(1, color='#64748b', linewidth=.8, linestyle='--')
    ax.set(
        xlabel='Moneyness K/S',
        ylabel='Annualised IV (%)',
        title='Biocon hypothetical combined IV smile — 30 September 2026',
    )
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.text(
        .01, .01,
        'Presentation proxy: matched-strike call and put IVs are averaged, then a liquidity-weighted quadratic is fitted. Not used in ATM IV or Merton.',
        fontsize=8.5, color='#475569',
    )
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(figures / 'iv_smile_combined_hypothetical_2026_09_30.png', dpi=180)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(summary.valuation_date), summary.atm_iv * 100, 'o-')
    ax.set(xlabel='Valuation date', ylabel='Annualised ATM IV (%)', title='Biocon monthly ATM implied volatility')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / 'atm_iv_monthly.png', dpi=180)
    plt.show()


if __name__ == '__main__':
    main()
