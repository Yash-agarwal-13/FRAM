"""Part A: Merton DD and PD using ATM IV as equity volatility."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from merton.calibration import kmv_iterative
from scipy.optimize import fsolve
from scipy.stats import norm

import input_data

ROOT = Path(__file__).resolve().parents[1]
PD_FLOOR = 0.0003

MARKET_FILE = input_data.file(input_data.MARKET_FILE)
FINANCIAL_FILE = input_data.file(input_data.FINANCIAL_FILE)

# Publication dates are required to preserve the information cutoff.
PUBLICATION_DATES = pd.to_datetime([
    '2025-08-07', '2025-11-11', '2026-02-12',
    '2026-05-07', '2026-08-05',
])
EXPECTED_BALANCE_DATES = pd.to_datetime([
    '2025-06-30', '2025-09-30', '2025-12-31',
    '2026-03-31', '2026-06-30',
])


def debt_schedule():
    raw = pd.read_excel(
        FINANCIAL_FILE, sheet_name='Quarterly balance sheet ', header=None
    )
    headers = raw.iloc[2, 1:6].astype(str)
    mask = raw.iloc[:, 0].astype(str).str.strip().eq('Borrowings')
    borrowings = raw.loc[mask, raw.columns[1:6]].apply(pd.to_numeric)
    assert borrowings.shape == (2, 5)
    reports = headers.str.split('\n').str[0].to_list()
    balance_dates = pd.to_datetime(
        headers.str.split('\n').str[-1].str.replace(',', '', regex=False)
    )
    debt = pd.DataFrame({
        'report': reports,
        'balance_sheet_date': balance_dates,
        'publication_date': PUBLICATION_DATES,
        'non_current_borrowings': borrowings.iloc[0].to_numpy(),
        'current_borrowings': borrowings.iloc[1].to_numpy(),
    })
    debt['debt'] = (
        debt.current_borrowings + 0.5 * debt.non_current_borrowings
    )
    assert debt.balance_sheet_date.tolist() == EXPECTED_BALANCE_DATES.tolist(), (
        'Balance-sheet dates or order changed; update PUBLICATION_DATES.'
    )
    assert len(PUBLICATION_DATES) == len(debt)
    return debt


def share_schedule():
    shares = pd.read_excel(MARKET_FILE, sheet_name='Shares')
    shares = shares.rename(columns={'effective_date': 'share_date'})
    shares['share_date'] = pd.to_datetime(shares.share_date)
    return shares[['share_date', 'shares_outstanding']].sort_values('share_date')


def equations(x, equity, debt, rate, sigma_e):
    """The two scaled Merton equations."""
    value, sigma_v = x
    if value <= 0 or sigma_v <= 0:
        return [1e6, 1e6]
    d1 = (np.log(value / debt) + rate + 0.5 * sigma_v**2) / sigma_v
    d2 = d1 - sigma_v
    model_equity = value * norm.cdf(d1) - debt * np.exp(-rate) * norm.cdf(d2)
    model_equity_vol = norm.cdf(d1) * sigma_v * value
    return [
        (model_equity - equity) / equity,
        (model_equity_vol - sigma_e * equity) / (sigma_e * equity),
    ]


def solve(equity, debt, rate, sigma_e):
    """Authoritative assignment solver using scipy.optimize.fsolve."""
    start = [equity + debt, sigma_e * equity / (equity + debt)]
    args = (equity, debt, rate, sigma_e)
    solution, _, _, _ = fsolve(
        equations, start, args=args, xtol=1e-11, full_output=True
    )
    residual = np.max(np.abs(equations(solution, *args)))
    success = min(solution) > 0 and residual < 1e-8
    return solution[0], solution[1], success, residual


def repository_check(equity, debt, rate, sigma_e):
    """Check-only calibration using the assignment-listed ItsSypher/merton."""
    fit = kmv_iterative(
        equity=equity, equity_vol=sigma_e, debt=debt, rf=rate, T=1,
        tol=1e-10, max_iter=500,
    )
    check_dd = dd(fit.asset_value, debt, fit.asset_vol, rate)
    check_pd = float(fit.diagnostics['risk_neutral_edf'])
    return fit.asset_value, fit.asset_vol, check_dd, check_pd, fit.converged


def dd(value, debt, sigma_v, drift):
    return (np.log(value / debt) + drift - 0.5 * sigma_v**2) / sigma_v


def inputs(volatility=None):
    """Build dated Merton inputs for ATM IV or another equity-volatility series."""
    if volatility is None:
        volatility = pd.read_csv(
            ROOT / 'data/processed/atm_implied_volatility.csv'
        )[['valuation_date', 'atm_iv']]
        volatility['sigma_e'] = volatility.atm_iv
    else:
        volatility = volatility[['valuation_date', 'sigma_e']].copy()
    volatility['valuation_date'] = pd.to_datetime(volatility.valuation_date)
    price = pd.read_csv(ROOT / 'data/processed/valuation_date_equity_prices.csv')
    price['valuation_date'] = pd.to_datetime(price.target_date)
    rate = pd.read_csv(ROOT / 'data/processed/valuation_date_tbill_rates.csv')
    rate = rate.loc[rate.maturity_days.eq(364)].copy()
    rate['valuation_date'] = pd.to_datetime(rate.target_date)

    data = volatility.merge(
        price[['valuation_date', 'close_price']], on='valuation_date'
    ).merge(
        rate[['valuation_date', 'date', 'yield_percent', 'continuous_rate']]
        .rename(columns={'date': 'rate_date'}), on='valuation_date'
    ).sort_values('valuation_date')

    shares = share_schedule()
    data = pd.merge_asof(data, shares, left_on='valuation_date',
                         right_on='share_date', direction='backward')

    debt = debt_schedule()
    data = pd.merge_asof(data, debt.sort_values('publication_date'),
                         left_on='valuation_date', right_on='publication_date',
                         direction='backward')

    history = pd.read_csv(ROOT / 'data/processed/biocon_equity_adjusted.csv')
    history['date'] = pd.to_datetime(history.date)
    data['physical_drift'] = [
        history.loc[history.date <= date, 'model_return'].dropna().mean() * 252
        for date in data.valuation_date
    ]
    data['drift_observations'] = [
        history.loc[history.date <= date, 'model_return'].notna().sum()
        for date in data.valuation_date
    ]
    data['equity_value'] = data.close_price * data.shares_outstanding / 1e6
    data['risk_free_rate'] = data.continuous_rate
    return data.reset_index(drop=True)


def calculate(data):
    rows = []
    for row in data.itertuples():
        main = solve(row.equity_value, row.debt, row.risk_free_rate, row.sigma_e)
        check = repository_check(
            row.equity_value, row.debt, row.risk_free_rate, row.sigma_e
        )
        rows.append({
            'asset_value': main[0], 'sigma_v': main[1],
            'solver_converged': main[2], 'max_scaled_residual': main[3],
            'check_asset_value': check[0], 'check_sigma_v': check[1],
            'check_dd_risk_neutral': check[2],
            'check_pd_risk_neutral_raw': check[3],
            'checker_converged': check[4],
        })
    result = pd.concat([data, pd.DataFrame(rows)], axis=1)
    result['asset_value_check_difference'] = result.asset_value - result.check_asset_value
    result['sigma_v_check_difference'] = result.sigma_v - result.check_sigma_v
    result['debt_to_asset'] = result.debt / result.asset_value
    result['dd_risk_neutral'] = dd(
        result.asset_value, result.debt, result.sigma_v, result.risk_free_rate
    )
    result['dd_check_difference'] = (
        result.dd_risk_neutral - result.check_dd_risk_neutral
    )
    result['dd_physical'] = dd(
        result.asset_value, result.debt, result.sigma_v, result.physical_drift
    )
    for case in ['risk_neutral', 'physical']:
        result[f'pd_{case}_raw'] = norm.cdf(-result[f'dd_{case}'])
        result[f'pd_{case}'] = result[f'pd_{case}_raw'].clip(lower=PD_FLOOR)
        result[f'pd_{case}_bp'] = result[f'pd_{case}'] * 10000
    result['pd_check_difference'] = (
        result.pd_risk_neutral_raw - result.check_pd_risk_neutral_raw
    )
    return result


def sensitivity(main):
    rows = []
    for sigma_e in np.arange(0.10, 0.801, 0.01):
        value, sigma_v, success, residual = solve(
            main.equity_value, main.debt, main.risk_free_rate, sigma_e
        )
        rows.append({
            'sigma_e': sigma_e, 'asset_value': value, 'sigma_v': sigma_v,
            'dd_risk_neutral': dd(value, main.debt, sigma_v,
                                  main.risk_free_rate),
            'solver_converged': success, 'max_scaled_residual': residual,
        })
    return pd.DataFrame(rows)


def figures(result, curve):
    folder = ROOT / 'outputs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(result.valuation_date)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, result.dd_risk_neutral, 'o-', label='Risk-neutral DD')
    ax.plot(dates, result.dd_physical, 's--', label='Physical DD')
    ax.set(xlabel='Valuation date', ylabel='Distance to default',
           title='Biocon Merton distance to default')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / 'merton_dd_over_time.png', dpi=180)

    main = result.iloc[-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve.sigma_e * 100, curve.dd_risk_neutral, linewidth=2.2)
    ax.scatter(main.sigma_e * 100, main.dd_risk_neutral, marker='X', s=90,
               color='#111827', label='30 Sep 2026 ATM IV')
    ax.set(xlabel='Equity volatility σE (%)', ylabel='Risk-neutral DD',
           title='Biocon Merton DD sensitivity_30 September 2026')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'merton_dd_vs_sigma_e_2026_09_30.png', dpi=180)
    plt.show()


def main():
    model_inputs = inputs()
    result = calculate(model_inputs)
    curve = sensitivity(result.iloc[-1])

    assert len(result) == 12 and model_inputs.notna().all().all()
    assert (result.publication_date <= result.valuation_date).all()
    assert result.solver_converged.all() and result.checker_converged.all()
    assert result.max_scaled_residual.max() < 1e-8
    assert result.asset_value_check_difference.abs().max() < 0.001
    assert result.sigma_v_check_difference.abs().max() < 1e-6
    assert result.dd_check_difference.abs().max() < 1e-5
    assert result.pd_check_difference.abs().max() < 1e-10
    assert curve.solver_converged.all()

    out = ROOT / 'data/processed'
    model_inputs.to_csv(out / 'merton_inputs.csv', index=False)
    result.to_csv(out / 'merton_results.csv', index=False)
    result[[
        'valuation_date', 'asset_value', 'check_asset_value',
        'asset_value_check_difference', 'sigma_v', 'check_sigma_v',
        'sigma_v_check_difference', 'dd_risk_neutral',
        'check_dd_risk_neutral', 'dd_check_difference',
        'pd_risk_neutral_raw', 'check_pd_risk_neutral_raw',
        'pd_check_difference', 'solver_converged', 'checker_converged',
    ]].to_csv(out / 'merton_solver_validation.csv', index=False)
    curve.to_csv(out / 'merton_sigma_e_sensitivity.csv', index=False)

    columns = [
        'valuation_date', 'equity_value', 'debt', 'sigma_e', 'asset_value',
        'sigma_v', 'dd_risk_neutral', 'pd_risk_neutral_bp',
        'dd_physical', 'pd_physical_bp',
    ]
    print(result[columns].to_string(index=False))

    # Make the assignment's low-leverage interpretation explicit.
    main_result = result.iloc[-1]
    floor_dates = int(result.pd_risk_neutral_raw.lt(PD_FLOOR).sum())
    print(
        f'\nInterpretation: 30 Sep 2026 D/V is '
        f'{main_result.debt_to_asset:.1%}. The high DD reflects Biocon\'s '
        'relatively low structural leverage under the Merton default point.'
    )
    print(
        f'PD floor: raw risk-neutral PD is below 3 bp on '
        f'{floor_dates}/{len(result)} dates, so those results are reported at '
        'the assignment floor of 3 bp (0.03%).'
    )
    print('\nItsSypher/merton checker: 12/12 passed')
    print('Maximum V difference:', result.asset_value_check_difference.abs().max())
    print('Maximum sigma_V difference:', result.sigma_v_check_difference.abs().max())
    print('Maximum DD difference:', result.dd_check_difference.abs().max())
    print('Maximum raw PD difference:', result.pd_check_difference.abs().max())
    figures(result, curve)


if __name__ == '__main__':
    main()
