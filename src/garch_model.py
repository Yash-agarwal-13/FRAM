"""Part B: rolling GARCH(1,1) volatility and Merton DD/PD."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

try:
    import merton_model
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Run the Part A Merton structural-model calculation first."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATES = pd.to_datetime([
    '2025-10-31', '2025-11-28', '2025-12-31', '2026-01-30',
    '2026-02-27', '2026-03-30', '2026-04-30', '2026-05-29',
    '2026-06-30', '2026-07-31', '2026-08-31', '2026-09-30',
])
WINDOW, HORIZON = 1000, 252


def annualised_volatility(fit):
    """Assignment formula for expected average variance over 252 days."""
    omega = fit.params['omega'] / 10000
    persistence = fit.params['alpha[1]'] + fit.params['beta[1]']
    long_run = omega / (1 - persistence)
    current = fit.conditional_volatility.iloc[-1] ** 2 / 10000
    speed = np.log(1 / persistence)
    weight = -np.expm1(-speed * HORIZON) / (speed * HORIZON)
    sigma_e = np.sqrt(HORIZON * (long_run + weight * (current - long_run)))
    return sigma_e, persistence, long_run, current


def fit_garch(window, date, distribution):
    returns = window.model_return * 100
    fit = arch_model(
        returns, mean='Constant', vol='GARCH', p=1, q=1,
        dist=distribution, rescale=False,
    ).fit(disp='off', show_warning=False)

    sigma_e, persistence, long_run, current = annualised_volatility(fit)
    direct_variance = fit.forecast(
        horizon=HORIZON, method='analytic', reindex=False
    ).variance.iloc[-1].to_numpy() / 10000
    direct_sigma_e = np.sqrt(HORIZON * direct_variance.mean())

    z = pd.Series(fit.std_resid).dropna()
    lb_z = acorr_ljungbox(z, lags=[10, 20], return_df=True)
    lb_z2 = acorr_ljungbox(z**2, lags=[10, 20], return_df=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='acorr_lm currently returns.*')
        lm_stat, lm_pvalue, _, _ = het_arch(returns - returns.mean(), nlags=10)

    p, se = fit.params, fit.std_err
    return {
        'valuation_date': date,
        'window_start': window.date.iloc[0],
        'window_end': window.date.iloc[-1],
        'observations': len(window),
        'distribution': distribution,
        'converged': fit.convergence_flag == 0,
        'mu': p.get('mu', np.nan) / 100,
        'mu_se': se.get('mu', np.nan) / 100,
        'omega': p['omega'] / 10000,
        'omega_se': se['omega'] / 10000,
        'alpha': p['alpha[1]'], 'alpha_se': se['alpha[1]'],
        'beta': p['beta[1]'], 'beta_se': se['beta[1]'],
        'nu': p.get('nu', np.nan), 'nu_se': se.get('nu', np.nan),
        'persistence': persistence,
        'long_run_daily_variance': long_run,
        'current_daily_variance': current,
        'half_life_days': np.log(0.5) / np.log(persistence),
        'log_likelihood': fit.loglikelihood, 'aic': fit.aic, 'bic': fit.bic,
        'arch_lm_stat_10': lm_stat, 'arch_lm_pvalue_10': lm_pvalue,
        'ljung_box_resid_pvalue_10': lb_z.loc[10, 'lb_pvalue'],
        'ljung_box_resid_pvalue_20': lb_z.loc[20, 'lb_pvalue'],
        'ljung_box_squared_pvalue_10': lb_z2.loc[10, 'lb_pvalue'],
        'ljung_box_squared_pvalue_20': lb_z2.loc[20, 'lb_pvalue'],
        'garch_sigma_e': sigma_e,
        'arch_sigma_e_check': direct_sigma_e,
        'sigma_e_check_difference': sigma_e - direct_sigma_e,
    }


def estimate_models():
    history = pd.read_csv(ROOT / 'data/processed/biocon_equity_adjusted.csv')
    history['date'] = pd.to_datetime(history.date)
    history = history.dropna(subset=['model_return']).sort_values('date')

    rows = []
    for date in DATES:
        window = history.loc[history.date <= date].tail(WINDOW)
        if len(window) != WINDOW:
            raise ValueError(f'{date.date()}: expected {WINDOW} returns')
        rows += [fit_garch(window, date, dist) for dist in ['normal', 't']]
    return pd.DataFrame(rows)


def select_student_t(models):
    """Use Student-t consistently because it fits heavy-tailed returns better."""
    chosen = models.query("distribution == 't'").copy()
    normal = models.query("distribution == 'normal'")[
        ['valuation_date', 'bic', 'garch_sigma_e']
    ].rename(columns={'bic': 'normal_bic', 'garch_sigma_e': 'normal_sigma_e'})
    chosen = chosen.merge(normal, on='valuation_date', validate='one_to_one')
    chosen = chosen.rename(columns={
        'bic': 'student_t_bic', 'garch_sigma_e': 'student_t_sigma_e'
    })
    chosen['student_t_bic_better'] = chosen.student_t_bic < chosen.normal_bic
    chosen['sigma_e'] = chosen.student_t_sigma_e
    return chosen


def plot_results(forecasts, merton):
    folder = ROOT / 'outputs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(forecasts.valuation_date)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, forecasts.normal_sigma_e * 100, 'o--', label='Normal GARCH')
    ax.plot(dates, forecasts.student_t_sigma_e * 100, 's-', label='Student-t GARCH')
    ax.set(xlabel='Valuation date', ylabel='Annualised equity volatility (%)',
           title='Biocon rolling 1,000-day GARCH(1,1) volatility')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / 'garch_sigma_e_over_time.png', dpi=180)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, merton.dd_risk_neutral, 'o-', label='Risk-neutral DD')
    ax.plot(dates, merton.dd_physical, 's--', label='Physical DD')
    ax.set(xlabel='Valuation date', ylabel='Distance to default',
           title='Biocon Merton DD using Student-t GARCH volatility')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / 'garch_merton_dd_over_time.png', dpi=180)
    plt.show()


def main():
    models = estimate_models()
    forecasts = select_student_t(models)
    inputs = merton_model.inputs(forecasts[['valuation_date', 'sigma_e']])
    merton = merton_model.calculate(inputs)

    assert len(models) == 24 and len(forecasts) == 12
    assert models.observations.eq(WINDOW).all()
    assert (models.window_end <= models.valuation_date).all()
    assert models.converged.all()
    assert (models.omega > 0).all()
    assert (models[['alpha', 'beta']] >= 0).all().all()
    assert models.persistence.between(0, 1, inclusive='neither').all()
    assert models.sigma_e_check_difference.abs().max() < 0.001
    assert merton.solver_converged.all() and merton.max_scaled_residual.max() < 1e-8
    assert merton.checker_converged.all()
    assert merton.asset_value_check_difference.abs().max() < 0.001
    assert merton.sigma_v_check_difference.abs().max() < 1e-6
    assert merton.dd_check_difference.abs().max() < 1e-5

    out = ROOT / 'data/processed'
    models.to_csv(out / 'garch_model_comparison.csv', index=False)
    forecasts.to_csv(out / 'garch_volatility_forecasts.csv', index=False)
    merton.to_csv(out / 'garch_merton_results.csv', index=False)

    columns = ['valuation_date', 'window_start', 'window_end', 'persistence',
               'half_life_days', 'normal_sigma_e', 'student_t_sigma_e']
    print('\nGARCH VOLATILITY FORECASTS')
    print(forecasts[columns].to_string(index=False))
    parameter_columns = [
        'valuation_date', 'omega', 'omega_se', 'alpha', 'alpha_se',
        'beta', 'beta_se', 'nu', 'nu_se', 'persistence',
    ]
    student_parameters = models.query("distribution == 't'")[parameter_columns]
    print('\nSTUDENT_T GARCH PARAMETERS AND STANDARD ERRORS')
    print(student_parameters.to_string(index=False))
    print(f"\nStudent-t BIC wins: {forecasts.student_t_bic_better.sum()}/12")
    print('Maximum volatility check difference:',
          models.sigma_e_check_difference.abs().max())

    selected = models.query("distribution == 't'")
    diagnostic_columns = [
        'valuation_date', 'persistence', 'long_run_daily_variance',
        'half_life_days', 'log_likelihood', 'aic', 'bic',
        'arch_lm_pvalue_10', 'ljung_box_resid_pvalue_10',
        'ljung_box_resid_pvalue_20', 'ljung_box_squared_pvalue_10',
        'ljung_box_squared_pvalue_20', 'garch_sigma_e',
    ]
    print('\nSTUDENT_T GARCH DIAGNOSTICS AND ANNUALISED VOLATILITY')
    print(selected[diagnostic_columns].to_string(index=False))
    print('Diagnostics below 5%: '
          f"ARCH-LM {(selected.arch_lm_pvalue_10 < .05).sum()}/12; "
          f"residual LB10 {(selected.ljung_box_resid_pvalue_10 < .05).sum()}/12; "
          f"squared LB10 {(selected.ljung_box_squared_pvalue_10 < .05).sum()}/12")

    result = merton.iloc[-1]
    print('\n30 SEPTEMBER 2026 - GARCH MERTON RESULT')
    print(result[['equity_value', 'debt', 'debt_to_asset', 'sigma_e', 'sigma_v',
                  'dd_risk_neutral', 'pd_risk_neutral_bp',
                  'dd_physical', 'pd_physical_bp']].to_string())
    plot_results(forecasts, merton)


if __name__ == '__main__':
    main()
