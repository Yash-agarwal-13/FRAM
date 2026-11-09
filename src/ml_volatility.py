"""Part B: Random Forest volatility forecasts and Merton DD/PD."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor

import merton_model
import input_data


ROOT = Path(__file__).resolve().parents[1]
MARKET_FILE = input_data.file(input_data.MARKET_FILE)
DATES = pd.to_datetime([
    '2025-10-31', '2025-11-28', '2025-12-31', '2026-01-30',
    '2026-02-27', '2026-03-30', '2026-04-30', '2026-05-29',
    '2026-06-30', '2026-07-31', '2026-08-31', '2026-09-30',
])
FEATURES = [
    'rv_5', 'rv_21', 'rv_63', 'return_1', 'return_5', 'return_21',
    'garch_sigma_t', 'india_vix', 'nifty_return_21', 'results_month',
]
WINDOW, MIN_GARCH = 1000, 252


def daily_garch(history):
    """Create known-at-date GARCH features and 21/252-day benchmarks."""
    cache = ROOT / 'data/processed/ml_daily_garch_features.csv'
    expected = len(history) - MIN_GARCH + 1
    if cache.exists():
        saved = pd.read_csv(cache, parse_dates=['date'])
        if (len(saved) == expected and saved.date.iloc[-1] == history.date.iloc[-1]):
            return saved

    rows = []
    returns = history.model_return.to_numpy()
    for i in range(MIN_GARCH - 1, len(history)):
        window = returns[max(0, i - WINDOW + 1):i + 1] * 100
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = arch_model(
                window, mean='Constant', vol='GARCH', p=1, q=1,
                dist='t', rescale=False,
            ).fit(disp='off', show_warning=False, options={'maxiter': 500})
        variance = fit.forecast(
            horizon=252, method='analytic', reindex=False
        ).variance.iloc[-1].to_numpy() / 10000
        rows.append({
            'date': history.date.iloc[i],
            'garch_sigma_t': fit.conditional_volatility[-1] / 100 * np.sqrt(252),
            'garch_sigma_21': np.sqrt(252 * variance[:21].mean()),
            'garch_sigma_252': np.sqrt(252 * variance.mean()),
            'garch_converged': fit.convergence_flag == 0,
        })
        if len(rows) % 500 == 0:
            print(f'Daily GARCH features: {len(rows)}/{expected}')

    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    return result


def build_panel():
    """Combine only information known by each Biocon trading date."""
    equity = pd.read_csv(ROOT / 'data/processed/biocon_equity_adjusted.csv')
    equity['date'] = pd.to_datetime(equity.date)
    equity = equity.dropna(subset=['model_return']).sort_values('date')

    vix = pd.read_excel(MARKET_FILE, sheet_name='India-VIX')
    vix.columns = vix.columns.str.strip()
    vix = vix.rename(columns={'Date': 'date', 'Close': 'india_vix'})
    vix['date'] = pd.to_datetime(vix.date)

    nifty = pd.read_excel(MARKET_FILE, sheet_name='Nifty-50')
    nifty.columns = nifty.columns.str.strip()
    nifty = nifty.rename(columns={'Date': 'date', 'Close': 'nifty_close'})
    nifty['date'] = pd.to_datetime(nifty.date)
    nifty = nifty.sort_values('date')
    nifty['nifty_return_21'] = np.log(
        nifty.nifty_close / nifty.nifty_close.shift(21)
    )

    panel = equity[['date', 'model_return']].merge(
        nifty[['date', 'nifty_close', 'nifty_return_21']],
        on='date', how='left', validate='one_to_one',
    ).merge(
        vix[['date', 'india_vix']], on='date', how='left', validate='one_to_one'
    )
    missing_vix = panel.loc[panel.india_vix.isna(), 'date']
    assert missing_vix.tolist() == [pd.Timestamp('2024-03-02')]
    panel['india_vix'] = panel.india_vix.ffill()

    result_dates = pd.read_excel(
        MARKET_FILE, sheet_name='Biocon_quartely-results-date'
    )['result_month'].astype(str)
    panel['results_month'] = (
        panel.date.dt.to_period('M').astype(str).isin(set(result_dates)).astype(int)
    )
    panel['return_1'] = panel.model_return
    for days in [5, 21]:
        panel[f'return_{days}'] = panel.model_return.rolling(days).sum()
    for days in [5, 21, 63]:
        panel[f'rv_{days}'] = np.sqrt(
            252 * panel.model_return.pow(2).rolling(days).mean()
        )

    panel = panel.merge(daily_garch(equity), on='date', how='left')
    for horizon in [21, 252]:
        future = panel.model_return.shift(-1)
        panel[f'target_sigma_{horizon}'] = np.sqrt(
            252 * future.pow(2).iloc[::-1].rolling(horizon).mean().iloc[::-1]
        )
        panel[f'target_end_{horizon}'] = panel.date.shift(-horizon)
    return panel


def metrics(actual, forecast, benchmark):
    """Assignment accuracy measures on annualised volatility."""
    actual, forecast, benchmark = map(
        np.asarray, (actual, forecast, benchmark)
    )
    ratio = actual**2 / np.maximum(forecast**2, 1e-12)
    return {
        'rmse': np.sqrt(np.mean((actual - forecast) ** 2)),
        'mae': np.mean(np.abs(actual - forecast)),
        'qlike': np.mean(ratio - np.log(ratio) - 1),
        'oos_r2': 1 - np.mean((actual - forecast) ** 2)
        / np.mean((actual - benchmark) ** 2),
    }


def time_split(panel, horizon):
    """Use 60/20/20 chronological splits and purge overlapping labels."""
    target = f'target_sigma_{horizon}'
    end = f'target_end_{horizon}'
    data = panel.dropna(subset=FEATURES + [target, end, f'garch_sigma_{horizon}'])
    data = data.loc[data[end] < DATES[0]].reset_index(drop=True)
    validation_start = data.date.iloc[int(len(data) * .60)]
    test_start = data.date.iloc[int(len(data) * .80)]
    train = data.loc[(data.date < validation_start) & (data[end] < validation_start)]
    validation = data.loc[
        (data.date >= validation_start) & (data.date < test_start)
        & (data[end] < test_start)
    ]
    test = data.loc[data.date >= test_start]
    return train, validation, test, validation_start, test_start


def new_model(params):
    return RandomForestRegressor(
        random_state=42, n_jobs=-1, max_features=.8, **params,
    )


def recency_weights(rows, half_life):
    """Give recent rows more weight; None keeps equal weights."""
    if half_life is None:
        return None
    age = len(rows) - 1 - np.arange(len(rows))
    return 0.5 ** (age / half_life)


def select_model(panel, horizon):
    """Tune on validation data, then evaluate once on untouched test data."""
    train, validation, test, validation_start, test_start = time_split(
        panel, horizon
    )
    target = f'target_sigma_{horizon}'
    grid = [
        {'n_estimators': trees, 'max_depth': depth, 'min_samples_leaf': leaf}
        for trees in [300, 600] for depth in [3, 5, None]
        for leaf in [5, 10, 20]
    ]
    tuning = []
    for params in grid:
        model = new_model(params)
        model.fit(train[FEATURES], np.log(train[target]))
        prediction = np.exp(model.predict(validation[FEATURES]))
        score = metrics(
            validation[target], prediction,
            validation[f'garch_sigma_{horizon}'],
        )
        tuning.append({**params, **score})
    tuning = pd.DataFrame(tuning).sort_values(['qlike', 'rmse'])
    winner = tuning.iloc[0]
    best = {
        'n_estimators': int(winner.n_estimators),
        'max_depth': None if pd.isna(winner.max_depth) else int(winner.max_depth),
        'min_samples_leaf': int(winner.min_samples_leaf),
    }

    weight_tuning = []
    for half_life in [None, 252, 504, 756, 1260]:
        model = new_model(best)
        model.fit(
            train[FEATURES], np.log(train[target]),
            sample_weight=recency_weights(train, half_life),
        )
        prediction = np.exp(model.predict(validation[FEATURES]))
        score = metrics(
            validation[target], prediction,
            validation[f'garch_sigma_{horizon}'],
        )
        weight_tuning.append({
            **best, 'weight_half_life': half_life or 0, **score,
        })
    weight_tuning = pd.DataFrame(weight_tuning).sort_values(['qlike', 'rmse'])
    # Recency weighting is a reported sensitivity, not a post-test model change.
    # The original equal-weight specification remains the locked final model.
    selected_half_life = None
    tuning['weight_half_life'] = np.nan
    tuning = pd.concat([tuning, weight_tuning], ignore_index=True)

    development = pd.concat([train, validation]).sort_values('date')
    model = new_model(best).fit(
        development[FEATURES], np.log(development[target]),
        sample_weight=recency_weights(development, selected_half_life),
    )
    ml_prediction = np.exp(model.predict(test[FEATURES]))
    accuracy = []
    for name, prediction in [
        ('Random Forest', ml_prediction),
        ('GARCH(1,1)', test[f'garch_sigma_{horizon}']),
    ]:
        accuracy.append({
            'horizon_days': horizon, 'model': name,
            'train_rows': len(train), 'validation_rows': len(validation),
            'test_rows': len(test), 'validation_start': validation_start,
            'test_start': test_start, 'test_end': test.target_end_252.max()
            if horizon == 252 else test.target_end_21.max(),
            **metrics(test[target], prediction, test[f'garch_sigma_{horizon}']),
        })
    return best, selected_half_life, tuning, pd.DataFrame(accuracy), test


def monthly_forecasts(panel, horizon, params, half_life):
    """Refit with labels observable by each assignment valuation date."""
    target, end = f'target_sigma_{horizon}', f'target_end_{horizon}'
    rows, final_model = [], None
    for date in DATES:
        training = panel.dropna(subset=FEATURES + [target, end])
        training = training.loc[training[end] <= date]
        current = panel.loc[panel.date.eq(date)]
        assert len(current) == 1 and current[FEATURES].notna().all(axis=None)
        final_model = new_model(params).fit(
            training[FEATURES], np.log(training[target]),
            sample_weight=recency_weights(training, half_life),
        )
        rows.append({
            'valuation_date': date, 'horizon_days': horizon,
            'training_rows': len(training),
            'latest_observable_target_end': training[end].max(),
            'ml_sigma_e': np.exp(final_model.predict(current[FEATURES]))[0],
            'garch_sigma_e': current[f'garch_sigma_{horizon}'].iloc[0],
        })
    importance = pd.DataFrame({
        'feature': FEATURES, 'feature_importance': final_model.feature_importances_
    }).sort_values('feature_importance', ascending=False)
    return pd.DataFrame(rows), importance


def figures(forecasts, importance, ml_merton):
    folder = ROOT / 'outputs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    main = forecasts.query('horizon_days == 252')

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(main.valuation_date, main.ml_sigma_e * 100, 'o-', label='Random Forest')
    ax.plot(main.valuation_date, main.garch_sigma_e * 100, 's--', label='GARCH')
    ax.set(xlabel='Valuation date', ylabel='Annualised equity volatility (%)',
           title='Biocon 252-day volatility forecasts')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / 'ml_vs_garch_sigma_e_over_time.png', dpi=180)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = importance.sort_values('feature_importance')
    ax.barh(ordered.feature, ordered.feature_importance, color='#2563eb')
    ax.set(xlabel='Random Forest feature importance',
           title='252-day model feature importance')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'ml_feature_importance_252.png', dpi=180)

    garch = pd.read_csv(ROOT / 'data/processed/garch_merton_results.csv')
    garch['valuation_date'] = pd.to_datetime(garch.valuation_date)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ml_merton.valuation_date, ml_merton.dd_risk_neutral,
            'o-', label='Random Forest DD')
    ax.plot(garch.valuation_date, garch.dd_risk_neutral,
            's--', label='GARCH DD')
    ax.set(xlabel='Valuation date', ylabel='Risk-neutral distance to default',
           title='Biocon Merton DD: Random Forest versus GARCH')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / 'ml_vs_garch_merton_dd.png', dpi=180)
    plt.show()


def main():
    panel = build_panel()
    all_forecasts, all_accuracy, all_tuning, importances = [], [], [], {}
    for horizon in [21, 252]:
        params, half_life, tuning, accuracy, _ = select_model(panel, horizon)
        forecasts, importance = monthly_forecasts(
            panel, horizon, params, half_life
        )
        forecasts['weight_half_life'] = half_life or 0
        tuning.insert(0, 'horizon_days', horizon)
        importance.insert(0, 'horizon_days', horizon)
        all_forecasts.append(forecasts)
        all_accuracy.append(accuracy)
        all_tuning.append(tuning)
        importances[horizon] = importance
        print(f'{horizon}-day selected parameters: {params}; '
              f'weight half-life: {half_life or "equal"}')

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    accuracy = pd.concat(all_accuracy, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    importance = pd.concat(importances.values(), ignore_index=True)

    selected = forecasts.query('horizon_days == 252').rename(
        columns={'ml_sigma_e': 'sigma_e'}
    )[['valuation_date', 'sigma_e']]
    ml_merton = merton_model.calculate(merton_model.inputs(selected))

    assert panel.garch_converged.dropna().all()
    assert len(forecasts) == 24 and len(accuracy) == 4
    assert forecasts[['ml_sigma_e', 'garch_sigma_e']].gt(0).all().all()
    assert (forecasts.latest_observable_target_end <= forecasts.valuation_date).all()
    assert ml_merton.solver_converged.all() and ml_merton.checker_converged.all()
    assert ml_merton.max_scaled_residual.max() < 1e-8
    assert ml_merton.dd_check_difference.abs().max() < 1e-5

    out = ROOT / 'data/processed'
    panel.to_csv(out / 'ml_feature_target_panel.csv', index=False)
    accuracy.to_csv(out / 'ml_test_accuracy.csv', index=False)
    tuning.to_csv(out / 'ml_validation_tuning.csv', index=False)
    forecasts.to_csv(out / 'ml_volatility_forecasts.csv', index=False)
    importance.to_csv(out / 'ml_feature_importance.csv', index=False)
    ml_merton.to_csv(out / 'ml_merton_results.csv', index=False)

    print('\nOUT-OF-SAMPLE ACCURACY')
    print(accuracy[['horizon_days', 'model', 'test_rows', 'rmse', 'mae',
                    'qlike', 'oos_r2']].to_string(index=False))
    print('\n252-DAY MONTHLY FORECASTS')
    print(forecasts.query('horizon_days == 252')[
        ['valuation_date', 'ml_sigma_e', 'garch_sigma_e']
    ].to_string(index=False))
    print('\n30 SEPTEMBER 2026 - RANDOM FOREST MERTON RESULT')
    print(ml_merton.iloc[-1][[
        'sigma_e', 'sigma_v', 'dd_risk_neutral', 'pd_risk_neutral_bp',
        'dd_physical', 'pd_physical_bp',
    ]].to_string())
    figures(forecasts, importances[252], ml_merton)


if __name__ == '__main__':
    main()
