"""Part D: Altman scores, rating-equivalent PD and implied volatility."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import merton_model
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Run the Part A Merton structural-model calculation first."
    ) from exc
try:
    import rating_model
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Run the Part C rating calculation first."
    ) from exc
import input_data


ROOT = Path(__file__).resolve().parents[1]
YEARS = [2022, 2023, 2024, 2025, 2026]
FINANCIAL_FILE = input_data.file(input_data.FINANCIAL_FILE)


def em_rating_map():
    frame = pd.read_excel(FINANCIAL_FILE, sheet_name='Altman Rating Map', header=3)
    frame = frame[[
        'rating_equivalent', 'lower_bound', 'upper_bound'
    ]].dropna(subset=['rating_equivalent'])
    frame.columns = ['rating', 'lower', 'upper']
    frame['lower'] = pd.to_numeric(frame.lower, errors='coerce').fillna(-np.inf)
    frame['upper'] = pd.to_numeric(frame.upper, errors='coerce').fillna(np.inf)
    return frame


def row_values(frame, label):
    """Read FY2022-FY2026 values from an exact statement row."""
    rows = frame[frame.iloc[:, 0].astype(str).str.strip().eq(label)]
    assert len(rows) == 1, f'Expected one row for {label}'
    values = pd.to_numeric(rows.iloc[0, 1:6], errors='raise').to_numpy(float)
    assert np.isfinite(values).all()
    return values


def rating_for_score(score):
    for row in em_rating_map().itertuples(index=False):
        if row.lower <= score < row.upper:
            return row.rating
    raise ValueError(f'No rating interval for EM score {score}')


def build_scores():
    source = FINANCIAL_FILE
    balance = pd.read_excel(source, sheet_name='Balance Sheet', header=None)
    profit = pd.read_excel(source, sheet_name='Profit & Loss', header=None)
    prices = pd.read_csv(
        ROOT / 'data/processed/fiscal_year_end_equity_prices.csv'
    ).set_index('fiscal_year').loc[YEARS]

    current_assets = row_values(balance, 'Total Current Assets')
    current_liabilities = row_values(balance, 'Total Current Liabilities')
    total_assets = row_values(balance, 'TOTAL ASSETS')
    retained_earnings = row_values(balance, 'Other equity')
    book_equity = row_values(balance, 'Total Equity')
    total_liabilities = row_values(balance, 'Total Liabilities')
    sales = row_values(profit, 'Revenue from operations')
    pbt = row_values(profit, 'Profit Before Tax')
    finance_cost = row_values(profit, 'Finance costs')
    ebit = pbt + finance_cost
    shares = pd.read_excel(
        input_data.file(input_data.MARKET_FILE), sheet_name='Shares'
    )
    shares.effective_date = pd.to_datetime(shares.effective_date)
    year_ends = pd.DataFrame({'date': pd.to_datetime([f'{y}-03-31' for y in YEARS])})
    annual_shares = pd.merge_asof(
        year_ends.sort_values('date'), shares.sort_values('effective_date'),
        left_on='date', right_on='effective_date', direction='backward',
    ).shares_outstanding.to_numpy()
    market_equity = prices.close_price.to_numpy() * annual_shares / 1e6

    data = pd.DataFrame({
        'fiscal_year': YEARS,
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'working_capital': current_assets - current_liabilities,
        'retained_earnings_other_equity': retained_earnings,
        'profit_before_tax': pbt,
        'finance_cost': finance_cost,
        'ebit': ebit,
        'sales': sales,
        'total_assets': total_assets,
        'total_liabilities': total_liabilities,
        'book_equity': book_equity,
        'shares_outstanding': annual_shares,
        'fiscal_year_end_date': prices.date.to_numpy(),
        'fiscal_year_end_price': prices.close_price.to_numpy(),
        'market_equity': market_equity,
    })
    data['x1_working_capital'] = data.working_capital / data.total_assets
    data['x2_retained_earnings'] = (
        data.retained_earnings_other_equity / data.total_assets
    )
    data['x3_ebit'] = data.ebit / data.total_assets
    data['x4_market_equity'] = data.market_equity / data.total_liabilities
    data['x4_book_equity'] = data.book_equity / data.total_liabilities
    data['x5_sales'] = data.sales / data.total_assets

    data['z_score'] = (
        1.2 * data.x1_working_capital
        + 1.4 * data.x2_retained_earnings
        + 3.3 * data.x3_ebit
        + 0.6 * data.x4_market_equity
        + data.x5_sales
    )
    data['z_double_prime'] = (
        6.56 * data.x1_working_capital
        + 3.26 * data.x2_retained_earnings
        + 6.72 * data.x3_ebit
        + 1.05 * data.x4_book_equity
    )
    data['em_score'] = data.z_double_prime + 3.25
    data['z_zone'] = pd.cut(
        data.z_score, [-np.inf, 1.81, 2.99, np.inf],
        labels=['Distress', 'Grey', 'Safe'], right=False,
    ).astype(str)
    data['z_double_prime_zone'] = pd.cut(
        data.z_double_prime, [-np.inf, 1.10, 2.60, np.inf],
        labels=['Distress', 'Grey', 'Safe'], right=False,
    ).astype(str)
    data['em_zone'] = pd.cut(
        data.em_score, [-np.inf, 4.35, 5.85, np.inf],
        labels=['Distress', 'Grey', 'Safe'], right=False,
    ).astype(str)
    data['em_rating_equivalent'] = data.em_score.map(rating_for_score)
    return data


def reverse_engineer(scores):
    """Map FY2026 EM rating to CRISIL PD and reverse-solve Merton."""
    final = scores.iloc[-1]
    rating_source = pd.read_excel(
        FINANCIAL_FILE, sheet_name='Rating Data', header=7, nrows=3,
        usecols='A:C',
    ).set_index('rating_category')
    collected = {'AAA': 'AAA', 'AA': 'AA', 'A': 'A'}
    base_category = final.em_rating_equivalent.rstrip('+-')
    if base_category not in collected:
        raise ValueError(
            f'EM rating {final.em_rating_equivalent} has no default rate in '
            'Rating Data; add the missing category before recalculating.'
        )
    category = collected[base_category]
    if category not in rating_source.index:
        raise ValueError(f'CRISIL default rate not collected for {category}')
    raw_pd = rating_source.loc[category, 'one_year_default_rate']
    target_pd = max(raw_pd, merton_model.PD_FLOOR)
    main = merton_model.inputs().iloc[-1]
    case = SimpleNamespace(
        equity_value=main.equity_value,
        debt=main.debt,
        risk_free_rate=main.risk_free_rate,
        physical_drift=main.physical_drift,
        target_dd=-norm.ppf(target_pd),
    )
    rows = []
    for drift_case, drift in [
        ('risk_neutral', case.risk_free_rate),
        ('physical', case.physical_drift),
    ]:
        solved = rating_model.solve_case(case, drift_case, drift)
        rows.append({
            'fiscal_year': int(final.fiscal_year),
            'em_score': final.em_score,
            'em_rating_equivalent': final.em_rating_equivalent,
            'crisil_pd_category': category,
            'crisil_pd_raw': raw_pd,
            'target_pd': target_pd,
            'target_pd_bp': target_pd * 10000,
            'target_dd': case.target_dd,
            'equity_value': main.equity_value,
            'debt': main.debt,
            'risk_free_rate': main.risk_free_rate,
            'physical_drift': main.physical_drift,
            **solved,
        })
    return pd.DataFrame(rows)


def score_figure(scores):
    specs = [
        ('z_score', 'Original Altman Z score', 'altman_z_score.png'),
        ('z_double_prime', 'Altman Z″ score', 'altman_z_double_prime.png'),
        ('em_score', 'Altman EM score', 'altman_em_score.png'),
    ]
    folder = ROOT / 'outputs/figures'
    folder.mkdir(parents=True, exist_ok=True)
    for column, title, filename in specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        values = scores[column]
        ax.plot(scores.fiscal_year, values, marker='o', linewidth=2.2)
        ax.annotate(
            f'FY2026: {values.iloc[-1]:.2f}',
            (scores.fiscal_year.iloc[-1], values.iloc[-1]),
            xytext=(-4, 10), textcoords='offset points', ha='right', fontsize=9,
        )
        ax.set(title=f'Biocon {title}, FY2022_FY2026', xlabel='Fiscal year',
               ylabel='Score', xticks=YEARS)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=180)
        plt.show()


def main():
    scores = build_scores()
    reverse = reverse_engineer(scores)

    assert len(scores) == 5
    assert scores.fiscal_year.tolist() == YEARS
    assert np.isfinite(scores.select_dtypes('number')).all().all()
    assert len(reverse) == 2
    assert reverse.fsolve_ier.eq(1).all()
    assert reverse.max_scaled_residual.max() < 1e-8
    assert reverse.check_success.all()
    assert reverse.asset_value_check_difference.abs().max() < 0.001
    assert reverse.sigma_v_check_difference.abs().max() < 1e-8

    folder = ROOT / 'data/processed'
    scores.to_csv(folder / 'altman_scores.csv', index=False)
    reverse.to_csv(folder / 'altman_reverse_engineered_volatility.csv', index=False)
    score_figure(scores)

    print('\nALTMAN SCORES')
    print(scores[[
        'fiscal_year', 'z_score', 'z_zone', 'z_double_prime',
        'z_double_prime_zone', 'em_score', 'em_zone',
        'em_rating_equivalent',
    ]].to_string(index=False))
    final = scores.iloc[-1]
    print(
        f'\nFY2026 INTERPRETATION: Original Z is {final.z_zone}, '
        f'Z-double-prime is {final.z_double_prime_zone}, and EM is '
        f'{final.em_zone}. The EM rating equivalent is '
        f'{final.em_rating_equivalent}.'
    )
    print('\nFY2026 EM RATING, PD AND REVERSE-ENGINEERED VOLATILITY')
    print(reverse[[
        'em_score', 'em_rating_equivalent', 'crisil_pd_category',
        'target_pd_bp', 'target_dd', 'drift_case', 'sigma_v', 'sigma_e',
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
