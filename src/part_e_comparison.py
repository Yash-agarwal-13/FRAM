"""Part E: compare the five default-risk routes on 30 September 2026."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
FIGURES = ROOT / 'outputs/figures'


def final_row(filename, drift='risk_neutral'):
    frame = pd.read_csv(DATA / filename)
    if 'drift_case' in frame:
        frame = frame.loc[frame.drift_case.eq(drift)]
    return frame.iloc[-1]


def build_table():
    iv = final_row('merton_results.csv')
    garch = final_row('garch_merton_results.csv')
    ml = final_row('ml_merton_results.csv')
    rating = pd.read_csv(DATA / 'rating_reverse_engineered_volatility.csv')
    rating = rating.query("drift_case == 'risk_neutral' and case == 'rating_point'").iloc[0]
    altman = final_row('altman_reverse_engineered_volatility.csv')

    rows = [
        ('A_ATM_implied_volatility', iv.sigma_e, iv.sigma_v,
         iv.dd_risk_neutral, iv.pd_risk_neutral_bp,
         'Risk_neutral', 'Point_in_time'),
        ('B_Student_t_GARCH', garch.sigma_e, garch.sigma_v,
         garch.dd_risk_neutral, garch.pd_risk_neutral_bp,
         'Risk_neutral', 'Point_in_time'),
        ('B_Random_Forest', ml.sigma_e, ml.sigma_v,
         ml.dd_risk_neutral, ml.pd_risk_neutral_bp,
         'Risk_neutral', 'Point_in_time'),
        ('C_CRISIL_rating', rating.sigma_e, rating.sigma_v,
         rating.target_dd, rating.target_pd_bp,
         'Risk_neutral_reverse_solve', 'Through_the_cycle'),
        ('D_Altman_EM', altman.sigma_e, altman.sigma_v,
         altman.target_dd, altman.target_pd_bp,
         'Risk_neutral_reverse_solve', 'Accounting_point_in_time'),
    ]
    columns = [
        'route', 'sigma_e', 'sigma_v', 'distance_to_default', 'pd_bp',
        'measure_basis', 'risk_perspective',
    ]
    return pd.DataFrame(rows, columns=columns)


def route_gaps(table):
    values = table.set_index('route')
    return pd.DataFrame({
        'comparison': [
            'ATM_IV_minus_GARCH_sigma_e',
            'ATM_IV_minus_Random_Forest_sigma_e',
            'CRISIL_minus_ATM_IV_sigma_e',
            'Altman_minus_ATM_IV_sigma_e',
        ],
        'difference': [
            values.loc['A_ATM_implied_volatility', 'sigma_e']
            - values.loc['B_Student_t_GARCH', 'sigma_e'],
            values.loc['A_ATM_implied_volatility', 'sigma_e']
            - values.loc['B_Random_Forest', 'sigma_e'],
            values.loc['C_CRISIL_rating', 'sigma_e']
            - values.loc['A_ATM_implied_volatility', 'sigma_e'],
            values.loc['D_Altman_EM', 'sigma_e']
            - values.loc['A_ATM_implied_volatility', 'sigma_e'],
        ],
    })


def figures(table):
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = ['ATM IV', 'GARCH', 'Random Forest', 'CRISIL', 'Altman EM']
    x = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colours = ['#2563eb', '#f97316', '#16a34a', '#dc2626', '#7c3aed']
    ax.bar(x, table.sigma_e * 100, color=colours)
    ax.set(xticks=x, xticklabels=labels, ylabel='Annualised equity volatility (%)',
           title='Part E_equity volatility comparison')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'part_e_volatility_comparison.png', dpi=180)
    plt.show()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, table.pd_bp, color=colours)
    ax.axhline(3, color='#111827', linestyle='--', linewidth=1.3,
               label='Assignment floor_3 bp')
    ax.set(xticks=x, xticklabels=labels, ylabel='Probability of default (bp)',
           title='Part E_reported default probability comparison')
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'part_e_pd_comparison.png', dpi=180)
    plt.show()


def main():
    table = build_table()
    gaps = route_gaps(table)
    assert len(table) == 5
    assert np.isfinite(table.select_dtypes('number')).all().all()
    assert table.pd_bp.ge(3 - 1e-10).all()
    table.to_csv(DATA / 'part_e_summary_table.csv', index=False)
    gaps.to_csv(DATA / 'part_e_route_gaps.csv', index=False)
    figures(table)
    display = table.copy()
    display[['sigma_e', 'sigma_v']] *= 100
    print('\nPART E_SUMMARY TABLE')
    print(display.to_string(index=False, formatters={
        'sigma_e': '{:.3f}%'.format,
        'sigma_v': '{:.3f}%'.format,
        'distance_to_default': '{:.4f}'.format,
        'pd_bp': '{:.3f}'.format,
    }))
    print('\nROUTE GAPS')
    print(gaps.to_string(index=False))


if __name__ == '__main__':
    main()
