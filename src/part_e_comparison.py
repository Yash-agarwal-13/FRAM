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
    if 'valuation_date' in frame:
        frame['valuation_date'] = pd.to_datetime(frame.valuation_date)
        target = pd.Timestamp('2026-09-30')
        selected = frame.loc[frame.valuation_date.eq(target)]
        assert len(selected) == 1, f'Expected one {target.date()} row in {filename}'
        return selected.iloc[0]
    if 'fiscal_year' in frame:
        selected = frame.loc[frame.fiscal_year.eq(2026)]
        assert len(selected) == 1, f'Expected one FY2026 row in {filename}'
        return selected.iloc[0]
    raise ValueError(f'No recognised date key in {filename}')


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
         'Risk_neutral_structural_PD', 'Point_in_time'),
        ('B_Student_t_GARCH', garch.sigma_e, garch.sigma_v,
         garch.dd_risk_neutral, garch.pd_risk_neutral_bp,
         'Risk_neutral_structural_PD', 'Point_in_time_forecast'),
        ('B_Random_Forest', ml.sigma_e, ml.sigma_v,
         ml.dd_risk_neutral, ml.pd_risk_neutral_bp,
         'Risk_neutral_structural_PD', 'Point_in_time_forecast'),
        ('C_CRISIL_rating', rating.sigma_e, rating.sigma_v,
         rating.target_dd, rating.target_pd_bp,
         'Risk_neutral_reverse_calibration', 'Through_the_cycle_rating'),
        ('D_Altman_EM', altman.sigma_e, altman.sigma_v,
         altman.target_dd, altman.target_pd_bp,
         'Risk_neutral_reverse_calibration', 'Accounting_point_in_time'),
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
    rating_comparison = pd.read_csv(DATA / 'rating_market_comparison.csv')
    band_lower = rating_comparison.rating_band_lower.iloc[0] * 100
    band_upper = rating_comparison.rating_band_upper.iloc[0] * 100
    assert 0 < band_lower <= band_upper

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colours = ['#2563eb', '#f97316', '#16a34a', '#dc2626', '#7c3aed']
    ax.bar(x, table.sigma_e * 100, color=colours)
    ax.axhspan(
        band_lower, band_upper, color='#dc2626', alpha=.18,
        label=f'Part C CRISIL band ({band_lower:.1f}% to {band_upper:.1f}%)',
    )
    ax.axhline(band_lower, color='#dc2626', linewidth=1.4)
    ax.axhline(band_upper, color='#dc2626', linewidth=1.4)
    ax.set(xticks=x, xticklabels=labels, ylabel='Annualised equity volatility (%)',
           title='Part E: Equity volatility comparison')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(FIGURES / 'part_e_volatility_comparison.png', dpi=180)
    plt.show()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, table.pd_bp, color='#334155', linewidth=1.8, zorder=2)
    ax.scatter(x, table.pd_bp, color=colours, s=85, zorder=3)
    ax.axhline(3, color='#111827', linestyle='--', linewidth=1.3,
               label='Assignment floor: 3 bp')
    for position, value in zip(x, table.pd_bp):
        ax.annotate(f'{value:.2f}', (position, value), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1, top=max(table.pd_bp) * 1.7)
    ax.set(xticks=x, xticklabels=labels, ylabel='Probability of default (bp)',
           title='Part E: Reported default probability comparison (log scale)')
    ax.grid(axis='y', which='both', alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'part_e_pd_comparison.png', dpi=180)
    plt.show()


def display_comparison(table):
    """Format the calculated Part E table for notebook or terminal output."""
    route_labels = {
        'A_ATM_implied_volatility': 'Part A: ATM implied volatility',
        'B_Student_t_GARCH': 'Part B: Student_t GARCH',
        'B_Random_Forest': 'Part B: Random Forest',
        'C_CRISIL_rating': 'Part C: CRISIL AA+',
        'D_Altman_EM': 'Part D: Altman EM',
    }
    input_labels = {
        'A_ATM_implied_volatility': 'Interpolated call/put ATM IV',
        'B_Student_t_GARCH': '252-day expected average variance',
        'B_Random_Forest': '252-day predicted realised volatility',
        'C_CRISIL_rating': 'AA category PD after assignment floor',
        'D_Altman_EM': 'A-equivalent category PD',
    }
    view = table.copy()
    view.insert(1, 'volatility_or_pd_input', view.route.map(input_labels))
    view['route'] = view.route.map(route_labels)
    view = view.rename(columns={
        'route': 'Route',
        'volatility_or_pd_input': 'Volatility or PD input',
        'sigma_e': 'sigma_E',
        'sigma_v': 'sigma_V',
        'distance_to_default': 'DD',
        'pd_bp': 'Reported PD',
        'measure_basis': 'Probability basis',
        'risk_perspective': 'Time perspective',
    })
    view['Probability basis'] = view['Probability basis'].str.replace('_', ' ')
    view['Time perspective'] = view['Time perspective'].str.replace('_', ' ')

    try:
        from IPython import get_ipython
        from IPython.display import display as notebook_display
        if get_ipython() is not None:
            styled = view.style.hide(axis='index').format({
                'sigma_E': '{:.3%}',
                'sigma_V': '{:.3%}',
                'DD': '{:.4f}',
                'Reported PD': '{:.2f} bp',
            })
            notebook_display(styled)
            return
    except ImportError:
        pass

    printable = view.copy()
    printable[['sigma_E', 'sigma_V']] *= 100
    print(printable.to_string(index=False, formatters={
        'sigma_E': '{:.3f}%'.format,
        'sigma_V': '{:.3f}%'.format,
        'DD': '{:.4f}'.format,
        'Reported PD': '{:.2f} bp'.format,
    }))


def main():
    table = build_table()
    gaps = route_gaps(table)
    assert len(table) == 5
    assert np.isfinite(table.select_dtypes('number')).all().all()
    assert table.pd_bp.ge(3 - 1e-10).all()
    table.to_csv(DATA / 'part_e_summary_table.csv', index=False)
    gaps.to_csv(DATA / 'part_e_route_gaps.csv', index=False)
    print('\nPART E_SUMMARY TABLE')
    display_comparison(table)
    figures(table)
    print('\nROUTE GAPS')
    print(gaps.to_string(index=False))


if __name__ == '__main__':
    main()
