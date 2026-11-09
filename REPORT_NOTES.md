# FRAM report notes

## Part A — Merton DD and PD interpretation

### Verified result

- On 30 September 2026, Biocon's Merton default-point debt is INR 96,530 million and estimated asset value is INR 694,387.569 million.
- D/V is approximately 13.9%.
- Risk-neutral DD is 9.281368 and physical DD is 9.396727.
- The raw risk-neutral and physical PDs are below the assignment's reporting floor.
- Report both PDs as 3 bp (0.03%), while retaining the raw model values in the supporting results.
- Across the 12 valuation dates, D/V ranges from approximately 12% to 18%, and the risk-neutral PD floor applies on all 12 dates.

### Explanation required in Part A results

Biocon's high distance to default is driven by its market equity and estimated asset value being large relative to the assignment default point. The raw Merton PD is therefore extremely small. In accordance with the assignment convention, values below 0.03% are reported at the 3 bp floor.

### Explanation required in Part E

The 3 bp figure is a reporting floor, not an assertion that Biocon's true one-year default probability is exactly 0.03%. The Merton result is market-implied and depends on the observed equity value, ATM equity volatility, the KMV default-point approximation and the model's simplifying assumptions. Compare it with the rating-implied and accounting-implied PDs before drawing the final credit-risk conclusion.

### Wording to avoid

Do not state that Biocon has "almost no borrowings." Its D/V is low relative to its market-implied asset value, but borrowings are material in absolute terms. Use "relatively low structural leverage under the Merton default-point definition."

### Presentation checks

- Label DD as unitless; never report DD in basis points.
- Label PD as `PD (bp)`.
- Show D/V in the Part A results table.
- State explicitly that the raw PD is below 3 bp and the displayed 3 bp is the assignment floor.
- Discuss why the floor binds in Part E.
- State that `ItsSypher/merton` version 1.0.2 is used only as an independent check of V, sigma_V, risk-neutral DD and raw PD; SciPy `fsolve` remains authoritative.

## Part B — GARCH(1,1)

### Implemented method

- Fit normal and Student-t GARCH(1,1) models separately on rolling 1,000-trading-day windows ending on each of the 12 valuation dates.
- Use only adjusted log returns available through the applicable valuation-date close.
- Report parameter estimates and standard errors, persistence, long-run variance, half-life, log-likelihood, AIC, BIC, ARCH-LM and Ljung-Box diagnostics.
- Annualise using the assignment's 252-day expected average-variance formula. The direct `arch` 252-step discrete forecast is retained as a numerical check.
- Use Student-t GARCH consistently for the Part B Merton route because its BIC is lower than normal GARCH on all 12 dates.

### Verified results

- All 24 fits converge and satisfy 0 < alpha + beta < 1.
- Student-t degrees of freedom range from approximately 4.92 to 5.36, supporting a heavy-tailed return specification.
- Student-t annualised equity volatility ranges from approximately 32.55% to 34.25%.
- On 30 September 2026, GARCH sigma_E is 32.6903%, Merton sigma_V is 28.4104%, risk-neutral DD is 7.0143 and physical DD is 7.1022. Both PDs remain below the assignment floor and are reported as 3 bp.
- The maximum difference between the assignment annualisation and the direct `arch` discrete forecast is approximately 0.000398 in annualised volatility.

### Diagnostics to explain

- ARCH-LM on raw returns rejects no-ARCH on all 12 dates, confirming volatility clustering and supporting use of GARCH.
- Ljung-Box tests on squared standardized residuals do not reject at lags 10 or 20 on any date, indicating that the GARCH variance equation captures the remaining volatility dependence adequately.
- Ljung-Box at lag 10 rejects for standardized residuals on 11 dates, although lag 20 does not reject. Disclose this as short-lag linear autocorrelation remaining in the mean equation; do not hide or reinterpret it as a variance-model failure.

## Part C — Rating-implied PD

- Use Biocon Limited's CRISIL AA+/Stable long-term rating, reaffirmed on 16 December 2025 and valid at the 30 September 2026 cutoff.
- CRISIL's FY2026 one-year default study reports 0.02% for AA, 0.00% for AAA and 0.07% for A. Apply the assignment's 0.03% floor before constructing the bucket.
- The rating point and lower bucket edge are both 3.00 bp. The upper edge is `sqrt(3 x 7) = 4.5826 bp`. Explain that the floor collapses the point and lower edge; this is not a coding error.
- Holding 30 September 2026 E, D, r and T fixed, the risk-neutral rating bucket implies equity volatility from 63.1241% to 65.0242%. The physical-drift range is 63.8451% to 65.7636%.
- The observed ATM IV of 24.9150% and Student-t GARCH forecast of 32.6903% are both below the risk-neutral rating-implied band. Present this as a difference between market-sensitive structural estimates and a through-the-cycle historical rating calibration, not as proof that either route is wrong.

## Part D — Altman accounting route

- Use consolidated statements. The assignment explicitly defines retained earnings as reserves and surplus, meaning `Other equity` under Ind AS. Do not replace this with the narrower retained-earnings reserve for the assignment calculation.
- Define EBIT consistently as profit before tax plus finance costs. Use revenue from operations as sales, total consolidated equity for book equity, and total liabilities as the denominator of X4.
- FY2026 results are original Z = 2.4284 (grey), Z″ = 3.5966 (safe), and EM = 6.8466 (safe). The EM model is the preferred accounting score because Biocon is an Indian emerging-market company and the original Z was calibrated on listed US manufacturers.
- EM 6.8466 maps to an A rating equivalent under Altman's published table. The same CRISIL default study used in Part C assigns the A category a one-year PD of 0.07%, or 7 bp. This gives DD = 3.1947.
- Holding the 30 September 2026 Merton inputs fixed, the Altman PD implies sigma_V = 58.3186% and sigma_E = 67.0979% under risk-neutral drift. Under physical drift, sigma_V = 58.9791% and sigma_E = 67.8570%.
- The accounting-equivalent A rating is four notches below Biocon's agency AA+ rating: A → A+ → AA− → AA → AA+. Discuss the difference as the accounting model reacting to weak FY2026 earnings and balance-sheet ratios, while the agency rating also incorporates qualitative, prospective, liquidity and group-support considerations.

## Part E_route comparison

- The final 30 September 2026 comparison is exported to `part_e_summary_table.csv`.
- ATM IV gives sigma_E 24.9150%, sigma_V 21.6530%, DD 9.2814 and reported PD 3 bp.
- Student_t GARCH gives sigma_E 32.6903%, sigma_V 28.4104%, DD 7.0143 and reported PD 3 bp.
- Random Forest gives sigma_E 31.4802%, sigma_V 27.3587%, DD 7.2946 and reported PD 3 bp.
- CRISIL AA+ gives reverse_solved sigma_E 63.1241%, sigma_V 54.8618%, DD 3.4316 and PD 3 bp at the rating point.
- Altman EM gives reverse_solved sigma_E 67.0979%, sigma_V 58.3186%, DD 3.1947 and PD 7 bp.
- The ATM IV minus GARCH volatility gap is -7.7753 percentage points. Describe this as a negative volatility risk premium proxy for this snapshot, with caution because the measures use different horizons and information sets.
