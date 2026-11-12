# FRAM Biocon Technical Report Blueprint

## Submission constraints

- Maximum length: 10 pages, excluding appendices.
- Page 1 must contain the executive summary, completed Part E comparison table and main conclusion.
- Number and title every table and figure.
- Cite the exact source file, report page or table for every input.
- Keep the notebook as the reproducible calculation record; the report should interpret results rather than reproduce code.
- Write the three Part E discussion answers separately in 100 to 150 words each.

## Recommended page plan

| Pages | Section | Main purpose |
|---|---|---|
| 1 | Executive summary | Present the five-route comparison and conclusion immediately. |
| 2 | Data, timing and assumptions | Explain inputs, information cutoffs, corporate actions and material BSM/Merton limitations. |
| 3-4 | Part A: IV and Merton | Present option cleaning, ATM IV, structural calibration, DD/PD and sensitivity. |
| 5-6 | Part B: GARCH and Random Forest | Present volatility models, diagnostics, test accuracy and Merton results. |
| 7 | Part C: Rating route | Present CRISIL PD bucket and reverse-engineered volatility band. |
| 8 | Part D: Altman route | Present accounting scores, rating equivalent, PD and reverse calibration. |
| 9-10 | Part E: comparison, discussion and conclusion | Explain route differences, answer the three required questions and state the recommended risk measure. |
| Appendices | Data and supporting evidence | Provide sources, cleaning audit, full tables, diagnostics and reproducibility information. |

## Page 1: Executive summary

### Required content

1. State the objective: estimate Biocon's one-year PD using market, forecast, rating and accounting routes at 30 September 2026.
2. Insert the completed comparison table.
3. Give the main conclusion in four to six sentences.
4. State explicitly that 3 bp is the assignment's reporting floor, not Biocon's precisely estimated true PD.

### Table 1. Comparison of one-year default-risk routes at 30 September 2026

| Route | sigma_E | sigma_V | DD | Reported PD | Basis | Perspective |
|---|---:|---:|---:|---:|---|---|
| ATM implied volatility | 24.915% | 21.653% | 9.281 | 3 bp | Risk-neutral structural PD | Point-in-time |
| Student-t GARCH | 32.690% | 28.410% | 7.014 | 3 bp | Risk-neutral structural PD | Point-in-time forecast |
| Random Forest | 31.480% | 27.359% | 7.295 | 3 bp | Risk-neutral structural PD | Point-in-time forecast |
| CRISIL rating | 63.124% | 54.862% | 3.432 | 3 bp | Risk-neutral reverse calibration | Through-the-cycle rating |
| Altman EM | 67.098% | 58.319% | 3.195 | 7 bp | Risk-neutral reverse calibration | Accounting point-in-time |

### Main conclusion to communicate

- Market-based routes indicate a large asset-value cushion relative to the KMV default point, so their raw PDs fall below the compulsory 3 bp floor.
- The 3 bp observations must not be described as equal economic risk merely because the reported PDs are clipped; DD preserves the large differences between the routes.
- GARCH implies more risk than ATM IV, while Random Forest is close to GARCH but does not beat it out of sample.
- CRISIL's through-the-cycle calibration implies a much higher equity-volatility band than current market measures.
- Altman EM is the most conservative route, producing an A-equivalent category and 7 bp PD, compared with Biocon's observed AA+ rating.
- For early warning, use a market-sensitive measure such as GARCH-Merton DD, monitored alongside rating and Altman indicators rather than replacing them.

## Section 1: Data, conventions and assumptions

### 1.1 Data and information cutoffs

Cover the following points concisely:

- Twelve valuation dates from 31 October 2025 to 30 September 2026, using the last trading day of each month.
- Biocon adjusted prices and adjusted returns are used directly and are not adjusted again.
- Unadjusted closing price is used for market capitalisation.
- Actual allotment-date share counts are used as a step function.
- Debt comes from the latest consolidated balance sheet publicly available on each date and is carried forward until the next publication.
- The 91-day RBI auction yield is used for BSM and the 364-day yield for one-year Merton; rates are converted using r = ln(1 + R/100).
- Trailing-twelve-month dividend yield equals verified cash dividends per share divided by spot.
- India VIX on the special 2 March 2024 session is filled using the last observation available before that date.
- All financial amounts are consolidated and expressed in INR million unless otherwise stated.

### 1.2 Model assumptions

Do not reproduce the entire notebook tables. Summarise the assumptions that materially affect interpretation:

- BSM assumes GBM, constant volatility and rates, continuous hedging, frictionless markets, no persistent arbitrage, European exercise and synchronous fair prices.
- These assumptions are approximations for Biocon; the observed smile, time-varying volatility and market frictions show their limitations.
- Merton assumes GBM for firm assets, constant asset volatility, one zero-coupon debt claim, default only at T when V is below D and equity as a European call on assets.
- Biocon's multiple maturities, refinancing needs and possibility of earlier distress make the structural PD a model-based indicator rather than a literal probability forecast.

### Suggested appendix cross-reference

State that the complete assumption-by-assumption assessment is reproduced in Appendix B.

## Section 2: Part A - Implied volatility and Merton PD

### 2.1 Option-chain cleaning and IV estimation

Explain:

- Near-month expiry is used when more than seven calendar days remain; otherwise, the following expiry is selected.
- Zero-volume contracts are removed.
- Only 0.90 <= K/S <= 1.10 is retained.
- Contracts violating the appropriate lower no-arbitrage bound are excluded; the upper bound is a diagnostic.
- Sequential cleaning reduces 1,541 raw F&O rows to 302 BSM-ready option contracts.
- The own solver uses Newton-Raphson from 30%, switches to bisection on [1%, 300%] if necessary and requires pricing error below INR 0.0001.
- `py_vollib` is check-only; all 302 contracts converge and the maximum IV difference is approximately 0.00000337.
- ATM call and put IVs are linearly interpolated around K/S = 1 and then averaged.

### Required exhibits

- **Table 2:** Sequential option-cleaning audit.
- **Figure 1:** 30 September 2026 call and put volatility smile.
- **Table 3:** ATM IV for the 12 valuation dates.
- **Figure 2:** ATM IV time series.

### 2.2 Merton calibration

State the equity-value and volatility equations, the starting values and the default point:

- E = price multiplied by shares.
- D = short-term borrowings + 0.5 multiplied by long-term borrowings.
- Initial V = E + D and initial sigma_V = sigma_E E/(E + D).
- Risk-neutral drift uses mu = r; physical drift uses the annualised historical mean adjusted log return.
- Each solution is substituted back into both equations and independently checked with the assignment-listed package.

### Required exhibits

- **Table 4:** Twelve-date Merton results containing E, D, D/V, sigma_E, sigma_V, risk-neutral DD/PD and physical DD/PD.
- **Figure 3:** Merton DD over time.
- **Figure 4:** DD versus sigma_E from 10% to 80% on 30 September 2026.

### Interpretation

At the main date, E is INR 603,475.738 million, D is INR 96,530 million, sigma_E is 24.915%, sigma_V is 21.653%, risk-neutral DD is 9.281 and reported PD is 3 bp. Explain that the high DD results from the market-implied asset value being large relative to the default point. Do not say that Biocon has almost no debt: observed D/V is approximately 12% to 18% across the dates.

## Section 3: Part B - GARCH and Random Forest

### 3.1 Student-t GARCH(1,1)

Explain:

- Normal and Student-t GARCH(1,1) are estimated on the previous 1,000 trading days at every valuation date.
- Student-t is selected because it has the lower BIC on all 12 dates and better accommodates heavy-tailed returns.
- Report omega, alpha, beta and nu with standard errors; persistence, long-run variance, half-life, log likelihood, AIC and BIC.
- Report ARCH-LM and Ljung-Box diagnostics.
- Disclose that squared-standardised-residual tests are satisfactory, but short-lag residual autocorrelation remains on most dates.
- Annualisation uses the assignment's expected average variance over 252 trading days.

### Required exhibits

- **Table 5:** Student-t GARCH parameters, standard errors and diagnostics.
- **Figure 5:** Annualised GARCH equity volatility over the 12 dates.

### 3.2 Random Forest volatility forecast

Explain the model choice using the modest sample, nonlinear relationships, ensemble robustness and interpretable feature importance. State:

- Targets are log annualised realised volatility over 21 and 252 days.
- Features include realised volatility over 5, 21 and 63 days; returns over 1, 5 and 21 days; filtered GARCH volatility; India VIX; Nifty 50 21-day return; and results-month indicator.
- The chronological split is 60% training, 20% validation and 20% test with purging of overlapping targets and no shuffling.
- Hyperparameters are chosen using validation QLIKE.
- The 252-day forecast enters Merton because it matches the one-year horizon.
- Equal weighting is the primary specification; recency weighting is only a sensitivity analysis.

### Required exhibits

- **Table 6:** Test-set RMSE, MAE, QLIKE and GARCH-relative out-of-sample R-squared for both horizons.
- **Figure 6:** Random Forest feature importance.
- **Figure 7:** Random Forest and GARCH volatility forecasts.
- **Table 7:** GARCH-Merton and Random-Forest-Merton results at 30 September 2026.
- **Figure 8:** Both DD series over the 12 dates.

### Interpretation

Report honestly that Random Forest did not beat GARCH out of sample. At 252 days its RMSE is 0.024302, MAE 0.018114, QLIKE 0.010764 and GARCH-relative R-squared -0.492144, compared with GARCH RMSE 0.019895, MAE 0.015199 and QLIKE 0.006013. A more elaborate forecasting model does not automatically provide a better PD because structural PD also depends on E, D, r and the nonlinear Merton transformation.

## Section 4: Part C - Rating-implied PD and volatility

Present:

- CRISIL AA+ Stable rating in force at the main date and the 16 December 2025 rationale date.
- Mapping to the CRISIL AA default category.
- Published one-year CDRs: AAA 0.00%, AA 0.02% and A 0.07%.
- After the 3 bp floor, the point and lower edge are 3 bp and the upper geometric edge is 4.5826 bp.
- Reverse calibration holds E, D, r and T fixed and solves for the volatility consistent with the target DD.

### Required exhibits

- **Table 8:** Rating, outlook, rationale date, category PD and bucket edges.
- **Table 9:** Reverse-engineered sigma_V and sigma_E for the lower edge, point and upper edge.
- **Figure 9:** Market volatility measures versus the CRISIL-implied equity-volatility band.

### Interpretation

The risk-neutral rating-consistent sigma_E band is approximately 63.124% to 65.024%. ATM IV and Student-t GARCH lie below the band. Explain that this is not automatically evidence that either route is wrong: market measures are point-in-time, whereas agency ratings and historical category default studies are through-the-cycle and incorporate qualitative judgement.

## Section 5: Part D - Altman accounting route

Present:

- Original Z, Z-double-prime and EM formulas and the ratios used.
- Consolidated FY2022 to FY2026 inputs.
- Other equity is used as reserves and surplus for X2, following the assignment instruction.
- EBIT is profit before tax plus finance costs.
- EM is preferred because Biocon is an emerging-market company.

### Required exhibits

- **Table 10:** FY2022-FY2026 ratios and Z, Z-double-prime and EM scores.
- **Figures 10-12:** Separate time-series plots for Z, Z-double-prime and EM.
- **Table 11:** FY2026 zone, rating-equivalent PD, DD and reverse-engineered volatility.

### Interpretation

For FY2026, Z is 2.428 (grey), Z-double-prime is 3.597 (safe) and EM is 6.847 (safe). EM maps mechanically to A, four notches below Biocon's observed AA+ rating. The corresponding PD is 7 bp, DD is 3.195 and reverse-engineered sigma_E is 67.098%. Explain the gap through model design: Altman uses a small set of historical accounting ratios, while CRISIL incorporates business profile, liquidity, management, industry conditions and forward-looking judgement.

## Section 6: Part E - Integrated comparison and discussion

### Required exhibits

- **Table 12:** Final route comparison, if Table 1 is not reused by cross-reference.
- **Figure 13:** Equity volatility from every route with the Part C band shaded.
- **Figure 14:** Reported PDs on a logarithmic scale with the 3 bp floor identified.
- A short paragraph comparing every gap in both DD and PD.

### Ordering and variance-risk-premium interpretation

Explain the observed ordering rather than comparing clipped PD alone. ATM IV produces the lowest sigma_E, followed by Random Forest and GARCH, with the rating and Altman reverse calibrations much higher. The IV-minus-GARCH difference is negative for Biocon at the main date, so the chosen short-dated ATM IV does not exhibit a positive variance-risk premium relative to the one-year GARCH measure. Qualify this comparison because the horizons differ: options expire within three months, while GARCH and Merton use one year.

### Three mandatory discussions: 100 to 150 words each

#### Discussion 1: Option maturity mismatch

Address how using options with less than three months to infer a one-year Merton volatility may bias DD. Cover term structure, short-dated event risk, mean reversion and the fact that annualising a short-horizon IV assumes persistence that may not hold.

#### Discussion 2: Did ML beat GARCH, and does better volatility imply better PD?

Use the actual negative out-of-sample R-squared and QLIKE results. Explain that forecast accuracy and PD quality are related but not identical because PD also depends on leverage, equity value, rates, structural assumptions and the volatility-to-PD nonlinear mapping.

#### Discussion 3: Early-warning recommendation

Recommend GARCH-Merton DD as the primary early-warning measure because it is timely, continuous and responsive to market conditions. Recommend using rating and Altman measures as slower-moving confirmation and challenge indicators. Mention sensitivity to market noise, model assumptions and the 3 bp floor.

## Final conclusion

Use one concise paragraph:

- State which route is most timely, which is most conservative and which is preferred for monitoring.
- Explain why the routes should not be expected to agree.
- Reiterate that floor-clipped PDs conceal meaningful DD differences.
- State the overall assessment of Biocon without claiming that default is impossible.

## Appendices

### Appendix A: Data sources and cleaning audit

- Exact source, file, report page/table and access date for every input.
- Corporate-action treatment.
- Shares-allotment schedule.
- Debt publication schedule.
- Dividend events.
- Option exclusions at each sequential cleaning stage.

### Appendix B: Complete assumption assessment

- Full BSM and Merton assumption tables from the notebook.

### Appendix C: Full model outputs

- Twelve-date ATM IV and Merton tables.
- Complete GARCH parameters, standard errors and diagnostics.
- ML hyperparameters, validation results and recency-weighting sensitivity.
- Rating reverse-calibration results.
- Altman input ratios and score mapping.

### Appendix D: Validation and reproducibility

- Own IV solver versus `py_vollib`.
- SciPy Merton solver versus `ItsSypher/merton`.
- Package versions.
- Input filenames and relevant checks.

## Source list to include

- Biocon annual reports, quarterly results and allotment filings.
- NSE historical prices, option bhavcopy and corporate actions.
- RBI 91-day and 364-day Treasury-bill auction workbooks and relevant release.
- CRISIL Biocon rating rationale and FY2026 default study.
- Altman references specified in the assignment.
- SciPy, `arch`, `statsmodels`, scikit-learn, `py_vollib` and `ItsSypher/merton`, with versions and stated use.

## Writing rules

- Use normal prose; do not describe calculations as "our model" when naming the actual method is clearer.
- Distinguish raw PD from the reported floor-adjusted PD.
- Distinguish point-in-time market measures from through-the-cycle rating measures.
- Do not claim causality from feature importance.
- Do not describe the Altman-equivalent rating as Biocon's actual credit rating.
- Do not claim Random Forest outperformed GARCH.
- Avoid copying the assignment wording verbatim; similarity above 20% is penalised.
- Keep detailed calculations in the notebook and appendix so the main report remains within 10 pages.

## Final pre-submission checklist

- [ ] Executive summary, final comparison table and main conclusion appear on page 1.
- [ ] Main report is no more than 10 pages excluding appendices.
- [ ] Every table and figure is numbered, titled and discussed in the text.
- [ ] Every external number has a source and exact page/table/file reference.
- [ ] All three Part E discussions are individually 100 to 150 words.
- [ ] GARCH parameters, standard errors and diagnostics are reported.
- [ ] Random Forest test performance is compared honestly with GARCH.
- [ ] The 3 bp reporting floor is explained.
- [ ] Low structural leverage and high DD are discussed without saying Biocon has no debt.
- [ ] CRISIL and Altman rating concepts are not conflated.
- [ ] The report figures and tables agree with the final notebook outputs.
- [ ] References include all datasets, papers and software repositories used.
