# Research Integrity and Reproducibility Audit

This document records the main implementation changes made to align the code with the submitted research question and with MScFE 690 source-code guidance.

## High-priority corrections implemented

1. **Volatility network alignment**: DAGMA is fitted to a residual-based volatility panel (`abs(Fama-French residual)`) rather than directly to residual returns.
2. **No-lookahead factor adjustment**: a walk-forward residualization path estimates post-initial-window factor betas using only historical observations.
3. **No backward filling**: equity and factor loaders no longer use `bfill`, which can propagate future information into earlier dates.
4. **Chronological structural-break thresholds**: break thresholds and drift z-scores are based only on prior drift observations. Historical events are descriptive labels, not calibration inputs.
5. **Untouched ML test period**: CatBoost early stopping uses a validation subset drawn from the pre-test period; the OOS test period is reserved for final evaluation.
6. **Forecast target corrected**: the CatBoost ablation defaults to forward realized volatility, matching the project design, rather than forward return.
7. **Comparable precision benchmark**: Graphical LASSO is fitted to the same volatility windows as DAGMA, and topology is thresholded using partial correlations rather than raw precision entries.
8. **Causal language softened**: low correlation and VIF are treated as empirical non-redundancy, not proof of mathematical orthogonality or causality.
9. **Rolling-window stability**: the primary DAGMA window is one trading year (252 days), updated approximately monthly. The submitted 60-day window remains available as a sensitivity analysis.
10. **Backtest leakage fixes**: no future drift is backward-filled, no graph is used before its estimation date, and drift scaling uses only historical observations. Transaction costs are applied to both actively rebalanced strategies.

## Repository status

This repository is private and owned by the connected capstone account. Group members should be given write access and the instructor/reviewer should be given the level of access required by the course.

## Data note

Research results should be treated as final only after they are reproduced from the documented real-data panel using the corrected pipeline. Synthetic fixtures are appropriate for automated tests only and should be labelled as such.
