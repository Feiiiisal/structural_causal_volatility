# Code changes made during review

The earlier version had a few places where future data could enter the analysis or where the code did not match the stated method.

The current version changes the following:

- Fama-French betas are updated through time instead of using the full sample.
- DAGMA is fitted to residual-based volatility, not directly to residual returns.
- Missing data are not backward-filled.
- The main rolling window is 252 trading days. The old 60-day setup can still be tested separately.
- Graphical LASSO is fitted on the same windows as DAGMA and compared using partial correlations.
- Network-break thresholds use past drift values only.
- CatBoost predicts forward volatility. Its test period is not used for early stopping.
- The backtest does not use a graph before that graph exists.
- Transaction costs are applied to both active portfolio rules.
- Low feature correlation is described as non-redundancy, not proof of causality or orthogonality.

The old reports and figures are kept as development records. Final results should come from a fresh run of the current pipeline on the final research dataset.
