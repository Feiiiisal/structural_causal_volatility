# Research design

## Objective

The project tests whether directed volatility networks contain incremental information for portfolio risk management after systematic factor exposure has been removed.

The empirical design is deliberately comparative. A directed network is not treated as useful simply because it is more complex. It must add information beyond correlation, Graphical LASSO and conventional volatility features, and that information must survive out-of-sample testing.

## Markets

Two equity settings are considered:

1. **Developed market:** large-cap U.S. equities.
2. **Emerging market:** Indian equities, using a liquid point-in-time universe such as the Nifty 50 constituent set available at each date.

The cross-market comparison uses the same model architecture, rolling-window design and evaluation metrics. Factor data must be market appropriate.

## Sample construction

The preferred common research window is 2016-2025 because it provides a long daily history and allows the two markets to be compared over the same calendar period.

Constituent membership should be handled point in time where the required history is available. A fixed end-of-sample constituent list can create survivorship bias and should be disclosed if used as a sensitivity sample.

## Stage 1: factor residualization

Daily excess returns are regressed on market, size and value factors. The model is re-estimated through time. Residuals therefore use only information available before or at their date.

The purpose is narrower than causal identification: residualization removes measured common-factor exposure. It cannot rule out all omitted common drivers.

## Stage 2: idiosyncratic volatility

The main graph input is the absolute residual,

\[
v_{i,t}=|\varepsilon_{i,t}|.
\]

Squared residuals can be used as a sensitivity check.

## Stage 3: directed network

A rolling Student-t neural DAG model is estimated under a DAGMA log-determinant acyclicity constraint. The primary window is 252 trading days and the graph is updated every 21 trading days.

Reported node summaries include in-degree, out-degree and eigenvector centrality. Out-degree is interpreted as a transmitter score only within the fitted model; it is not treated as structural causality without qualification.

## Stage 4: dependence benchmarks

The same residual-volatility window is used to estimate:

- a correlation network;
- a Graphical LASSO precision matrix converted to partial correlations.

Directed and undirected centralities are compared using Pearson/Spearman association and variance-inflation diagnostics.

## Stage 5: structural change

Network drift is measured by the Frobenius distance between consecutive adjacency matrices. The break threshold is estimated from past drift observations. Historical crisis labels are used after detection for interpretation, not for setting the threshold.

## Stage 6: volatility forecasting

The forecast target is five-day forward realized volatility. The baseline CatBoost model contains momentum, rolling volatility, correlation and precision-network features. The augmented model adds directed-network features and structural-change indicators.

The final test period is not used for early stopping.

## Stage 7: portfolio evaluation

The network-aware allocation is compared with equal-weight buy and hold and inverse-volatility risk parity.

Evaluation focuses on:

- annualized return;
- annualized volatility;
- Sharpe ratio;
- maximum drawdown;
- Calmar ratio;
- turnover and transaction costs.

The portfolio test is a risk-management experiment. It is not framed as proof that causal discovery creates alpha.

## Cross-market comparison

The final comparison should report the same diagnostics for the United States and India:

- residualization effect;
- network sparsity and centrality distribution;
- network/precision non-redundancy;
- structural-break behaviour;
- forecast ablation;
- portfolio risk metrics.

A cross-market conclusion is supported only if both panels are estimated under comparable chronology and data-quality rules.

## Sensitivity analysis

The main results should be checked against a small set of pre-specified alternatives:

- graph windows of 126, 252 and 504 trading days;
- absolute and squared residual volatility;
- alternative Graphical LASSO penalties selected by cross-validation;
- transaction-cost assumptions above and below the base case;
- different hub-pruning counts;
- alternative structural-break percentiles;
- subperiods that separate calm and stressed market regimes.

The purpose is not to search for the best-performing specification. It is to test whether the main conclusion survives reasonable design choices.

## Threats to validity

**Causal interpretation.** The graph is estimated from observational data. Factor residualization reduces measured common exposure but does not remove every latent confounder.

**Survivorship bias.** A fixed end-of-sample constituent list can overstate historical performance. Point-in-time membership is preferred.

**Factor mismatch.** The U.S. and Indian panels require market-appropriate factors. Reusing U.S. factors for India would make the residualization stage difficult to defend.

**Non-stationarity.** Network structure, volatility dynamics and factor exposures can change across regimes. Rolling estimation limits but does not remove this problem.

**Model-selection risk.** Window length, graph sparsity, break thresholds and portfolio rules can be overfit. Sensitivity tests are therefore reported alongside the main specification.

**Economic significance.** A statistically detectable forecasting gain may be too small to matter after turnover and trading costs. Forecast and portfolio results are evaluated separately.

## Method references

- Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Friedman, J., Hastie, T. and Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441.
- Diebold, F. X. and Yilmaz, K. (2014). On the network topology of variance decompositions. *Journal of Econometrics*, 182(1), 119-134.
- Pamfil, R. et al. (2020). DYNOTEARS: Structure learning from time-series data. *Proceedings of Machine Learning Research*, 108, 1595-1605.
- Bello, K., Aragam, B. and Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization. *Advances in Neural Information Processing Systems*, 35.
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.

