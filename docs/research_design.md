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
