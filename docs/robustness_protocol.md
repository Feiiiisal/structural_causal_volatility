# Robustness and uncertainty protocol

This analysis was added in response to peer-review comments about uncertainty, model sensitivity and sample dependence. The final base specification is not re-selected after seeing these results.

## Forecast uncertainty

The baseline and network-augmented CatBoost forecasts are compared on the same out-of-sample dates and assets.

Two checks are reported:

- a paired 20-trading-day moving-block bootstrap with 2,000 replications for the percentage RMSE reduction;
- a HAC loss-differential test using the daily cross-sectional mean squared-error difference and five lags, reflecting the overlapping five-day forecast horizon.

A positive RMSE-reduction interval means the network model improves the baseline forecast. An interval crossing zero means the gain is not stable enough to distinguish from sampling variation at the 95% level.

## Portfolio uncertainty

Daily risk-parity and network-overlay returns are resampled together using 20-trading-day moving blocks. The bootstrap reports 95% intervals for:

- the Sharpe-ratio difference, network minus risk parity;
- the maximum-drawdown difference, network minus risk parity.

Because drawdowns are negative, a positive drawdown difference means the network overlay has a less severe drawdown.

## Sensitivity checks

The following reviewer-motivated alternatives are evaluated without replacing the frozen base specification:

1. 126-day graph window;
2. 504-day graph window;
3. squared residuals instead of absolute residuals;
4. 90th-percentile structural-break threshold;
5. 97.5th-percentile structural-break threshold;
6. 2.5-basis-point one-way transaction cost;
7. 10-basis-point one-way transaction cost;
8. prune one high-transmitter asset instead of two;
9. prune three high-transmitter assets instead of two;
10. a 15-stock composition check that excludes the five stocks with the highest volatility during the pre-2021 training period.

The composition check is not a cure for survivorship bias. It tests whether the main result is unusually dependent on the most volatile names while preserving the test period. Full point-in-time index reconstruction remains outside the available frozen data and is therefore retained as a limitation rather than imputed.
