# Final empirical results

Frozen research window: 2016-2025.

The U.S. and Indian runs use the same 252-day graph window, 21-day update cadence,
chronological factor residualization, five-day volatility target and out-of-sample split.

## Cross-market summary

| market | assets | observations | baseline_rmse | network_rmse | rmse_reduction_pct | network_sharpe | risk_parity_sharpe | network_max_drawdown | risk_parity_max_drawdown |
| ------ | ------ | ------------ | ------------- | ------------ | ------------------ | -------------- | ------------------ | -------------------- | ------------------------ |
| us     | 20     | 2513         | 0.141363      | 0.173384     | -22.652200         | 1.035000       | 1.077000           | -20.100000           | -20.990000               |
| india  | 20     | 2469         | 0.111221      | 0.104830     | 5.746127           | 1.060000       | 1.648000           | -18.870000           | -16.430000               |

Each market folder contains the run metrics, tables, figures and the SHA-256 manifest
for the frozen inputs. The raw frozen data are retained in the private GitHub Actions
artifacts rather than committed to the repository.
