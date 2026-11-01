# Capstone pipeline

The `catboost_dagma` package contains the empirical pipeline used in the project.

## Modules

- `data/`: market data loading and Fama-French residualization
- `dagma/`: rolling directed network estimation
- `benchmark/`: Graphical LASSO and network-feature comparisons
- `breaks/`: network-drift and structural-change detection
- `ml/`: forecasting features and CatBoost ablation
- `strategy/`: portfolio experiment
- `visualization/`: report figures
- `tests/`: statistical and chronology checks
- `pipeline.py`: end-to-end orchestration

The main graph window is 252 trading days with a 21-day update step. DAGMA and Graphical LASSO use the same residual-volatility windows. Structural-break thresholds use past observations only. The CatBoost test period is held out from early stopping and model selection.

Run the full test suite with:

```bash
pytest -q catboost_dagma/tests
```

Run an experiment with:

```bash
python catboost_dagma/scripts/run_experiment.py \
  --returns-path /path/to/equity_returns.pkl \
  --factor-path /path/to/factors.csv
```
