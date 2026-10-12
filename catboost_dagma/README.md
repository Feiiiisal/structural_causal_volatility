# catboost_dagma

This folder contains the current capstone pipeline.

## Files

- `data/` loads returns and removes Fama-French factor exposure.
- `dagma/` fits the rolling directed network.
- `benchmark/` builds the Graphical LASSO comparison.
- `breaks/` measures changes in the network.
- `ml/` builds features and runs the CatBoost ablation.
- `strategy/` contains the portfolio test.
- `pipeline.py` runs the steps together.

## Current settings

The main DAGMA window is 252 trading days and is updated every 21 trading days.

DAGMA and Graphical LASSO use the same residual-volatility windows. Structural-break thresholds use past observations only. CatBoost predicts forward volatility and keeps the test period separate from model selection.

## Run

Set `SP100_DATA_PATH` first, then call the pipeline from Python:

```python
from catboost_dagma.pipeline import CatBoostDagmaPipeline

result = CatBoostDagmaPipeline().run()
```

For the focused tests:

```bash
pytest -q catboost_dagma/tests/test_research_integrity_patch.py
```

The notebook is only a small runner. Generate report figures from a fresh pipeline run before using numerical results.
