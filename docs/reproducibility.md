# Reproducibility

## Environment

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Inputs

The pipeline expects:

1. a daily equity-return panel with dates in the index and tickers in columns;
2. a daily factor panel containing `Mkt-RF`, `SMB`, `HML` and `RF`.

CSV and pickle files are supported.

Set:

```bash
export EQUITY_DATA_PATH=/path/to/equity_returns.pkl
export FF_DATA_PATH=/path/to/factors.csv
```

`EQUITY_DATA_PATH` can point to either the developed- or emerging-market return panel used for a given run.

## Run

```bash
python catboost_dagma/scripts/run_experiment.py \
  --returns-path /path/to/equity_returns.pkl \
  --factor-path /path/to/factors.csv
```

The Python entry point is:

```python
from catboost_dagma.pipeline import CatBoostDagmaPipeline

results = CatBoostDagmaPipeline(
    returns_path="/path/to/equity_returns.pkl",
    factor_path="/path/to/factors.csv",
).run()
```

## Validation rules

The empirical workflow follows these chronology rules:

- no backward filling of returns or factors;
- factor betas are estimated with historical data only;
- graph windows end before their forecast targets begin;
- structural-break thresholds use previous drift observations only;
- CatBoost validation data come from the pre-test period;
- the test set is not used for early stopping;
- transaction costs are applied to active portfolio rules.

## Tests

```bash
pytest -q catboost_dagma/tests
```

The tests cover data alignment, residualization, Student-t DAG fitting, precision-network construction, network feature comparisons, structural-break chronology, forecasting splits and portfolio mechanics.

GitHub Actions runs the test suite automatically.

## Randomness

Synthetic tests use fixed NumPy seeds. CatBoost uses a fixed random seed from `catboost_dagma/config.py`.

## Data provenance

Market data should be archived outside the repository with:

- source;
- download date;
- market;
- constituent definition;
- adjustment method;
- currency;
- date range;
- missing-data treatment.

The final paper should identify the exact frozen datasets used for each reported table and figure.

## Results archive

For a final research run, save the following together:

- configuration used;
- commit SHA;
- data manifest;
- summary tables;
- generated figures;
- model diagnostics.

This makes each reported result traceable to one code version and one data snapshot.
