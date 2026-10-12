# Structural Causal Volatility

MScFE 690 capstone project.

**Title:** Structural Causal Volatility Networks for Adaptive Risk Management in Equity Portfolios: Evidence Across Developed and Emerging Markets

**Group:** Halkano Molu Guracha, Feisel Hassan Ali, Tushar Gahlot

## What we are testing

We want to know whether a directed volatility network adds useful risk information after common market factors have been removed.

The main comparison is:

- DAGMA network
- Graphical LASSO network
- standard correlation features

The network features are then tested in a volatility forecast and a portfolio risk overlay.

## Pipeline

```text
returns
  -> Fama-French residuals
  -> residual volatility
  -> rolling DAGMA network
  -> Graphical LASSO benchmark
  -> network drift
  -> CatBoost volatility forecast
  -> portfolio test
```

The main code is in `catboost_dagma/`.

## Data

Set the return and factor files before running offline:

```bash
export SP100_DATA_PATH=/path/to/sp100_daily_returns_2016_2026.pkl
export FF_DATA_PATH=/path/to/fama_french_daily.csv
```

CSV and pickle files are supported. If `FF_DATA_PATH` is not set, the Fama-French series can be downloaded by running the pipeline with `offline=False`.

The research data are not stored in the repo.

## Setup

```bash
git clone https://github.com/Feiiiisal/structural_causal_volatility.git
cd structural_causal_volatility

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows:

```powershell
.venv\Scripts\activate
```

## Tests

```bash
pytest -q catboost_dagma/tests/test_research_integrity_patch.py
```

## Notes

This repo was rebuilt from the group's earlier implementation. The old code and generated outputs remain available in Git history, but they are not part of the current working tree.

Final numerical results should come from a fresh run on the final dataset.

The main review fixes are listed in [docs/reports/research_integrity_audit.md](docs/reports/research_integrity_audit.md).
