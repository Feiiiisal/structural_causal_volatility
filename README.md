# Structural Causal Volatility Networks for Adaptive Risk Management

MScFE 690 Capstone Project

**Project title:** Structural Causal Volatility Networks for Adaptive Risk Management in Equity Portfolios: Evidence Across Developed and Emerging Markets

**Contributors**
- Halkano Molu Guracha
- Feisel Hassan Ali
- Tushar Gahlot

## Research question

The project asks whether a directed, time-varying volatility network contains useful risk information beyond common systematic factor exposure and conventional symmetric dependence measures, and whether that information improves out-of-sample portfolio risk management.

## Current implementation status

This repository was initialized from the earlier project implementation in `abuhaby/causal_graph_volatility` and then revised to address the research-integrity issues identified during the capstone review.

The corrected research pipeline is now the primary implementation. Figures, notebooks and numerical results imported from the earlier repository are retained as historical development artifacts. **They should not be treated as final capstone results until they are reproduced with the corrected no-lookahead pipeline and documented real-data panel.**

The developed-market implementation currently uses a U.S. equity universe. The emerging-market comparison remains part of the capstone design and should be added before making final cross-market claims.

## Corrected research pipeline

```text
Equity returns
    ↓
Chronological Fama-French factor adjustment
    ↓
Idiosyncratic residuals
    ↓
Residual-based volatility measure |εᵢ,t|
    ↓
Rolling Student-t DAGMA / DYNOTEARS network
    ↓
Graphical LASSO partial-correlation benchmark
    ↓
Past-only network-drift / structural-change features
    ↓
CatBoost forward-volatility ablation
    ↓
Network-aware portfolio risk overlay
    ↓
Strict out-of-sample evaluation
```

### Main methodological safeguards

- **Factor adjustment without future information.** After the initial historical window, factor betas are re-estimated only from observations available at the time.
- **Volatility-network alignment.** DAGMA is fitted to residual-based volatility rather than raw or residual returns.
- **No backward filling.** Missing observations are not filled with future values.
- **Stable rolling specification.** The primary DAGMA window is 252 trading days with a 21-day update cadence. The earlier 60-day specification is retained only as a sensitivity test.
- **Comparable dependence benchmark.** Graphical LASSO is evaluated on the same volatility windows and converted to partial correlations before topological thresholding.
- **Past-only structural-change thresholds.** Frobenius network drift is compared with thresholds estimated only from prior drift observations.
- **Untouched test period.** CatBoost validation/early stopping is performed before the out-of-sample test period.
- **Volatility forecasting target.** The ML ablation predicts forward realized volatility rather than forward return.
- **Backtest safeguards.** No graph is used before its estimation date, future drift is not backward-filled, and transaction costs are applied consistently to actively rebalanced strategies.
- **Careful causal language.** Low correlation or low VIF between network features is reported as empirical non-redundancy, not as mathematical proof of orthogonality or causality.

See [docs/reports/research_integrity_audit.md](docs/reports/research_integrity_audit.md) for the audit trail.

## Data

The real equity return panel is intentionally not hard-coded to one contributor's computer and is not committed to the repository.

Set the path before running the Version 2 pipeline:

```bash
export SP100_DATA_PATH=/absolute/path/to/sp100_daily_returns_2016_2026.pkl
```

A CSV return panel can also be supplied. Automated-test fixtures are synthetic and are for software validation only; they are not a substitute for the research dataset.

## Installation

```bash
git clone https://github.com/Feiiiisal/structural_causal_volatility.git
cd structural_causal_volatility

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Running the research-integrity tests

```bash
pytest -q catboost_dagma/tests/test_research_integrity_patch.py
```

The broader legacy test suite remains in `tests/` and `catboost_dagma/tests/`. Some legacy integration tests depend on datasets that are not committed to the repository and should be run only when those inputs are configured.

## Main project directories

```text
catboost_dagma/
├── benchmark/        # Graphical LASSO and empirical network comparison
├── breaks/           # Past-only structural-change detection
├── dagma/            # Student-t DAGMA / DYNOTEARS implementation
├── data/             # Data loading and factor residualization
├── ml/               # Feature engineering and CatBoost ablation
├── strategy/         # Network-aware portfolio application
├── tests/            # Version 2 and research-integrity tests
└── pipeline.py       # Corrected end-to-end research pipeline

src/causal_volatility/ # Earlier modular causal-volatility framework
notebooks/             # Earlier exploratory and validation notebooks
figures/               # Earlier generated diagnostic figures
docs/                  # Specifications, reports and audit notes
```

## Reproducibility and interpretation

The project separates **software tests** from **empirical research results**. Passing unit tests establishes that the implementation behaves as intended on controlled inputs; it does not validate the economic hypothesis. Final numerical claims in the capstone report should be tied to a dated pipeline run, documented data inputs, and an untouched out-of-sample evaluation.

## Repository provenance

The initial codebase was imported from the group's earlier repository and preserved here so that the capstone has a private, writable repository under the connected group account. Research-integrity corrections are recorded in the commit history of this repository.

## License

MIT, consistent with the imported project metadata.
