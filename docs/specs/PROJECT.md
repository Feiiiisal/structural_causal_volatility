# Project: Causal Graph Volatility Refactoring & Framework Expansion

## Architecture
The `causal_volatility` package refactors monolithic research notebooks and scripts into an enterprise-grade, modular quantitative finance library for systematic risk and volatility analysis.

### System Flow
```
Data Ingestion (Yahoo Finance / FRED with 3-tier fallback & offline caching)
  │
  ▼
Data Processing & Realized Volatility Estimators (Garman-Klass, Yang-Zhang, Parkinson, etc.)
  │
  ▼
Stationarity Transformations & Dual Diagnostic Suite (ADF + KPSS, Ljung-Box, Engle ARCH)
  │
  ▼
Conditional Volatility Residualization (AR-GARCH, EGARCH, GJR-GARCH, Student-t GARCH)
  │
  ▼
Structural Causal Discovery (Rank-Deficiency-Free Bivariate Granger DAG with Momentary Controls)
  │
  ▼
Adaptive Causal Multiplier (Z-score weighting, EWM smoothing, 252-day rolling percentile)
  │
  ▼
Backtesting Engine (Vectorized Trailing Stop Ratchet State Machine with Regime Re-entry)
  │
  ▼
Performance & Risk Metrics Evaluator (Sharpe, Volatility, Returns, Drawdowns, Stop-outs)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Package Config & Env Setup | Modern `pyproject.toml`, clean venv with dependencies, package root structure | M1 | Survey |
| 2 | Systematic Risk Data Ingestion | Multi-source fetcher (OEF, VIX from Yahoo Finance, Moody's BAA10Y from FRED) with 3-tier fallback | M1 | Survey |
| 3 | Market Calendar Alignment & Cleaning | Trading day filtering, forward/backfill of macro series, liquidity index | M1 | Survey |
| 4 | Realized Volatility Estimators | Base estimator ABC, Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang, Close-to-Close | M1 | Survey |
| 5 | Offline Fixture Caching | Frozen offline historical data fixtures for zero-network testing | M1 | Survey |
| 6 | Stationarity Transformations | Log/first differences with $\epsilon=10^{-8}$ safety against zero-variance/flat days | M2 | Survey |
| 7 | Dual Stationarity Diagnostic Suite | Automated ADF and KPSS testing with clean lookup table warning handling | M2 | Survey |
| 8 | Serial & ARCH Dependence Diagnostics | Ljung-Box Q-test and Engle ARCH LM-test on returns, volatilities, and residuals | M2 | Survey |
| 9 | Optimal AR Lag Selector | Automated AIC/BIC search over autoregressive lag orders $p \in [1, 21]$ | M2 | Survey |
| 10 | Volatility Residualization Engines | BaseVolatilityModel ABC, standard AR(p)-GARCH(1,1), Student-t GARCH | M2 | Survey |
| 11 | Advanced Asymmetric Volatility Models | EGARCH (Nelson leverage effect) and GJR-GARCH (threshold asymmetry) | M2 | Survey |
| 12 | Structural Causal Discovery (R2 Fix) | Bivariate Granger regressions with target self-lag collinearity fix (zero rank deficiency) | M3 | Survey |
| 13 | Causal Graph & Pathway Analysis | Pathway filtering at $\alpha=0.05$, parent extraction, DAG edge summaries | M3 | Survey |
| 14 | Dynamic Causal Multiplier | Composite shock z-scoring, EWM-10 smoothing, 252-day rolling percentile mapping to $[\lambda_{min}, \lambda_0]$ | M3 | Survey |
| 15 | Vectorized Ratchet Trailing Stop Engine | High-performance 1D numpy state machine with calm/stormy regime re-entry logic | M4 | Survey |
| 16 | Comprehensive Risk Metrics | Annualized return, annualized volatility, Sharpe ratio, max drawdown, stop-out event counting | M4 | Survey |
| 17 | Multi-Horizon Validation Suite | 10-year 50/50 OOS split, 25-year 60/20/20 split, and 5-fold expanding walk-forward validation | M4 | Survey |
| 18 | Unified Pipeline Facade & CLI | Programmatic pipeline facade and CLI entry point (`causal-volatility`) | M4 | Survey |
| 19 | E2E Test Suite & Baseline Matching | 4-tier verification suite matching `output.log` and OOS metrics within $\le 10^{-2}$ tolerance | M5 (Final) | Survey |
| 20 | Adversarial Coverage Hardening | White-box stress testing, extreme edge cases, synthetic market shocks | M5 (Final) | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Package Foundation & Data Layer | Pyproject setup, environment, data ingestion (YF+FRED+fallbacks), data processing, realized volatility estimators (GK, Parkinson, Yang-Zhang, etc.), offline fixtures | none | DONE |
| M2 | Stationarity & Volatility Models (R3) | Stationarity transforms, ADF/KPSS diagnostics, AR lag selection, BaseVolatilityModel ABC, standard GARCH, Student-t GARCH, EGARCH, GJR-GARCH | M1 | DONE |
| M3 | Causal Discovery & Multiplier (R2) | Rank-deficiency-free Granger causal discovery, graph extraction, dynamic adaptive risk multiplier | M2 | PLANNED |
| M4 | Backtest Engine & CLI Pipeline | Vectorized trailing stop ratchet engine, risk metrics, validation schemes (OOS, 60/20/20, walk-forward), CLI runner | M3 | PLANNED |
| M5 | E2E Test Pass & Adversarial Hardening | Pass 100% of the 4-tier E2E test suite (Phase 1) + Adversarial coverage hardening (Phase 2) | M4, E2E Track | PLANNED |

## Parallel Track: E2E Testing Track
| Track | Scope | Dependencies | Status |
|-------|------|-------------|--------|
| E2E Testing Track | Independent opaque-box test suite derivation from requirements: test infra, Tier 1-4 tests (Category-Partition, BVA, Pairwise, Workload, baseline matching vs output.log), publishes `TEST_READY.md` | none (runs in parallel) | DONE |

## Interface Contracts

### Data Layer ↔ Estimators
- Data Ingestion outputs `pd.DataFrame` indexed by `pd.DatetimeIndex` with columns:
  `['SP100_Open', 'SP100_High', 'SP100_Low', 'SP100_Close', 'SP100_Volume', 'VIX_Close', 'Credit_Spread']`
- Estimator signatures: `estimate(high: pd.Series, low: pd.Series, close: pd.Series, open: pd.Series) -> pd.Series`
- Realized volatility series is strictly non-negative (`clip(lower=0.0)`).

### Estimators ↔ Stationarity & Volatility Models
- Data Processor produces aligned `DataFrame` with columns:
  `['SP100_Close', 'Garman_Klass_Vol', 'VIX_Close', 'Credit_Spread', 'Liquidity_Proxy']`
- Stationarity Transformer outputs `stationary_df` with columns:
  `['SP100_Returns', 'GK_Vol_Diff', 'VIX_Diff', 'Credit_Spread_Diff', 'Liquidity_Diff']`
- `BaseVolatilityModel` contract:
  * `fit(vol_series: pd.Series) -> BaseVolatilityModel`
  * `get_standardized_residuals() -> pd.Series` (standardized innovations $z_t$)
  * `get_conditional_volatility() -> pd.Series` ($\sigma_t$)
  * `get_diagnostics(lags: int = 10) -> dict` (`ljung_box_p`, `arch_lm_p`, `is_white_noise`, `aic`, `bic`)

### Volatility Models ↔ Causal Discovery
- Causal Discovery input: `pd.DataFrame` containing `Vol_Innovations` (standardized residuals from any volatility model) along with macro shocks `['VIX_Diff', 'Credit_Spread_Diff', 'Liquidity_Diff']`.
- Function signature:
  `execute_structural_causal_discovery(df: pd.DataFrame, target_var: str = 'Vol_Innovations', tau_max: int = 5, alpha_thresh: float = 0.05) -> dict`
- Returns dictionary with:
  `p_matrix: np.ndarray (n_nodes, n_nodes, tau_max+1)`
  `val_matrix: np.ndarray (n_nodes, n_nodes, tau_max+1)`
  `var_names: list[str]`
- Guarantee: **Zero** `SingularMatrixWarning` or rank-deficient OLS errors.

### Causal Discovery ↔ Multiplier & Backtest
- Multiplier signature:
  `construct_causal_multiplier(df_causal: pd.DataFrame, causal_output: dict, lambda_0: float = 2.0, lambda_min: float = 1.3) -> pd.Series`
- Returns dynamic multiplier series bounded in $[1.3, 2.0]$.
- Backtest engine signature:
  `run_trailing_stop(prices: pd.Series, vol_series: pd.Series, multiplier: pd.Series) -> pd.DataFrame`
- Returns backtest history:
  `['Close', 'Base_Vol', 'Multiplier', 'Close_MA5', 'Stop_Level', 'Invested_State']`
- Risk metrics signature:
  `evaluate_risk_metrics(backtest_df: pd.DataFrame) -> dict`
  Returns `{'annual_return': float, 'annual_volatility': float, 'sharpe_ratio': float, 'max_drawdown': float, 'stop_outs': int}`

## Code Layout
```
/home/oem/Documents/causal graph volatility/
├── pyproject.toml
├── README.md
├── src/
│   └── causal_volatility/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetcher.py
│       │   ├── processor.py
│       │   └── storage.py
│       ├── estimators/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── garman_klass.py
│       │   ├── parkinson.py
│       │   ├── rogers_satchell.py
│       │   ├── yang_zhang.py
│       │   └── close_to_close.py
│       ├── stationarity/
│       │   ├── __init__.py
│       │   ├── transform.py
│       │   └── diagnostics.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── selection.py
│       │   ├── garch.py
│       │   ├── egarch.py
│       │   └── gjr_garch.py
│       ├── causal/
│       │   ├── __init__.py
│       │   ├── discovery.py
│       │   ├── pathways.py
│       │   └── multiplier.py
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── metrics.py
│       │   └── validation.py
│       ├── pipeline.py
│       └── cli.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── tier1_unit/
    ├── tier2_integration/
    ├── tier3_regression/
    └── tier4_e2e/
```
