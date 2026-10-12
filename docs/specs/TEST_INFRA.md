# TEST_INFRA.md: 4-Tier Test Architecture & Verification Infrastructure

## 1. Overview & Testing Philosophy

The `causal_volatility` test suite is designed as an independent, requirement-driven, opaque-box testing framework. It rigorously validates the quantitative finance library across four distinct test tiers, ensuring mathematical precision, architectural robustness, numerical stability, and exact regression alignment with published historical baselines.

### 1.1 Scope & Traceability
The test infrastructure directly validates the core project requirements outlined in `PROJECT.md` and `ORIGINAL_REQUEST.md`:
- **R1 (Modular Refactoring)**: Clean public interfaces, modular separation of data ingestion, feature calculation, stationarity, volatility modeling, causal discovery, and backtesting.
- **R2 (Bug Fixes & Stability)**: Elimination of statsmodels `SingularMatrixWarning` / rank deficiency in structural Granger causal discovery through regressor matrix purification, verified to emit exactly zero warnings.
- **R3 (Framework Expansion)**: Alternative volatility residualization engines (standard AR-GARCH, EGARCH, GJR-GARCH, Student-$t$ GARCH) adhering to a unified `BaseVolatilityModel` contract and pluggable into the downstream causal DAG pipeline.
- **Verification Baseline**: Exact numerical reproduction of in-sample and out-of-sample risk metrics recorded in `output.log` and `causal_volatility_framework_OOS.txt` within a relative tolerance $\le 10^{-2}$ or absolute tolerance $\le 0.002$.

---

## 2. Directory Layout & Test Suite Hierarchy

```
tests/
├── conftest.py                       # Global test harness, fixtures, mock generators, tolerance assertions
├── fixtures/                         # Cached offline market data and macro series
│   └── market_data_2016_2026.csv     # Frozen historical OHLCV, VIX, and Credit Spread
├── tier1_unit/                       # Tier 1: Fast isolated unit tests (< 2s total)
│   ├── test_realized_volatility.py   # Analytical math, zero vol, negative variance clipping
│   ├── test_stationarity_math.py     # Log differences, epsilon safety against zero-variance
│   ├── test_ols_rank_condition.py    # Regressor matrix condition number, zero SingularMatrixWarning
│   ├── test_multiplier_bounds.py     # Dynamic risk multiplier strictly bounded in [1.3, 2.0]
│   └── test_ratchet_state_machine.py # Trailing stop ratchet invariant, regime re-entry logic
├── tier2_integration/                # Tier 2: Boundary & integration tests (< 15s total)
│   ├── test_fred_fallback.py         # 3-tier FRED fallback hierarchy (API key -> CSV -> synthetic proxy)
│   ├── test_stationarity_pipeline.py # Dual stationarity (ADF + KPSS) & dependence tests
│   ├── test_volatility_contracts.py  # BaseVolatilityModel contract across GARCH, EGARCH, GJR, Student-t
│   └── test_causal_dag_recovery.py   # Ground-truth causal recovery on synthetic VAR(1) processes
├── tier3_regression/                 # Tier 3: Baseline numerical regression & warning audit
│   ├── test_output_log_metrics.py    # Exact in-sample and out-of-sample metrics matching output.log
│   └── test_warning_free_run.py      # Full pipeline execution asserting zero SingularMatrixWarnings
└── tier4_e2e/                        # Tier 4: Real-world workload & system tests (< 60s total)
    ├── test_cli_execution.py         # CLI entry point (`causal-volatility`) in offline mode
    ├── test_full_pipeline_run.py     # End-to-end ingestion -> discovery -> backtest workflow
    ├── test_multi_model_run.py       # Comparative backtest execution across all 4 volatility models
    └── test_walk_forward_run.py      # 5-fold expanding window walk-forward validation
```

---

## 3. Test Tiers Specification

### Tier 1: Unit Tests (`tests/tier1_unit/`)
- **Execution Target**: $< 2$ seconds, 100% deterministic, zero network I/O.
- **Coverage Criteria**: $\ge 5$ test cases per feature covering boundary, typical, and degenerate inputs.
- **Features Tested**:
  1. `test_realized_volatility.py`:
     - Analytical accuracy on known closed-form OHLC inputs for Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang, and Close-to-Close estimators.
     - Zero-volatility flat trading days ($H=L=O=C \implies \sigma=0.0$).
     - Overnight gap edge condition: $(2\ln 2 - 1)(\ln(C/O))^2 > 0.5(\ln(H/L))^2$ asserting explicit clipping to $0.0$ rather than NaN.
     - Non-negative invariant: $\forall t, \sigma_t \ge 0.0$.
     - Extreme market swings (e.g. 50% intraday drop) validating finite non-overflow outputs.
  2. `test_stationarity_math.py`:
     - First-order logarithmic differencing $\Delta \ln P_t$.
     - Zero-variance safety: verifying $\ln(\sigma_t + \epsilon)$ with $\epsilon = 10^{-8}$ prevents $-\infty$ or NaN on flat days.
     - Preservation of time index alignment following differencing.
     - Liquidity proxy scaling by $10^7$: $L_t = \text{Volume}_t / 10^7$.
  3. `test_ols_rank_condition.py`:
     - Regressor matrix assembly for bivariate Granger causality testing.
     - Self-causality exclusion verification: when $source\_var == target\_var$, candidate shock is skipped or autoregressive lag $\tau$ is isolated, preventing bitwise identical columns.
     - Column rank verification: $\text{rank}(X) == \text{cols}(X)$.
     - Condition number bounds: $\kappa(X) < 10^3$.
     - Explicit warning check: `warnings.catch_warnings()` asserting zero `SingularMatrixWarning` or `Rank-deficient` messages.
  4. `test_multiplier_bounds.py`:
     - Causal multiplier mapping under extreme positive shocks ($z \gg 3$) yielding $\lambda_{min} = 1.3$.
     - Multiplier mapping under zero or negative shock ($z \le 0$) yielding $\lambda_0 = 2.0$.
     - Monotonicity: higher causal systemic stress monotonically decreases multiplier $\lambda_t$.
     - Strict boundary invariant: $\forall t, 1.3 \le \lambda_t \le 2.0$.
     - Smoothness test: EWM-10 smoothing produces bounded first-differences.
  5. `test_ratchet_state_machine.py`:
     - Ratchet non-decreasing invariant: While invested ($state=1$), $\text{Stop}_t \ge \text{Stop}_{t-1}$.
     - Stop-out trigger: When $\text{Close}_t < \text{Stop}_{t-1}$, state transitions to $0$ on day $t$.
     - Calm regime re-entry ($\lambda_t \ge 1.8$): Re-entry occurs when $\text{Close}_t > \text{MA5}_t$.
     - Stormy regime re-entry ($\lambda_t < 1.8$): Re-entry requires both $\text{Close}_t > \text{MA5}_t$ and $\text{Close}_t > \text{Close}_{t-1}$.
     - Extreme gap-down handling: Order execution and stop-level reset behavior.

---

### Tier 2: Integration Tests (`tests/tier2_integration/`)
- **Execution Target**: $< 15$ seconds, uses local fixtures and mock HTTP transports.
- **Coverage Criteria**: $\ge 5$ test cases per interface boundary.
- **Features Tested**:
  1. `test_fred_fallback.py`:
     - Mode 1: Valid FRED API key yields parsed corporate bond credit spread DataFrame.
     - Mode 2: Missing / invalid key falls back to direct FRED CSV download stream.
     - Mode 3: Network disconnection / HTTP 429/500 falls back to deterministic synthetic proxy series emitting a descriptive warning.
     - Caching verification: Once fetched, subsequent calls load from disk cache without network requests.
     - Calendar alignment: Re-indexing macro series against NYSE equity trading calendar.
  2. `test_stationarity_pipeline.py`:
     - Execution of automated Augmented Dickey-Fuller (ADF) test.
     - Execution of Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test with lookup table warning suppression.
     - Ljung-Box autocorrelation test across 10 lags.
     - Engle ARCH LM heteroskedasticity test across 10 lags.
     - Pipeline diagnostic summary table compilation matching expected schema.
  3. `test_volatility_contracts.py`:
     - BaseVolatilityModel contract verification across 4 implementations:
       * `ARGARCHModel` (Gaussian AR(p)-GARCH(1,1))
       * `EGARCHModel` (Nelson Exponential GARCH with asymmetry parameter $\gamma$)
       * `GJRGARCHModel` (Threshold GJR-GARCH with leverage indicator)
       * `StudentTGARCHModel` (Heavy-tailed Student-$t$ innovations)
     - For each model, verify:
       * `fit(vol_series)` returns model instance.
       * `get_standardized_residuals()` has mean $\approx 0.0$ and unit variance $\approx 1.0$.
       * `get_conditional_volatility()` produces strictly positive $\sigma_t > 0.0$.
       * `get_diagnostics()` returns `ljung_box_p`, `arch_lm_p`, `aic`, `bic`.
  4. `test_causal_dag_recovery.py`:
     - Synthetic bivariate VAR(1) process with known causal edge $X_{t-1} \to Y_t$ ($\beta = 1.5, p < 0.001$).
     - Non-causal noise series $Z_t \not\to Y_t$ ($\beta \approx 0.0, p > 0.05$).
     - Verify `execute_structural_causal_discovery` correctly identifies $X$ as causal parent at lag 1 with $p < 0.05$.
     - Verify $Z$ is rejected.
     - Verify all regressions complete with zero `SingularMatrixWarning`.

---

### Tier 3: Regression Tests (`tests/tier3_regression/`)
- **Execution Target**: Deterministic verification against authoritative frozen baseline `output.log`.
- **Tolerance**: Relative tolerance $\le 10^{-2}$ (1%) or absolute tolerance $\le 0.002$ on ratios.
- **Reference Targets (from `output.log`)**:
  1. **In-Sample Strategy Comparison (TRAIN: 2016-01-05 to 2020-12-29, 1256 rows)**:
     - **Standard Baseline ($\lambda = 2.0$)**:
       * Annualized Return: $11.66\% \pm 0.15\%$
       * Annualized Volatility: $15.22\% \pm 0.15\%$
       * Sharpe Ratio: $0.766 \pm 0.015$
       * Maximum Drawdown: $-28.97\% \pm 0.15\%$
       * Stop-Out Count: $106 \pm 1$
     - **Causal Adaptive Strategy (AR(8)-GARCH(1,1))**:
       * Annualized Return: $13.76\% \pm 0.20\%$
       * Annualized Volatility: $11.64\% \pm 0.20\%$
       * Sharpe Ratio: $1.182 \pm 0.025$
       * Maximum Drawdown: $-8.50\% \pm 0.20\%$
       * Stop-Out Count: $106 \pm 2$
  2. **Out-of-Sample Strategy Comparison (TEST: 2020-12-30 to 2025-12-31, 1257 rows)**:
     - **Standard Baseline ($\lambda = 2.0$)**:
       * Annualized Return: $7.43\% \pm 0.15\%$
       * Annualized Volatility: $14.93\% \pm 0.15\%$
       * Sharpe Ratio: $0.498 \pm 0.015$
       * Maximum Drawdown: $-22.22\% \pm 0.15\%$
       * Stop-Out Count: $135 \pm 1$
     - **Causal Adaptive Strategy (AR(8)-GARCH(1,1))**:
       * Annualized Return: $5.76\% \pm 0.20\%$
       * Annualized Volatility: $13.17\% \pm 0.20\%$
       * Sharpe Ratio: $0.437 \pm 0.025$
       * Maximum Drawdown: $-19.34\% \pm 0.20\%$
       * Stop-Out Count: $129 \pm 2$
  3. **Discovered Causal Parents (TRAIN Window)**:
     - `VIX_Diff` Lag 1: $\beta \approx 2.542, p < 10^{-10}$
     - `VIX_Diff` Lag 2: $\beta \approx 1.512, p < 10^{-4}$
     - `VIX_Diff` Lag 3: $\beta \approx 1.029, p < 0.01$
     - `Credit_Spread_Diff` Lag 1: $\beta \approx 2.996, p < 0.01$
  4. **Warning-Free Execution**:
     - Complete pipeline run must emit zero `SingularMatrixWarning` instances.

---

### Tier 4: E2E System Tests (`tests/tier4_e2e/`)
- **Execution Target**: $< 60$ seconds, comprehensive integration of all system components.
- **Scenarios Tested**:
  1. `test_cli_execution.py`:
     - CLI execution via `causal-volatility --start 2016-01-01 --end 2026-01-01 --model garch --split oos --offline --output-dir <tmpdir>`.
     - Exit code 0, generated JSON metrics file, and formatted markdown summary table.
  2. `test_full_pipeline_run.py`:
     - Programmatic invocation of `CausalVolatilityPipeline.run()`.
     - Output validation: contains raw matrix, cleaned matrix, stationary matrix, GARCH model result, causal DAG, multiplier series, and backtest history.
  3. `test_multi_model_run.py`:
     - Automated execution across model suite: `garch`, `egarch`, `gjr`, `student_t`.
     - Verification that alternative models (EGARCH, GJR-GARCH, Student-t) successfully fit and generate comparative backtest metrics.
  4. `test_walk_forward_run.py`:
     - 5-fold expanding window walk-forward validation.
     - Verifies fold date alignment, non-lookahead training, and aggregation of fold Sharpe ratios and drawdowns.

---

## 4. Test Runner Commands & Configuration

### 4.1 Running the Test Suite
```bash
# Run entire test suite
pytest -v

# Run specific tiers
pytest -v tests/tier1_unit/
pytest -v tests/tier2_integration/
pytest -v tests/tier3_regression/
pytest -v tests/tier4_e2e/

# Run with coverage report
pytest --cov=causal_volatility --cov-report=term-missing tests/

# Run only fast unit tests (budget < 2s)
pytest -m "unit"

# Run tests in strict offline mode
pytest -m "offline"
```

### 4.2 Pytest Configuration (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: fast deterministic unit tests (Tier 1)",
    "integration: subsystem and boundary tests (Tier 2)",
    "regression: exact baseline numerical matching (Tier 3)",
    "e2e: full pipeline and CLI execution (Tier 4)",
    "offline: zero-network execution using cached fixtures",
]
filterwarnings = [
    "error::statsmodels.tools.sm_exceptions.SingularMatrixWarning",
    "ignore::statsmodels.tools.sm_exceptions.InterpolationWarning",
    "ignore::FutureWarning",
    "ignore::UserWarning",
]
```

---

## 5. Fixture Management & Anti-Cheating Protocol

### 5.1 Shared Fixtures (`tests/conftest.py`)
- `synthetic_market_data`: Deterministic generator providing 2514 daily bars of OHLCV, VIX, and credit spread with known volatility dynamics.
- `offline_market_data`: Loads `tests/fixtures/market_data_2016_2026.csv` or falls back to synthetic data matching `output.log` dimensions.
- `mock_fred_responses`: Intercepts external requests to FRED API / CSV downloads and returns deterministic series.
- `mock_yfinance_responses`: Intercepts yfinance downloads and returns historical price matrices.
- `tolerance_assert`: Unified helper asserting relative tolerance $\le 10^{-2}$ or absolute tolerance $\le 0.002$.

### 5.2 Mandatory Integrity & Anti-Cheating Rules
1. **No Facade Tests**: Tests must execute actual mathematical models, regressions, and backtest loops. No test may be written to assert trivial `assert True` or mock away the core algorithm under test.
2. **Authoritative Expected Output**: All expected metrics in Tier 3 are sourced verbatim from `output.log` and `causal_volatility_framework_OOS.txt`.
3. **Independent Verification**: Test code is strictly isolated from `src/causal_volatility/` and subject to audit by `teamwork_preview_auditor`.
