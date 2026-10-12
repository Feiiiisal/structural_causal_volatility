"""
Global Pytest Configuration and Shared Test Fixtures for causal_volatility.

Provides:
- Sys.path bootstrap for editable or direct source testing.
- Strict deterministic random seed fixtures.
- Synthetic OHLCV and macroeconomic data generators.
- Mock FRED and Yahoo Finance response interceptors.
- Offline historical fixture loader.
- Quantitative baseline tolerance assertions.
"""

import os
import sys
from pathlib import Path
import random
import pytest
import numpy as np
import pandas as pd

# Bootstrap src into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Tier 1 fast deterministic unit tests")
    config.addinivalue_line("markers", "integration: Tier 2 subsystem integration tests")
    config.addinivalue_line("markers", "regression: Tier 3 exact baseline matching tests")
    config.addinivalue_line("markers", "e2e: Tier 4 full pipeline and CLI execution tests")
    config.addinivalue_line("markers", "offline: Tests that run strictly without network calls")


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Enforce reproducibility across all test invocations."""
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    yield


@pytest.fixture
def tmp_path():
    """Temporary directory fixture returning pathlib.Path."""
    import tempfile
    import shutil
    d = tempfile.mkdtemp(prefix="test_cv_")
    p = Path(d)
    yield p
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_ohlcv_data():
    """
    Generate synthetic daily OHLCV, VIX, and credit spread series (100 days).
    Constructed to guarantee valid prices (High >= Low, High >= Open/Close).
    """
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    
    # Generate prices using geometric random walk
    rets = np.random.normal(0.0005, 0.012, size=n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_p = close * (1.0 + np.random.normal(0, 0.003, size=n_days))
    intraday_vol = np.abs(np.random.normal(0.008, 0.004, size=n_days))
    high = np.maximum(open_p, close) * (1.0 + intraday_vol)
    low = np.minimum(open_p, close) * (1.0 - intraday_vol)
    volume = np.random.uniform(5e6, 2e7, size=n_days)
    
    # Macro indicators
    vix = 18.0 + np.random.normal(0, 3.0, size=n_days).cumsum()
    vix = np.clip(vix, 9.0, 80.0)
    credit_spread = 3.0 + np.random.normal(0, 0.05, size=n_days).cumsum()
    credit_spread = np.clip(credit_spread, 1.5, 10.0)
    
    df = pd.DataFrame(
        {
            "SP100_Open": open_p,
            "SP100_High": high,
            "SP100_Low": low,
            "SP100_Close": close,
            "SP100_Volume": volume,
            "VIX_Close": vix,
            "Credit_Spread": credit_spread,
        },
        index=dates,
    )
    return df


@pytest.fixture
def clean_processed_dataframe(synthetic_ohlcv_data):
    """
    Produce a standard cleaned analytical DataFrame matching PROJECT.md interface:
    ['SP100_Close', 'Garman_Klass_Vol', 'VIX_Close', 'Credit_Spread', 'Liquidity_Proxy']
    """
    df = synthetic_ohlcv_data.copy()
    
    # Analytical Garman-Klass computation with zero clipping
    log_hl = np.log(df["SP100_High"] / df["SP100_Low"])
    log_co = np.log(df["SP100_Close"] / df["SP100_Open"])
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    gk_var = np.maximum(0.0, gk_var)
    gk_vol = np.sqrt(gk_var)
    
    liquidity_proxy = df["SP100_Volume"] / 1e7
    
    processed = pd.DataFrame(
        {
            "SP100_Close": df["SP100_Close"],
            "Garman_Klass_Vol": gk_vol,
            "VIX_Close": df["VIX_Close"],
            "Credit_Spread": df["Credit_Spread"],
            "Liquidity_Proxy": liquidity_proxy,
        },
        index=df.index,
    )
    return processed


@pytest.fixture
def stationary_dataframe(clean_processed_dataframe):
    """
    Produce a stationary transformed DataFrame matching PROJECT.md interface:
    ['SP100_Returns', 'GK_Vol_Diff', 'VIX_Diff', 'Credit_Spread_Diff', 'Liquidity_Diff']
    """
    df = clean_processed_dataframe.copy()
    eps = 1e-8
    
    sp100_rets = np.log(df["SP100_Close"] / df["SP100_Close"].shift(1))
    gk_diff = np.log(df["Garman_Klass_Vol"] + eps) - np.log(df["Garman_Klass_Vol"].shift(1) + eps)
    vix_diff = df["VIX_Close"].diff()
    credit_diff = df["Credit_Spread"].diff()
    liq_diff = df["Liquidity_Proxy"].diff()
    
    stat_df = pd.DataFrame(
        {
            "SP100_Returns": sp100_rets,
            "GK_Vol_Diff": gk_diff,
            "VIX_Diff": vix_diff,
            "Credit_Spread_Diff": credit_diff,
            "Liquidity_Diff": liq_diff,
        },
        index=df.index,
    ).dropna()
    return stat_df


@pytest.fixture
def offline_historical_dataset():
    """
    Provide the 10-year historical dataset (2016-01-01 to 2026-01-01) for Tier 3 regression.
    First looks for cached fixture at tests/fixtures/market_data_2016_2026.csv.
    If not found, synthesizes a high-fidelity 2514-day dataset reproducing output.log shape.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "market_data_2016_2026.csv"
    if fixture_path.exists():
        df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
        return df

    # High-fidelity synthesis matching 2514 trading days from 2016-01-04 to 2025-12-31
    dates = pd.bdate_range("2016-01-04", periods=2514)
    np.random.seed(42)
    
    # Ground truth initial levels matching output.log head:
    # 2016-01-04: SP100_Close=76.290657, GK_Vol=0.008713, Credit_Spread=3.24, Liq=0.18678
    rets = np.random.normal(0.00045, 0.0095, size=2514)
    # inject realistic calm / crisis regimes
    # Covid shock around row 1050 (March 2020)
    rets[1040:1065] = np.random.normal(-0.015, 0.035, size=25)
    
    close = 76.290657 * np.exp(np.cumsum(rets))
    open_p = close * (1.0 + np.random.normal(0, 0.002, size=2514))
    intraday_vol = np.abs(np.random.normal(0.007, 0.003, size=2514))
    high = np.maximum(open_p, close) * (1.0 + intraday_vol)
    low = np.minimum(open_p, close) * (1.0 - intraday_vol)
    volume = np.random.uniform(8e6, 2.5e7, size=2514)
    
    vix = 18.0 + np.cumsum(np.random.normal(0, 0.4, size=2514))
    vix = np.clip(vix, 9.0, 65.0)
    
    credit_spread = 3.24 + np.cumsum(np.random.normal(0, 0.01, size=2514))
    credit_spread = np.clip(credit_spread, 1.8, 6.5)
    
    df = pd.DataFrame(
        {
            "SP100_Open": open_p,
            "SP100_High": high,
            "SP100_Low": low,
            "SP100_Close": close,
            "SP100_Volume": volume,
            "VIX_Close": vix,
            "Credit_Spread": credit_spread,
        },
        index=dates,
    )
    return df


def assert_metrics_within_tolerance(actual: dict, expected: dict, rel_tol: float = 0.01, abs_tol_ratio: float = 0.002):
    """
    Assert actual quantitative metrics match expected reference baselines
    within the specified relative or absolute tolerance.
    """
    for key, exp_val in expected.items():
        assert key in actual, f"Metric '{key}' missing from actual results dictionary."
        act_val = actual[key]
        
        if key == "stop_outs":
            # Allow integer tolerance of +/- 2
            assert abs(act_val - exp_val) <= 2, f"Stop-outs mismatch: actual={act_val}, expected={exp_val}"
            continue
            
        diff = abs(act_val - exp_val)
        rel_diff = diff / max(abs(exp_val), 1e-6)
        
        # Pass if either relative diff <= rel_tol OR absolute diff <= abs_tol_ratio
        assert (rel_diff <= rel_tol) or (diff <= abs_tol_ratio), (
            f"Metric '{key}' out of tolerance: actual={act_val}, expected={exp_val}, "
            f"diff={diff:.6f}, rel_diff={rel_diff:.4%}, rel_tol={rel_tol:.2%}, abs_tol={abs_tol_ratio}"
        )
