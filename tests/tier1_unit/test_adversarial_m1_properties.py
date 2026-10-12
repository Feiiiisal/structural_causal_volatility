"""
Tier 1: Adversarial Property-Based and Boundary Tests for Realized Volatility Estimators and DataProcessor.
Authored by Challenger 2 for Milestone M1.
"""

import pytest
import numpy as np
import pandas as pd

from causal_volatility.estimators import (
    GarmanKlassEstimator,
    ParkinsonEstimator,
    RogersSatchellEstimator,
    YangZhangEstimator,
    CloseToCloseEstimator,
)
from causal_volatility.data.processor import DataProcessor


@pytest.mark.unit
def test_yang_zhang_vs_independent_reference_formula():
    """Verify Yang-Zhang estimator exactly matches independent reference implementation."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    open_ = close * (1.0 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.006, n)))
    
    h = pd.Series(high, index=dates)
    l = pd.Series(low, index=dates)
    c = pd.Series(close, index=dates)
    o = pd.Series(open_, index=dates)
    
    for win in [2, 5, 10, 20]:
        yz = YangZhangEstimator().estimate(h, l, c, o, window=win)
        
        # Independent reference calculation
        log_oc = np.log(o / c.shift(1))
        log_co = np.log(c / o)
        u = np.log(h / o)
        d = np.log(l / o)
        c_ret = np.log(c / o)
        rs = u * (u - c_ret) + d * (d - c_ret)
        
        k = 0.34 / (1.34 + (win + 1.0) / (win - 1.0))
        var_o = log_oc.rolling(win).var()
        var_c = log_co.rolling(win).var()
        var_rs = rs.rolling(win).mean()
        ref_var = np.maximum(0.0, var_o + k * var_c + (1.0 - k) * var_rs)
        ref_vol = np.sqrt(ref_var).fillna(0.0)
        
        assert np.isclose(yz, ref_vol, atol=1e-12).all()


@pytest.mark.unit
def test_yang_zhang_drift_invariance():
    """Verify Yang-Zhang estimator is invariant across varying drift regimes."""
    rng = np.random.default_rng(123)
    n_days = 600
    n_substeps = 200
    sigma = 0.20
    dt_day = 1.0 / 252.0
    dt_sub = dt_day / n_substeps
    
    results = {}
    for mu in [-0.5, 0.0, 0.5]:
        current_p = 100.0
        records = []
        for _ in range(n_days):
            shocks = sigma * np.sqrt(dt_sub) * rng.standard_normal(n_substeps)
            drift = (mu - 0.5 * sigma**2) * dt_sub
            path = current_p * np.exp(np.cumsum(drift + shocks))
            high_p = max(current_p, np.max(path))
            low_p = min(current_p, np.min(path))
            close_p = path[-1]
            records.append((current_p, high_p, low_p, close_p))
            current_p = close_p
            
        df = pd.DataFrame(records, columns=["open", "high", "low", "close"])
        yz = YangZhangEstimator().estimate(df["high"], df["low"], df["close"], df["open"], window=20)
        results[mu] = yz.iloc[20:].mean()
        
    vals = list(results.values())
    max_dev = (max(vals) - min(vals)) / np.mean(vals)
    assert max_dev < 0.03, f"Yang-Zhang showed drift variation: {max_dev:.2%}"


@pytest.mark.unit
def test_parkinson_efficiency_vs_close_to_close():
    """Verify Parkinson estimator variance is strictly lower than close-to-close under pure diffusion."""
    rng = np.random.default_rng(777)
    n_days = 5000
    n_substeps = 500
    sigma = 0.20
    dt_day = 1.0 / 252.0
    dt_sub = dt_day / n_substeps
    
    current_p = 100.0
    records = []
    for _ in range(n_days):
        open_p = current_p
        shocks = sigma * np.sqrt(dt_sub) * rng.standard_normal(n_substeps)
        drift = -0.5 * (sigma ** 2) * dt_sub
        path = open_p * np.exp(np.cumsum(drift + shocks))
        records.append((open_p, max(open_p, np.max(path)), min(open_p, np.min(path)), path[-1]))
        current_p = path[-1]
        
    df = pd.DataFrame(records, columns=["open", "high", "low", "close"])
    
    pk_var = (ParkinsonEstimator().estimate(df["high"], df["low"], df["close"], df["open"])) ** 2
    cc_var = (CloseToCloseEstimator().estimate(df["high"], df["low"], df["close"], df["open"], window=1)) ** 2
    
    var_pk = np.var(pk_var.iloc[1:], ddof=1)
    var_cc = np.var(cc_var.iloc[1:], ddof=1)
    eff_ratio = var_cc / var_pk
    
    assert var_pk < var_cc, f"Parkinson variance {var_pk} not lower than CC {var_cc}"
    assert eff_ratio > 4.0, f"Efficiency ratio {eff_ratio:.2f} too low (expected ~5.2)"


@pytest.mark.unit
def test_estimator_price_scale_invariance():
    """Verify scale equivariance: multiplying prices by constant scale does not change volatility."""
    rng = np.random.default_rng(888)
    n = 100
    dates = pd.bdate_range("2023-01-01", periods=n)
    
    c = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    o = c * (1.0 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    l = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    
    estimators = [
        GarmanKlassEstimator(),
        ParkinsonEstimator(),
        RogersSatchellEstimator(),
        YangZhangEstimator(),
        CloseToCloseEstimator(),
    ]
    
    for est in estimators:
        base = est.estimate(h, l, c, o)
        scaled = est.estimate(h * 50.0, l * 50.0, c * 50.0, o * 50.0)
        assert np.isclose(base, scaled, atol=1e-9).all()


@pytest.mark.unit
def test_data_processor_corrupted_inputs():
    """Verify DataProcessor resilience to NaN rows, zero/negative volume, and edge cases."""
    processor = DataProcessor()
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    
    # 1. Zero and negative volume handling
    df = pd.DataFrame(
        {
            "SP100_Open": [100.0] * 6,
            "SP100_High": [105.0] * 6,
            "SP100_Low": [95.0] * 6,
            "SP100_Close": [102.0] * 6,
            "SP100_Volume": [1e7, 0.0, -1e6, 1e7, np.nan, 2e7],
            "VIX_Close": [20.0] * 6,
            "Credit_Spread": [3.0] * 6,
        },
        index=dates,
    )
    res = processor.process(df)
    # Only rows 0, 3, 5 have Volume > 0 and Close notna
    assert len(res) == 3
    assert (res["Liquidity_Proxy"] > 0).all()
    
    # 2. All-NaN DataFrame
    df_all_nan = pd.DataFrame(
        {
            "SP100_Open": [np.nan] * 5,
            "SP100_High": [np.nan] * 5,
            "SP100_Low": [np.nan] * 5,
            "SP100_Close": [np.nan] * 5,
            "SP100_Volume": [np.nan] * 5,
            "VIX_Close": [np.nan] * 5,
            "Credit_Spread": [np.nan] * 5,
        },
        index=dates[:5],
    )
    res_nan = processor.process(df_all_nan)
    assert len(res_nan) == 0
