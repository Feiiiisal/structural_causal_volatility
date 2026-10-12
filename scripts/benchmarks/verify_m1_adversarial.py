"""
Milestone M1 Challenger 2 Empirical Verification Harness.
Executes adversarial property-based testing and boundary analysis across:
1. Yang-Zhang estimator vs independent reference implementation & drift invariance.
2. Parkinson variance efficiency vs Close-to-Close under pure continuous diffusion.
3. DataProcessor robustness under corrupted inputs (all-NaN, weekends, zero/negative volume).
4. Property-based tests across all 5 estimators (scale invariance, non-negativity, zero-range).
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

from causal_volatility.estimators import (
    GarmanKlassEstimator,
    ParkinsonEstimator,
    RogersSatchellEstimator,
    YangZhangEstimator,
    CloseToCloseEstimator,
)
from causal_volatility.data.processor import DataProcessor


def generate_gbm_ohlc(
    n_days: int = 500,
    n_substeps: int = 500,
    mu: float = 0.0,
    sigma: float = 0.20,
    s0: float = 100.0,
    overnight_vol_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate discrete daily OHLC by sampling a continuous Geometric Brownian Motion."""
    rng = np.random.default_rng(seed)
    dt_day = 1.0 / 252.0
    dt_sub = dt_day / n_substeps
    
    current_price = s0
    records = []
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    
    for i in range(n_days):
        # Overnight jump
        if i > 0 and overnight_vol_ratio > 0:
            sigma_o = sigma * overnight_vol_ratio * np.sqrt(dt_day)
            drift_o = mu * dt_day * 0.2
            jump = drift_o + sigma_o * rng.standard_normal()
            open_p = current_price * np.exp(jump)
        else:
            open_p = current_price
            
        # Intraday Brownian motion
        sub_drifts = (mu - 0.5 * sigma**2) * dt_sub
        sub_shocks = sigma * np.sqrt(dt_sub) * rng.standard_normal(n_substeps)
        sub_log_returns = sub_drifts + sub_shocks
        sub_prices = open_p * np.exp(np.cumsum(sub_log_returns))
        
        high_p = max(open_p, np.max(sub_prices))
        low_p = min(open_p, np.min(sub_prices))
        close_p = sub_prices[-1]
        
        current_price = close_p
        records.append({
            "SP100_Open": open_p,
            "SP100_High": high_p,
            "SP100_Low": low_p,
            "SP100_Close": close_p,
            "SP100_Volume": 1e7,
            "VIX_Close": 20.0,
            "Credit_Spread": 3.0,
        })
        
    return pd.DataFrame(records, index=dates)


def independent_reference_yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 5,
) -> pd.Series:
    """Independent implementation of Yang-Zhang (2000) from first principles."""
    n = window
    log_oc = np.log(open_ / close.shift(1))
    log_co = np.log(close / open_)
    
    u = np.log(high / open_)
    d = np.log(low / open_)
    c = np.log(close / open_)
    rs = u * (u - c) + d * (d - c)
    
    if n > 1:
        k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        var_o = log_oc.rolling(n).var()
        var_c = log_co.rolling(n).var()
        var_rs = rs.rolling(n).mean()
    else:
        k = 0.34 / 2.34
        var_o = log_oc ** 2
        var_c = log_co ** 2
        var_rs = rs
        
    total_var = var_o + k * var_c + (1.0 - k) * var_rs
    total_var = np.maximum(0.0, total_var)
    return pd.Series(np.sqrt(total_var), index=open_.index, name="Ref_Yang_Zhang").fillna(0.0)


def run_task1_yang_zhang():
    print("=" * 70)
    print("TASK 1: Yang-Zhang Estimator Benchmark & Drift Invariance")
    print("=" * 70)
    
    # 1. Compare against independent reference across multiple seeds and windows
    max_diffs = []
    windows = [2, 5, 10, 21]
    for win in windows:
        for seed in [1, 42, 123]:
            df = generate_gbm_ohlc(n_days=500, mu=0.05, sigma=0.25, seed=seed)
            yz_pkg = YangZhangEstimator().estimate(
                high=df["SP100_High"],
                low=df["SP100_Low"],
                close=df["SP100_Close"],
                open=df["SP100_Open"],
                window=win,
            )
            yz_ref = independent_reference_yang_zhang(
                open_=df["SP100_Open"],
                high=df["SP100_High"],
                low=df["SP100_Low"],
                close=df["SP100_Close"],
                window=win,
            )
            diff = np.max(np.abs(yz_pkg - yz_ref))
            max_diffs.append(diff)
            assert np.isclose(yz_pkg, yz_ref, atol=1e-12).all(), f"Mismatch for win={win}, seed={seed}"
            
    print(f"[PASS] Exact match with independent reference Yang-Zhang across all windows & seeds. Max diff: {max(max_diffs):.2e}")

    # 2. Check Drift Invariance
    # Compare Yang-Zhang, Rogers-Satchell, Parkinson, Garman-Klass, Close-to-Close across mu in [-1.0, 1.0]
    drifts = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
    true_daily_sigma = 0.20 * np.sqrt(1.0 / 252.0)
    print(f"True daily sigma: {true_daily_sigma:.6f}")
    
    results = []
    for mu in drifts:
        # 1000 days pure continuous diffusion with drift
        df = generate_gbm_ohlc(n_days=1000, n_substeps=1000, mu=mu, sigma=0.20, overnight_vol_ratio=0.0, seed=42)
        
        yz = YangZhangEstimator().estimate(df["SP100_High"], df["SP100_Low"], df["SP100_Close"], df["SP100_Open"], window=20)
        rs = RogersSatchellEstimator().estimate(df["SP100_High"], df["SP100_Low"], df["SP100_Close"], df["SP100_Open"])
        pk = ParkinsonEstimator().estimate(df["SP100_High"], df["SP100_Low"], df["SP100_Close"], df["SP100_Open"])
        gk = GarmanKlassEstimator().estimate(df["SP100_High"], df["SP100_Low"], df["SP100_Close"], df["SP100_Open"])
        cc = CloseToCloseEstimator().estimate(df["SP100_High"], df["SP100_Low"], df["SP100_Close"], df["SP100_Open"], window=20)
        
        # Mean estimated daily volatility (skipping initial warmup)
        results.append({
            "mu": mu,
            "YZ_mean": yz.iloc[20:].mean(),
            "RS_mean": rs.iloc[20:].mean(),
            "PK_mean": pk.iloc[20:].mean(),
            "GK_mean": gk.iloc[20:].mean(),
            "CC_mean": cc.iloc[20:].mean(),
        })
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Check that Yang-Zhang remains stable across drift
    yz_vals = res_df["YZ_mean"].values
    yz_variation = (np.max(yz_vals) - np.min(yz_vals)) / np.mean(yz_vals)
    print(f"Yang-Zhang variation across mu in [-1.0, 1.0]: {yz_variation:.4%}")
    assert yz_variation < 0.05, f"Yang-Zhang showed excessive drift sensitivity: {yz_variation:.4%}"
    print("[PASS] Yang-Zhang demonstrates empirical drift independence!")


def run_task2_parkinson_efficiency():
    print("=" * 70)
    print("TASK 2: Parkinson Variance Efficiency vs Close-to-Close")
    print("=" * 70)
    
    # Pure continuous diffusion (zero drift, no jumps)
    # Parkinson (1980): The efficiency ratio Var(sigma^2_CC) / Var(sigma^2_P) is approx 5.2
    n_sims = 2000
    n_steps = 2000
    sigma = 0.20
    dt = 1.0 / 252.0
    dt_step = dt / n_steps
    
    rng = np.random.default_rng(12345)
    
    # Simulate single-day continuous paths
    # d ln S = -0.5*sigma^2*dt + sigma*dW
    shocks = sigma * np.sqrt(dt_step) * rng.standard_normal((n_sims, n_steps))
    drift = -0.5 * (sigma ** 2) * dt_step
    log_paths = np.cumsum(drift + shocks, axis=1)
    # Prepend 0 for start of day
    log_paths = np.hstack([np.zeros((n_sims, 1)), log_paths])
    
    open_p = np.ones(n_sims)
    high_p = np.exp(np.max(log_paths, axis=1))
    low_p = np.exp(np.min(log_paths, axis=1))
    close_p = np.exp(log_paths[:, -1])
    
    high_s = pd.Series(high_p)
    low_s = pd.Series(low_p)
    close_s = pd.Series(close_p)
    open_s = pd.Series(open_p)
    
    # Compute Parkinson variance for each day
    pk_vol = ParkinsonEstimator().estimate(high_s, low_s, close_s, open_s)
    pk_var = pk_vol ** 2
    
    # Compute Close-to-close variance for each day: r^2
    # Note: CloseToCloseEstimator with win=1 computes |r_t|, so squared is r_t^2
    cc_vol = CloseToCloseEstimator().estimate(high_s, low_s, close_s, open_s, window=1)
    cc_var = cc_vol ** 2
    
    # Statistical properties
    mean_pk_var = np.mean(pk_var)
    mean_cc_var = np.mean(cc_var)
    true_var = (sigma ** 2) * dt
    
    var_pk = np.var(pk_var, ddof=1)
    var_cc = np.var(cc_var, ddof=1)
    
    eff_ratio = var_cc / var_pk
    
    print(f"True day variance:                {true_var:.8f}")
    print(f"Mean Parkinson variance:           {mean_pk_var:.8f} (bias: {(mean_pk_var - true_var)/true_var:.2%})")
    print(f"Mean Close-to-Close variance:      {mean_cc_var:.8f} (bias: {(mean_cc_var - true_var)/true_var:.2%})")
    print(f"Variance of Parkinson estimator:   {var_pk:.12f}")
    print(f"Variance of CC estimator:          {var_cc:.12f}")
    print(f"Empirical Efficiency Ratio (Var_CC / Var_PK): {eff_ratio:.3f} (theoretical: ~5.2)")
    
    # Assertions
    assert var_pk < var_cc, f"Parkinson variance {var_pk} not strictly lower than CC variance {var_cc}!"
    assert eff_ratio > 4.0, f"Parkinson efficiency ratio {eff_ratio:.2f} is significantly below theoretical 5.2!"
    print("[PASS] Parkinson variance is strictly lower than CC variance (efficiency ratio ~ 5x confirmed)!")


def run_task3_corrupted_inputs():
    print("=" * 70)
    print("TASK 3: DataProcessor Robustness Under Corrupted Inputs")
    print("=" * 70)
    
    processor = DataProcessor()
    
    # 1. All-NaN rows
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df_all_nan = pd.DataFrame(
        {
            "SP100_Open": [np.nan] * 10,
            "SP100_High": [np.nan] * 10,
            "SP100_Low": [np.nan] * 10,
            "SP100_Close": [np.nan] * 10,
            "SP100_Volume": [np.nan] * 10,
            "VIX_Close": [np.nan] * 10,
            "Credit_Spread": [np.nan] * 10,
        },
        index=dates,
    )
    res_nan = processor.process(df_all_nan)
    assert len(res_nan) == 0, f"Expected empty DataFrame for all-NaN input, got len={len(res_nan)}"
    print("[PASS] All-NaN DataFrame processed gracefully to empty output without unhandled exception.")

    # 2. Weekend dates with zero volume
    # 7 consecutive days starting Monday: 5 weekdays + 2 weekend days
    dates_7 = pd.date_range("2023-01-02", periods=7, freq="D") # Jan 2 2023 is Monday, Jan 7-8 Sat-Sun
    df_weekends = pd.DataFrame(
        {
            "SP100_Open": [100.0] * 5 + [100.0, 100.0],
            "SP100_High": [105.0] * 5 + [100.0, 100.0],
            "SP100_Low": [95.0] * 5 + [100.0, 100.0],
            "SP100_Close": [102.0] * 5 + [100.0, 100.0],
            "SP100_Volume": [1e7] * 5 + [0.0, 0.0], # Weekend has zero volume
            "VIX_Close": [20.0] * 7,
            "Credit_Spread": [3.0] * 7,
        },
        index=dates_7,
    )
    res_weekends = processor.process(df_weekends)
    assert len(res_weekends) == 5, f"Expected 5 trading days, got {len(res_weekends)}"
    assert not any(d.dayofweek >= 5 for d in res_weekends.index), "Weekend dates were not dropped!"
    print("[PASS] Weekend dates with zero volume correctly dropped.")

    # 3. Weekend dates with non-zero volume (Corrupted feed)
    # What happens if a corrupt data source puts Saturday/Sunday in with positive volume?
    df_corrupt_weekend = pd.DataFrame(
        {
            "SP100_Open": [100.0] * 7,
            "SP100_High": [105.0] * 7,
            "SP100_Low": [95.0] * 7,
            "SP100_Close": [102.0] * 7,
            "SP100_Volume": [1e7] * 7, # Corrupt: positive volume on weekends!
            "VIX_Close": [20.0] * 7,
            "Credit_Spread": [3.0] * 7,
        },
        index=dates_7,
    )
    res_corrupt_wk = processor.process(df_corrupt_weekend)
    print(f"Note: DataProcessor with corrupt weekend (volume > 0) output rows: {len(res_corrupt_wk)}")
    # In processor.py, valid_mask = SP100_Close.notna() & (SP100_Volume > 0).
    # If corrupt weekend has volume > 0, does it survive?
    has_weekend = any(d.dayofweek >= 5 for d in res_corrupt_wk.index)
    print(f"Does corrupt weekend with positive volume survive processor? {has_weekend}")

    # 4. Zero and negative volume
    df_zero_vol = pd.DataFrame(
        {
            "SP100_Open": [100.0, 100.0, 100.0],
            "SP100_High": [105.0, 105.0, 105.0],
            "SP100_Low": [95.0, 95.0, 95.0],
            "SP100_Close": [102.0, 102.0, 102.0],
            "SP100_Volume": [1e7, 0.0, -1e6], # zero and negative
            "VIX_Close": [20.0, 20.0, 20.0],
            "Credit_Spread": [3.0, 3.0, 3.0],
        },
        index=dates_7[:3],
    )
    res_zero = processor.process(df_zero_vol)
    assert len(res_zero) == 1, f"Expected exactly 1 row surviving, got {len(res_zero)}"
    print("[PASS] Zero and negative volume rows correctly dropped.")

    # 5. Missing required columns
    try:
        processor.process(pd.DataFrame({"SP100_Close": [100.0]}))
        assert False, "Should have raised ValueError for missing columns"
    except ValueError as e:
        print(f"[PASS] Missing required columns caught cleanly: {e}")

    # 6. Macro series with NaNs at start / middle / end
    df_macro_nans = pd.DataFrame(
        {
            "SP100_Open": [100.0] * 5,
            "SP100_High": [105.0] * 5,
            "SP100_Low": [95.0] * 5,
            "SP100_Close": [102.0] * 5,
            "SP100_Volume": [1e7] * 5,
            "VIX_Close": [np.nan, 20.0, np.nan, 22.0, np.nan],
            "Credit_Spread": [np.nan, np.nan, 3.2, np.nan, 3.5],
        },
        index=dates_7[:5],
    )
    res_macro = processor.process(df_macro_nans)
    assert len(res_macro) == 5, f"Expected 5 rows after ffill/bfill, got {len(res_macro)}"
    assert not res_macro.isna().any().any(), "Found NaNs in processed output!"
    print("[PASS] Macro series with interspersed and boundary NaNs correctly ffilled/bfilled.")


def run_task4_property_based_testing():
    print("=" * 70)
    print("TASK 4: Property-Based Testing & Boundary Invariants Across All Estimators")
    print("=" * 70)
    
    estimators = {
        "Garman-Klass": GarmanKlassEstimator(),
        "Parkinson": ParkinsonEstimator(),
        "Rogers-Satchell": RogersSatchellEstimator(),
        "Yang-Zhang": YangZhangEstimator(),
        "Close-to-Close": CloseToCloseEstimator(),
    }
    
    rng = np.random.default_rng(999)
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    
    # 1. Scale Equivariance / Invariance
    # In finance, volatility is measured on log returns: sigma(lambda * P) == sigma(P)
    base_close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    base_open = base_close * (1.0 + rng.normal(0, 0.002, n))
    base_high = np.maximum(base_open, base_close) * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    base_low = np.minimum(base_open, base_close) * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    
    h = pd.Series(base_high, index=dates)
    l = pd.Series(base_low, index=dates)
    c = pd.Series(base_close, index=dates)
    o = pd.Series(base_open, index=dates)
    
    for scale in [0.01, 10.0, 1000.0]:
        for name, est in estimators.items():
            res_orig = est.estimate(h, l, c, o)
            res_scaled = est.estimate(h * scale, l * scale, c * scale, o * scale)
            max_diff = np.max(np.abs(res_orig - res_scaled))
            assert np.isclose(res_orig, res_scaled, atol=1e-10).all(), (
                f"{name} failed scale invariance at scale {scale}! Max diff: {max_diff}"
            )
    print("[PASS] All 5 estimators strictly satisfy Price Scale Invariance (scale-equivariant returns)!")

    # 2. Boundary Condition: Flat days (H == L == O == C)
    flat_h = pd.Series([100.0] * 5, index=dates[:5])
    flat_l = pd.Series([100.0] * 5, index=dates[:5])
    flat_c = pd.Series([100.0] * 5, index=dates[:5])
    flat_o = pd.Series([100.0] * 5, index=dates[:5])
    
    for name, est in estimators.items():
        res = est.estimate(flat_h, flat_l, flat_c, flat_o)
        assert (res == 0.0).all(), f"{name} failed to return 0.0 on flat days!"
        assert not res.isna().any(), f"{name} produced NaN on flat days!"
    print("[PASS] All 5 estimators produce exact 0.0 without NaN on flat days.")

    # 3. Boundary Condition: Zero Range Intraday (H == L but C != O)
    # This happens in tick-compressed or synthetic anomalies: H=L=100, but O=98, C=102
    anom_h = pd.Series([100.0] * 5, index=dates[:5])
    anom_l = pd.Series([100.0] * 5, index=dates[:5])
    anom_o = pd.Series([98.0] * 5, index=dates[:5])
    anom_c = pd.Series([102.0] * 5, index=dates[:5])
    for name, est in estimators.items():
        res = est.estimate(anom_h, anom_l, anom_c, anom_o)
        assert not res.isna().any(), f"{name} produced NaN on anomalous H=L, O!=C bar!"
        assert (res >= 0.0).all(), f"{name} produced negative volatility on anomalous bar!"
    print("[PASS] All 5 estimators survive zero intraday range with gap (clipped to non-negative without NaN).")

    # 4. Inverted Bars (H < L or Low > High) - Stress testing mathematical stability
    inv_h = pd.Series([95.0] * 5, index=dates[:5])
    inv_l = pd.Series([105.0] * 5, index=dates[:5])
    inv_o = pd.Series([100.0] * 5, index=dates[:5])
    inv_c = pd.Series([100.0] * 5, index=dates[:5])
    for name, est in estimators.items():
        res = est.estimate(inv_h, inv_l, inv_c, inv_o)
        assert not res.isna().any(), f"{name} produced NaN on inverted bar!"
        assert (res >= 0.0).all(), f"{name} produced negative on inverted bar!"
    print("[PASS] All estimators handle inverted bar stress test without NaN or crash.")

    # 5. Window boundary testing for rolling estimators (Yang-Zhang and Close-to-Close)
    for win in [1, 2, 5, 20, 500, 1000]:
        yz_res = YangZhangEstimator().estimate(h, l, c, o, window=win)
        assert len(yz_res) == n
        assert not yz_res.isna().any()
        assert (yz_res >= 0.0).all()
        
        cc_res = CloseToCloseEstimator().estimate(h, l, c, o, window=win)
        assert len(cc_res) == n
        assert not cc_res.isna().any()
        assert (cc_res >= 0.0).all()
    print("[PASS] Window boundary tests (win=1 up to win=len) passed cleanly.")


if __name__ == "__main__":
    run_task1_yang_zhang()
    run_task2_parkinson_efficiency()
    run_task3_corrupted_inputs()
    run_task4_property_based_testing()
    print("\nALL ADVERSARIAL VERIFICATION SUITES COMPLETED SUCCESSFULLY!")
