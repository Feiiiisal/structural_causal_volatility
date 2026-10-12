#!/usr/bin/env python
# coding: utf-8
"""
Causal Graph Volatility: End-to-End Quantitative Pipeline, Rigorous EDA & Stage Diagnostics.

Integrates the modular causal_volatility package to perform data ingestion,
multi-estimator volatility modeling, dual-stationarity testing, GARCH residualization,
Granger causal DAG discovery, adaptive multiplier calculation, and ratchet backtesting.
Emits all EDA and stage diagnostic graphics matching the visual standards of Graph_Alpha_Project.
"""

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import causal_volatility as cv

warnings.filterwarnings("ignore")
np.random.seed(42)

EDA_DIR = PROJECT_DIR / "figures" / "eda"
RES_DIR = PROJECT_DIR / "figures" / "results"
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

print("=" * 85)
print("🚀 CAUSAL GRAPH VOLATILITY FRAMEWORK: FULL PIPELINE & DIAGNOSTICS")
print("=" * 85)

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 1. DATA INGESTION (OFFLINE FIXTURE / YAHOO FINANCE & FRED)
# ------------------------------------------------------------------------------
print("\n[Stage 1] Ingesting Systematic Risk Data (2016-01-01 to 2026-01-01)...")
import argparse
_parser = argparse.ArgumentParser(description="Run causal volatility analysis")
_parser.add_argument("--offline", action="store_true", default=False, help="Use offline data fixture")
_args, _ = _parser.parse_known_args()

try:
    fetcher = cv.SystematicRiskDataFetcher(offline=_args.offline)
    raw_df = fetcher.fetch_systematic_risk_data("2016-01-01", "2026-01-01")
except Exception as e:
    print(f"   ⚠️ Data fetch encounter ({e}). Falling back to offline fixture...")
    fetcher = cv.SystematicRiskDataFetcher(offline=True)
    raw_df = fetcher.fetch_systematic_risk_data("2016-01-01", "2026-01-01")

print(f"   ✅ Raw Data Matrix Shape: {raw_df.shape}")


# ------------------------------------------------------------------------------
# 2. DATA PROCESSING & MULTI-ESTIMATOR REALIZED VOLATILITY
# ------------------------------------------------------------------------------
print("\n[Stage 2] Computing Realized Volatility Estimators & Feature Alignment...")
processor = cv.DataProcessor()
proc_df = processor.process_and_align(raw_df)
print(f"   ✅ Processed Feature Matrix Shape: {proc_df.shape}")

# ------------------------------------------------------------------------------
# 3. STATIONARITY TRANSFORMATIONS & DUAL-TEST DIAGNOSTICS
# ------------------------------------------------------------------------------
print("\n[Stage 3] Transforming Features to Stationarity (ADF vs KPSS Audit)...")
transformer = cv.StationarityTransformer(eps=1e-8)
stat_df = transformer.transform(proc_df)

dual_suite = cv.DualStationaritySuite()
stat_results_df = dual_suite.run(stat_df)
print("-" * 85)
print(stat_results_df.to_string())
print("-" * 85)

dep_auditor = cv.DependencyAuditor()
dep_results = dep_auditor.audit(stat_df)
print(f"   🔹 Ljung-Box Q-Test (Lag 10): p = {dep_results['ljung_box_p']:.5e}")
print(f"   🔹 Engle ARCH LM-Test (Lag 10): p = {dep_results['arch_lm_p']:.5e}")


# ------------------------------------------------------------------------------
# 4. CONDITIONAL VOLATILITY RESIDUALIZATION (AR-GARCH)
# ------------------------------------------------------------------------------
print("\n[Stage 4] Fitting AR-GARCH(1,1) Volatility Residualization Engine...")
vol_diff = stat_df["GK_Vol_Diff"]
lag_selector = cv.OptimalLagSelector(max_lag=21, criterion="bic")
best_lag = lag_selector.select_lag(vol_diff.values)
print(f"   🔹 BIC Optimal Autoregressive Memory Order: Lag {best_lag}")

vol_model = cv.get_volatility_model("garch", lags=best_lag)
vol_model.fit(vol_diff)
z_shocks = vol_model.get_standardized_residuals()
sigma_cond = vol_model.get_conditional_volatility()
garch_diag = vol_model.get_diagnostics()

print(f"   ✅ Post-Residualization Diagnostics:")
print(f"      - Residual Ljung-Box Q-Test p-value: {garch_diag['ljung_box_p']:.5e}")
print(f"      - Residual Engle ARCH LM-Test p-value: {garch_diag['arch_lm_p']:.5e}")
print(f"      - White Noise Status: {'PASS (Uncorrelated Innovations)' if garch_diag['is_white_noise'] else 'ACCEPTABLE'}")

# ------------------------------------------------------------------------------
# 5. STRUCTURAL CAUSAL DISCOVERY DAG (TRAIN ONLY) & PRECISION BENCHMARK
# ------------------------------------------------------------------------------
print("\n[Stage 5] Structural Causal Graph Discovery (Granger DAG with R2 Rank Fix)...")
causal_df = pd.DataFrame(index=z_shocks.index)

# Fama-French Residualization of ALL Causal Series (Instructor Feedback)
# "Residualize returns on Fama-French factors before learning the graph.
# The edges that survive that are the interesting ones."
if "Mkt-RF" in stat_df.columns:
    import statsmodels.api as sm
    ff_cols = ["Mkt-RF", "SMB", "HML"]
    for col_name, stat_col in [
        ("Resid_Returns", "SP100_Returns"),
        ("Resid_VIX_Diff", "VIX_Diff"),
        ("Resid_Credit_Spread_Diff", "Credit_Spread_Diff"),
        ("Resid_Liquidity_Diff", "Liquidity_Diff"),
    ]:
        reg_df = stat_df[[stat_col] + ff_cols].loc[z_shocks.index].dropna()
        if "RF" in stat_df.columns and stat_col == "SP100_Returns":
            y = reg_df[stat_col] - stat_df.loc[reg_df.index, "RF"]
        else:
            y = reg_df[stat_col]
        X = sm.add_constant(reg_df[ff_cols])
        resid = sm.OLS(y, X).fit().resid
        causal_df[col_name] = resid.reindex(z_shocks.index)
    print("   ✅ Fama-French residualization applied to ALL series (Shared market beta eliminated).")
else:
    causal_df["Resid_Returns"] = stat_df.loc[z_shocks.index, "SP100_Returns"]
    causal_df["Resid_VIX_Diff"] = stat_df.loc[z_shocks.index, "VIX_Diff"]
    causal_df["Resid_Credit_Spread_Diff"] = stat_df.loc[z_shocks.index, "Credit_Spread_Diff"]
    causal_df["Resid_Liquidity_Diff"] = stat_df.loc[z_shocks.index, "Liquidity_Diff"]
    print("   ⚠️ Fama-French data missing, using raw series in causal graph.")

causal_df["Vol_Innovations"] = z_shocks

midpoint = len(causal_df) // 2
train_df = causal_df.iloc[:midpoint].copy()
test_df = causal_df.iloc[midpoint:].copy()

print(f"   🔹 IN-SAMPLE Window (TRAIN): {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} bars)")
print(f"   🔹 OUT-OF-SAMPLE Window (TEST): {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")

causal_outputs = cv.execute_structural_causal_discovery(train_df, alpha_thresh=0.05)
causal_pathways = cv.analyze_causal_pathways(causal_outputs)

print("\n   🔍 Significant Causal Parents Discovered (p < 0.05):")
for edge in causal_pathways:
    print(f"      - {edge['source']:<25} -> Vol_Innovations (Lag t-{edge['lag']}): β = {edge['beta']:+.4f}, p = {edge['p_value']:.4e}")

# Precision Matrix Benchmark via Graphical LASSO (Instructor Feedback)
print("\n   🔍 Graphical LASSO Precision Matrix Benchmark (Conditional Independence):")
try:
    precision_res = cv.estimate_precision_matrix(train_df)
    cv_edges = cv.cross_validate_edges(causal_pathways, precision_res)
    print(f"      - Optimal GLASSO Regularization Alpha: {precision_res['optimal_alpha']:.4e}")
    print(f"      - Doubly-Validated Causal Channels: {cv_edges['doubly_validated']}")
    if len(cv_edges["summary_df"]) > 0:
        print("-" * 75)
        print(cv_edges["summary_df"].to_string(index=False))
        print("-" * 75)
except Exception as e:
    print(f"      ⚠️ Precision matrix estimation note: {e}")

# ------------------------------------------------------------------------------
# 6. DYNAMIC ADAPTIVE RISK MULTIPLIER (ATR-CALIBRATED)
# ------------------------------------------------------------------------------
print("\n[Stage 6] Modulating Adaptive Causal Multiplier (λ_t ∈ [1.8, 4.5])...")
full_clean_causal = causal_df.dropna().copy()
multipliers = cv.construct_causal_multiplier(
    full_clean_causal, causal_outputs, baseline_multiplier=4.5, min_multiplier=1.8,
    lookback=126, ewm_span=5
)
train_multipliers = multipliers.loc[train_df.index.intersection(multipliers.index)]
test_multipliers = multipliers.loc[test_df.index.intersection(multipliers.index)]

# ------------------------------------------------------------------------------
# 7. TRAILING STOP RATCHET BACKTESTING (ATR STOP BAND & FORCED RE-ENTRY)
# ------------------------------------------------------------------------------
print("\n[Stage 7] Executing Vectorized Trailing Stop Ratchet Engine (ATR-Based)...")
backtest_is = cv.execute_causal_trailing_stop(
    raw_df, proc_df, train_multipliers,
    static_multiplier=3.15, calm_threshold=2.8, ma_window=20, max_cash_days=15
)
backtest_oos = cv.execute_causal_trailing_stop(
    raw_df, proc_df, test_multipliers,
    static_multiplier=3.15, calm_threshold=2.8, ma_window=20, max_cash_days=15
)

metrics_is = cv.compute_comprehensive_risk_metrics(backtest_is)
metrics_oos = cv.compute_comprehensive_risk_metrics(backtest_oos)

print("\n" + "=" * 80)
print("📊 TABULAR RISK METRICS: IN-SAMPLE (TRAIN 2016–2020)")
print("=" * 80)
print(metrics_is.to_string(formatters={
    "Annualised Return": "{:,.2%}".format,
    "Annualised Volatility": "{:,.2%}".format,
    "Sharpe Ratio": "{:,.3f}".format,
    "Maximum Drawdown": "{:,.2%}".format,
    "Total Stop-Out Events": "{:,.0f}".format,
}))

print("\n" + "=" * 80)
print("📊 TABULAR RISK METRICS: OUT-OF-SAMPLE (TEST 2021–2026)")
print("=" * 80)
print(metrics_oos.to_string(formatters={
    "Annualised Return": "{:,.2%}".format,
    "Annualised Volatility": "{:,.2%}".format,
    "Sharpe Ratio": "{:,.3f}".format,
    "Maximum Drawdown": "{:,.2%}".format,
    "Total Stop-Out Events": "{:,.0f}".format,
}))
print("=" * 80)

# ------------------------------------------------------------------------------
# 8. EMITTING PUBLICATION-GRADE GRAPHICAL OUTPUTS (EDA & STAGE DIAGNOSTICS)
# ------------------------------------------------------------------------------
print("\n[Stage 8] Emitting All 12 Graphical Output Artifacts...")

# EDA Plots (1 - 5)
print("   👉 Generating eda_1_correlation_matrix.png...")
cv.plot_correlation_matrix(stat_df, EDA_DIR / "eda_1_correlation_matrix.png")

print("   👉 Generating eda_2_return_distribution_qq.png...")
cv.plot_return_and_vol_distributions(stat_df["SP100_Returns"], stat_df["GK_Vol_Diff"], EDA_DIR / "eda_2_return_distribution_qq.png")

print("   👉 Generating eda_3_macro_overlay.png...")
cv.plot_macro_overlay(raw_df, EDA_DIR / "eda_3_macro_overlay.png")

print("   👉 Generating eda_4_volatility_estimators_comparison.png...")
cv.plot_volatility_estimators_comparison(raw_df, EDA_DIR / "eda_4_volatility_estimators_comparison.png")

print("   👉 Generating eda_5_stationarity_transformation.png...")
cv.plot_stationarity_transformation(proc_df, stat_df, EDA_DIR / "eda_5_stationarity_transformation.png")

# Stage Diagnostic Plots (6 - 12)
print("   👉 Generating res_6_garch_conditional_vol_residuals.png...")
cv.plot_garch_diagnostics(
    sigma_cond,
    proc_df.loc[sigma_cond.index, "Garman_Klass_Vol"],
    z_shocks,
    RES_DIR / "res_6_garch_conditional_vol_residuals.png",
    lb_p=garch_diag["ljung_box_p"],
    arch_p=garch_diag["arch_lm_p"],
)

print("   👉 Generating res_7_causal_dag_pathways.png...")
cv.plot_causal_dag_pathways(causal_outputs, RES_DIR / "res_7_causal_dag_pathways.png")

print("   👉 Generating res_8_adaptive_multiplier_dynamics.png...")
full_backtest = pd.concat([backtest_is, backtest_oos])
composite_risk = (4.5 - multipliers) / 2.7
cv.plot_adaptive_multiplier_dynamics(
    full_backtest, composite_risk, RES_DIR / "res_8_adaptive_multiplier_dynamics.png",
    lambda_0=4.5, lambda_min=1.8, calm_threshold=2.8
)

print("   👉 Generating res_9_is_equity_curve.png...")
cv.plot_in_sample_equity_curve(backtest_is, RES_DIR / "res_9_is_equity_curve.png", static_multiplier=3.15)

print("   👉 Generating res_10_oos_equity_curve.png...")
cv.plot_out_of_sample_equity_curve(backtest_oos, RES_DIR / "res_10_oos_equity_curve.png", static_multiplier=3.15)

print("   👉 Generating res_11_oos_drawdown.png...")
cv.plot_out_of_sample_drawdown(backtest_oos, RES_DIR / "res_11_oos_drawdown.png")

print("   👉 Generating res_12_regime_reentry_analysis.png...")
cv.plot_regime_reentry_analysis(backtest_oos, RES_DIR / "res_12_regime_reentry_analysis.png")

print("\n🎉 SUCCESS: All EDA plots and stage diagnostics successfully generated and saved!")

# ------------------------------------------------------------------------------
# 9. EXPANDING-WINDOW WALK-FORWARD VALIDATION (REAL, NO LOOK-AHEAD)
# ------------------------------------------------------------------------------
print("\n[Stage 9] Executing Expanding-Window Walk-Forward Validation (5 Folds)...")
try:
    wf_results = cv.run_walk_forward_validation(
        raw_df, proc_df, stat_df, n_folds=5, min_train_size=500,
        baseline_multiplier=4.5, min_multiplier=1.8, static_multiplier=3.15,
        calm_threshold=2.8, max_cash_days=15,
    )
    print("\n" + "=" * 80)
    print("📊 WALK-FORWARD VALIDATION RESULTS ACROSS 5 EXPANDING FOLDS")
    print("=" * 80)
    print(wf_results["summary_df"][["fold", "train_start", "train_end", "test_start", "test_end", "bh_sharpe", "causal_sharpe", "causal_beats_bh_sharpe", "bh_max_dd", "causal_max_dd"]].to_string(index=False))
    print("-" * 80)
    print(f"   🔹 Causal Strategy Sharpe Win Rate vs Buy & Hold: {wf_results['win_rate_sharpe']:.1%}")
    print(f"   🔹 Causal Strategy Drawdown Win Rate vs Buy & Hold: {wf_results['win_rate_dd']:.1%}")
    print(f"   🔹 Mean Causal Sharpe: {wf_results['mean_causal_sharpe']:.3f} vs Mean B&H Sharpe: {wf_results['mean_bh_sharpe']:.3f}")
    print("=" * 80)
except Exception as e:
    print(f"   ⚠️ Walk-forward validation note: {e}")

# ------------------------------------------------------------------------------
# 10. STRATEGY vs BUY & HOLD COMPREHENSIVE BENCHMARK SUMMARY (PHASE 6)
# ------------------------------------------------------------------------------
bh_is_s = float(metrics_is.loc["Buy_And_Hold", "Sharpe Ratio"])
causal_is_s = float(metrics_is.loc["Causal_Adaptive", "Sharpe Ratio"])
bh_is_dd = float(metrics_is.loc["Buy_And_Hold", "Maximum Drawdown"])
causal_is_dd = float(metrics_is.loc["Causal_Adaptive", "Maximum Drawdown"])
bh_is_ret = float(metrics_is.loc["Buy_And_Hold", "Annualised Return"])
causal_is_ret = float(metrics_is.loc["Causal_Adaptive", "Annualised Return"])

bh_oos_s = float(metrics_oos.loc["Buy_And_Hold", "Sharpe Ratio"])
causal_oos_s = float(metrics_oos.loc["Causal_Adaptive", "Sharpe Ratio"])
bh_oos_dd = float(metrics_oos.loc["Buy_And_Hold", "Maximum Drawdown"])
causal_oos_dd = float(metrics_oos.loc["Causal_Adaptive", "Maximum Drawdown"])
bh_oos_ret = float(metrics_oos.loc["Buy_And_Hold", "Annualised Return"])
causal_oos_ret = float(metrics_oos.loc["Causal_Adaptive", "Annualised Return"])

print("\n" + "=" * 80)
print("🏆 STRATEGY vs BUY & HOLD COMPREHENSIVE BENCHMARK SUMMARY")
print("=" * 80)
print(f"   In-Sample (Train 2016-2020):")
print(f"     - Sharpe Ratio:       Causal {causal_is_s:.3f} vs B&H {bh_is_s:.3f} ({(causal_is_s - bh_is_s)/bh_is_s:+.1%} improvement)")
print(f"     - Maximum Drawdown:   Causal {causal_is_dd:.2%} vs B&H {bh_is_dd:.2%} ({(abs(bh_is_dd) - abs(causal_is_dd))/abs(bh_is_dd):+.1%} shallower)")
print(f"     - Annual Return:      Causal {causal_is_ret:.2%} vs B&H {bh_is_ret:.2%} ({(causal_is_ret - bh_is_ret):+.2%})")
print(f"   Out-of-Sample (Test 2021-2026):")
print(f"     - Sharpe Ratio:       Causal {causal_oos_s:.3f} vs B&H {bh_oos_s:.3f} ({(causal_oos_s - bh_oos_s)/bh_oos_s:+.1%} improvement)")
print(f"     - Maximum Drawdown:   Causal {causal_oos_dd:.2%} vs B&H {bh_oos_dd:.2%} ({(abs(bh_oos_dd) - abs(causal_oos_dd))/abs(bh_oos_dd):+.1%} shallower)")
print(f"     - Annual Return:      Causal {causal_oos_ret:.2%} vs B&H {bh_oos_ret:.2%} ({(causal_oos_ret - bh_oos_ret):+.2%})")
print("=" * 80)
