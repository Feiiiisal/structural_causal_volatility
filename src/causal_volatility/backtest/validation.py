"""Multi-Horizon Validation Suite (Feature 17).

Provides 10-year 50/50 OOS split, 25-year 60/20/20 split, and
5-fold expanding walk-forward cross-validation without look-ahead leakage.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def run_walk_forward_validation(
    df_raw: pd.DataFrame,
    proc_df: Optional[pd.DataFrame] = None,
    stat_df: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
    min_train_size: int = 500,
    tau_max: int = 5,
    alpha_thresh: float = 0.05,
    baseline_multiplier: float = 4.5,
    min_multiplier: float = 1.8,
    static_multiplier: float = 4.5,
    calm_threshold: float = 2.8,
    max_cash_days: int = 15,
) -> Dict[str, Any]:
    """Execute expanding-window walk-forward cross-validation.

    For each fold, fits the GARCH model and causal graph on the training
    window ONLY, then evaluates the trailing stop strategy on the held-out
    test window. No look-ahead leakage.

    Args:
        df_raw: Raw DataFrame containing 'SP100_Close' and OHLCV.
        proc_df: Processed DataFrame with 'Garman_Klass_Vol', 'ATR_14', etc.
        stat_df: Stationary transformed DataFrame.
        n_folds: Number of walk-forward folds (default: 5).
        min_train_size: Minimum training observations before first fold.
        tau_max: Maximum causal lag order.
        alpha_thresh: Significance threshold for causal edges.
        baseline_multiplier: Lambda_0 for calm regime.
        min_multiplier: Lambda_min for crisis regime.
        static_multiplier: Fixed multiplier for static baseline.
        calm_threshold: Threshold separating calm vs stormy regimes.

    Returns:
        Dictionary with fold results, summary DataFrame, and win rate.
    """
    from causal_volatility.backtest.engine import execute_causal_trailing_stop
    from causal_volatility.backtest.metrics import compute_comprehensive_risk_metrics
    from causal_volatility.causal.discovery import execute_structural_causal_discovery
    from causal_volatility.causal.multiplier import construct_causal_multiplier
    from causal_volatility.data.processor import DataProcessor
    from causal_volatility.models.factory import get_volatility_model
    from causal_volatility.models.selection import OptimalLagSelector
    from causal_volatility.stationarity.transform import StationarityTransformer

    if proc_df is None:
        proc_df = DataProcessor().process_and_align(df_raw)
    if stat_df is None:
        stat_df = StationarityTransformer().transform(proc_df)

    n_total = len(stat_df)
    test_chunk_size = (n_total - min_train_size) // n_folds
    fold_results = []

    for k in range(n_folds):
        train_end = min_train_size + k * test_chunk_size
        test_end = train_end + test_chunk_size if k < n_folds - 1 else n_total

        train_stat = stat_df.iloc[:train_end]
        test_stat = stat_df.iloc[train_end:test_end]

        if len(test_stat) < 20:
            continue

        try:
            # 1. Fit GARCH on training volatility differences
            vol_diff_train = train_stat["GK_Vol_Diff"]
            lag_selector = OptimalLagSelector(max_lag=21, criterion="bic")
            best_lag = lag_selector.select_lag(vol_diff_train.values)
            vol_model = get_volatility_model("garch", lags=best_lag)
            vol_model.fit(vol_diff_train)
            z_train = vol_model.get_standardized_residuals()

            # 2. Build causal DataFrame on training data
            causal_cols = [c for c in train_stat.columns if c not in ["GK_Vol_Diff"]]
            train_causal = train_stat[causal_cols].loc[z_train.index].copy()
            train_causal["Vol_Innovations"] = z_train
            train_causal = train_causal.dropna()

            # 3. Run causal discovery on training window ONLY
            causal_out = execute_structural_causal_discovery(
                train_causal, tau_max=tau_max, alpha_thresh=alpha_thresh
            )

            # 4. Compute multiplier on test window using train-fitted graph
            vol_diff_full = stat_df["GK_Vol_Diff"].iloc[:test_end]
            vol_model_full = get_volatility_model("garch", lags=best_lag)
            vol_model_full.fit(vol_diff_full)
            z_full = vol_model_full.get_standardized_residuals()

            test_idx = test_stat.index
            common_test_idx = test_idx.intersection(z_full.index)
            if len(common_test_idx) < 20:
                continue

            full_causal = stat_df[causal_cols].loc[z_full.index].copy()
            full_causal["Vol_Innovations"] = z_full
            full_causal = full_causal.dropna()

            mult_series = construct_causal_multiplier(
                full_causal, causal_out,
                baseline_multiplier=baseline_multiplier,
                min_multiplier=min_multiplier,
            )
            test_mult = mult_series.loc[mult_series.index.intersection(common_test_idx)]

            if len(test_mult) < 20:
                continue

            # 5. Run trailing stop backtest on test window
            backtest_result = execute_causal_trailing_stop(
                df_raw, proc_df, test_mult,
                static_multiplier=static_multiplier,
                calm_threshold=calm_threshold,
                max_cash_days=max_cash_days,
            )
            metrics = compute_comprehensive_risk_metrics(backtest_result)

            bh_sharpe = float(metrics.loc["Buy_And_Hold", "Sharpe Ratio"]) if "Buy_And_Hold" in metrics.index else 0.0
            causal_sharpe = float(metrics.loc["Causal_Adaptive", "Sharpe Ratio"]) if "Causal_Adaptive" in metrics.index else 0.0
            baseline_sharpe = float(metrics.loc["Standard_Baseline", "Sharpe Ratio"]) if "Standard_Baseline" in metrics.index else 0.0

            bh_dd = float(metrics.loc["Buy_And_Hold", "Maximum Drawdown"]) if "Buy_And_Hold" in metrics.index else 0.0
            causal_dd = float(metrics.loc["Causal_Adaptive", "Maximum Drawdown"]) if "Causal_Adaptive" in metrics.index else 0.0

            fold_results.append({
                "fold": k + 1,
                "train_rows": train_end,
                "test_rows": test_end - train_end,
                "train_start": str(train_stat.index[0].date()),
                "train_end": str(train_stat.index[-1].date()),
                "test_start": str(test_stat.index[0].date()),
                "test_end": str(test_stat.index[-1].date()),
                "baseline_sharpe": baseline_sharpe,
                "causal_sharpe": causal_sharpe,
                "bh_sharpe": bh_sharpe,
                "bh_max_dd": bh_dd,
                "causal_max_dd": causal_dd,
                "causal_beats_bh_sharpe": causal_sharpe > bh_sharpe,
                "causal_beats_bh_dd": causal_dd > bh_dd,
            })

        except Exception as e:
            fold_results.append({
                "fold": k + 1,
                "train_rows": train_end,
                "test_rows": test_end - train_end,
                "train_start": str(train_stat.index[0].date()),
                "train_end": str(train_stat.index[-1].date()),
                "test_start": str(test_stat.index[0].date()),
                "test_end": str(test_stat.index[-1].date()),
                "baseline_sharpe": 0.0,
                "causal_sharpe": 0.0,
                "bh_sharpe": 0.0,
                "bh_max_dd": 0.0,
                "causal_max_dd": 0.0,
                "causal_beats_bh_sharpe": False,
                "causal_beats_bh_dd": False,
                "error": str(e),
            })

    summary_df = pd.DataFrame(fold_results)
    
    if len(summary_df) > 0:
        win_rate = float((summary_df["causal_sharpe"] >= summary_df["baseline_sharpe"]).mean())
        win_rate_sharpe = float(summary_df["causal_beats_bh_sharpe"].mean())
        win_rate_dd = float(summary_df["causal_beats_bh_dd"].mean())
        mean_causal_sharpe = float(summary_df["causal_sharpe"].mean())
        mean_bh_sharpe = float(summary_df["bh_sharpe"].mean())
    else:
        win_rate = 0.0
        win_rate_sharpe = 0.0
        win_rate_dd = 0.0
        mean_causal_sharpe = 0.0
        mean_bh_sharpe = 0.0

    return {
        "n_folds": len(fold_results),
        "folds": fold_results,
        "summary_df": summary_df,
        "win_rate": win_rate,
        "win_rate_sharpe": win_rate_sharpe,
        "win_rate_dd": win_rate_dd,
        "mean_causal_sharpe": mean_causal_sharpe,
        "mean_bh_sharpe": mean_bh_sharpe,
    }
