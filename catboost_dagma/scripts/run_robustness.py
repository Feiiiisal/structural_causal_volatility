#!/usr/bin/env python
"""Reviewer-motivated uncertainty and sensitivity analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_dagma.analysis.uncertainty import (
    forecast_uncertainty,
    portfolio_uncertainty,
)
from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector
from catboost_dagma.config import STRATEGY_CONFIG
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel
from catboost_dagma.ml.feature_engineering import FusionFeatureEngineer
from catboost_dagma.pipeline import CatBoostDagmaPipeline
from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--returns-path", required=True)
    parser.add_argument("--factor-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def prediction_frame(result):
    ablation = result["ablation_results"]
    df = ablation["test_df"][["date", "asset", "forward_volatility"]].copy()
    df["pred_baseline"] = ablation["pred_test_base"]
    df["pred_network"] = ablation["pred_test_aug"]
    return df


def row_from_result(label, result, **settings):
    ablation = result["ablation_results"]
    portfolio = result["strategy_results"]["stats_summary"]
    return {
        "variant": label,
        "assets": len(result["returns_df"].columns),
        "observations": len(result["returns_df"]),
        "rmse_reduction_pct": ablation["improvement"]["rmse_reduction_pct"],
        "baseline_rmse": ablation["baseline"]["test_rmse"],
        "network_rmse": ablation["augmented"]["test_rmse"],
        "baseline_r2": ablation["baseline"]["test_r2"],
        "network_r2": ablation["augmented"]["test_r2"],
        "risk_parity_sharpe": portfolio["Standard_Risk_Parity"]["Sharpe_Ratio"],
        "network_sharpe": portfolio["Causal_Contagion_Pruned"]["Sharpe_Ratio"],
        "risk_parity_mdd": portfolio["Standard_Risk_Parity"]["Max_Drawdown"],
        "network_mdd": portfolio["Causal_Contagion_Pruned"]["Max_Drawdown"],
        **settings,
    }


def run_full(args, *, assets=None, window_size=252, volatility_proxy="absolute"):
    return CatBoostDagmaPipeline(
        assets=assets,
        start_date="2016-01-01",
        end_date="2026-01-01",
        window_size=window_size,
        step_size=21,
        n_workers=args.workers,
        returns_path=args.returns_path,
        factor_path=args.factor_path,
        offline=True,
        volatility_proxy=volatility_proxy,
        break_percentile=95.0,
    ).run(verbose=True)


def break_variant(base, percentile):
    detector = StructuralBreakDetector(
        percentile_threshold=percentile,
        min_history=20,
        history_window=252,
    )
    drift = base["drift_series"]
    breaks = detector.fit_detect(drift)
    break_features = detector.generate_break_features(drift)

    features = FusionFeatureEngineer(forward_horizon=5).build_ml_dataset(
        base["returns_df"],
        base["dagma_results"],
        base["precision_results"],
        break_features,
    )
    ablation = CatBoostFusionModel(
        target_col="forward_volatility"
    ).run_ablation_study(features, split_date="2021-01-01")

    strategy = CausalContagionStrategy().backtest(
        base["returns_df"],
        base["dagma_results"],
        breaks,
        drift_series=drift,
        evaluation_start="2021-01-01",
    )

    result = dict(base)
    result["breaks_df"] = breaks
    result["break_features_df"] = break_features
    result["df_ml"] = features
    result["ablation_results"] = ablation
    result["strategy_results"] = strategy
    return result


def strategy_variant(base, *, transaction_cost=None, prune_k=None):
    cfg = STRATEGY_CONFIG.copy()
    if transaction_cost is not None:
        cfg["transaction_cost"] = transaction_cost
    if prune_k is not None:
        cfg["contagion_hub_prune_k"] = prune_k

    strategy = CausalContagionStrategy(cfg).backtest(
        base["returns_df"],
        base["dagma_results"],
        base["breaks_df"],
        drift_series=base["drift_series"],
        evaluation_start="2021-01-01",
    )

    result = dict(base)
    result["strategy_results"] = strategy
    return result


def core15_assets(base):
    returns = base["returns_df"]
    training = returns.loc[returns.index < pd.Timestamp("2021-01-01")]
    annual_vol = training.std(ddof=1) * np.sqrt(252)
    drop = annual_vol.nlargest(5).index
    return [asset for asset in returns.columns if asset not in set(drop)]


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[{args.market}] base specification")
    base = run_full(args)

    predictions = prediction_frame(base)
    predictions.to_csv(output / "base_forecast_predictions.csv", index=False)
    base["strategy_results"]["df_equity_evaluation"].to_csv(
        output / "base_portfolio_equity.csv"
    )

    uncertainty = {
        "forecast": forecast_uncertainty(predictions),
        "portfolio": portfolio_uncertainty(
            base["strategy_results"]["df_equity_evaluation"]
        ),
    }
    (output / "uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2),
        encoding="utf-8",
    )

    rows = [
        row_from_result(
            "base",
            base,
            window_size=252,
            volatility_proxy="absolute",
            break_percentile=95.0,
            transaction_cost=0.0005,
            prune_k=2,
            sample_rule="balanced_20",
        )
    ]

    print(f"[{args.market}] 126-day window")
    short_window = run_full(args, window_size=126)
    rows.append(row_from_result(
        "window_126",
        short_window,
        window_size=126,
        volatility_proxy="absolute",
        break_percentile=95.0,
        transaction_cost=0.0005,
        prune_k=2,
        sample_rule="balanced_20",
    ))

    print(f"[{args.market}] 504-day window")
    long_window = run_full(args, window_size=504)
    rows.append(row_from_result(
        "window_504",
        long_window,
        window_size=504,
        volatility_proxy="absolute",
        break_percentile=95.0,
        transaction_cost=0.0005,
        prune_k=2,
        sample_rule="balanced_20",
    ))

    print(f"[{args.market}] squared-residual volatility proxy")
    squared = run_full(args, volatility_proxy="squared")
    rows.append(row_from_result(
        "squared_residual",
        squared,
        window_size=252,
        volatility_proxy="squared",
        break_percentile=95.0,
        transaction_cost=0.0005,
        prune_k=2,
        sample_rule="balanced_20",
    ))

    for percentile in (90.0, 97.5):
        print(f"[{args.market}] break percentile {percentile}")
        variant = break_variant(base, percentile)
        rows.append(row_from_result(
            f"break_{str(percentile).replace('.', '_')}",
            variant,
            window_size=252,
            volatility_proxy="absolute",
            break_percentile=percentile,
            transaction_cost=0.0005,
            prune_k=2,
            sample_rule="balanced_20",
        ))

    for cost in (0.00025, 0.001):
        print(f"[{args.market}] transaction cost {cost}")
        variant = strategy_variant(base, transaction_cost=cost)
        rows.append(row_from_result(
            f"cost_{cost}",
            variant,
            window_size=252,
            volatility_proxy="absolute",
            break_percentile=95.0,
            transaction_cost=cost,
            prune_k=2,
            sample_rule="balanced_20",
        ))

    for prune_k in (1, 3):
        print(f"[{args.market}] prune k={prune_k}")
        variant = strategy_variant(base, prune_k=prune_k)
        rows.append(row_from_result(
            f"prune_{prune_k}",
            variant,
            window_size=252,
            volatility_proxy="absolute",
            break_percentile=95.0,
            transaction_cost=0.0005,
            prune_k=prune_k,
            sample_rule="balanced_20",
        ))

    selected = core15_assets(base)
    print(f"[{args.market}] training-period composition check: {selected}")
    composition = run_full(args, assets=selected)
    rows.append(row_from_result(
        "training_core15",
        composition,
        window_size=252,
        volatility_proxy="absolute",
        break_percentile=95.0,
        transaction_cost=0.0005,
        prune_k=2,
        sample_rule="exclude_5_highest_pre2021_vol",
    ))

    table = pd.DataFrame(rows)
    table.to_csv(output / "sensitivity.csv", index=False)

    metadata = {
        "market": args.market,
        "base_assets": list(base["returns_df"].columns),
        "core15_assets": selected,
        "uncertainty_method": {
            "forecast": "paired 20-trading-day moving-block bootstrap over test dates; 2000 replications; HAC loss-differential test with lag 5",
            "portfolio": "paired 20-trading-day moving-block bootstrap of daily strategy returns; 2000 replications",
        },
        "interpretation": (
            "These checks are reviewer-motivated robustness analyses. "
            "They are not used to select a replacement base specification."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
