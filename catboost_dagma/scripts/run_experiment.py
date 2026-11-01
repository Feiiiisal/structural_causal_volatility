#!/usr/bin/env python
"""Run the capstone experiment and save report-ready outputs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_dagma.config import DAGMA_CONFIG
from catboost_dagma.pipeline import CatBoostDagmaPipeline
from catboost_dagma.visualization.diagnostics import (
    plot_catboost_ablation_and_importance,
    plot_contagion_strategy_performance,
    plot_ff3_residualization_audit,
    plot_network_comparison,
    plot_structural_breaks_causal_drift,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns-path", required=True)
    parser.add_argument("--factor-path")
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--market", default="market")
    parser.add_argument("--run-name")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-01-01")
    parser.add_argument("--window-size", type=int, default=DAGMA_CONFIG["window_size"])
    parser.add_argument("--step-size", type=int, default=DAGMA_CONFIG["step_size"])
    parser.add_argument("--workers", type=int, default=DAGMA_CONFIG["n_workers"])
    parser.add_argument("--online-factors", action="store_true")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_outputs(results, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    results["exposure_df"].to_csv(tables / "factor_exposure.csv")
    results["network_comparison_df"].to_csv(tables / "network_comparison.csv", index=False)
    results["vif_df"].to_csv(tables / "vif.csv", index=False)
    results["breaks_df"].to_csv(tables / "structural_breaks.csv", index=False)
    results["strategy_results"]["df_equity_evaluation"].to_csv(tables / "portfolio_equity.csv")
    results["ablation_results"]["feature_importance_df"].to_csv(
        tables / "catboost_feature_importance.csv", index=False
    )

    metrics = {
        "market": args.market,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "assets": list(results["returns_df"].columns),
        "observations": int(len(results["returns_df"])),
        "ablation": {
            "target": results["ablation_results"]["target"],
            "baseline": results["ablation_results"]["baseline"],
            "augmented": results["ablation_results"]["augmented"],
            "improvement": results["ablation_results"]["improvement"],
        },
        "portfolio": results["strategy_results"]["stats_summary"],
        "elapsed_seconds": results["elapsed_seconds"],
        "created_utc": datetime.now(timezone.utc),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_ready),
        encoding="utf-8",
    )

    plot_ff3_residualization_audit(
        results["returns_df"],
        results["residuals_df"],
        results["exposure_df"],
        figures / "factor_residualization.png",
    )
    plot_network_comparison(
        results["network_comparison_df"],
        results["vif_df"],
        figures / "network_comparison.png",
    )
    plot_structural_breaks_causal_drift(
        results["drift_series"],
        results["breaks_df"],
        results["break_features_df"]["drift_threshold"],
        figures / "network_drift.png",
    )
    plot_catboost_ablation_and_importance(
        results["ablation_results"],
        figures / "catboost_ablation.png",
    )
    plot_contagion_strategy_performance(
        results["strategy_results"],
        figures / "portfolio_performance.png",
    )


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.market}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.output_dir) / run_name

    pipeline = CatBoostDagmaPipeline(
        assets=args.assets,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size,
        step_size=args.step_size,
        n_workers=args.workers,
        sp100_path=args.returns_path,
        factor_path=args.factor_path,
        offline=not args.online_factors,
    )
    results = pipeline.run(verbose=True)
    save_outputs(results, out_dir, args)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
