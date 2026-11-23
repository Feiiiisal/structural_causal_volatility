"""Verify that results, robustness outputs, and frozen inputs use one snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> None:
    for market in ("us", "india"):
        final = root / "results" / "final" / market
        robust = root / "results" / "robustness" / market
        frozen = root / "frozen_data" / market
        metrics = json.loads((final / "metrics.json").read_text())
        manifest = json.loads((final / "data" / "manifest.json").read_text())
        uncertainty = json.loads((robust / "uncertainty.json").read_text())
        with (robust / "sensitivity.csv").open(newline="", encoding="utf-8") as f:
            variants = list(csv.DictReader(f))
        if not variants or variants[0]["variant"] != "base":
            raise ValueError(f"{market}: missing sensitivity base row")

        checks = [
            ("baseline RMSE", metrics["ablation"]["baseline"]["test_rmse"], uncertainty["forecast"]["baseline_rmse"]),
            ("network RMSE", metrics["ablation"]["augmented"]["test_rmse"], uncertainty["forecast"]["network_rmse"]),
            ("RMSE change", metrics["ablation"]["improvement"]["rmse_reduction_pct"], uncertainty["forecast"]["rmse_reduction_pct"]),
            ("sensitivity baseline", metrics["ablation"]["baseline"]["test_rmse"], float(variants[0]["baseline_rmse"])),
            ("sensitivity network", metrics["ablation"]["augmented"]["test_rmse"], float(variants[0]["network_rmse"])),
            ("sensitivity Sharpe", metrics["portfolio"]["Causal_Contagion_Pruned"]["Sharpe_Ratio"], float(variants[0]["network_sharpe"])),
        ]
        for label, expected, actual in checks:
            tol = 0.002 if "Sharpe" in label else 0.0001
            if abs(float(expected) - float(actual)) > tol:
                raise ValueError(f"{market} {label} mismatch: {expected} vs {actual}")
        for filename, details in manifest["files"].items():
            actual_hash = hashlib.sha256((frozen / filename).read_bytes()).hexdigest()
            if actual_hash != details["sha256"]:
                raise ValueError(f"{market} {filename}: frozen data hash mismatch")
        print(f"{market}: results and all three frozen inputs verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    verify(parser.parse_args().base_dir)
