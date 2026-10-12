#!/usr/bin/env python
# coding: utf-8
"""
Standalone Execution Script for Version 2: CatBoost + DAGMA Causal Fusion Pipeline.
Runs the complete workflow, prints quantitative results, and saves all diagnostic graphics.
"""

import argparse
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
CATBOOST_DAGMA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CATBOOST_DAGMA_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CATBOOST_DAGMA_DIR) not in sys.path:
    sys.path.insert(0, str(CATBOOST_DAGMA_DIR))

from catboost_dagma.pipeline import CatBoostDagmaPipeline
from catboost_dagma.config import DEFAULT_SECTOR_ASSETS, FIGURES_DIR


def main():
    parser = argparse.ArgumentParser(description="Run CatBoost + DAGMA Version 2 Pipeline")
    parser.add_argument("--assets", nargs="+", default=None, help="List of asset tickers (defaults to 20 representative multi-sector stocks)")
    parser.add_argument("--window-size", type=int, default=60, help="Rolling window size in days")
    parser.add_argument("--step-size", type=int, default=5, help="Step size between rolling windows")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for parallel DAGMA")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR), help="Output directory for figures")
    args = parser.parse_args()

    pipeline = CatBoostDagmaPipeline(
        assets=args.assets,
        window_size=args.window_size,
        step_size=args.step_size,
        n_workers=args.workers,
        output_dir=args.output_dir,
    )

    results = pipeline.run(verbose=True)
    print("\n✅ All V2 Pipeline deliverables successfully generated!")


if __name__ == "__main__":
    main()
