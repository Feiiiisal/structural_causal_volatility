#!/usr/bin/env python
"""Run the capstone pipeline."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost_dagma.config import DAGMA_CONFIG
from catboost_dagma.pipeline import CatBoostDagmaPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--window-size", type=int, default=DAGMA_CONFIG["window_size"])
    parser.add_argument("--step-size", type=int, default=DAGMA_CONFIG["step_size"])
    parser.add_argument("--workers", type=int, default=DAGMA_CONFIG["n_workers"])
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-01-01")
    args = parser.parse_args()

    pipeline = CatBoostDagmaPipeline(
        assets=args.assets,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size,
        step_size=args.step_size,
        n_workers=args.workers,
    )
    pipeline.run(verbose=True)


if __name__ == "__main__":
    main()
