"""Configuration for the capstone DAGMA + CatBoost research pipeline.

Defaults are chosen for defensible out-of-sample research. All paths and
window lengths remain configurable so the submitted 60-day specification can
still be reproduced as a sensitivity check.
"""
from pathlib import Path
import os
import torch

CATBOOST_DAGMA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CATBOOST_DAGMA_DIR.parent
DATA_DIR = CATBOOST_DAGMA_DIR / "data"
FIGURES_DIR = CATBOOST_DAGMA_DIR / "figures"
DOCS_DIR = CATBOOST_DAGMA_DIR / "docs"
REPORTS_DIR = CATBOOST_DAGMA_DIR / "reports"

for _p in (FIGURES_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Never hard-code one researcher's home directory. Set SP100_DATA_PATH when running.
_env_sp100 = os.environ.get("SP100_DATA_PATH")
SP100_PICKLE_PATH = Path(_env_sp100).expanduser() if _env_sp100 else PROJECT_ROOT / "data" / "sp100_daily_returns_2016_2026.pkl"

DEFAULT_SECTOR_ASSETS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    "XOM", "CVX", "COP",
    "JNJ", "UNH", "PFE", "ABBV",
    "AMZN", "PG", "CAT", "BA", "HD",
]

HISTORICAL_EVENTS = {
    "COVID_Crash": {"start": "2020-02-01", "end": "2020-04-30", "peak_date": "2020-03-23"},
    "Fed_Hikes_2022": {"start": "2021-12-01", "end": "2022-10-31", "peak_date": "2022-06-13"},
    "SVB_Banking_Collapse": {"start": "2023-02-15", "end": "2023-04-15", "peak_date": "2023-03-10"},
}

FF_FACTORS = ["Mkt-RF", "SMB", "HML"]
FF_RISK_FREE = "RF"

DAGMA_CONFIG = {
    # One trading year is the primary specification; 60 days remains a robustness case.
    "window_size": 252,
    "step_size": 21,
    "p_lags": 1,
    "hidden_dim": 16,
    "loss_type": "student-t",
    "learnable_nu": True,
    "init_nu": 3.0,
    "lambda1": 0.02,
    "T": 4,
    "mu_init": 0.1,
    "mu_factor": 0.1,
    "s": 1.0,
    "warm_iter": 100,
    "max_iter": 200,
    "lr": 0.005,
    "tol": 1e-6,
    "checkpoint": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float64,
    "n_workers": min(8, os.cpu_count() or 4),
}

PRECISION_CONFIG = {
    "alpha_grid": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "cv_folds": 3,
    # Threshold is applied to partial correlations, not raw precision entries.
    "partial_corr_threshold": 0.10,
}

BREAK_DETECTION_CONFIG = {
    "percentile_threshold": 95.0,
    "min_event_distance": 10,
    "min_history": 20,
}

CATBOOST_CONFIG = {
    "task_type": "GPU" if torch.cuda.is_available() else "CPU",
    "devices": "0" if torch.cuda.is_available() else None,
    "gpu_ram_part": 0.70,
    "iterations": 800,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "train_ratio": 0.70,
    "validation_ratio": 0.20,
    "verbose": 0,
}

STRATEGY_CONFIG = {
    "initial_capital": 10000.0,
    "transaction_cost": 0.0005,
    "rebalance_cadence": 5,
    "contagion_hub_prune_k": 2,
    "risk_parity_lookback": 60,
    "lambda_min": 2.5,
    "lambda_max": 4.0,
    "ma_window": 20,
    "drift_history": 252,
}
