"""
Configuration settings for Version 2: DAGMA-DYNOTEARS + CatBoost Fusion Pipeline.
Fuses non-linear continuous causal discovery with Student-t log-likelihood,
Fama-French 3-factor residualization, Graphical LASSO precision matrix benchmarks,
and structural break detection for risk prediction and adaptive portfolio management.
"""

from pathlib import Path
import os
import torch

# Base Directories
CATBOOST_DAGMA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CATBOOST_DAGMA_DIR.parent
DATA_DIR = CATBOOST_DAGMA_DIR / "data"
FIGURES_DIR = CATBOOST_DAGMA_DIR / "figures"
DOCS_DIR = CATBOOST_DAGMA_DIR / "docs"
REPORTS_DIR = CATBOOST_DAGMA_DIR / "reports"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Primary External Data Sources
SP100_PICKLE_PATH = Path("/home/oem/Documents/proj/data/sp100_daily_returns_2018_2026.pkl")

# Default Asset Universe: Representative S&P 100 Multi-Sector Basket
DEFAULT_SECTOR_ASSETS = [
    # Technology / Growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    # Financials / Banks (Vital for SVB break detection)
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    # Energy / Commodities
    "XOM", "CVX", "COP",
    # Healthcare / Defensive
    "JNJ", "UNH", "PFE", "ABBV",
    # Industrials & Consumer
    "AMZN", "PG", "CAT", "BA", "HD"
]

# Historical Stress Events for Calibration and Structural Break Verification
HISTORICAL_EVENTS = {
    "COVID_Crash": {
        "start": "2020-02-01",
        "end": "2020-04-30",
        "peak_date": "2020-03-23",
        "description": "Global liquidity contraction and rapid market selloff"
    },
    "Fed_Hikes_2022": {
        "start": "2021-12-01",
        "end": "2022-10-31",
        "peak_date": "2022-06-13",
        "description": "Macro factor shock: aggressive interest rate hikes and inflation peak"
    },
    "SVB_Banking_Collapse": {
        "start": "2023-02-15",
        "end": "2023-04-15",
        "peak_date": "2023-03-10",
        "description": "Regional banking contagion triggered by Silicon Valley Bank failure"
    }
}

# Fama-French Residualization Settings
FF_FACTORS = ["Mkt-RF", "SMB", "HML"]
FF_RISK_FREE = "RF"

# DAGMA-DYNOTEARS Model & Optimization Parameters
DAGMA_CONFIG = {
    "window_size": 60,           # Rolling estimation window (trading days)
    "step_size": 5,              # Cadence between rolling fits
    "p_lags": 1,                 # Autoregressive lag order for lagged block A
    "hidden_dim": 16,            # Hidden units in non-linear MLP
    "loss_type": "student-t",    # Heavy-tailed loss: 'student-t' or 'gaussian'
    "learnable_nu": True,        # Dynamically estimate tail degrees-of-freedom nu
    "init_nu": 3.0,              # Initial degrees of freedom (leptokurtic prior)
    "lambda1": 0.02,             # L1 sparsity penalty
    "T": 4,                      # Outer central-path stages
    "mu_init": 0.1,              # Initial central-path barrier parameter
    "mu_factor": 0.1,            # Barrier decay factor per stage
    "s": 1.0,                    # Domain safety parameter for log-det acyclicity
    "warm_iter": 100,            # Inner iterations for warm stages
    "max_iter": 200,             # Inner iterations for final stage
    "lr": 0.005,                 # Learning rate for Adam optimizer
    "tol": 1e-6,                 # Early exit tolerance on loss change
    "checkpoint": 50,            # Logging cadence
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float64,      # Double precision essential for log-det stability
    "n_workers": min(20, os.cpu_count() or 4) # ProcessPoolExecutor worker count
}

# Precision Matrix Benchmark Settings
PRECISION_CONFIG = {
    "method": "graphical_lasso", # 'graphical_lasso' or 'ledoit_wolf'
    "alpha_grid": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "cv_folds": 3,
    "sparsity_threshold": 0.05
}

# Structural Break Detection Settings
BREAK_DETECTION_CONFIG = {
    "metric": "frobenius",       # ||W_t - W_{t-1}||_F
    "percentile_threshold": 95.0,# 95th percentile historical drift
    "min_event_distance": 10     # Minimum bars between distinct break events
}

# CatBoost Machine Learning & Ablation Settings
CATBOOST_CONFIG = {
    "task_type": "GPU" if torch.cuda.is_available() else "CPU",
    "devices": "0" if torch.cuda.is_available() else None,
    "gpu_ram_part": 0.70,        # Sized safely for 8GB RTX 4060
    "iterations": 800,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "train_ratio": 0.70,
    "verbose": 0
}

# Downstream Portfolio Strategy Settings
STRATEGY_CONFIG = {
    "initial_capital": 10000.0,
    "transaction_cost": 0.0005,  # 5 bps per trade
    "rebalance_cadence": 5,      # Days between regular rebalance
    "contagion_hub_prune_k": 2,  # Number of top out-degree nodes to prune during stress
    "risk_parity_lookback": 60,  # Volatility estimation lookback
    "lambda_min": 2.5,           # Tightest multiplier for trailing stop during peak stress
    "lambda_max": 4.0,           # Loosest multiplier for trailing stop during calm regime
    "ma_window": 20              # Moving average confirmation window for cash re-entry
}
