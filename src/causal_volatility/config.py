"""Configuration dataclasses for the causal_volatility package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class DataConfig:
    """Configuration parameters for data ingestion and preprocessing."""

    start_date: str = "2016-01-01"
    end_date: str = "2026-01-01"
    equity_ticker: str = "OEF"
    vix_ticker: str = "^VIX"
    fred_series_id: str = "BAA10Y"
    fred_api_key: Optional[str] = None
    offline: bool = False
    fixture_path: Optional[Union[str, Path]] = None
    cache_dir: Optional[Union[str, Path]] = None
    volume_scale: float = 1e7
    epsilon: float = 1e-8


@dataclass
class ModelConfig:
    """Configuration parameters for volatility modeling and causal discovery."""

    model_type: str = "garch"  # Options: 'garch', 'egarch', 'gjr', 'student_t'
    mean_model: str = "AR"
    p_lags: int = 8
    q_lags: int = 1
    o_lags: int = 0
    dist: str = "normal"  # Options: 'normal', 't', 'skewt'
    scale_factor: float = 100.0
    auto_select_lag: bool = True
    max_ar_lag: int = 21
    tau_max: int = 5
    alpha_thresh: float = 0.05
    target_var: str = "Vol_Innovations"


@dataclass
class BacktestConfig:
    """Configuration parameters for ratchet trailing stop backtesting and validation."""

    lambda_0: float = 4.5
    lambda_min: float = 1.8
    ewm_span: int = 5
    rolling_percentile_window: int = 126
    rolling_min_periods: int = 20
    ma_window: int = 20
    reentry_calm_threshold: float = 2.8
    max_cash_days: int = 15
    atr_window: int = 14
    split_type: str = "oos"  # Options: 'oos', '60_20_20', 'walkforward'
    train_ratio: float = 0.5
    val_ratio: float = 0.0
    test_ratio: float = 0.5
    trading_days_per_year: int = 252
