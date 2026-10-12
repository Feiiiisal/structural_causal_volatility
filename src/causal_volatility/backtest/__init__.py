"""Backtest package for causal volatility analysis."""

from causal_volatility.backtest.engine import execute_causal_trailing_stop, run_trailing_stop
from causal_volatility.backtest.metrics import compute_comprehensive_risk_metrics
from causal_volatility.backtest.validation import run_walk_forward_validation

__all__ = [
    "run_trailing_stop",
    "execute_causal_trailing_stop",
    "compute_comprehensive_risk_metrics",
    "run_walk_forward_validation",
]
