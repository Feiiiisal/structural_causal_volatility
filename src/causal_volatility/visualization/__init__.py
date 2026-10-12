"""Visualization and quantitative graphics package for causal volatility analysis."""

from causal_volatility.visualization.eda import (
    plot_correlation_matrix,
    plot_macro_overlay,
    plot_return_and_vol_distributions,
    plot_stationarity_transformation,
    plot_volatility_estimators_comparison,
)
from causal_volatility.visualization.stage_diagnostics import (
    plot_adaptive_multiplier_dynamics,
    plot_causal_dag_pathways,
    plot_garch_diagnostics,
    plot_in_sample_equity_curve,
    plot_out_of_sample_drawdown,
    plot_out_of_sample_equity_curve,
    plot_regime_reentry_analysis,
)

__all__ = [
    "plot_correlation_matrix",
    "plot_return_and_vol_distributions",
    "plot_macro_overlay",
    "plot_volatility_estimators_comparison",
    "plot_stationarity_transformation",
    "plot_garch_diagnostics",
    "plot_causal_dag_pathways",
    "plot_adaptive_multiplier_dynamics",
    "plot_in_sample_equity_curve",
    "plot_out_of_sample_equity_curve",
    "plot_out_of_sample_drawdown",
    "plot_regime_reentry_analysis",
]
