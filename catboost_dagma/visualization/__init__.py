"""
Visualization module for CatBoost-DAGMA Version 2.
"""
from catboost_dagma.visualization.diagnostics import (
    plot_ff3_residualization_audit,
    plot_empirical_orthogonality_benchmark,
    plot_structural_breaks_causal_drift,
    plot_catboost_ablation_and_importance,
    plot_contagion_strategy_performance,
)

__all__ = [
    "plot_ff3_residualization_audit",
    "plot_empirical_orthogonality_benchmark",
    "plot_structural_breaks_causal_drift",
    "plot_catboost_ablation_and_importance",
    "plot_contagion_strategy_performance",
]
