"""Causal Graph Volatility Analysis & Framework Expansion Package."""

from causal_volatility.causal.discovery import (
    StructuralCausalDiscovery,
    execute_structural_causal_discovery,
)
from causal_volatility.causal.multiplier import (
    CausalMultiplier,
    construct_causal_multiplier,
)
from causal_volatility.causal.pathways import (
    CausalEdge,
    CausalGraph,
    analyze_causal_pathways,
)
from causal_volatility.causal.precision_benchmark import (
    cross_validate_edges,
    estimate_precision_matrix,
)
from causal_volatility.config import BacktestConfig, DataConfig, ModelConfig
from causal_volatility.data.fetcher import SystematicRiskDataFetcher
from causal_volatility.data.processor import DataProcessor
from causal_volatility.data.storage import DataStorage
from causal_volatility.estimators.base import BaseVolatilityEstimator
from causal_volatility.estimators.close_to_close import CloseToCloseEstimator
from causal_volatility.estimators.garman_klass import GarmanKlassEstimator
from causal_volatility.estimators.parkinson import ParkinsonEstimator
from causal_volatility.estimators.rogers_satchell import RogersSatchellEstimator
from causal_volatility.estimators.yang_zhang import YangZhangEstimator
from causal_volatility.models.base import BaseVolatilityModel
from causal_volatility.models.egarch import EGARCHModel
from causal_volatility.models.factory import get_volatility_model
from causal_volatility.models.garch import ARGARCHModel, StudentTGARCHModel
from causal_volatility.models.gjr_garch import GJRGARCHModel
from causal_volatility.models.selection import OptimalLagSelector
from causal_volatility.stationarity.diagnostics import (
    DependencyAuditor,
    DualStationaritySuite,
    run_dependence_tests,
    run_dual_stationarity_tests,
)
from causal_volatility.stationarity.transform import StationarityTransformer
from causal_volatility.backtest.engine import execute_causal_trailing_stop, run_trailing_stop
from causal_volatility.backtest.metrics import compute_comprehensive_risk_metrics
from causal_volatility.backtest.validation import run_walk_forward_validation
from causal_volatility.pipeline import CausalVolatilityPipeline
try:
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
except ImportError:
    pass

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "ModelConfig",
    "BacktestConfig",
    "SystematicRiskDataFetcher",
    "DataProcessor",
    "DataStorage",
    "BaseVolatilityEstimator",
    "GarmanKlassEstimator",
    "ParkinsonEstimator",
    "RogersSatchellEstimator",
    "YangZhangEstimator",
    "CloseToCloseEstimator",
    "StationarityTransformer",
    "DualStationaritySuite",
    "DependencyAuditor",
    "run_dual_stationarity_tests",
    "run_dependence_tests",
    "BaseVolatilityModel",
    "OptimalLagSelector",
    "ARGARCHModel",
    "StudentTGARCHModel",
    "EGARCHModel",
    "GJRGARCHModel",
    "get_volatility_model",
    "execute_structural_causal_discovery",
    "StructuralCausalDiscovery",
    "analyze_causal_pathways",
    "estimate_precision_matrix",
    "cross_validate_edges",
    "CausalGraph",
    "CausalEdge",
    "construct_causal_multiplier",
    "CausalMultiplier",
    "run_trailing_stop",
    "execute_causal_trailing_stop",
    "compute_comprehensive_risk_metrics",
    "run_walk_forward_validation",
    "CausalVolatilityPipeline",
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


