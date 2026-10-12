"""
CatBoost + DAGMA Version 2: Advanced Non-Linear Causal Discovery and Volatility Fusion Framework.
"""

from catboost_dagma.pipeline import CatBoostDagmaPipeline
from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer
from catboost_dagma.dagma.model import DeepDynotearsMLP
from catboost_dagma.dagma.solver import train_dagma_dynotears, run_parallel_rolling_dagma
from catboost_dagma.benchmark.precision_matrix import PrecisionMatrixEstimator
from catboost_dagma.benchmark.orthogonality import EmpiricalOrthogonalityAuditor
from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector
from catboost_dagma.ml.feature_engineering import FusionFeatureEngineer
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel
from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy

__all__ = [
    "CatBoostDagmaPipeline",
    "UnifiedDataLoader",
    "FamaFrenchResidualizer",
    "DeepDynotearsMLP",
    "train_dagma_dynotears",
    "run_parallel_rolling_dagma",
    "PrecisionMatrixEstimator",
    "EmpiricalOrthogonalityAuditor",
    "StructuralBreakDetector",
    "FusionFeatureEngineer",
    "CatBoostFusionModel",
    "CausalContagionStrategy",
]
