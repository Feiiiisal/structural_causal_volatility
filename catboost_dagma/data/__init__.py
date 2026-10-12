"""
Data module for CatBoost-DAGMA Version 2.
"""
from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer

__all__ = ["UnifiedDataLoader", "FamaFrenchResidualizer"]
