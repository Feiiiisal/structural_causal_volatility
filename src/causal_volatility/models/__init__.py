"""Volatility residualization models and optimal lag selection (R3)."""

from causal_volatility.models.base import BaseVolatilityModel
from causal_volatility.models.egarch import EGARCHModel
from causal_volatility.models.factory import (
    get_volatility_model,
    list_available_models,
    register_volatility_model,
)
from causal_volatility.models.garch import ARGARCHModel, StudentTGARCHModel
from causal_volatility.models.gjr_garch import GJRGARCHModel
from causal_volatility.models.selection import OptimalLagSelector

__all__ = [
    "BaseVolatilityModel",
    "OptimalLagSelector",
    "ARGARCHModel",
    "StudentTGARCHModel",
    "EGARCHModel",
    "GJRGARCHModel",
    "get_volatility_model",
    "register_volatility_model",
    "list_available_models",
]
