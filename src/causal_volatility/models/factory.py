"""Factory and registry for volatility residualization models."""

from typing import Dict, List, Type, Union
from causal_volatility.models.base import BaseVolatilityModel
from causal_volatility.models.egarch import EGARCHModel
from causal_volatility.models.garch import ARGARCHModel, StudentTGARCHModel
from causal_volatility.models.gjr_garch import GJRGARCHModel

# Canonical model registry mapping model identifiers to classes
_MODEL_REGISTRY: Dict[str, Type[BaseVolatilityModel]] = {
    "garch": ARGARCHModel,
    "ar_garch": ARGARCHModel,
    "argarch": ARGARCHModel,
    "egarch": EGARCHModel,
    "gjr": GJRGARCHModel,
    "gjr_garch": GJRGARCHModel,
    "gjrgarch": GJRGARCHModel,
    "student_t": StudentTGARCHModel,
    "studentt": StudentTGARCHModel,
    "t_garch": StudentTGARCHModel,
}


def register_volatility_model(name: str, model_cls: Type[BaseVolatilityModel]) -> None:
    """Register a new volatility model class into the factory registry.

    Parameters
    ----------
    name : str
        Identifier key for the model.
    model_cls : Type[BaseVolatilityModel]
        Class implementing BaseVolatilityModel contract.
    """
    if not issubclass(model_cls, BaseVolatilityModel):
        raise TypeError(f"Class '{model_cls.__name__}' must inherit from BaseVolatilityModel.")
    _MODEL_REGISTRY[name.lower().replace("-", "_")] = model_cls


def list_available_models() -> List[str]:
    """List all registered volatility model identifier keys.

    Returns
    -------
    List[str]
        List of registered model keys.
    """
    return sorted(list(_MODEL_REGISTRY.keys()))


def get_volatility_model(
    name: Union[str, BaseVolatilityModel],
    **kwargs,
) -> BaseVolatilityModel:
    """Factory function instantiating a volatility model by name.

    Parameters
    ----------
    name : str or BaseVolatilityModel
        Model name ('garch', 'egarch', 'gjr', 'student_t') or pre-instantiated model.
    **kwargs
        Hyperparameters passed to the model constructor (e.g. p, q, dist, mean).

    Returns
    -------
    BaseVolatilityModel
        Instantiated volatility model complying with BaseVolatilityModel contract.
    """
    if isinstance(name, BaseVolatilityModel):
        return name

    normalized_key = str(name).lower().replace("-", "_")
    if normalized_key not in _MODEL_REGISTRY:
        available = list_available_models()
        raise ValueError(
            f"Unknown volatility model '{name}'. Registered options: {available}"
        )

    model_cls = _MODEL_REGISTRY[normalized_key]
    if "lags" in kwargs and "ar_lags" not in kwargs:
        kwargs["ar_lags"] = kwargs.pop("lags")
        if "mean" not in kwargs and kwargs["ar_lags"]:
            kwargs["mean"] = "AR"
    return model_cls(**kwargs)

