"""Abstract base interface for realized volatility estimators."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseVolatilityEstimator(ABC):
    """Abstract base class for all realized price volatility estimators."""

    @abstractmethod
    def estimate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        open: Optional[pd.Series] = None,
        open_: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """Compute the realized volatility series from daily OHLC prices.

        Parameters
        ----------
        high : pd.Series
            Daily high price series.
        low : pd.Series
            Daily low price series.
        close : pd.Series
            Daily close price series.
        open : pd.Series
            Daily open price series.
        **kwargs : Any
            Additional estimator-specific parameters (e.g. rolling window).

        Returns
        -------
        pd.Series
            Strictly non-negative realized volatility series aligned with input index.
        """
        pass
