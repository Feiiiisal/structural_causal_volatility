"""Parkinson Realized Volatility Estimator.

Implements the Parkinson (1980) extreme-value volatility estimator using
high and low prices.
"""

from typing import Optional
import numpy as np
import pandas as pd

from causal_volatility.estimators.base import BaseVolatilityEstimator


class ParkinsonEstimator(BaseVolatilityEstimator):
    """Parkinson (1980) range-based volatility estimator.

    Mathematical formulation:
    $$\\sigma_P^2 = \\frac{\\left(\\ln\\frac{H}{L}\\right)^2}{4\\ln 2}$$
    $$\\sigma_P = \\sqrt{\\max(0, \\sigma_P^2)}$$
    """

    def estimate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        open: Optional[pd.Series] = None,
        open_: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """Calculate daily Parkinson volatility."""
        h = high.astype(float)
        l = low.astype(float)

        log_hl = np.log(h / l)
        var = (log_hl ** 2) / (4.0 * np.log(2.0))
        var = np.maximum(0.0, var)
        vol = np.sqrt(var)

        return pd.Series(vol, index=high.index, name="Parkinson_Vol")
