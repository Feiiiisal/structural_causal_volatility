"""Garman-Klass Realized Volatility Estimator.

Implements the classical Garman-Klass (1980) extreme-value volatility estimator
incorporating opening, closing, high, and low prices, with explicit analytical
zero-clipping to prevent negative variance on anomalous gap days.
"""

from typing import Optional
import numpy as np
import pandas as pd

from causal_volatility.estimators.base import BaseVolatilityEstimator


class GarmanKlassEstimator(BaseVolatilityEstimator):
    """Garman-Klass (1980) intraday price volatility estimator.

    Mathematical formulation:
    $$\\sigma_{GK}^2 = \\max\\left(0, 0.5 \\left(\\ln\\frac{H}{L}\\right)^2 - (2\\ln 2 - 1) \\left(\\ln\\frac{C}{O}\\right)^2\\right)$$
    $$\\sigma_{GK} = \\sqrt{\\sigma_{GK}^2}$$

    The negative variance clipping is essential: on days with a compressed intraday
    range relative to a large opening gap, $(2\\ln 2 - 1)(\\ln C/O)^2 > 0.5(\\ln H/L)^2$,
    which would otherwise yield NaN under the square root.
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
        """Calculate daily Garman-Klass volatility."""
        o_series = open if open is not None else open_
        if o_series is None:
            raise ValueError("Must provide 'open' or 'open_' price series.")

        h = high.astype(float)
        l = low.astype(float)
        c = close.astype(float)
        o = o_series.astype(float)

        log_hl = np.log(h / l)
        log_co = np.log(c / o)

        var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        # Analytical zero-clipping to prevent negative variance on gap-dominated bars
        var = np.maximum(0.0, var)
        vol = np.sqrt(var)

        return pd.Series(vol, index=high.index, name="Garman_Klass_Vol")
