"""Rogers-Satchell Realized Volatility Estimator.

Implements the Rogers and Satchell (1991) volatility estimator, which allows
for non-zero drift in price processes while utilizing high, low, close, and open.
"""

from typing import Optional
import numpy as np
import pandas as pd

from causal_volatility.estimators.base import BaseVolatilityEstimator


class RogersSatchellEstimator(BaseVolatilityEstimator):
    """Rogers-Satchell (1991) realized volatility estimator.

    Mathematical formulation:
    $$\\sigma_{RS}^2 = \\ln\\frac{H}{C}\\ln\\frac{H}{O} + \\ln\\frac{L}{C}\\ln\\frac{L}{O}$$
    $$\\sigma_{RS} = \\sqrt{\\max(0, \\sigma_{RS}^2)}$$
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
        """Calculate daily Rogers-Satchell volatility."""
        o_series = open if open is not None else open_
        if o_series is None:
            raise ValueError("Must provide 'open' or 'open_' price series.")

        h = high.astype(float)
        l = low.astype(float)
        c = close.astype(float)
        o = o_series.astype(float)

        log_hc = np.log(h / c)
        log_ho = np.log(h / o)
        log_lc = np.log(l / c)
        log_lo = np.log(l / o)

        var = log_hc * log_ho + log_lc * log_lo
        var = np.maximum(0.0, var)
        vol = np.sqrt(var)

        return pd.Series(vol, index=high.index, name="Rogers_Satchell_Vol")
