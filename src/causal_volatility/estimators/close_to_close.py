"""Close-to-Close Realized Volatility Estimator.

Standard classical sample volatility benchmark based on close-to-close log returns.
"""

from typing import Optional
import numpy as np
import pandas as pd

from causal_volatility.estimators.base import BaseVolatilityEstimator


class CloseToCloseEstimator(BaseVolatilityEstimator):
    """Close-to-Close historical volatility estimator.

    Computes absolute log returns or sample rolling standard deviation
    of log returns:
    $$r_t = \\ln\\left(\\frac{C_t}{C_{t-1}}\\right)$$
    $$\\sigma_{CC, t} = |r_t| \\quad (\\text{or } \\text{std}(r_{t-n+1:t}))$$
    """

    def __init__(self, window: int = 1):
        self.window = window

    def estimate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        open: Optional[pd.Series] = None,
        open_: Optional[pd.Series] = None,
        window: int = 1,
        **kwargs,
    ) -> pd.Series:
        """Calculate close-to-close volatility."""
        c = close.astype(float)
        log_ret = np.log(c / c.shift(1))

        win = window if window is not None else self.window
        if win > 1:
            vol = log_ret.rolling(win).std().fillna(0.0)
        else:
            vol = np.abs(log_ret).fillna(0.0)

        return pd.Series(vol, index=close.index, name="Close_To_Close_Vol")
