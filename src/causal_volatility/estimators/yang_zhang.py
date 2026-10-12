"""Yang-Zhang Realized Volatility Estimator.

Implements the Yang and Zhang (2000) minimum-variance unbiased estimator,
which handles both continuous intraday drift and discrete overnight jumps.
"""

from typing import Optional
import numpy as np
import pandas as pd

from causal_volatility.estimators.base import BaseVolatilityEstimator


class YangZhangEstimator(BaseVolatilityEstimator):
    """Yang-Zhang (2000) realized volatility estimator.

    Combines overnight jump variance, open-to-close continuous variance,
    and Rogers-Satchell intraday variance with an optimal weighting coefficient k.

    Formulation:
    $$\\text{var}_o = \\text{Var}\\left(\\ln\\frac{O_t}{C_{t-1}}\\right)$$
    $$\\text{var}_c = \\text{Var}\\left(\\ln\\frac{C_t}{O_t}\\right)$$
    $$\\text{var}_{rs} = \\text{Mean}\\left(\\ln\\frac{H_t}{C_t}\\ln\\frac{H_t}{O_t} + \\ln\\frac{L_t}{C_t}\\ln\\frac{L_t}{O_t}\\right)$$
    $$k = \\frac{0.34}{1.34 + \\frac{n + 1}{n - 1}}$$
    $$\\sigma_{YZ}^2 = \\text{var}_o + k \\cdot \\text{var}_c + (1 - k) \\cdot \\text{var}_{rs}$$
    $$\\sigma_{YZ} = \\sqrt{\\max(0, \\sigma_{YZ}^2)}$$
    """

    def __init__(self, default_window: int = 5):
        self.default_window = default_window

    def estimate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        open: Optional[pd.Series] = None,
        open_: Optional[pd.Series] = None,
        window: int = 5,
        **kwargs,
    ) -> pd.Series:
        """Calculate rolling Yang-Zhang volatility."""
        o_series = open if open is not None else open_
        if o_series is None:
            raise ValueError("Must provide 'open' or 'open_' price series.")

        h = high.astype(float)
        l = low.astype(float)
        c = close.astype(float)
        o = o_series.astype(float)

        win = window if window is not None else self.default_window

        log_oc = np.log(o / c.shift(1))
        log_co = np.log(c / o)
        log_hc = np.log(h / c)
        log_ho = np.log(h / o)
        log_lc = np.log(l / c)
        log_lo = np.log(l / o)

        rs = log_hc * log_ho + log_lc * log_lo

        if win > 1:
            k = 0.34 / (1.34 + (win + 1.0) / (win - 1.0))
            var_o = log_oc.rolling(win).var()
            var_c = log_co.rolling(win).var()
            var_rs = rs.rolling(win).mean()
        else:
            k = 0.34 / 2.34
            var_o = log_oc ** 2
            var_c = log_co ** 2
            var_rs = rs

        total_var = var_o + k * var_c + (1.0 - k) * var_rs
        total_var = np.maximum(0.0, total_var)
        vol = pd.Series(np.sqrt(total_var), index=high.index, name="Yang_Zhang_Vol").fillna(0.0)

        return vol
