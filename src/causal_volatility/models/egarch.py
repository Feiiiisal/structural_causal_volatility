"""Exponential GARCH (EGARCH) model capturing asymmetric leverage effects."""

import warnings
from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model

from causal_volatility.models.base import BaseVolatilityModel


class EGARCHModel(BaseVolatilityModel):
    """Exponential GARCH (EGARCH) Model (Nelson 1991).

    Models the logarithm of conditional variance, allowing asymmetric volatility responses
    to positive and negative innovations (leverage effect) without non-negativity parameter
    constraints.
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "Constant",
        scale_factor: float = 100.0,
    ):
        """Initialize EGARCHModel.

        Parameters
        ----------
        p : int, default 1
            Lag order of the symmetric innovations.
        o : int, default 1
            Lag order of the asymmetric leverage terms.
        q : int, default 1
            Lag order of the lagged log conditional variance.
        dist : str, default 'normal'
            Error distribution ('normal', 't', 'skewt').
        mean : str, default 'Constant'
            Mean model specification.
        scale_factor : float, default 100.0
            Variance scale multiplier.
        """
        super().__init__(scale_factor=scale_factor)
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.mean = mean

    def fit(self, vol_series: pd.Series) -> "EGARCHModel":
        """Fit EGARCH model to volatility series.

        Parameters
        ----------
        vol_series : pd.Series
            Stationary volatility series.

        Returns
        -------
        EGARCHModel
            Fitted model instance (self).
        """
        if not isinstance(vol_series, pd.Series):
            vol_series = pd.Series(vol_series)

        clean_vol = vol_series.dropna()
        self.index_ = clean_vol.index
        scaled_series = clean_vol * self.scale_factor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                scaled_series,
                mean=self.mean,
                vol="EGARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
            )
            self.fit_result_ = am.fit(disp="off", show_warning=False)

        # Standardized residuals
        z = self.fit_result_.std_resid.dropna()
        self.standardized_residuals_ = pd.Series(
            z.values,
            index=self.index_[-len(z):],
            name="Vol_Innovations",
        )

        # Conditional volatility restored to original scale
        sig = self.fit_result_.conditional_volatility.dropna() / self.scale_factor
        self.conditional_volatility_ = pd.Series(
            sig.values,
            index=self.index_[-len(sig):],
            name="Cond_Vol",
        )

        return self

    def get_standardized_residuals(self) -> pd.Series:
        """Get standardized innovations series."""
        if self.standardized_residuals_ is None:
            raise RuntimeError("Model must be fitted before extracting residuals.")
        return self.standardized_residuals_

    def get_conditional_volatility(self) -> pd.Series:
        """Get conditional volatility series in original scale."""
        if self.conditional_volatility_ is None:
            raise RuntimeError("Model must be fitted before extracting conditional volatility.")
        return self.conditional_volatility_
