"""Glosten-Jagannathan-Runkle GARCH (GJR-GARCH) model with threshold asymmetry."""

import warnings
from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model

from causal_volatility.models.base import BaseVolatilityModel


class GJRGARCHModel(BaseVolatilityModel):
    """GJR-GARCH Volatility Model (Glosten, Jagannathan, and Runkle 1993).

    Extends standard GARCH by adding a threshold indicator term for negative innovations
    gamma * I_{t-1} * epsilon_{t-1}^2, modeling asymmetric volatility clustering during market downturns.
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
        """Initialize GJRGARCHModel.

        Parameters
        ----------
        p : int, default 1
            Order of symmetric innovations in variance equation.
        o : int, default 1
            Order of asymmetric (threshold) terms.
        q : int, default 1
            Order of lagged conditional variance terms.
        dist : str, default 'normal'
            Innovation distribution ('normal', 't', 'skewt').
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

    def fit(self, vol_series: pd.Series) -> "GJRGARCHModel":
        """Fit GJR-GARCH model to volatility series.

        Parameters
        ----------
        vol_series : pd.Series
            Stationary volatility series.

        Returns
        -------
        GJRGARCHModel
            Fitted model instance (self).
        """
        if not isinstance(vol_series, pd.Series):
            vol_series = pd.Series(vol_series)

        clean_vol = vol_series.dropna()
        self.index_ = clean_vol.index
        scaled_series = clean_vol * self.scale_factor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # In the arch library, GJR-GARCH is specified as vol='GARCH' with o > 0
            am = arch_model(
                scaled_series,
                mean=self.mean,
                vol="GARCH",
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
