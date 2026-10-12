"""Autoregressive GARCH and Student-t GARCH volatility models."""

import warnings
from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model

from causal_volatility.models.base import BaseVolatilityModel
from causal_volatility.models.selection import OptimalLagSelector


class ARGARCHModel(BaseVolatilityModel):
    """AR(p)-GARCH(1,1) Volatility Residualization Model.

    Fits an Autoregressive mean model combined with a Generalized Autoregressive
    Conditional Heteroskedasticity (GARCH) process under Gaussian, Student-t, or
    Skew-t innovation distributions. Standardizes innovations into z_t = epsilon_t / sigma_t
    for downstream causal DAG modeling.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        ar_lags: Optional[int] = 0,
        dist: str = "normal",
        mean: str = "Constant",
        scale_factor: float = 100.0,
        auto_select_lag: bool = False,
        max_ar_lag: int = 21,
    ):
        """Initialize ARGARCHModel.

        Parameters
        ----------
        p : int, default 1
            Lag order of the symmetric innovations in the GARCH variance equation.
        q : int, default 1
            Lag order of the lagged conditional variance in the GARCH variance equation.
        ar_lags : Optional[int], default 0
            Number of autoregressive lags in the mean equation. If 0 or None,
            constant mean model is utilized unless auto_select_lag is True.
        dist : str, default 'normal'
            Distribution assumption for standardized residuals ('normal', 't', 'skewt').
        mean : str, default 'Constant'
            Mean model specification ('Constant' or 'AR').
        scale_factor : float, default 100.0
            Variance scale multiplier to eliminate optimization flat gradients.
        auto_select_lag : bool, default False
            If True, runs OptimalLagSelector before fitting to identify optimal AR lag.
        max_ar_lag : int, default 21
            Maximum AR lag to scan if auto_select_lag is True.
        """
        super().__init__(scale_factor=scale_factor)
        self.p = p
        self.q = q
        self.ar_lags = ar_lags
        self.dist = dist
        self.mean = mean
        self.auto_select_lag = auto_select_lag
        self.max_ar_lag = max_ar_lag
        self.effective_ar_lag_: Optional[int] = None

    def fit(self, vol_series: pd.Series) -> "ARGARCHModel":
        """Fit the GARCH model to stationary volatility differences.

        Parameters
        ----------
        vol_series : pd.Series
            Stationary volatility series (e.g. GK_Vol_Diff).

        Returns
        -------
        ARGARCHModel
            Fitted model instance (self).
        """
        if not isinstance(vol_series, pd.Series):
            vol_series = pd.Series(vol_series)

        clean_vol = vol_series.dropna()
        self.index_ = clean_vol.index

        # Determine mean model and AR lags
        if self.auto_select_lag:
            selector = OptimalLagSelector(max_lag=self.max_ar_lag)
            opt_lag = selector.select_lag(clean_vol)
            self.effective_ar_lag_ = opt_lag
            mean_type = "AR"
        elif self.ar_lags is not None and self.ar_lags > 0:
            self.effective_ar_lag_ = self.ar_lags
            mean_type = "AR"
        elif self.mean.upper() == "AR":
            self.effective_ar_lag_ = self.ar_lags if self.ar_lags else 1
            mean_type = "AR"
        else:
            self.effective_ar_lag_ = 0
            mean_type = "Constant"

        # Pre-scale series by scale_factor (default 100.0)
        scaled_series = clean_vol * self.scale_factor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mean_type == "AR":
                am = arch_model(
                    scaled_series,
                    mean="AR",
                    lags=self.effective_ar_lag_,
                    vol="GARCH",
                    p=self.p,
                    q=self.q,
                    dist=self.dist,
                )
            else:
                am = arch_model(
                    scaled_series,
                    mean="Constant",
                    vol="GARCH",
                    p=self.p,
                    q=self.q,
                    dist=self.dist,
                )

            self.fit_result_ = am.fit(disp="off", show_warning=False)

        # Extract standardized residuals z_t = epsilon_t / sigma_t
        z = self.fit_result_.std_resid.dropna()
        self.standardized_residuals_ = pd.Series(
            z.values,
            index=self.index_[-len(z):],
            name="Vol_Innovations",
        )

        # Extract conditional volatility restored to original scale
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


class StudentTGARCHModel(ARGARCHModel):
    """Heavy-tailed Student-t AR(p)-GARCH(1,1) Volatility Model.

    Accommodates excess kurtosis and fat-tailed shocks in financial volatility by estimating
    the degrees-of-freedom parameter nu alongside conditional variance parameters.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        ar_lags: Optional[int] = 0,
        dist: str = "t",
        mean: str = "Constant",
        scale_factor: float = 100.0,
        auto_select_lag: bool = False,
        max_ar_lag: int = 21,
    ):
        super().__init__(
            p=p,
            q=q,
            ar_lags=ar_lags,
            dist=dist,
            mean=mean,
            scale_factor=scale_factor,
            auto_select_lag=auto_select_lag,
            max_ar_lag=max_ar_lag,
        )
