"""Abstract base class contract for volatility residualization models."""

from abc import ABC, abstractmethod
import warnings
from typing import Optional
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


class BaseVolatilityModel(ABC):
    """Abstract Base Class defining the unified Volatility Model contract (R3).

    All volatility models in the causal framework (standard AR-GARCH, Student-t GARCH,
    EGARCH, GJR-GARCH) inherit from this ABC and provide uniform interfaces for fitting,
    extracting standardized innovations (z_t = epsilon_t / sigma_t), recovering conditional
    volatility, and extracting econometric diagnostics.
    """

    def __init__(self, scale_factor: float = 100.0):
        """Initialize BaseVolatilityModel.

        Parameters
        ----------
        scale_factor : float, default 100.0
            Multiplicative scale factor applied to input series before optimization
            to prevent vanishing gradients on small daily volatility scales (~0.01).
        """
        self.scale_factor = scale_factor
        self.fit_result_ = None
        self.index_: Optional[pd.Index] = None
        self.standardized_residuals_: Optional[pd.Series] = None
        self.conditional_volatility_: Optional[pd.Series] = None

    @abstractmethod
    def fit(self, vol_series: pd.Series) -> "BaseVolatilityModel":
        """Fit the volatility model to a stationary volatility series.

        Parameters
        ----------
        vol_series : pd.Series
            Stationary volatility series (e.g. GK_Vol_Diff).

        Returns
        -------
        BaseVolatilityModel
            Fitted model instance (self).
        """
        pass

    @abstractmethod
    def get_standardized_residuals(self) -> pd.Series:
        """Extract standardized innovations (z_t = epsilon_t / sigma_t).

        Returns
        -------
        pd.Series
            Standardized residuals named 'Vol_Innovations' with mean ~ 0 and std ~ 1.
        """
        pass

    @abstractmethod
    def get_conditional_volatility(self) -> pd.Series:
        """Extract conditional volatility series in the original scale.

        Returns
        -------
        pd.Series
            Conditional standard deviation named 'Cond_Vol' (strictly positive).
        """
        pass

    def get_diagnostics(self, lags: int = 10) -> dict:
        """Calculate econometric diagnostics on the standardized residuals.

        Audits whether standardized residuals behave as Gaussian or heavy-tailed white noise
        free of serial correlation (Ljung-Box) and ARCH clustering (Engle LM).

        Parameters
        ----------
        lags : int, default 10
            Number of lags to evaluate.

        Returns
        -------
        dict
            Dictionary containing:
            - 'ljung_box_p': Ljung-Box test p-value
            - 'arch_lm_p': Engle ARCH LM test p-value
            - 'is_white_noise': bool, True if both p-values >= 0.05
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'log_likelihood': Log-likelihood of fitted model
        """
        if self.fit_result_ is None:
            raise RuntimeError("Model must be fitted before calling get_diagnostics().")

        z = self.get_standardized_residuals().dropna()
        if len(z) < lags + 2:
            raise ValueError(
                f"Insufficient residual observations ({len(z)}) to test {lags} lags."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb = acorr_ljungbox(z, lags=[lags], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
            try:
                arch = het_arch(z, nlags=lags)
            except TypeError:
                arch = het_arch(z, maxlag=lags)
            arch_p = float(arch[1])

        aic = float(getattr(self.fit_result_, "aic", np.nan))
        bic = float(getattr(self.fit_result_, "bic", np.nan))
        log_likelihood = float(getattr(self.fit_result_, "loglikelihood", np.nan))

        return {
            "ljung_box_p": lb_p,
            "arch_lm_p": arch_p,
            "is_white_noise": bool(lb_p >= 0.05 and arch_p >= 0.05),
            "aic": aic,
            "bic": bic,
            "log_likelihood": log_likelihood,
        }
