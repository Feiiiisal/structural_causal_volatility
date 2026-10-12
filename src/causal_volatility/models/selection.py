"""Optimal autoregressive lag selection using information criteria."""

import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class OptimalLagSelector:
    """Automated lag order selection for autoregressive mean models.

    Scans autoregressive orders p in [1, max_lag] using statsmodels AutoReg,
    evaluating each model under the Bayesian Information Criterion (BIC) or
    Akaike Information Criterion (AIC) to select the parsimonious lag order
    that filters serial persistence without overfitting.
    """

    def __init__(
        self,
        max_lag: int = 21,
        criterion: str = "bic",
        trend: str = "c",
    ):
        """Initialize OptimalLagSelector.

        Parameters
        ----------
        max_lag : int, default 21
            Maximum autoregressive lag order to evaluate.
        criterion : str, default 'bic'
            Information criterion to minimize ('bic' or 'aic').
        trend : str, default 'c'
            Trend parameter for AutoReg ('c' for constant, 'n' for none, 'ct' for linear trend).
        """
        self.max_lag = max_lag
        self.criterion = criterion.lower()
        if self.criterion not in ["bic", "aic"]:
            raise ValueError(f"Criterion must be 'bic' or 'aic', got '{criterion}'")
        self.trend = trend
        self.best_lag_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.lag_scores_: dict[int, float] = {}

    def select_lag(self, series: Union[pd.Series, np.ndarray]) -> int:
        """Evaluate lags 1 through max_lag and return optimal lag order.

        Parameters
        ----------
        series : pd.Series or np.ndarray
            Stationary volatility series to fit.

        Returns
        -------
        int
            Optimal autoregressive lag order p in [1, max_lag].
        """
        if isinstance(series, pd.Series):
            clean_series = series.dropna().values
        else:
            clean_series = np.asarray(series)
            clean_series = clean_series[np.isfinite(clean_series)]

        n_obs = len(clean_series)
        if n_obs < 10:
            raise ValueError(f"Series too short ({n_obs} observations) for lag selection.")

        # Ensure we do not exceed degree-of-freedom limits
        effective_max = max(1, min(self.max_lag, n_obs // 3))

        best_score = np.inf
        best_lag = 1
        self.lag_scores_ = {}

        for lag in range(1, effective_max + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = AutoReg(clean_series, lags=lag, trend=self.trend).fit()
                    score = float(model.bic if self.criterion == "bic" else model.aic)

                self.lag_scores_[lag] = score
                if score < best_score:
                    best_score = score
                    best_lag = lag
            except Exception:
                # If fitting fails at this lag order, skip
                continue

        self.best_lag_ = best_lag
        self.best_score_ = best_score
        return self.best_lag_

    def __call__(self, series: Union[pd.Series, np.ndarray]) -> int:
        """Call instance directly to select lag."""
        return self.select_lag(series)
