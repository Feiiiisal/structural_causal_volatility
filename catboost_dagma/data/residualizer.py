"""Fama-French residuals."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm

from catboost_dagma.config import FF_FACTORS, FF_RISK_FREE


class FamaFrenchResidualizer:
    def __init__(
        self,
        factors: Optional[List[str]] = None,
        risk_free_col: str = FF_RISK_FREE,
        add_constant: bool = True,
    ):
        self.factors = factors or FF_FACTORS
        self.risk_free_col = risk_free_col
        self.add_constant = add_constant
        self.regression_summaries_: Dict[str, Dict[str, float]] = {}

    def _prepare(self, returns: pd.DataFrame, factors: pd.DataFrame):
        dates = returns.index.intersection(factors.index).sort_values()
        y = returns.loc[dates].astype(float).copy()
        f = factors.loc[dates].astype(float).copy()

        if self.risk_free_col in f:
            y = y.sub(f[self.risk_free_col], axis=0)

        X = f[self.factors].copy()
        if self.add_constant:
            X = sm.add_constant(X, has_constant="add")

        return y, X

    def fit_transform(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        y, X = self._prepare(returns, factors)
        residuals, summary = {}, {}

        for asset in y:
            fit = sm.OLS(y[asset].values, X.values).fit()
            residuals[asset] = fit.resid

            names = ["const"] + self.factors if self.add_constant else self.factors
            row = dict(zip(names, fit.params))
            row["R2"] = float(fit.rsquared)
            row["Resid_Vol"] = float(np.std(fit.resid, ddof=1) * np.sqrt(252))
            summary[asset] = row

        self.regression_summaries_ = summary
        return pd.DataFrame(residuals, index=y.index)

    def fit_transform_walk_forward(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        min_train_size: int = 252,
        refit_every: int = 21,
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        y, X = self._prepare(returns, factors)
        if len(y) <= min_train_size:
            raise ValueError(f"Need more than {min_train_size} rows.")

        Y, Xv = y.values, X.values
        residuals = np.full_like(Y, np.nan, dtype=float)

        beta = np.linalg.lstsq(Xv[:min_train_size], Y[:min_train_size], rcond=None)[0]
        residuals[:min_train_size] = Y[:min_train_size] - Xv[:min_train_size] @ beta
        self._save_summary(y, beta, residuals, min_train_size)

        for start in range(min_train_size, len(y), refit_every):
            end = min(start + refit_every, len(y))
            first = max(0, start - lookback) if lookback else 0
            beta = np.linalg.lstsq(Xv[first:start], Y[first:start], rcond=None)[0]
            residuals[start:end] = Y[start:end] - Xv[start:end] @ beta

        return pd.DataFrame(residuals, index=y.index, columns=y.columns)

    def _save_summary(self, y, beta, residuals, n):
        names = ["const"] + self.factors if self.add_constant else self.factors
        fitted = y.values[:n] - residuals[:n]
        ss_res = np.square(residuals[:n]).sum(axis=0)
        ss_tot = np.square(y.values[:n] - y.values[:n].mean(axis=0)).sum(axis=0)

        summary = {}
        for j, asset in enumerate(y.columns):
            row = {name: float(beta[k, j]) for k, name in enumerate(names)}
            row["R2"] = float(1 - ss_res[j] / ss_tot[j]) if ss_tot[j] else 0.0
            row["Resid_Vol"] = float(np.std(residuals[:n, j], ddof=1) * np.sqrt(252))
            summary[asset] = row

        self.regression_summaries_ = summary

    def transform_rolling_window(self, returns_window, factors_window):
        if len(returns_window) != len(factors_window):
            raise ValueError("Returns and factors must have the same number of rows.")

        X = np.column_stack([np.ones(len(factors_window)), factors_window])
        beta = np.linalg.lstsq(X, returns_window, rcond=None)[0]
        return returns_window - X @ beta

    def get_factor_exposure_report(self) -> pd.DataFrame:
        if not self.regression_summaries_:
            raise RuntimeError("Fit the residualizer first.")
        return pd.DataFrame.from_dict(self.regression_summaries_, orient="index")
