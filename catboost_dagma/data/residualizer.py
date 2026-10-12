"""Fama-French residualization utilities.

Residualization reduces measured exposure to included systematic factors. It does
not, by itself, prove that remaining relationships are causal.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost_dagma.config import FF_FACTORS, FF_RISK_FREE


class FamaFrenchResidualizer:
    def __init__(self, factors: Optional[List[str]] = None, risk_free_col: str = FF_RISK_FREE, add_constant: bool = True):
        self.factors = factors or FF_FACTORS
        self.risk_free_col = risk_free_col
        self.add_constant = add_constant
        self.regression_summaries_: Dict[str, Dict[str, float]] = {}

    def _aligned_excess_and_design(self, returns_df: pd.DataFrame, factors_df: pd.DataFrame):
        dates = returns_df.index.intersection(factors_df.index).sort_values()
        y_panel = returns_df.loc[dates].astype(float).copy()
        f_panel = factors_df.loc[dates].astype(float).copy()
        if self.risk_free_col in f_panel.columns:
            y_panel = y_panel.sub(f_panel[self.risk_free_col], axis=0)
        X = f_panel[self.factors].copy()
        if self.add_constant:
            X = sm.add_constant(X, has_constant="add")
        return y_panel, X

    def fit_transform(self, returns_df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
        """In-sample residualization for diagnostics only.

        For out-of-sample model evaluation use :meth:`fit_transform_walk_forward`.
        """
        y_panel, X = self._aligned_excess_and_design(returns_df, factors_df)
        residuals = {}
        summaries = {}
        for col in y_panel.columns:
            model = sm.OLS(y_panel[col].values, X.values).fit()
            residuals[col] = model.resid
            names = ["const"] + self.factors if self.add_constant else self.factors
            info = dict(zip(names, model.params))
            info["R2"] = float(model.rsquared)
            info["Resid_Vol"] = float(np.std(model.resid, ddof=1) * np.sqrt(252))
            summaries[col] = info
        self.regression_summaries_ = summaries
        return pd.DataFrame(residuals, index=y_panel.index)

    def fit_transform_walk_forward(
        self,
        returns_df: pd.DataFrame,
        factors_df: pd.DataFrame,
        min_train_size: int = 252,
        refit_every: int = 21,
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """Chronological factor residualization without future information.

        The initial training sample is residualized in-sample because it is entirely
        historical by the first forecast date. Thereafter factor betas are estimated
        only on observations available before each prediction block.
        """
        y_panel, X = self._aligned_excess_and_design(returns_df, factors_df)
        n = len(y_panel)
        if n <= min_train_size:
            raise ValueError(f"Need more than min_train_size={min_train_size} observations; got {n}.")

        Y = y_panel.values
        Xv = X.values
        resid = np.full_like(Y, np.nan, dtype=float)
        columns = list(y_panel.columns)

        def _beta(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
            # lstsq is numerically safer than explicitly inverting X'X.
            return np.linalg.lstsq(x_train, y_train, rcond=None)[0]

        # Initial historical training block.
        b0 = _beta(Xv[:min_train_size], Y[:min_train_size])
        resid[:min_train_size] = Y[:min_train_size] - Xv[:min_train_size] @ b0

        # Store initial-window diagnostics for reporting.
        pred0 = Xv[:min_train_size] @ b0
        ss_res = ((Y[:min_train_size] - pred0) ** 2).sum(axis=0)
        ss_tot = ((Y[:min_train_size] - Y[:min_train_size].mean(axis=0)) ** 2).sum(axis=0)
        names = ["const"] + self.factors if self.add_constant else self.factors
        summaries = {}
        for j, col in enumerate(columns):
            info = {name: float(b0[k, j]) for k, name in enumerate(names)}
            info["R2"] = float(1.0 - ss_res[j] / ss_tot[j]) if ss_tot[j] > 0 else 0.0
            info["Resid_Vol"] = float(np.std(resid[:min_train_size, j], ddof=1) * np.sqrt(252))
            summaries[col] = info
        self.regression_summaries_ = summaries

        for start in range(min_train_size, n, refit_every):
            end = min(start + refit_every, n)
            train_start = max(0, start - lookback) if lookback else 0
            beta = _beta(Xv[train_start:start], Y[train_start:start])
            resid[start:end] = Y[start:end] - Xv[start:end] @ beta

        return pd.DataFrame(resid, index=y_panel.index, columns=columns)

    def transform_rolling_window(self, returns_window: np.ndarray, factors_window: np.ndarray) -> np.ndarray:
        if returns_window.shape[0] != factors_window.shape[0]:
            raise ValueError("Returns and factors must have the same number of rows.")
        X = np.column_stack([np.ones(len(factors_window)), factors_window])
        beta = np.linalg.lstsq(X, returns_window, rcond=None)[0]
        return returns_window - X @ beta

    def get_factor_exposure_report(self) -> pd.DataFrame:
        if not self.regression_summaries_:
            raise RuntimeError("Call fit_transform or fit_transform_walk_forward first.")
        return pd.DataFrame.from_dict(self.regression_summaries_, orient="index")
