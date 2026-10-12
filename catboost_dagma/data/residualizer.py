"""
Fama-French 3-Factor Residualization Engine.
Addresses Capstone Instructor Feedback (a):
Strips out shared systemic market, size, and value beta from returns before causal discovery,
ensuring that discovered DAGMA edges represent genuine idiosyncratic causal channels.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost_dagma.config import FF_FACTORS, FF_RISK_FREE


class FamaFrenchResidualizer:
    """
    Performs multi-factor OLS regression to eliminate shared systemic beta:
    R_{i,t} - R_{f,t} = alpha_i + beta_{i,MKT}*(R_{M,t} - R_{f,t}) + beta_{i,SMB}*SMB_t + beta_{i,HML}*HML_t + epsilon_{i,t}
    
    The resulting residual panel E = [epsilon_1, ..., epsilon_N] contains purely idiosyncratic
    innovations orthogonal to systemic market drivers.
    """

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

    def fit_transform(
        self,
        returns_df: pd.DataFrame,
        factors_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Static/in-sample full-window residualization across all assets.
        Returns DataFrame of residuals with the same index and column names as returns_df.
        """
        aligned_dates = returns_df.index.intersection(factors_df.index).sort_values()
        y_panel = returns_df.loc[aligned_dates].copy()
        f_panel = factors_df.loc[aligned_dates].copy()

        # Risk-free adjustment
        if self.risk_free_col in f_panel.columns:
            rf = f_panel[self.risk_free_col]
            excess_returns = y_panel.sub(rf, axis=0)
        else:
            excess_returns = y_panel

        # Factor Design Matrix X
        X = f_panel[self.factors].copy()
        if self.add_constant:
            X = sm.add_constant(X)

        residuals_dict = {}
        summaries = {}

        for col in y_panel.columns:
            y = excess_returns[col].values
            # Fit OLS
            model = sm.OLS(y, X.values).fit()
            resid = model.resid
            residuals_dict[col] = resid

            # Store factor exposures and R^2 for diagnostic reporting
            param_names = ["const"] + self.factors if self.add_constant else self.factors
            exposures = dict(zip(param_names, model.params))
            exposures["R2"] = float(model.rsquared)
            exposures["Resid_Vol"] = float(np.std(resid) * np.sqrt(252))
            summaries[col] = exposures

        self.regression_summaries_ = summaries
        residuals_df = pd.DataFrame(residuals_dict, index=aligned_dates)
        return residuals_df

    def transform_rolling_window(
        self,
        returns_window: np.ndarray,
        factors_window: np.ndarray,
    ) -> np.ndarray:
        """
        High-performance vectorized residualization for rolling-window DAGMA fits.
        Fits OLS on factors_window and projects out shared factor variation.
        
        Args:
            returns_window: Array of shape (T, N)
            factors_window: Array of shape (T, K) where K is number of factors
        Returns:
            residuals_window: Array of shape (T, N)
        """
        T, N = returns_window.shape
        T_f, K = factors_window.shape
        assert T == T_f, f"Mismatched window lengths: returns={T}, factors={T_f}"

        # Design matrix X: add intercept column of ones
        ones = np.ones((T, 1), dtype=factors_window.dtype)
        X = np.hstack([ones, factors_window]) # Shape (T, K+1)

        # Vectorized OLS: Beta = (X^T X)^{-1} X^T Y
        # Compute projection matrix P = X (X^T X)^{-1} X^T
        try:
            XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(K + 1))
            Beta = XtX_inv @ X.T @ returns_window # Shape (K+1, N)
            residuals = returns_window - X @ Beta # Shape (T, N)
            return residuals
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            Beta = np.linalg.pinv(X) @ returns_window
            return returns_window - X @ Beta

    def get_factor_exposure_report(self) -> pd.DataFrame:
        """
        Returns DataFrame detailing estimated factor betas, alpha, and R^2 for all assets.
        """
        if not self.regression_summaries_:
            raise RuntimeError("Must call fit_transform before generating exposure report.")
        return pd.DataFrame.from_dict(self.regression_summaries_, orient="index")
