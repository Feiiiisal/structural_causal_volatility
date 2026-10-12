"""
Structural Causal Discovery Engine (Requirement R2 Fix).

Maps out the time-lagged structural causal graph driving volatility innovations
via bivariate Granger conditional independence regressions. Eliminates statsmodels
SingularMatrixWarning and rank deficiency by removing collinear self-lag duplicates.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm


def execute_structural_causal_discovery(
    df: pd.DataFrame,
    target_var: str = "Vol_Innovations",
    tau_max: int = 5,
    alpha_thresh: float = 0.05,
    include_self_lags: bool = False,
) -> Dict[str, Any]:
    """
    Execute time-series structural causal discovery via conditional Granger regressions.

    For each candidate driver variable and time lag tau in [1, tau_max], evaluates
    whether the past shock X_{t-tau} Granger-causes target innovations Y_t after
    conditioning on target autoregressive memory [Y_{t-1}, ..., Y_{t-tau_max}].

    R2 Collinearity Fix:
        When source_name == target_var, Column 1 (Y_{t-tau}) is duplicated in
        Column tau + 1 by the conditioning loop over lags 1..tau_max.
        - If include_self_lags is False (default): Skips regressions where
          source_name == target_var, as self-autoregressive memory is already
          residualized by the preceding volatility model (e.g. AR-GARCH).
        - If include_self_lags is True: Omits lag == tau from the conditioning set,
          preserving full column rank and preventing identical columns.

    Args:
        df: Input DataFrame containing target_var and candidate macroeconomic drivers.
        target_var: Column name of target residualized innovations (default: 'Vol_Innovations').
        tau_max: Maximum lag order to test (default: 5).
        alpha_thresh: Statistical significance threshold alpha (default: 0.05).
        include_self_lags: Whether to test target self-lags (default: False).

    Returns:
        Dictionary containing:
            - 'p_matrix': (n_nodes, n_nodes, tau_max + 1) ndarray of p-values
            - 'val_matrix': (n_nodes, n_nodes, tau_max + 1) ndarray of path coefficients (beta)
            - 'var_names': list of variable names in df
            - 'clean_df': DataFrame after dropna
            - 'tau_max': maximum lag order evaluated
            - 'target_var': name of target variable
            - 'alpha_thresh': alpha threshold used
    """
    clean_causal_df = df.dropna().copy()
    var_names = list(clean_causal_df.columns)

    if target_var not in var_names:
        raise ValueError(f"Target variable '{target_var}' not found in columns: {var_names}")

    target_idx = var_names.index(target_var)
    n_nodes = len(var_names)

    # Initialize tensors: p-values default to 1.0 (insignificant), coefficients default to 0.0
    p_matrix = np.ones((n_nodes, n_nodes, tau_max + 1), dtype=float)
    val_matrix = np.zeros((n_nodes, n_nodes, tau_max + 1), dtype=float)

    y_series = clean_causal_df[target_var].values

    for source_idx, source_name in enumerate(var_names):
        # R2 Fix: Skip self-regression when include_self_lags is False
        if source_name == target_var and not include_self_lags:
            continue

        for tau in range(1, tau_max + 1):
            x_shifted = clean_causal_df[source_name].shift(tau).values

            # Assemble design matrix: candidate shock followed by target history lags
            X_matrix = [x_shifted]
            for lag in range(1, tau_max + 1):
                # R2 Fix: If evaluating self-lags, omit tested lag to prevent collinearity
                if source_name == target_var and lag == tau:
                    continue
                X_matrix.append(clean_causal_df[target_var].shift(lag).values)

            X_arr = np.column_stack(X_matrix)

            valid_mask = np.all(np.isfinite(X_arr), axis=1) & np.isfinite(y_series)
            n_valid = int(np.sum(valid_mask))

            # Minimum sample size requirement: must exceed regressor dimension + constant
            min_required = X_arr.shape[1] + 2
            if n_valid > 50 or (n_valid > min_required and n_valid >= 20):
                y_cropped = y_series[valid_mask]
                X_cropped = X_arr[valid_mask]
                X_cropped_with_c = sm.add_constant(X_cropped, has_constant="add")

                # Defensively assert matrix rank to prevent any SingularMatrixWarning
                cols = X_cropped_with_c.shape[1]
                matrix_rank = np.linalg.matrix_rank(X_cropped_with_c)

                if matrix_rank == cols:
                    model = sm.OLS(y_cropped, X_cropped_with_c).fit()
                    p_matrix[source_idx, target_idx, tau] = model.pvalues[1]
                    val_matrix[source_idx, target_idx, tau] = model.params[1]
                else:
                    # If degenerate data causes rank deficiency, record non-significant edge
                    p_matrix[source_idx, target_idx, tau] = 1.0
                    val_matrix[source_idx, target_idx, tau] = 0.0

    return {
        "p_matrix": p_matrix,
        "val_matrix": val_matrix,
        "var_names": var_names,
        "clean_df": clean_causal_df,
        "tau_max": tau_max,
        "target_var": target_var,
        "alpha_thresh": alpha_thresh,
    }


class StructuralCausalDiscovery:
    """
    Object-oriented interface for Structural Causal Discovery.

    Encapsulates configuration, fitting, and access to causal tensors.
    """

    def __init__(
        self,
        target_var: str = "Vol_Innovations",
        tau_max: int = 5,
        alpha_thresh: float = 0.05,
        include_self_lags: bool = False,
    ):
        self.target_var = target_var
        self.tau_max = tau_max
        self.alpha_thresh = alpha_thresh
        self.include_self_lags = include_self_lags
        self.results_: Optional[Dict[str, Any]] = None

    def fit(self, df: pd.DataFrame) -> "StructuralCausalDiscovery":
        """Fit structural causal discovery model to input data."""
        self.results_ = execute_structural_causal_discovery(
            df=df,
            target_var=self.target_var,
            tau_max=self.tau_max,
            alpha_thresh=self.alpha_thresh,
            include_self_lags=self.include_self_lags,
        )
        return self

    @property
    def p_matrix(self) -> np.ndarray:
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.results_["p_matrix"]

    @property
    def val_matrix(self) -> np.ndarray:
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.results_["val_matrix"]

    @property
    def var_names(self) -> List[str]:
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.results_["var_names"]

    def to_dict(self) -> Dict[str, Any]:
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.results_
