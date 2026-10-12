"""Compare network features."""
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


class NetworkFeatureAudit:
    def audit_topological_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = [
            ("causal_in_deg", "corr_deg", "Causal in-degree vs correlation degree"),
            ("causal_out_deg", "corr_deg", "Causal out-degree vs correlation degree"),
            ("causal_eigen", "corr_eigen", "Causal eigen vs correlation eigen"),
            ("causal_in_deg", "precision_deg", "Causal in-degree vs precision degree"),
            ("causal_out_deg", "precision_deg", "Causal out-degree vs precision degree"),
            ("causal_eigen", "precision_eigen", "Causal eigen vs precision eigen"),
            ("precision_deg", "corr_deg", "Precision degree vs correlation degree"),
            ("precision_eigen", "corr_eigen", "Precision eigen vs correlation eigen"),
        ]

        rows = []
        for left, right, label in pairs:
            if left not in df or right not in df:
                continue

            values = df[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) <= 10 or values[left].std() <= 1e-10 or values[right].std() <= 1e-10:
                pearson = spearman = p_pearson = p_spearman = np.nan
            else:
                pearson, p_pearson = pearsonr(values[left], values[right])
                spearman, p_spearman = spearmanr(values[left], values[right])

            rows.append({
                "Comparison": label,
                "Feature_A": left,
                "Feature_B": right,
                "Pearson_r": pearson,
                "Pearson_p": p_pearson,
                "Spearman_rho": spearman,
                "Spearman_p": p_spearman,
                "Empirical_Verdict": self._label(pearson),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _label(value: float) -> str:
        if not np.isfinite(value):
            return "Insufficient variation"
        value = abs(value)
        if value < 0.35:
            return "Low association"
        if value < 0.70:
            return "Moderate association"
        return "High association"

    def audit_variance_inflation_factors(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        cols = feature_cols or [
            col for col in [
                "corr_deg", "corr_eigen",
                "precision_deg", "precision_eigen",
                "causal_in_deg", "causal_out_deg",
                "causal_eigen", "causal_drift",
            ] if col in df
        ]

        X = df[cols].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if X.empty:
            return pd.DataFrame(columns=["Feature", "VIF", "Multicollinearity_Status"])

        X = X.copy()
        X["const"] = 1.0
        rows = []

        for i, col in enumerate(cols):
            try:
                vif = float(variance_inflation_factor(X.values, i))
            except Exception:
                vif = np.nan

            if not np.isfinite(vif):
                status = "Undefined"
            elif vif < 5:
                status = "Low"
            elif vif < 10:
                status = "Moderate"
            else:
                status = "High"

            rows.append({
                "Feature": col,
                "VIF": vif,
                "Multicollinearity_Status": status,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def compute_directed_asymmetry_index(W: np.ndarray) -> float:
        scale = np.linalg.norm(W, "fro")
        if scale < 1e-12:
            return 0.0
        return float(np.linalg.norm(W - W.T, "fro") / scale)


EmpiricalOrthogonalityAuditor = NetworkFeatureAudit
