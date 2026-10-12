"""Empirical comparison of directed-network and dependence-network features.

The module deliberately avoids using low pairwise correlation as proof of
mathematical orthogonality or causal independence.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


class EmpiricalOrthogonalityAuditor:
    """Backwards-compatible class name for the empirical non-redundancy audit."""

    def __init__(self, significance_alpha: float = 0.05):
        self.significance_alpha = significance_alpha

    def audit_topological_correlations(self, df_features: pd.DataFrame) -> pd.DataFrame:
        pairs = [
            ("causal_in_deg", "corr_deg", "Causal In-Degree vs Correlation Degree"),
            ("causal_out_deg", "corr_deg", "Causal Out-Degree vs Correlation Degree"),
            ("causal_eigen", "corr_eigen", "Causal Eigen vs Correlation Eigen"),
            ("causal_in_deg", "precision_deg", "Causal In-Degree vs Precision Degree"),
            ("causal_out_deg", "precision_deg", "Causal Out-Degree vs Precision Degree"),
            ("causal_eigen", "precision_eigen", "Causal Eigen vs Precision Eigen"),
            ("precision_deg", "corr_deg", "Precision Degree vs Correlation Degree"),
            ("precision_eigen", "corr_eigen", "Precision Eigen vs Correlation Eigen"),
        ]
        rows = []
        for a, b, label in pairs:
            if a not in df_features or b not in df_features:
                continue
            valid = df_features[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 10 and valid[a].std() > 1e-10 and valid[b].std() > 1e-10:
                rp, pp = pearsonr(valid[a], valid[b])
                rs, ps = spearmanr(valid[a], valid[b])
            else:
                rp, pp, rs, ps = np.nan, np.nan, np.nan, np.nan

            abs_r = abs(rp) if np.isfinite(rp) else np.nan
            if np.isnan(abs_r):
                verdict = "Insufficient variation"
            elif abs_r < 0.35:
                verdict = "Low linear association"
            elif abs_r < 0.70:
                verdict = "Moderate linear association"
            else:
                verdict = "High linear association"

            rows.append({
                "Comparison": label,
                "Feature_A": a,
                "Feature_B": b,
                "Pearson_r": float(rp) if np.isfinite(rp) else np.nan,
                "Pearson_p": float(pp) if np.isfinite(pp) else np.nan,
                "Spearman_rho": float(rs) if np.isfinite(rs) else np.nan,
                "Spearman_p": float(ps) if np.isfinite(ps) else np.nan,
                "Empirical_Verdict": verdict,
            })
        return pd.DataFrame(rows)

    def audit_variance_inflation_factors(self, df_features: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        cols = feature_cols or [
            c for c in [
                "corr_deg", "corr_eigen", "precision_deg", "precision_eigen",
                "causal_in_deg", "causal_out_deg", "causal_eigen", "causal_drift",
            ] if c in df_features.columns
        ]
        X = df_features[cols].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
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
            rows.append({"Feature": col, "VIF": vif, "Multicollinearity_Status": status})
        return pd.DataFrame(rows)

    def compute_directed_asymmetry_index(self, W: np.ndarray) -> float:
        """Describe structural asymmetry; this is not a test or proof of causality."""
        W = np.asarray(W, dtype=float)
        denom = np.linalg.norm(W, "fro")
        if denom < 1e-12:
            return 0.0
        return float(np.linalg.norm(W - W.T, "fro") / denom)
