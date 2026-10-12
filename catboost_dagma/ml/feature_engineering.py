"""
Feature Engineering Engine for CatBoost + DAGMA Causal Fusion.
Constructs the tabular dataset by fusing:
1. Baseline momentum & volatility features
2. Standard correlation graph centralities
3. Graphical LASSO precision matrix centralities (conditional independence benchmark)
4. Non-linear DAGMA causal topologies (in-degree, out-degree, eigenvector centrality, drift, break flags)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from catboost_dagma.dagma.solver import get_eigenvector_centrality


class FusionFeatureEngineer:
    """
    Transforms rolling window causal graphs, precision matrices, and asset return series
    into an aligned tabular machine learning dataset suitable for CatBoost training.
    """

    def __init__(
        self,
        forward_horizon: int = 5,
        corr_threshold: float = 0.40,
    ):
        self.forward_horizon = forward_horizon
        self.corr_threshold = corr_threshold

    def build_ml_dataset(
        self,
        returns_df: pd.DataFrame,
        rolling_dagma_results: List[Dict[str, Any]],
        rolling_precision_results: Optional[List[Dict[str, Any]]] = None,
        drift_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Iterates across rolling windows and assets to assemble a structured dataframe.
        """
        assets = returns_df.columns.tolist()
        N = len(assets)
        dates = returns_df.index

        rows = []
        for i, res in enumerate(rolling_dagma_results):
            end_idx = res["end_idx"]
            date = res["date"]

            # Guard against edge of dataset for forward returns
            if end_idx + self.forward_horizon > len(returns_df):
                continue

            # 1. Forward target: cumulative forward return over horizon h
            forward_slice = returns_df.iloc[end_idx : end_idx + self.forward_horizon]
            forward_returns = forward_slice.sum(axis=0).values
            forward_vol = forward_slice.std(axis=0).values * np.sqrt(252)

            # 2. Historical baseline features (past 20 and 60 days)
            hist_60 = returns_df.iloc[max(0, end_idx - 60) : end_idx]
            hist_20 = returns_df.iloc[max(0, end_idx - 20) : end_idx]
            mom_20 = hist_20.sum(axis=0).values
            vol_20 = hist_20.std(axis=0).values * np.sqrt(252)
            vol_60 = hist_60.std(axis=0).values * np.sqrt(252)

            # 3. Correlation graph centralities
            corr_mat = np.corrcoef(hist_60.values.T)
            np.fill_diagonal(corr_mat, 0.0)
            in_degree_corr = (np.abs(corr_mat) > self.corr_threshold).sum(axis=0).astype(float)
            eigen_corr = get_eigenvector_centrality(corr_mat, threshold=self.corr_threshold)

            # 4. Precision matrix centralities
            if rolling_precision_results and i < len(rolling_precision_results):
                p_res = rolling_precision_results[i]
                prec_deg = p_res["precision_degree"]
                prec_eigen = p_res["precision_eigen"]
            else:
                prec_deg = np.zeros(N)
                prec_eigen = np.zeros(N)

            # 5. DAGMA Causal Topologies
            causal_in = res["in_degree_causal"]
            causal_out = res["out_degree_causal"]
            causal_eigen = res["eigen_causal"]
            nu = res.get("nu", 3.0)

            # 6. Structural Break and Drift Features
            if drift_df is not None and date in drift_df.index:
                drift_val = float(drift_df.loc[date, "causal_drift"])
                is_break = int(drift_df.loc[date, "is_structural_break"])
                days_since_break = int(drift_df.loc[date, "days_since_last_break"])
            else:
                drift_val = 0.0
                is_break = 0
                days_since_break = 999

            for asset_idx, asset_name in enumerate(assets):
                rows.append({
                    "window_idx": i,
                    "date": date,
                    "asset": asset_name,
                    # Prediction Targets
                    "forward_return": float(forward_returns[asset_idx]),
                    "forward_volatility": float(forward_vol[asset_idx]),
                    # Baseline Asset Features
                    "momentum_20d": float(mom_20[asset_idx]),
                    "volatility_20d": float(vol_20[asset_idx]),
                    "volatility_60d": float(vol_60[asset_idx]),
                    # Correlation Graph Centralities
                    "corr_deg": float(in_degree_corr[asset_idx]),
                    "corr_eigen": float(eigen_corr[asset_idx]),
                    # Precision Matrix Centralities (Instructor Feedback b)
                    "precision_deg": float(prec_deg[asset_idx]),
                    "precision_eigen": float(prec_eigen[asset_idx]),
                    # DAGMA Non-Linear Causal Centralities (Instructor Feedback a & b)
                    "causal_in_deg": float(causal_in[asset_idx]),
                    "causal_out_deg": float(causal_out[asset_idx]),
                    "causal_eigen": float(causal_eigen[asset_idx]),
                    "causal_drift": float(drift_val),
                    "is_structural_break": int(is_break),
                    "days_since_last_break": int(days_since_break),
                    "dagma_nu": float(nu),
                })

        df_ml = pd.DataFrame(rows)
        return df_ml
