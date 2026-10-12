"""Build the CatBoost dataset."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from catboost_dagma.dagma.solver import get_eigenvector_centrality


class FusionFeatureEngineer:
    def __init__(self, forward_horizon: int = 5, corr_threshold: float = 0.40):
        self.forward_horizon = forward_horizon
        self.corr_threshold = corr_threshold

    def build_ml_dataset(
        self,
        returns_df: pd.DataFrame,
        rolling_dagma_results: List[Dict[str, Any]],
        rolling_precision_results: Optional[List[Dict[str, Any]]] = None,
        drift_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        assets = returns_df.columns.tolist()
        rows = []

        for i, result in enumerate(rolling_dagma_results):
            end = int(result["end_idx"])
            date = pd.Timestamp(result["date"])

            if end + self.forward_horizon > len(returns_df):
                continue

            future = returns_df.iloc[end:end + self.forward_horizon]
            forward_return = future.sum().values
            forward_vol = np.sqrt(
                (252.0 / self.forward_horizon) * np.square(future.values).sum(axis=0)
            )

            hist20 = returns_df.iloc[max(0, end - 20):end]
            hist60 = returns_df.iloc[max(0, end - 60):end]

            momentum = hist20.sum().values
            vol20 = hist20.std(ddof=1).values * np.sqrt(252)
            vol60 = hist60.std(ddof=1).values * np.sqrt(252)

            corr = np.corrcoef(hist60.values.T)
            corr = np.nan_to_num(corr)
            np.fill_diagonal(corr, 0.0)
            corr_degree = (np.abs(corr) > self.corr_threshold).sum(axis=0)
            corr_eigen = get_eigenvector_centrality(corr, self.corr_threshold)

            precision_degree = np.zeros(len(assets))
            precision_eigen = np.zeros(len(assets))
            if rolling_precision_results and i < len(rolling_precision_results):
                precision_degree = rolling_precision_results[i]["precision_degree"]
                precision_eigen = rolling_precision_results[i]["precision_eigen"]

            drift = self._drift_row(drift_df, date)

            for j, asset in enumerate(assets):
                rows.append({
                    "window_idx": i,
                    "date": date,
                    "asset": asset,
                    "forward_return": float(forward_return[j]),
                    "forward_volatility": float(forward_vol[j]),
                    "momentum_20d": float(momentum[j]),
                    "volatility_20d": float(vol20[j]),
                    "volatility_60d": float(vol60[j]),
                    "corr_deg": float(corr_degree[j]),
                    "corr_eigen": float(corr_eigen[j]),
                    "precision_deg": float(precision_degree[j]),
                    "precision_eigen": float(precision_eigen[j]),
                    "causal_in_deg": float(result["in_degree_causal"][j]),
                    "causal_out_deg": float(result["out_degree_causal"][j]),
                    "causal_eigen": float(result["eigen_causal"][j]),
                    "causal_drift": drift["causal_drift"],
                    "drift_threshold": drift["drift_threshold"],
                    "is_structural_break": drift["is_structural_break"],
                    "days_since_last_break": drift["days_since_last_break"],
                    "dagma_nu": float(result.get("nu", np.nan)),
                })

        return pd.DataFrame(rows)

    @staticmethod
    def _drift_row(drift_df: Optional[pd.DataFrame], date: pd.Timestamp) -> Dict[str, float]:
        blank = {
            "causal_drift": np.nan,
            "drift_threshold": np.nan,
            "is_structural_break": 0,
            "days_since_last_break": np.nan,
        }
        if drift_df is None or date not in drift_df.index:
            return blank

        row = drift_df.loc[date]
        return {
            "causal_drift": float(row.get("causal_drift", np.nan)),
            "drift_threshold": float(row.get("drift_threshold", np.nan)),
            "is_structural_break": int(row.get("is_structural_break", 0)),
            "days_since_last_break": float(row.get("days_since_last_break", np.nan)),
        }
