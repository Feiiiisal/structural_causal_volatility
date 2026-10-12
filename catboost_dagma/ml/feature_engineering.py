"""Feature engineering for volatility forecasting and network ablation."""
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
        n_assets = len(assets)
        rows = []

        for i, res in enumerate(rolling_dagma_results):
            end_idx = int(res["end_idx"])  # first index not used in graph estimation
            asof_date = pd.Timestamp(res["date"])
            if end_idx + self.forward_horizon > len(returns_df):
                continue

            future = returns_df.iloc[end_idx:end_idx + self.forward_horizon]
            # Realized-volatility proxy over the forecast horizon; no future value enters predictors.
            forward_return = future.sum(axis=0).values
            forward_vol = np.sqrt((252.0 / self.forward_horizon) * np.square(future.values).sum(axis=0))

            hist_60 = returns_df.iloc[max(0, end_idx - 60):end_idx]
            hist_20 = returns_df.iloc[max(0, end_idx - 20):end_idx]
            mom_20 = hist_20.sum(axis=0).values
            vol_20 = hist_20.std(axis=0, ddof=1).values * np.sqrt(252)
            vol_60 = hist_60.std(axis=0, ddof=1).values * np.sqrt(252)

            corr_mat = np.corrcoef(hist_60.values.T)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr_mat, 0.0)
            corr_deg = (np.abs(corr_mat) > self.corr_threshold).sum(axis=0).astype(float)
            corr_eigen = get_eigenvector_centrality(corr_mat, threshold=self.corr_threshold)

            if rolling_precision_results and i < len(rolling_precision_results):
                p = rolling_precision_results[i]
                prec_deg = p["precision_degree"]
                prec_eigen = p["precision_eigen"]
            else:
                prec_deg = np.zeros(n_assets)
                prec_eigen = np.zeros(n_assets)

            causal_in = res["in_degree_causal"]
            causal_out = res["out_degree_causal"]
            causal_eigen = res["eigen_causal"]
            nu = float(res.get("nu", np.nan))

            drift_val = np.nan
            threshold = np.nan
            is_break = 0
            days_since = np.nan
            if drift_df is not None and asof_date in drift_df.index:
                row = drift_df.loc[asof_date]
                drift_val = float(row.get("causal_drift", np.nan))
                threshold = float(row.get("drift_threshold", np.nan))
                is_break = int(row.get("is_structural_break", 0))
                days_since = float(row.get("days_since_last_break", np.nan))

            for j, asset in enumerate(assets):
                rows.append({
                    "window_idx": i,
                    "date": asof_date,
                    "asset": asset,
                    "forward_return": float(forward_return[j]),
                    "forward_volatility": float(forward_vol[j]),
                    "momentum_20d": float(mom_20[j]),
                    "volatility_20d": float(vol_20[j]),
                    "volatility_60d": float(vol_60[j]),
                    "corr_deg": float(corr_deg[j]),
                    "corr_eigen": float(corr_eigen[j]),
                    "precision_deg": float(prec_deg[j]),
                    "precision_eigen": float(prec_eigen[j]),
                    "causal_in_deg": float(causal_in[j]),
                    "causal_out_deg": float(causal_out[j]),
                    "causal_eigen": float(causal_eigen[j]),
                    "causal_drift": drift_val,
                    "drift_threshold": threshold,
                    "is_structural_break": is_break,
                    "days_since_last_break": days_since,
                    "dagma_nu": nu,
                })
        return pd.DataFrame(rows)
