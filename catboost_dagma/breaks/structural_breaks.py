"""
Structural Break Detection Engine via Causal Topological Drift.
Measures the geometric Frobenius distance between consecutive DAGMA causal graphs:
Drift_t = ||W_t - W_{t-1}||_F
Flags systemic structural regime shifts including the March 2023 SVB collapse and March 2020 COVID shock.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from catboost_dagma.config import HISTORICAL_EVENTS, BREAK_DETECTION_CONFIG


class StructuralBreakDetector:
    """
    Detects market-wide topological regime shifts by tracking the rate of change
    in the learned non-linear causal graph W_t over rolling estimation windows.
    """

    def __init__(
        self,
        percentile_threshold: float = 95.0,
        min_event_distance: int = 10,
    ):
        self.percentile_threshold = percentile_threshold
        self.min_event_distance = min_event_distance
        self.threshold_: float = 0.0
        self.drift_series_: Optional[pd.Series] = None
        self.detected_breaks_: List[Dict[str, Any]] = []

    def compute_causal_drift(
        self,
        rolling_results: List[Dict[str, Any]],
    ) -> pd.Series:
        """
        Computes the Frobenius norm distance between consecutive rolling causal graphs:
        Drift_t = ||W_t - W_{t-1}||_F
        """
        dates = [res["date"] for res in rolling_results]
        drift_values = [0.0]  # First window has zero drift

        for i in range(1, len(rolling_results)):
            W_prev = rolling_results[i - 1]["W"]
            W_curr = rolling_results[i]["W"]
            dist = float(np.linalg.norm(W_curr - W_prev, ord="fro"))
            drift_values.append(dist)

        drift_series = pd.Series(drift_values, index=pd.to_datetime(dates), name="causal_drift")
        self.drift_series_ = drift_series
        return drift_series

    def fit_detect(
        self,
        drift_series: Optional[pd.Series] = None,
        rolling_results: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Calculates the percentile threshold and identifies structural break timestamps.
        Matches identified breaks against known historical stress events (e.g. SVB collapse).
        """
        if drift_series is None:
            if rolling_results is None:
                raise ValueError("Must provide either drift_series or rolling_results.")
            drift_series = self.compute_causal_drift(rolling_results)

        # Compute empirical percentile threshold
        valid_drifts = drift_series.iloc[1:]  # Exclude first 0.0 element
        self.threshold_ = float(np.percentile(valid_drifts, self.percentile_threshold))

        # Identify break candidates
        break_mask = drift_series > self.threshold_
        break_indices = np.where(break_mask.values)[0]

        detected_events = []
        last_added_idx = -self.min_event_distance

        for idx in break_indices:
            if idx - last_added_idx >= self.min_event_distance:
                event_date = drift_series.index[idx]
                drift_mag = float(drift_series.iloc[idx])
                
                # Check for historical alignment
                matched_historical = None
                for name, meta in HISTORICAL_EVENTS.items():
                    start = pd.to_datetime(meta["start"])
                    end = pd.to_datetime(meta["end"])
                    if start <= event_date <= end:
                        matched_historical = name
                        break

                detected_events.append({
                    "date": event_date,
                    "window_idx": idx,
                    "causal_drift": round(drift_mag, 4),
                    "threshold": round(self.threshold_, 4),
                    "matched_historical_event": matched_historical or "Idiosyncratic_Regime_Shift",
                })
                last_added_idx = idx

        self.detected_breaks_ = detected_events
        return pd.DataFrame(detected_events)

    def generate_break_features(
        self,
        drift_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Generates ML feature columns:
        - causal_drift: continuous Frobenius norm drift
        - is_structural_break: binary indicator (1 if drift > threshold)
        - days_since_last_break: count of trading days since the previous break
        - drift_zscore: rolling z-score of causal drift
        """
        if self.threshold_ == 0.0:
            self.threshold_ = float(np.percentile(drift_series.iloc[1:], self.percentile_threshold))

        is_break = (drift_series > self.threshold_).astype(int)

        # Compute time since last break
        time_since_break = []
        current_counter = 999  # Large initial number

        for b in is_break.values:
            if b == 1:
                current_counter = 0
            else:
                current_counter += 1
            time_since_break.append(current_counter)

        # Drift z-score
        mean_drift = drift_series.mean()
        std_drift = drift_series.std() if drift_series.std() > 1e-6 else 1.0
        drift_z = (drift_series - mean_drift) / std_drift

        df_break_features = pd.DataFrame({
            "causal_drift": drift_series.values,
            "is_structural_break": is_break.values,
            "days_since_last_break": time_since_break,
            "drift_zscore": drift_z.values,
        }, index=drift_series.index)

        return df_break_features
