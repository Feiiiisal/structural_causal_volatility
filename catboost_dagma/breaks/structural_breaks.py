"""Track changes in the rolling network."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from catboost_dagma.config import HISTORICAL_EVENTS


class StructuralBreakDetector:
    def __init__(
        self,
        percentile_threshold: float = 95.0,
        min_event_distance: int = 10,
        min_history: int = 20,
        history_window: Optional[int] = 252,
    ):
        self.percentile_threshold = percentile_threshold
        self.min_event_distance = min_event_distance
        self.min_history = min_history
        self.history_window = history_window

        self.threshold_ = np.nan
        self.threshold_series_ = None
        self.drift_series_ = None
        self.detected_breaks_: List[Dict[str, Any]] = []

    def compute_causal_drift(self, results: List[Dict[str, Any]]) -> pd.Series:
        if not results:
            return pd.Series(dtype=float, name="causal_drift")

        values = [0.0]
        for i in range(1, len(results)):
            change = results[i]["W"] - results[i - 1]["W"]
            values.append(float(np.linalg.norm(change, ord="fro")))

        dates = pd.to_datetime([item["date"] for item in results])
        self.drift_series_ = pd.Series(values, index=dates, name="causal_drift").sort_index()
        return self.drift_series_

    def _thresholds(self, drift: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=drift.index, dtype=float, name="drift_threshold")

        for i in range(1, len(drift)):
            start = max(1, i - self.history_window) if self.history_window else 1
            history = drift.iloc[start:i].dropna()

            if len(history) >= self.min_history:
                out.iloc[i] = np.percentile(history, self.percentile_threshold)

        self.threshold_series_ = out
        valid = out.dropna()
        self.threshold_ = float(valid.iloc[-1]) if not valid.empty else np.nan
        return out

    def fit_detect(
        self,
        drift_series: Optional[pd.Series] = None,
        rolling_results: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        if drift_series is None:
            if rolling_results is None:
                raise ValueError("Provide drift_series or rolling_results.")
            drift_series = self.compute_causal_drift(rolling_results)

        drift = drift_series.sort_index()
        thresholds = self._thresholds(drift)
        candidates = np.flatnonzero(((drift > thresholds) & thresholds.notna()).values)

        events = []
        last = -self.min_event_distance

        for idx in candidates:
            if idx - last < self.min_event_distance:
                continue

            date = drift.index[idx]
            events.append({
                "date": date,
                "window_idx": int(idx),
                "causal_drift": float(drift.iloc[idx]),
                "threshold": float(thresholds.iloc[idx]),
                "matched_historical_event": self._event_name(date),
            })
            last = idx

        self.detected_breaks_ = events
        return pd.DataFrame(events)

    def _event_name(self, date: pd.Timestamp) -> str:
        for name, event in HISTORICAL_EVENTS.items():
            if pd.Timestamp(event["start"]) <= date <= pd.Timestamp(event["end"]):
                return name
        return "Unlabelled_Structural_Change"

    def generate_break_features(self, drift_series: pd.Series) -> pd.DataFrame:
        drift = drift_series.sort_index().astype(float)
        thresholds = self._thresholds(drift)
        flags = ((drift > thresholds) & thresholds.notna()).astype(int)

        since = []
        counter = np.nan
        for flag in flags:
            if flag:
                counter = 0.0
            elif np.isfinite(counter):
                counter += 1.0
            since.append(counter)

        zscores = []
        for i in range(len(drift)):
            start = max(1, i - self.history_window) if self.history_window else 1
            history = drift.iloc[start:i].dropna()

            if len(history) < self.min_history or history.std(ddof=1) < 1e-12:
                zscores.append(np.nan)
            else:
                zscores.append(float((drift.iloc[i] - history.mean()) / history.std(ddof=1)))

        return pd.DataFrame({
            "causal_drift": drift.values,
            "drift_threshold": thresholds.values,
            "is_structural_break": flags.values,
            "days_since_last_break": since,
            "drift_zscore": zscores,
        }, index=drift.index)
