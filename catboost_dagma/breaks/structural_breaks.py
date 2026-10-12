"""Structural change detection using causal-network drift without look-ahead."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from catboost_dagma.config import HISTORICAL_EVENTS


class StructuralBreakDetector:
    def __init__(self, percentile_threshold: float = 95.0, min_event_distance: int = 10, min_history: int = 20, history_window: Optional[int] = 252):
        self.percentile_threshold = percentile_threshold
        self.min_event_distance = min_event_distance
        self.min_history = min_history
        self.history_window = history_window
        self.threshold_: float = np.nan
        self.threshold_series_: Optional[pd.Series] = None
        self.drift_series_: Optional[pd.Series] = None
        self.detected_breaks_: List[Dict[str, Any]] = []

    def compute_causal_drift(self, rolling_results: List[Dict[str, Any]]) -> pd.Series:
        if not rolling_results:
            return pd.Series(dtype=float, name="causal_drift")
        dates = pd.to_datetime([res["date"] for res in rolling_results])
        values = [0.0]
        for i in range(1, len(rolling_results)):
            values.append(float(np.linalg.norm(rolling_results[i]["W"] - rolling_results[i - 1]["W"], ord="fro")))
        out = pd.Series(values, index=dates, name="causal_drift").sort_index()
        self.drift_series_ = out
        return out

    def _past_thresholds(self, drift: pd.Series) -> pd.Series:
        vals = drift.astype(float)
        thresholds = pd.Series(np.nan, index=vals.index, dtype=float, name="drift_threshold")
        for i in range(1, len(vals)):
            start = 1
            if self.history_window is not None:
                start = max(1, i - self.history_window)
            hist = vals.iloc[start:i].dropna()  # strictly prior observations
            if len(hist) >= self.min_history:
                thresholds.iloc[i] = float(np.percentile(hist.values, self.percentile_threshold))
        self.threshold_series_ = thresholds
        valid = thresholds.dropna()
        self.threshold_ = float(valid.iloc[-1]) if len(valid) else np.nan
        return thresholds

    def fit_detect(self, drift_series: Optional[pd.Series] = None, rolling_results: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        if drift_series is None:
            if rolling_results is None:
                raise ValueError("Provide drift_series or rolling_results.")
            drift_series = self.compute_causal_drift(rolling_results)
        drift_series = drift_series.sort_index()
        thresholds = self._past_thresholds(drift_series)
        break_mask = (drift_series > thresholds) & thresholds.notna()
        break_indices = np.flatnonzero(break_mask.values)

        events = []
        last_idx = -self.min_event_distance
        for idx in break_indices:
            if idx - last_idx < self.min_event_distance:
                continue
            date = drift_series.index[idx]
            matched = None
            # Historical event labels are descriptive only; they do not set the threshold.
            for name, meta in HISTORICAL_EVENTS.items():
                if pd.Timestamp(meta["start"]) <= date <= pd.Timestamp(meta["end"]):
                    matched = name
                    break
            events.append({
                "date": date,
                "window_idx": int(idx),
                "causal_drift": float(drift_series.iloc[idx]),
                "threshold": float(thresholds.iloc[idx]),
                "matched_historical_event": matched or "Unlabelled_Structural_Change",
            })
            last_idx = idx
        self.detected_breaks_ = events
        return pd.DataFrame(events)

    def generate_break_features(self, drift_series: pd.Series) -> pd.DataFrame:
        drift = drift_series.sort_index().astype(float)
        thresholds = self._past_thresholds(drift)
        is_break = ((drift > thresholds) & thresholds.notna()).astype(int)

        days_since = []
        count = np.nan
        for flag in is_break.values:
            if flag == 1:
                count = 0.0
            elif np.isnan(count):
                count = np.nan
            else:
                count += 1.0
            days_since.append(count)

        zscores = []
        for i in range(len(drift)):
            start = 1
            if self.history_window is not None:
                start = max(1, i - self.history_window)
            hist = drift.iloc[start:i].dropna()
            if len(hist) < self.min_history or hist.std(ddof=1) < 1e-12:
                zscores.append(np.nan)
            else:
                zscores.append(float((drift.iloc[i] - hist.mean()) / hist.std(ddof=1)))

        return pd.DataFrame({
            "causal_drift": drift.values,
            "drift_threshold": thresholds.values,
            "is_structural_break": is_break.values,
            "days_since_last_break": days_since,
            "drift_zscore": zscores,
        }, index=drift.index)
