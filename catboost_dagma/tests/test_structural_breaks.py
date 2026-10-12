"""
Unit tests for Structural Break Detection and Frobenius Causal Drift.
"""

import numpy as np
import pandas as pd
import pytest
from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector


def test_structural_break_detector():
    dates = pd.date_range("2023-01-01", periods=10, freq="5D")
    rolling_results = []
    for i, d in enumerate(dates):
        # Create shifting W matrix
        W = np.zeros((3, 3))
        if i == 5:
            W[0, 1] = 2.5  # Large shock / structural break
        else:
            W[0, 1] = 0.2 + i * 0.01
        rolling_results.append({
            "date": d,
            "W": W,
        })

    detector = StructuralBreakDetector(percentile_threshold=80.0, min_event_distance=2)
    drift_series = detector.compute_causal_drift(rolling_results)
    assert len(drift_series) == 10
    assert drift_series.iloc[5] > 1.0

    breaks_df = detector.fit_detect(drift_series)
    assert len(breaks_df) >= 1
    assert "causal_drift" in breaks_df.columns
    assert "matched_historical_event" in breaks_df.columns

    features_df = detector.generate_break_features(drift_series)
    assert "is_structural_break" in features_df.columns
    assert "days_since_last_break" in features_df.columns
    assert features_df.loc[dates[5], "is_structural_break"] == 1
