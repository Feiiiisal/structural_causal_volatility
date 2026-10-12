import numpy as np
import pandas as pd

from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector


def test_structural_break_detector():
    dates = pd.date_range("2023-01-01", periods=40, freq="5D")
    results = []

    for i, date in enumerate(dates):
        W = np.zeros((3, 3))
        W[0, 1] = 2.5 if i == 25 else 0.2 + i * 0.002
        results.append({"date": date, "W": W})

    detector = StructuralBreakDetector(
        percentile_threshold=90,
        min_event_distance=2,
        min_history=5,
        history_window=20,
    )

    drift = detector.compute_causal_drift(results)
    breaks = detector.fit_detect(drift)
    features = detector.generate_break_features(drift)

    assert drift.iloc[25] > 1.0
    assert not breaks.empty
    assert "causal_drift" in breaks
    assert features.loc[dates[25], "is_structural_break"] == 1
