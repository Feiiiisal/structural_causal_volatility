"""
Unit tests for Feature Engineering and CatBoost Ablation.
"""

import numpy as np
import pandas as pd
import pytest
from catboost_dagma.ml.feature_engineering import FusionFeatureEngineer
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel


def test_catboost_ablation_pipeline():
    np.random.seed(42)
    n = 120
    df_ml = pd.DataFrame({
        "window_idx": np.repeat(range(30), 4),
        "date": pd.date_range("2022-01-01", periods=120, freq="D"),
        "asset": ["A", "B", "C", "D"] * 30,
        "forward_return": np.random.normal(0.01, 0.03, n),
        "forward_volatility": np.random.uniform(0.15, 0.35, n),
        "momentum_20d": np.random.normal(0.02, 0.05, n),
        "volatility_20d": np.random.uniform(0.15, 0.35, n),
        "volatility_60d": np.random.uniform(0.15, 0.35, n),
        "corr_deg": np.random.uniform(0, 3, n),
        "corr_eigen": np.random.uniform(0, 1, n),
        "precision_deg": np.random.uniform(0, 3, n),
        "precision_eigen": np.random.uniform(0, 1, n),
        "causal_in_deg": np.random.uniform(0, 3, n),
        "causal_out_deg": np.random.uniform(0, 3, n),
        "causal_eigen": np.random.uniform(0, 1, n),
        "causal_drift": np.random.uniform(0, 1.5, n),
        "is_structural_break": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "days_since_last_break": np.random.randint(0, 50, n),
        "dagma_nu": [3.0] * n,
    })

    model = CatBoostFusionModel(config={"iterations": 20, "depth": 3, "verbose": 0, "task_type": "CPU"})
    res = model.run_ablation_study(df_ml, split_date="2022-03-01")

    assert "baseline" in res
    assert "augmented" in res
    assert "feature_importance_df" in res
    assert "test_rmse" in res["baseline"]
    assert "test_rmse" in res["augmented"]
    assert len(res["feature_importance_df"]) == len(model.augmented_features)
