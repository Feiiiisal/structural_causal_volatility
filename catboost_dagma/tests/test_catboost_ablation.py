import numpy as np
import pandas as pd

from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel


def test_catboost_ablation_runs():
    rng = np.random.default_rng(42)
    n = 120

    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "asset": ["A", "B", "C", "D"] * 30,
        "forward_volatility": rng.uniform(0.15, 0.35, n),
        "momentum_20d": rng.normal(0.02, 0.05, n),
        "volatility_20d": rng.uniform(0.15, 0.35, n),
        "volatility_60d": rng.uniform(0.15, 0.35, n),
        "corr_deg": rng.uniform(0, 3, n),
        "corr_eigen": rng.uniform(0, 1, n),
        "precision_deg": rng.uniform(0, 3, n),
        "precision_eigen": rng.uniform(0, 1, n),
        "causal_in_deg": rng.uniform(0, 3, n),
        "causal_out_deg": rng.uniform(0, 3, n),
        "causal_eigen": rng.uniform(0, 1, n),
        "causal_drift": rng.uniform(0, 1.5, n),
        "is_structural_break": rng.integers(0, 2, n),
        "days_since_last_break": rng.integers(0, 50, n),
        "dagma_nu": 3.0,
    })

    model = CatBoostFusionModel(config={
        "iterations": 20,
        "depth": 3,
        "verbose": 0,
        "task_type": "CPU",
        "validation_ratio": 0.2,
    })
    result = model.run_ablation_study(df, split_date="2022-03-01")

    assert result["target"] == "forward_volatility"
    assert "test_rmse" in result["baseline"]
    assert "test_rmse" in result["augmented"]
    assert len(result["feature_importance_df"]) == len(model.augmented_features)
