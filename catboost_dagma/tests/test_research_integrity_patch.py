import numpy as np
import pandas as pd

from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector
from catboost_dagma.data.residualizer import FamaFrenchResidualizer
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel
from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy


def test_future_returns_do_not_change_past_residuals():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=420, freq="B")

    factors = pd.DataFrame({
        "Mkt-RF": rng.normal(0, 0.01, len(dates)),
        "SMB": rng.normal(0, 0.006, len(dates)),
        "HML": rng.normal(0, 0.006, len(dates)),
        "RF": 0.0,
    }, index=dates)

    returns = pd.DataFrame({
        "A": 1.1 * factors["Mkt-RF"] + rng.normal(0, 0.01, len(dates)),
        "B": 0.8 * factors["Mkt-RF"] + rng.normal(0, 0.01, len(dates)),
    }, index=dates)

    first = FamaFrenchResidualizer().fit_transform_walk_forward(
        returns, factors, min_train_size=100, refit_every=20
    )

    changed = returns.copy()
    changed.iloc[300:] += 10.0
    second = FamaFrenchResidualizer().fit_transform_walk_forward(
        changed, factors, min_train_size=100, refit_every=20
    )

    np.testing.assert_allclose(
        first.iloc[:300].values,
        second.iloc[:300].values,
        atol=1e-10,
    )


def test_break_threshold_uses_past_values():
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    drift = pd.Series(np.linspace(0.1, 0.5, 50), index=dates)

    first = StructuralBreakDetector(
        percentile_threshold=90,
        min_history=10,
        history_window=20,
    ).generate_break_features(drift)

    changed = drift.copy()
    changed.iloc[-1] = 1000.0

    second = StructuralBreakDetector(
        percentile_threshold=90,
        min_history=10,
        history_window=20,
    ).generate_break_features(changed)

    pd.testing.assert_series_equal(
        first["drift_threshold"].iloc[:-1],
        second["drift_threshold"].iloc[:-1],
    )


def test_catboost_uses_volatility_target_and_separate_test_period():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=90, freq="B")
    rows = []

    for date in dates:
        for asset in ["A", "B"]:
            rows.append(_ml_row(rng, date, asset))

    model = CatBoostFusionModel(config={
        "iterations": 5,
        "depth": 2,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "verbose": 0,
        "task_type": "CPU",
        "validation_ratio": 0.2,
    })

    result = model.run_ablation_study(
        pd.DataFrame(rows),
        split_date="2020-04-01",
    )

    assert result["target"] == "forward_volatility"
    assert pd.Timestamp(result["test_date_range"][0]) >= pd.Timestamp("2020-04-01")
    assert result["validation_samples"] > 0


def _ml_row(rng, date, asset):
    return {
        "date": date,
        "asset": asset,
        "forward_volatility": abs(rng.normal(0.2, 0.03)),
        "momentum_20d": rng.normal(),
        "volatility_20d": rng.random(),
        "volatility_60d": rng.random(),
        "corr_deg": rng.random(),
        "corr_eigen": rng.random(),
        "precision_deg": rng.random(),
        "precision_eigen": rng.random(),
        "causal_in_deg": rng.random(),
        "causal_out_deg": rng.random(),
        "causal_eigen": rng.random(),
        "causal_drift": rng.random(),
        "is_structural_break": 0,
        "days_since_last_break": rng.integers(1, 30),
        "dagma_nu": 3.0,
    }


def test_strategy_can_start_before_first_graph():
    rng = np.random.default_rng(10)
    dates = pd.date_range("2022-01-03", periods=180, freq="B")
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (180, 3)),
        index=dates,
        columns=list("ABC"),
    )

    W = np.array([
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0],
    ])
    graphs = [
        {"date": dates[120], "W": W, "end_idx": 121},
        {"date": dates[150], "W": W * 1.1, "end_idx": 151},
    ]

    result = CausalContagionStrategy(config={
        "risk_parity_lookback": 30,
        "rebalance_cadence": 5,
        "contagion_hub_prune_k": 1,
        "lambda_min": 2.0,
        "lambda_max": 4.0,
        "ma_window": 20,
        "drift_history": 60,
    }).backtest(
        returns,
        graphs,
        pd.DataFrame(columns=["date"]),
        evaluation_start=None,
    )

    assert not result["df_equity"].isna().any().any()
