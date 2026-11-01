import numpy as np
import pandas as pd

from catboost_dagma.analysis.uncertainty import (
    forecast_uncertainty,
    portfolio_uncertainty,
)


def test_forecast_uncertainty_detects_clear_improvement():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    rows = []

    for date in dates:
        for asset in ["A", "B", "C"]:
            y = abs(rng.normal(0.2, 0.03))
            rows.append({
                "date": date,
                "asset": asset,
                "forward_volatility": y,
                "pred_baseline": y + rng.normal(0, 0.05),
                "pred_network": y + rng.normal(0, 0.01),
            })

    result = forecast_uncertainty(
        pd.DataFrame(rows),
        n_boot=200,
        block_length=10,
        seed=7,
    )

    assert result["rmse_reduction_pct"] > 0
    assert result["rmse_reduction_ci_low"] > 0


def test_portfolio_uncertainty_is_paired_and_finite():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2022-01-03", periods=200, freq="B")
    rp = rng.normal(0.0005, 0.01, len(dates))
    net = rp - 0.0001 + rng.normal(0, 0.001, len(dates))

    equity = pd.DataFrame({
        "Standard_Risk_Parity": 10000 * np.cumprod(1 + rp),
        "Causal_Contagion_Pruned": 10000 * np.cumprod(1 + net),
    }, index=dates)

    result = portfolio_uncertainty(
        equity,
        n_boot=200,
        block_length=10,
        seed=3,
    )

    assert np.isfinite(result["sharpe_difference"])
    assert np.isfinite(result["sharpe_difference_ci_low"])
    assert np.isfinite(result["max_drawdown_difference_ci_high"])
