import numpy as np
import pandas as pd

from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy


def test_strategy_returns_equity_and_stats():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=250, freq="B")
    assets = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]

    returns = pd.DataFrame(
        rng.normal(0.0005, 0.015, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )

    graphs = []
    for i in range(60, len(dates), 10):
        W = rng.uniform(0, 0.2, size=(len(assets), len(assets)))
        np.fill_diagonal(W, 0.0)
        graphs.append({"date": dates[i], "W": W, "end_idx": i})

    breaks = pd.DataFrame({"date": [dates[100], dates[180]]})

    result = CausalContagionStrategy(config={
        "risk_parity_lookback": 30,
        "rebalance_cadence": 5,
        "contagion_hub_prune_k": 1,
        "lambda_min": 2.0,
        "lambda_max": 4.0,
        "drift_history": 60,
    }).backtest(
        returns,
        graphs,
        breaks,
        evaluation_start=None,
    )

    equity = result["df_equity"]
    assert list(equity.columns) == [
        "Buy_And_Hold",
        "Standard_Risk_Parity",
        "Causal_Contagion_Pruned",
    ]
    assert not equity.isna().any().any()

    for stats in result["stats_summary"].values():
        assert {"Annualized_Return", "Sharpe_Ratio", "Max_Drawdown"}.issubset(stats)
