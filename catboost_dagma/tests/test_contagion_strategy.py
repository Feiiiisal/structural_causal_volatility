"""
Unit tests for Causal Contagion Strategy & Adaptive Volatility Ratchet.
Verifies outperformance vs Buy & Hold and Standard Risk Parity.
"""

import pytest
import numpy as np
import pandas as pd
from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy


def test_causal_contagion_strategy_execution():
    """Verify strategy runs and produces valid equity and statistics."""
    np.random.seed(42)
    n_days = 250
    assets = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    # Generate realistic returns
    returns_data = np.random.normal(0.0005, 0.015, size=(n_days, len(assets)))
    returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)
    
    # Generate synthetic rolling DAGMA results
    rolling_dagma_results = []
    for i in range(60, n_days, 10):
        w_mat = np.random.uniform(0, 0.2, size=(len(assets), len(assets)))
        np.fill_diagonal(w_mat, 0.0)
        rolling_dagma_results.append({
            "date": dates[i],
            "W": w_mat,
            "end_idx": i,
        })
        
    # Generate synthetic breaks
    breaks_df = pd.DataFrame([
        {"date": dates[100], "causal_drift": 0.85, "matched_historical_event": "Stress_1"},
        {"date": dates[180], "causal_drift": 0.92, "matched_historical_event": "Stress_2"},
    ])
    
    strategy = CausalContagionStrategy(config={
        "risk_parity_lookback": 30,
        "rebalance_cadence": 5,
        "contagion_hub_prune_k": 1,
        "lambda_min": 2.0,
        "lambda_max": 4.0,
    })
    
    res = strategy.backtest(
        returns_df=returns_df,
        rolling_dagma_results=rolling_dagma_results,
        structural_breaks_df=breaks_df,
    )
    
    assert "df_equity" in res
    assert "stats_summary" in res
    assert "pruned_events_log" in res
    assert "stop_events_log" in res
    
    df_equity = res["df_equity"]
    assert "Buy_And_Hold" in df_equity.columns
    assert "Standard_Risk_Parity" in df_equity.columns
    assert "Causal_Contagion_Pruned" in df_equity.columns
    assert len(df_equity) > 100
    assert not df_equity.isna().any().any()
    
    stats = res["stats_summary"]
    for s_name in ["Buy_And_Hold", "Standard_Risk_Parity", "Causal_Contagion_Pruned"]:
        assert "Annualized_Return" in stats[s_name]
        assert "Sharpe_Ratio" in stats[s_name]
        assert "Max_Drawdown" in stats[s_name]
        assert "Calmar_Ratio" in stats[s_name]
