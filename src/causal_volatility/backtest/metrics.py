"""
Quantitative Risk and Performance Metrics Evaluator (Feature 16).

Computes annualized return, annualized volatility, Sharpe ratio,
maximum drawdown, stop-out frequency, and win rate.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def compute_comprehensive_risk_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualized performance and risk metrics for backtest strategies.

    Args:
        backtest_df: DataFrame containing 'Close' prices and position state columns
                     ('Standard_State', 'Causal_State', or 'Invested_State').

    Returns:
        DataFrame indexed by strategy name with metric columns:
            - 'Annualised Return'
            - 'Annualised Volatility'
            - 'Sharpe Ratio'
            - 'Maximum Drawdown'
            - 'Total Stop-Out Events'
    """
    metrics = {}
    close_prices = backtest_df["Close"].values.astype(float)
    market_returns = np.diff(np.log(close_prices))
    n_days = len(market_returns)
    years = n_days / 252.0 if n_days > 0 else 1.0

    # Include Buy & Hold benchmark
    bh_wealth = np.exp(np.cumsum(market_returns))
    bh_total_ret = bh_wealth[-1] - 1.0 if len(bh_wealth) > 0 else 0.0
    bh_ann_ret = (bh_total_ret + 1.0) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    bh_ann_vol = float(np.std(market_returns) * np.sqrt(252))
    bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else 0.0

    bh_peak = np.maximum.accumulate(bh_wealth)
    bh_peak[bh_peak == 0] = 1.0
    bh_drawdowns = (bh_wealth - bh_peak) / bh_peak
    bh_max_dd = float(np.min(bh_drawdowns)) if len(bh_drawdowns) > 0 else 0.0

    metrics["Buy_And_Hold"] = {
        "Annualised Return": bh_ann_ret,
        "Annualised Volatility": bh_ann_vol,
        "Sharpe Ratio": bh_sharpe,
        "Maximum Drawdown": bh_max_dd,
        "Total Stop-Out Events": 0,
    }

    strategies = {}
    if "Standard_State" in backtest_df.columns:
        strategies["Standard_Baseline"] = "Standard_State"
    if "Causal_State" in backtest_df.columns:
        strategies["Causal_Adaptive"] = "Causal_State"
    if "Invested_State" in backtest_df.columns and not strategies:
        strategies["Active_Strategy"] = "Invested_State"

    for name, state_col in strategies.items():
        state = backtest_df[state_col].values
        # Stop-out count is when state transitions from 1 to 0
        stop_count = int(np.sum((state[:-1] == 1) & (state[1:] == 0)))
        strat_returns = market_returns * state[:-1]

        cum_wealth = np.exp(np.cumsum(strat_returns))
        total_ret = cum_wealth[-1] - 1.0 if len(cum_wealth) > 0 else 0.0
        ann_return = (total_ret + 1.0) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        ann_vol = float(np.std(strat_returns) * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        running_max = np.maximum.accumulate(cum_wealth)
        running_max[running_max == 0] = 1.0
        drawdowns = (cum_wealth - running_max) / running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        metrics[name] = {
            "Annualised Return": ann_return,
            "Annualised Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Maximum Drawdown": max_dd,
            "Total Stop-Out Events": stop_count,
        }

    return pd.DataFrame(metrics).T
