"""Portfolio test using the network signal."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from catboost_dagma.config import STRATEGY_CONFIG


class CausalContagionStrategy:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or STRATEGY_CONFIG).copy()
        self.initial_capital = cfg.get("initial_capital", 10000.0)
        self.cost = cfg.get("transaction_cost", 0.0005)
        self.rebalance_cadence = cfg.get("rebalance_cadence", 5)
        self.prune_k = cfg.get("contagion_hub_prune_k", 2)
        self.lookback = cfg.get("risk_parity_lookback", 60)
        self.lambda_min = cfg.get("lambda_min", 2.5)
        self.lambda_max = cfg.get("lambda_max", 4.0)
        self.ma_window = cfg.get("ma_window", 20)
        self.drift_history = cfg.get("drift_history", 252)

    def backtest(
        self,
        returns_df: pd.DataFrame,
        rolling_dagma_results: List[Dict[str, Any]],
        structural_breaks_df: pd.DataFrame,
        drift_series: Optional[pd.Series] = None,
        evaluation_start: Optional[str] = "2021-01-01",
    ) -> Dict[str, Any]:
        if returns_df.empty:
            raise ValueError("returns_df is empty")

        dates = pd.DatetimeIndex(returns_df.index)
        assets = returns_df.columns.tolist()
        n_assets = len(assets)

        drift = self._daily_drift(dates, rolling_dagma_results, drift_series)
        drift_rank = self._drift_rank(drift)
        graphs = self._graphs_by_day(dates, rolling_dagma_results)

        break_dates = set()
        if "date" in structural_breaks_df:
            break_dates = set(pd.to_datetime(structural_breaks_df["date"]).dt.date)

        market_return = returns_df.mean(axis=1)
        market_price = (1 + market_return).cumprod()
        market_ma = market_price.rolling(self.ma_window, min_periods=1).mean()
        market_vol = market_return.rolling(
            self.ma_window,
            min_periods=max(5, self.ma_window // 4),
        ).std().fillna(0.0)

        stop_scale = self.lambda_max - (
            self.lambda_max - self.lambda_min
        ) * drift_rank

        equal = np.ones(n_assets) / n_assets
        w_bh = equal.copy()
        w_rp = equal.copy()
        w_net = equal.copy()

        wealth_bh = [self.initial_capital]
        wealth_rp = [self.initial_capital]
        wealth_net = [self.initial_capital]

        rp_turnover = 0.0
        net_turnover = 0.0
        pruned, stops = [], []

        invested = True
        first_price = float(market_price.iloc[self.lookback])
        first_vol = float(market_vol.iloc[self.lookback])
        stop = first_price * (1 - stop_scale[self.lookback] * first_vol)

        for t in range(self.lookback, len(returns_df)):
            ret = returns_df.iloc[t].values.astype(float)
            day = dates[t].date()

            wealth_bh.append(wealth_bh[-1] * (1 + float(w_bh @ ret)))
            wealth_rp.append(wealth_rp[-1] * (1 + float(w_rp @ ret)))
            net_ret = float(w_net @ ret) if invested else 0.0
            wealth_net.append(wealth_net[-1] * (1 + net_ret))

            invested, stop = self._update_stop(
                t, day, invested, stop, stop_scale,
                market_price, market_ma, market_vol, stops,
            )

            is_break = day in break_dates
            if t % self.rebalance_cadence and not is_break:
                continue

            new_rp = self._risk_parity_weights(returns_df.iloc[t - self.lookback:t])
            wealth_rp[-1], turn = self._apply_cost(wealth_rp[-1], w_rp, new_rp)
            rp_turnover += turn
            w_rp = new_rp

            new_net, event = self._network_weights(
                graphs[t], assets, is_break, drift_rank[t], day, drift.iloc[t]
            )
            if event:
                pruned.append(event)

            wealth_net[-1], turn = self._apply_cost(wealth_net[-1], w_net, new_net)
            net_turnover += turn
            w_net = new_net

        equity = pd.DataFrame({
            "Buy_And_Hold": wealth_bh,
            "Standard_Risk_Parity": wealth_rp,
            "Causal_Contagion_Pruned": wealth_net,
        }, index=dates[self.lookback - 1:])

        evaluation = equity
        if evaluation_start:
            evaluation = equity.loc[equity.index >= pd.Timestamp(evaluation_start)].copy()
            if len(evaluation) > 1:
                evaluation = evaluation / evaluation.iloc[0] * self.initial_capital

        return {
            "df_equity": equity,
            "df_equity_evaluation": evaluation,
            "stats_summary": {
                name: self._stats(evaluation[name])
                for name in evaluation.columns
            },
            "pruned_events_log": pd.DataFrame(pruned),
            "stop_events_log": pd.DataFrame(stops),
            "total_turnover": net_turnover,
            "risk_parity_turnover": rp_turnover,
        }

    def _daily_drift(self, dates, results, drift_series):
        if drift_series is None:
            values = {}
            for i in range(1, len(results)):
                date = pd.Timestamp(results[i]["date"])
                change = results[i]["W"] - results[i - 1]["W"]
                values[date] = float(np.linalg.norm(change, "fro"))
            drift_series = pd.Series(values, dtype=float)

        return drift_series.sort_index().reindex(dates).ffill().fillna(0.0)

    def _drift_rank(self, drift):
        values = drift.values.astype(float)
        ranks = np.zeros(len(values))

        for i, value in enumerate(values):
            start = max(0, i - self.drift_history)
            history = values[start:i]
            history = history[np.isfinite(history)]

            if len(history) >= 5 and np.isfinite(value):
                ranks[i] = np.mean(history <= value)

        return ranks

    @staticmethod
    def _graphs_by_day(dates, results):
        ordered = sorted(results, key=lambda item: pd.Timestamp(item["date"]))
        graph_dates = [pd.Timestamp(item["date"]).date() for item in ordered]
        graph_values = [item["W"] for item in ordered]

        out = []
        pos = -1
        for date in dates:
            while pos + 1 < len(graph_dates) and graph_dates[pos + 1] <= date.date():
                pos += 1
            out.append(graph_values[pos] if pos >= 0 else None)

        return out

    def _update_stop(
        self,
        t,
        day,
        invested,
        stop,
        stop_scale,
        market_price,
        market_ma,
        market_vol,
        log,
    ):
        price = float(market_price.iloc[t])
        raw_stop = price * (1 - stop_scale[t] * float(market_vol.iloc[t]))

        if invested:
            stop = max(stop, raw_stop)
            if price < stop:
                log.append({"date": day, "type": "STOP_OUT", "price": price})
                return False, raw_stop
            return True, stop

        previous = float(market_price.iloc[t - 1])
        if price > float(market_ma.iloc[t]) and price > previous:
            log.append({"date": day, "type": "RE_ENTRY", "price": price})
            return True, raw_stop

        return False, stop

    @staticmethod
    def _risk_parity_weights(history):
        vol = history.std(ddof=1).values + 1e-8
        inv = 1 / vol
        return inv / inv.sum()

    def _network_weights(self, W, assets, is_break, drift_rank, day, drift):
        if W is None:
            return np.ones(len(assets)) / len(assets), None

        out_degree = np.abs(W).sum(axis=1)
        scale = out_degree.max() or 1.0
        weights = np.exp(-(out_degree / scale))

        event = None
        if is_break or drift_rank > 0.60:
            ids = np.argsort(out_degree)[-min(self.prune_k, len(assets)):]
            weights[ids] = 0.0
            event = {
                "date": day,
                "pruned_assets": [assets[i] for i in ids],
                "causal_drift": float(drift),
            }

        if weights.sum() == 0:
            weights = np.ones(len(assets))

        return weights / weights.sum(), event

    def _apply_cost(self, wealth, old_weights, new_weights):
        turnover = float(np.abs(new_weights - old_weights).sum())
        return wealth * max(0.0, 1 - turnover * self.cost), turnover

    @staticmethod
    def _stats(series: pd.Series) -> Dict[str, float]:
        if len(series) < 2:
            return {
                "Annualized_Return": np.nan,
                "Annualized_Vol": np.nan,
                "Sharpe_Ratio": np.nan,
                "Max_Drawdown": np.nan,
                "Calmar_Ratio": np.nan,
            }

        returns = series.pct_change().dropna()
        annual_return = (series.iloc[-1] / series.iloc[0]) ** (
            252 / max(len(returns), 1)
        ) - 1
        annual_vol = returns.std(ddof=1) * np.sqrt(252)
        drawdown = series / series.cummax() - 1
        max_drawdown = float(drawdown.min())

        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

        return {
            "Annualized_Return": round(annual_return * 100, 2),
            "Annualized_Vol": round(float(annual_vol) * 100, 2),
            "Sharpe_Ratio": round(float(sharpe), 3) if np.isfinite(sharpe) else np.nan,
            "Max_Drawdown": round(max_drawdown * 100, 2),
            "Calmar_Ratio": round(float(calmar), 3) if np.isfinite(calmar) else np.nan,
        }
