"""Network-aware portfolio risk overlay with chronological safeguards.

The implementation is an empirical strategy benchmark, not an assertion that the
causal overlay must outperform conventional portfolios.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from catboost_dagma.config import STRATEGY_CONFIG


class CausalContagionStrategy:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = (config or STRATEGY_CONFIG).copy()
        self.initial_capital = self.config.get("initial_capital", 10000.0)
        self.transaction_cost = self.config.get("transaction_cost", 0.0005)
        self.rebalance_cadence = self.config.get("rebalance_cadence", 5)
        self.prune_k = self.config.get("contagion_hub_prune_k", 2)
        self.lookback = self.config.get("risk_parity_lookback", 60)
        self.lambda_min = self.config.get("lambda_min", 2.5)
        self.lambda_max = self.config.get("lambda_max", 4.0)
        self.ma_window = self.config.get("ma_window", 20)
        self.drift_history = self.config.get("drift_history", 252)

    def _online_drift_percentile(self, drift: pd.Series) -> np.ndarray:
        vals = drift.astype(float).values
        out = np.zeros(len(vals), dtype=float)
        for i, x in enumerate(vals):
            start = max(0, i - self.drift_history)
            hist = vals[start:i]
            hist = hist[np.isfinite(hist)]
            if len(hist) >= 5 and np.isfinite(x):
                out[i] = float(np.mean(hist <= x))
            else:
                out[i] = 0.0
        return out

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
        assets = returns_df.columns.tolist()
        n_assets = len(assets)
        dates = pd.DatetimeIndex(returns_df.index)
        break_dates = set(pd.to_datetime(structural_breaks_df.get("date", pd.Series(dtype="datetime64[ns]")).dropna()).dt.date)

        if drift_series is None:
            records = {}
            for i in range(1, len(rolling_dagma_results)):
                date = pd.Timestamp(rolling_dagma_results[i]["date"])
                records[date] = float(np.linalg.norm(rolling_dagma_results[i]["W"] - rolling_dagma_results[i - 1]["W"], ord="fro"))
            drift_series = pd.Series(records, dtype=float)
        drift_series = drift_series.sort_index()

        # Forward-fill only after a drift observation exists. Never backward-fill a future signal.
        drift_daily = drift_series.reindex(dates).ffill().fillna(0.0)
        norm_drift = self._online_drift_percentile(drift_daily)

        # Most recent graph available by each date; no graph is used before the first estimate exists.
        dagma_sorted = sorted(rolling_dagma_results, key=lambda r: pd.Timestamp(r["date"]))
        dagma_dates = [pd.Timestamp(r["date"]).date() for r in dagma_sorted]
        dagma_Ws = [r["W"] for r in dagma_sorted]
        W_for_day = []
        idx = -1
        for d in dates:
            while idx + 1 < len(dagma_dates) and dagma_dates[idx + 1] <= d.date():
                idx += 1
            W_for_day.append(dagma_Ws[idx] if idx >= 0 else None)

        mkt_rets = returns_df.mean(axis=1)
        mkt_price = (1.0 + mkt_rets).cumprod()
        mkt_ma = mkt_price.rolling(self.ma_window, min_periods=1).mean()
        mkt_vol = mkt_rets.rolling(self.ma_window, min_periods=max(5, self.ma_window // 4)).std().fillna(0.0)
        lambda_t = self.lambda_max - (self.lambda_max - self.lambda_min) * norm_drift

        wealth_bh = [self.initial_capital]
        wealth_rp = [self.initial_capital]
        wealth_causal = [self.initial_capital]
        weights_bh = np.ones(n_assets) / n_assets
        weights_rp = np.ones(n_assets) / n_assets
        weights_causal = np.ones(n_assets) / n_assets
        turnover_causal = 0.0
        turnover_rp = 0.0
        pruned_events, stop_events = [], []

        invested = 1
        p0 = float(mkt_price.iloc[self.lookback])
        v0 = float(mkt_vol.iloc[self.lookback])
        curr_stop = p0 * (1.0 - lambda_t[self.lookback] * v0)

        for t in range(self.lookback, len(returns_df)):
            current_date = dates[t].date()
            ret_t = returns_df.iloc[t].values.astype(float)

            # Today's return is earned using weights chosen previously.
            wealth_bh.append(wealth_bh[-1] * (1.0 + float(weights_bh @ ret_t)))
            wealth_rp.append(wealth_rp[-1] * (1.0 + float(weights_rp @ ret_t)))
            wealth_causal.append(wealth_causal[-1] * (1.0 + float(weights_causal @ ret_t) * invested))

            p_t = float(mkt_price.iloc[t])
            raw_stop = p_t * (1.0 - lambda_t[t] * float(mkt_vol.iloc[t]))
            if invested:
                curr_stop = max(curr_stop, raw_stop)
                if p_t < curr_stop:
                    invested = 0
                    stop_events.append({"date": current_date, "type": "STOP_OUT", "price": p_t, "lambda_t": float(lambda_t[t])})
            else:
                prev_p = float(mkt_price.iloc[t - 1])
                if p_t > float(mkt_ma.iloc[t]) and p_t > prev_p:
                    invested = 1
                    curr_stop = raw_stop
                    stop_events.append({"date": current_date, "type": "RE_ENTRY", "price": p_t})

            is_rebalance = (t % self.rebalance_cadence == 0)
            is_break = current_date in break_dates
            if not (is_rebalance or is_break):
                continue

            hist = returns_df.iloc[t - self.lookback:t].values
            vol = np.std(hist, axis=0, ddof=1) + 1e-8
            inv = 1.0 / vol
            new_rp = inv / inv.sum()
            rp_turnover = float(np.abs(new_rp - weights_rp).sum())
            wealth_rp[-1] *= max(0.0, 1.0 - rp_turnover * self.transaction_cost)
            turnover_rp += rp_turnover
            weights_rp = new_rp

            raw_w = np.ones(n_assets, dtype=float)
            pruned_names = []
            W_t = W_for_day[t]
            if W_t is not None:
                out_degree = np.abs(W_t).sum(axis=1)
                denom = out_degree.max() if out_degree.max() > 0 else 1.0
                raw_w *= np.exp(-(out_degree / denom))
                if is_break or norm_drift[t] > 0.60:
                    k = min(self.prune_k, n_assets)
                    for asset_idx in np.argsort(out_degree)[-k:]:
                        raw_w[asset_idx] = 0.0
                        pruned_names.append(assets[asset_idx])
                    pruned_events.append({
                        "date": current_date,
                        "pruned_assets": pruned_names,
                        "max_out_degree": float(out_degree.max()),
                        "causal_drift": float(drift_daily.iloc[t]),
                        "drift_percentile": float(norm_drift[t]),
                    })

            new_causal = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(n_assets) / n_assets
            causal_turnover = float(np.abs(new_causal - weights_causal).sum())
            wealth_causal[-1] *= max(0.0, 1.0 - causal_turnover * self.transaction_cost)
            turnover_causal += causal_turnover
            weights_causal = new_causal

        equity_index = dates[self.lookback - 1:]
        equity = pd.DataFrame({
            "Buy_And_Hold": wealth_bh,
            "Standard_Risk_Parity": wealth_rp,
            "Causal_Contagion_Pruned": wealth_causal,
        }, index=equity_index)

        eval_equity = equity
        if evaluation_start is not None:
            eval_equity = equity.loc[equity.index >= pd.Timestamp(evaluation_start)].copy()
            if len(eval_equity) >= 2:
                eval_equity = eval_equity / eval_equity.iloc[0] * self.initial_capital

        def stats(series: pd.Series) -> Dict[str, float]:
            if len(series) < 2:
                return {"Annualized_Return": np.nan, "Annualized_Vol": np.nan, "Sharpe_Ratio": np.nan, "Max_Drawdown": np.nan, "Calmar_Ratio": np.nan}
            rets = series.pct_change().dropna()
            ann_ret = (series.iloc[-1] / series.iloc[0]) ** (252.0 / max(len(rets), 1)) - 1.0
            ann_vol = float(rets.std(ddof=1) * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            dd = series / series.cummax() - 1.0
            max_dd = float(dd.min())
            calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
            return {
                "Annualized_Return": round(ann_ret * 100, 2),
                "Annualized_Vol": round(ann_vol * 100, 2),
                "Sharpe_Ratio": round(float(sharpe), 3) if np.isfinite(sharpe) else np.nan,
                "Max_Drawdown": round(max_dd * 100, 2),
                "Calmar_Ratio": round(float(calmar), 3) if np.isfinite(calmar) else np.nan,
            }

        summary = {name: stats(eval_equity[name]) for name in eval_equity.columns}
        return {
            "df_equity": equity,
            "df_equity_evaluation": eval_equity,
            "stats_summary": summary,
            "pruned_events_log": pd.DataFrame(pruned_events),
            "stop_events_log": pd.DataFrame(stop_events),
            "total_turnover": turnover_causal,
            "risk_parity_turnover": turnover_rp,
        }
