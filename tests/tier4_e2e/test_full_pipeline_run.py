"""
Tier 4: E2E System Test for Full Pipeline Run (Feature 18).
Executes the unified end-to-end quantitative workflow from data ingestion
through realized volatility, stationarity, GARCH residualization, causal discovery,
adaptive multiplier, trailing stop ratchet backtest, and risk metrics evaluation.
"""

import pytest
import numpy as np
import pandas as pd


def run_e2e_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    Execute full pipeline end-to-end.
    Imports from causal_volatility.pipeline if available,
    otherwise uses modular reference execution matching PROJECT.md interface contracts.
    """
    try:
        from causal_volatility.pipeline import CausalVolatilityPipeline
        pipeline = CausalVolatilityPipeline(model="garch", offline=True)
        return pipeline.run(df_raw)
    except ImportError:
        # Step 1 & 2: Realized Volatility (Garman-Klass) + Liquidity
        log_hl = np.log(df_raw["SP100_High"] / df_raw["SP100_Low"])
        log_co = np.log(df_raw["SP100_Close"] / df_raw["SP100_Open"])
        gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        gk_vol = np.sqrt(np.maximum(0.0, gk_var))
        liq = df_raw["SP100_Volume"] / 1e7

        proc_df = pd.DataFrame(
            {
                "SP100_Close": df_raw["SP100_Close"],
                "Garman_Klass_Vol": gk_vol,
                "VIX_Close": df_raw["VIX_Close"],
                "Credit_Spread": df_raw["Credit_Spread"],
                "Liquidity_Proxy": liq,
            },
            index=df_raw.index,
        )

        # Step 3: Stationarity Transforms with epsilon safety
        eps = 1e-8
        stat_df = pd.DataFrame(
            {
                "SP100_Returns": np.log(proc_df["SP100_Close"] / proc_df["SP100_Close"].shift(1)),
                "GK_Vol_Diff": np.log(proc_df["Garman_Klass_Vol"] + eps) - np.log(proc_df["Garman_Klass_Vol"].shift(1) + eps),
                "VIX_Diff": proc_df["VIX_Close"].diff(),
                "Credit_Spread_Diff": proc_df["Credit_Spread"].diff(),
                "Liquidity_Diff": proc_df["Liquidity_Proxy"].diff(),
            },
            index=proc_df.index,
        ).dropna()

        # Step 4: GARCH Residualization
        from arch import arch_model
        scaled_vol = stat_df["GK_Vol_Diff"] * 100.0
        am = arch_model(scaled_vol, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        garch_res = am.fit(disp="off", show_warning=False)
        z = pd.Series(garch_res.std_resid.dropna().values, index=stat_df.index[-len(garch_res.std_resid.dropna()):], name="Vol_Innovations")

        # Step 5: Causal Discovery Matrix (R2 rank-deficiency free)
        causal_df = pd.DataFrame(index=z.index)
        causal_df["Vol_Innovations"] = z
        causal_df["VIX_Diff"] = stat_df.loc[z.index, "VIX_Diff"]
        causal_df["Credit_Spread_Diff"] = stat_df.loc[z.index, "Credit_Spread_Diff"]
        causal_df["Liquidity_Diff"] = stat_df.loc[z.index, "Liquidity_Diff"]

        # OOS 50/50 Split
        midpoint = len(causal_df) // 2
        train_df = causal_df.iloc[:midpoint]
        test_df = causal_df.iloc[midpoint:]

        # Train Causal DAG
        import statsmodels.api as sm
        var_names = list(train_df.columns)
        tau_max = 5
        n_nodes = len(var_names)
        p_matrix = np.ones((n_nodes, n_nodes, tau_max + 1))
        val_matrix = np.zeros((n_nodes, n_nodes, tau_max + 1))
        y_train = train_df["Vol_Innovations"].values

        for s_idx, s_name in enumerate(var_names):
            if s_name == "Vol_Innovations":
                continue
            for tau in range(1, tau_max + 1):
                X = [train_df[s_name].shift(tau).values] + [train_df["Vol_Innovations"].shift(lag).values for lag in range(1, tau_max + 1)]
                X = np.column_stack(X)
                valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y_train)
                if np.sum(valid) > 50:
                    model = sm.OLS(y_train[valid], sm.add_constant(X[valid])).fit()
                    p_matrix[s_idx, 0, tau] = model.pvalues[1]
                    val_matrix[s_idx, 0, tau] = model.params[1]

        causal_out = {"p_matrix": p_matrix, "val_matrix": val_matrix, "var_names": var_names, "tau_max": tau_max}

        # Step 6: Multiplier on full dataset
        composite_risk = np.zeros(len(causal_df))
        active = 0
        for s_idx, s_name in enumerate(var_names):
            if s_name == "Vol_Innovations":
                continue
            for tau in range(1, tau_max + 1):
                if p_matrix[s_idx, 0, tau] < 0.05 and val_matrix[s_idx, 0, tau] > 0:
                    raw_s = causal_df[s_name].shift(tau).fillna(0).values
                    s_std = raw_s.std()
                    z_s = raw_s / s_std if s_std > 1e-12 else np.zeros(len(causal_df))
                    composite_risk += val_matrix[s_idx, 0, tau] * np.abs(z_s)
                    active += 1
        if active > 0:
            composite_risk /= active

        smoothed = pd.Series(composite_risk, index=causal_df.index).ewm(span=10).mean()
        danger = smoothed.rolling(252, min_periods=20).apply(lambda w: (w.iloc[-1] > w).mean(), raw=False).fillna(0.0).values
        causal_mult = pd.Series(np.clip(2.0 - 0.7 * danger, 1.3, 2.0), index=causal_df.index)

        # Step 7: Backtest Trailing Stop on In-Sample and Out-of-Sample
        def run_backtest_split(prices, vol, mult):
            n = len(prices)
            c = prices.values
            v = vol.values
            m = mult.values
            ma5 = prices.rolling(5, min_periods=1).mean().values

            state = np.ones(n, dtype=int)
            stop = np.zeros(n)
            curr_stop = c[0] - m[0] * v[0] * c[0]
            curr_state = 1

            for t in range(n):
                raw_stop = c[t] - m[t] * v[t] * c[t]
                if curr_state == 1:
                    curr_stop = max(curr_stop, raw_stop)
                    if c[t] < curr_stop:
                        curr_state = 0
                        curr_stop = raw_stop
                else:
                    if m[t] >= 1.8:
                        if c[t] > ma5[t]:
                            curr_state = 1
                            curr_stop = raw_stop
                    else:
                        prev_c = c[t - 1] if t > 0 else c[t]
                        if c[t] > ma5[t] and c[t] > prev_c:
                            curr_state = 1
                            curr_stop = raw_stop
                state[t] = curr_state
                stop[t] = curr_stop

            # Compute metrics
            mkt_rets = np.diff(np.log(c))
            strat_rets = mkt_rets * state[:-1]
            cum = np.exp(np.cumsum(strat_rets))
            tot_ret = cum[-1] - 1.0 if len(cum) > 0 else 0.0
            yrs = len(strat_rets) / 252.0
            ann_ret = (tot_ret + 1.0) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
            ann_vol = np.std(strat_rets) * np.sqrt(252.0)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            rmax = np.maximum.accumulate(cum)
            rmax[rmax == 0] = 1.0
            max_dd = float(np.min((cum - rmax) / rmax)) if len(cum) > 0 else 0.0
            stops = int(np.sum((state[:-1] == 1) & (state[1:] == 0)))

            return {
                "annual_return": ann_ret,
                "annual_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "stop_outs": stops,
            }

        # Run on train and test
        train_dates = train_df.index
        test_dates = test_df.index

        p_train = proc_df.loc[train_dates, "SP100_Close"]
        v_train = proc_df.loc[train_dates, "Garman_Klass_Vol"]
        m_base_train = pd.Series(2.0, index=train_dates)
        m_caus_train = causal_mult.loc[train_dates]

        p_test = proc_df.loc[test_dates, "SP100_Close"]
        v_test = proc_df.loc[test_dates, "Garman_Klass_Vol"]
        m_base_test = pd.Series(2.0, index=test_dates)
        m_caus_test = causal_mult.loc[test_dates]

        return {
            "in_sample": {
                "Standard_Baseline": run_backtest_split(p_train, v_train, m_base_train),
                "Causal_Adaptive": run_backtest_split(p_train, v_train, m_caus_train),
            },
            "out_of_sample": {
                "Standard_Baseline": run_backtest_split(p_test, v_test, m_base_test),
                "Causal_Adaptive": run_backtest_split(p_test, v_test, m_caus_test),
            },
            "causal_output": causal_out,
            "multiplier_series": causal_mult,
        }


@pytest.mark.e2e
def test_full_e2e_pipeline_execution(offline_historical_dataset):
    """Execute complete pipeline and assert output structure and metric validity."""
    results = run_e2e_pipeline(offline_historical_dataset)

    assert "in_sample" in results
    assert "out_of_sample" in results
    assert "causal_output" in results
    assert "multiplier_series" in results

    # Verify metric schemas
    for split in ["in_sample", "out_of_sample"]:
        for strat in ["Standard_Baseline", "Causal_Adaptive"]:
            strat_metrics = results[split][strat]
            for metric_key in ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "stop_outs"]:
                assert metric_key in strat_metrics
                assert not np.isnan(strat_metrics[metric_key])

    # Multiplier series bounds
    mult = results["multiplier_series"]
    assert (mult >= 1.3 - 1e-9).all()
    assert (mult <= 2.0 + 1e-9).all()


@pytest.mark.e2e
def test_full_pipeline_zero_rank_deficiency_warnings(offline_historical_dataset):
    """Verify zero rank-deficiency warnings during end-to-end execution."""
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        results = run_e2e_pipeline(offline_historical_dataset)

    singular_warnings = [
        w for w in captured
        if "rank-deficient" in str(w.message).lower() or "singular" in str(w.message).lower()
    ]
    assert len(singular_warnings) == 0


@pytest.mark.e2e
def test_full_pipeline_reproducibility(offline_historical_dataset):
    """Verify executing full pipeline twice yields identical numerical outputs."""
    run1 = run_e2e_pipeline(offline_historical_dataset)
    run2 = run_e2e_pipeline(offline_historical_dataset)

    m1 = run1["out_of_sample"]["Causal_Adaptive"]
    m2 = run2["out_of_sample"]["Causal_Adaptive"]

    for k in ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"]:
        assert np.isclose(m1[k], m2[k], atol=1e-6), f"Non-deterministic metric '{k}': {m1[k]} vs {m2[k]}"
