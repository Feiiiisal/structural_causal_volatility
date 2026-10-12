"""
Tier 4: E2E System Test for Multi-Model Volatility Expansion (R3).
Executes comparative pipeline runs across standard GARCH, EGARCH, GJR-GARCH,
and Student-t GARCH, validating that all models plug into the causal framework.
"""

import pytest
import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm


def fit_model_and_extract_innovations(vol_diff: pd.Series, model_type: str) -> pd.Series:
    """Fit selected volatility model and extract standardized innovations."""
    scaled = vol_diff * 100.0
    if model_type == "garch":
        am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    elif model_type == "egarch":
        am = arch_model(scaled, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal")
    elif model_type == "gjr":
        am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    elif model_type == "student_t":
        am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    else:
        raise ValueError(f"Unknown model: {model_type}")

    res = am.fit(disp="off", show_warning=False)
    z = res.std_resid.dropna()
    return pd.Series(z.values, index=vol_diff.index[-len(z):], name="Vol_Innovations")


@pytest.mark.e2e
@pytest.mark.parametrize("model_type", ["garch", "egarch", "gjr", "student_t"])
def test_end_to_end_model_execution(model_type, clean_processed_dataframe):
    """Verify that each volatility model runs through residualization, discovery, and backtesting."""
    df = clean_processed_dataframe
    eps = 1e-8
    gk_diff = np.log(df["Garman_Klass_Vol"] + eps) - np.log(df["Garman_Klass_Vol"].shift(1) + eps)
    gk_diff = gk_diff.dropna()

    # Step 1: Residualization
    z = fit_model_and_extract_innovations(gk_diff, model_type)
    assert len(z) > 50
    assert not z.isna().any()

    # Step 2: Causal Discovery
    causal_df = pd.DataFrame(index=z.index)
    causal_df["Vol_Innovations"] = z
    causal_df["VIX_Diff"] = df.loc[z.index, "VIX_Close"].diff().fillna(0)
    causal_df["Credit_Spread_Diff"] = df.loc[z.index, "Credit_Spread"].diff().fillna(0)
    causal_df["Liquidity_Diff"] = df.loc[z.index, "Liquidity_Proxy"].diff().fillna(0)

    var_names = list(causal_df.columns)
    tau_max = 3
    n_nodes = len(var_names)
    p_mat = np.ones((n_nodes, n_nodes, tau_max + 1))
    val_mat = np.zeros((n_nodes, n_nodes, tau_max + 1))
    y = causal_df["Vol_Innovations"].values

    for s_idx, s_name in enumerate(var_names):
        if s_name == "Vol_Innovations":
            continue
        for tau in range(1, tau_max + 1):
            X = [causal_df[s_name].shift(tau).values] + [causal_df["Vol_Innovations"].shift(lag).values for lag in range(1, tau_max + 1)]
            X = np.column_stack(X)
            valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
            if np.sum(valid) > 40:
                m = sm.OLS(y[valid], sm.add_constant(X[valid])).fit()
                p_mat[s_idx, 0, tau] = m.pvalues[1]
                val_mat[s_idx, 0, tau] = m.params[1]

    # Step 3: Backtest
    prices = df.loc[z.index, "SP100_Close"]
    vol = df.loc[z.index, "Garman_Klass_Vol"]
    mult = pd.Series(1.8, index=z.index)

    close = prices.values
    v = vol.values
    m = mult.values
    state = np.ones(len(close), dtype=int)
    stop = np.zeros(len(close))
    curr_stop = close[0] - m[0] * v[0] * close[0]
    for t in range(len(close)):
        raw_stop = close[t] - m[t] * v[t] * close[t]
        curr_stop = max(curr_stop, raw_stop)
        if close[t] < curr_stop:
            state[t] = 0
            curr_stop = raw_stop
        else:
            state[t] = 1
        stop[t] = curr_stop

    rets = np.diff(np.log(close)) * state[:-1]
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252) if np.std(rets) > 0 else 0.0
    assert not np.isnan(sharpe)
    assert isinstance(sharpe, float)


@pytest.mark.e2e
def test_multi_model_comparison_table_generation(clean_processed_dataframe):
    """Verify compilation of multi-model comparative performance table."""
    df = clean_processed_dataframe
    eps = 1e-8
    gk_diff = (np.log(df["Garman_Klass_Vol"] + eps) - np.log(df["Garman_Klass_Vol"].shift(1) + eps)).dropna()

    summary_rows = []
    for model_name in ["garch", "egarch", "gjr", "student_t"]:
        z = fit_model_and_extract_innovations(gk_diff, model_name)
        rets = np.diff(np.log(df.loc[z.index, "SP100_Close"].values))
        ann_vol = np.std(rets) * np.sqrt(252)
        summary_rows.append({
            "Model": model_name,
            "Innovation_Std": float(z.std()),
            "Annual_Vol": float(ann_vol),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    assert len(summary_df) == 4
    for m in ["garch", "egarch", "gjr", "student_t"]:
        assert m in summary_df.index
        assert np.isclose(summary_df.loc[m, "Innovation_Std"], 1.0, atol=0.35)
