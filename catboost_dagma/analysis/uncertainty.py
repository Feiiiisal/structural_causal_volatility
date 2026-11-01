"""Paired uncertainty estimates for the final out-of-sample tests."""
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)

    block = max(1, min(block_length, n))
    out = []

    while len(out) < n:
        start = int(rng.integers(0, n - block + 1))
        out.extend(range(start, start + block))

    return np.asarray(out[:n], dtype=int)


def _hac_mean_test(values: np.ndarray, max_lag: int = 5) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        return {"statistic": np.nan, "p_value": np.nan}

    centered = x - x.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0

    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * weight * gamma

    var_mean = max(long_run / n, 0.0)
    if var_mean <= 0:
        return {"statistic": np.nan, "p_value": np.nan}

    stat = float(x.mean() / np.sqrt(var_mean))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(stat))))
    return {"statistic": stat, "p_value": p_value}


def forecast_uncertainty(
    predictions: pd.DataFrame,
    n_boot: int = 2000,
    block_length: int = 20,
    seed: int = 42,
) -> Dict[str, float]:
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["sqerr_baseline"] = np.square(
        df["forward_volatility"] - df["pred_baseline"]
    )
    df["sqerr_network"] = np.square(
        df["forward_volatility"] - df["pred_network"]
    )

    daily = df.groupby("date")[["sqerr_baseline", "sqerr_network"]].mean()
    base_rmse = float(np.sqrt(daily["sqerr_baseline"].mean()))
    net_rmse = float(np.sqrt(daily["sqerr_network"].mean()))
    reduction = 100.0 * (base_rmse - net_rmse) / base_rmse

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    arr = daily.to_numpy()

    for i in range(n_boot):
        idx = _block_indices(len(arr), block_length, rng)
        sample = arr[idx]
        b = float(np.sqrt(sample[:, 0].mean()))
        n = float(np.sqrt(sample[:, 1].mean()))
        boot[i] = 100.0 * (b - n) / b if b > 0 else np.nan

    loss_diff = daily["sqerr_baseline"].to_numpy() - daily["sqerr_network"].to_numpy()
    dm = _hac_mean_test(loss_diff, max_lag=5)

    return {
        "baseline_rmse": base_rmse,
        "network_rmse": net_rmse,
        "rmse_reduction_pct": float(reduction),
        "rmse_reduction_ci_low": float(np.nanpercentile(boot, 2.5)),
        "rmse_reduction_ci_high": float(np.nanpercentile(boot, 97.5)),
        "dm_hac_statistic": dm["statistic"],
        "dm_hac_p_value": dm["p_value"],
        "bootstrap_repetitions": int(n_boot),
        "block_length_days": int(block_length),
    }


def _portfolio_stats(returns: np.ndarray) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 2:
        return {"sharpe": np.nan, "max_drawdown": np.nan}

    wealth = np.cumprod(1.0 + r)
    annual_return = wealth[-1] ** (252.0 / len(r)) - 1.0
    annual_vol = np.std(r, ddof=1) * np.sqrt(252.0)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    running_max = np.maximum.accumulate(wealth)
    max_drawdown = float(np.min(wealth / running_max - 1.0))

    return {"sharpe": float(sharpe), "max_drawdown": max_drawdown}


def portfolio_uncertainty(
    equity: pd.DataFrame,
    n_boot: int = 2000,
    block_length: int = 20,
    seed: int = 42,
) -> Dict[str, float]:
    needed = ["Standard_Risk_Parity", "Causal_Contagion_Pruned"]
    missing = [col for col in needed if col not in equity]
    if missing:
        raise ValueError(f"Missing portfolio columns: {missing}")

    returns = equity[needed].pct_change().dropna()
    rp = returns["Standard_Risk_Parity"].to_numpy()
    net = returns["Causal_Contagion_Pruned"].to_numpy()

    rp_stats = _portfolio_stats(rp)
    net_stats = _portfolio_stats(net)

    rng = np.random.default_rng(seed)
    sharpe_diff = np.empty(n_boot, dtype=float)
    mdd_diff = np.empty(n_boot, dtype=float)
    paired = np.column_stack([rp, net])

    for i in range(n_boot):
        idx = _block_indices(len(paired), block_length, rng)
        sample = paired[idx]
        s_rp = _portfolio_stats(sample[:, 0])
        s_net = _portfolio_stats(sample[:, 1])
        sharpe_diff[i] = s_net["sharpe"] - s_rp["sharpe"]
        mdd_diff[i] = s_net["max_drawdown"] - s_rp["max_drawdown"]

    return {
        "risk_parity_sharpe": rp_stats["sharpe"],
        "network_sharpe": net_stats["sharpe"],
        "sharpe_difference": net_stats["sharpe"] - rp_stats["sharpe"],
        "sharpe_difference_ci_low": float(np.nanpercentile(sharpe_diff, 2.5)),
        "sharpe_difference_ci_high": float(np.nanpercentile(sharpe_diff, 97.5)),
        "risk_parity_max_drawdown": rp_stats["max_drawdown"],
        "network_max_drawdown": net_stats["max_drawdown"],
        "max_drawdown_difference": (
            net_stats["max_drawdown"] - rp_stats["max_drawdown"]
        ),
        "max_drawdown_difference_ci_low": float(np.nanpercentile(mdd_diff, 2.5)),
        "max_drawdown_difference_ci_high": float(np.nanpercentile(mdd_diff, 97.5)),
        "bootstrap_repetitions": int(n_boot),
        "block_length_days": int(block_length),
    }
