"""Plots used in the report."""
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig, out_path: Union[str, Path]):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def set_publication_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    })


def plot_ff3_residualization_audit(
    raw_returns: pd.DataFrame,
    residuals_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    out_path: Union[str, Path],
):
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    asset = raw_returns.columns[0]

    sns.kdeplot(raw_returns[asset], ax=axes[0, 0], label="raw")
    sns.kdeplot(residuals_df[asset], ax=axes[0, 0], label="residual")
    axes[0, 0].set_title(f"{asset}: returns before and after factor adjustment")
    axes[0, 0].legend()

    if "Mkt-RF" in exposure_df:
        sns.histplot(exposure_df["Mkt-RF"], ax=axes[0, 1], bins=15)
        axes[0, 1].set_title("Market beta")

    if "R2" in exposure_df:
        subset = exposure_df["R2"].head(15)
        axes[1, 0].bar(subset.index, subset.values)
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].set_title("Fama-French R²")

    raw_corr = raw_returns.iloc[:, :10].corr().values
    residual_corr = residuals_df.iloc[:, :10].corr().values
    np.fill_diagonal(raw_corr, np.nan)
    np.fill_diagonal(residual_corr, np.nan)

    values = [
        np.nanmean(np.abs(raw_corr)),
        np.nanmean(np.abs(residual_corr)),
    ]
    axes[1, 1].bar(["raw", "residual"], values)
    axes[1, 1].set_title("Mean absolute cross-asset correlation")

    _save(fig, out_path)


def plot_network_comparison(
    orthogonality_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    out_path: Union[str, Path],
):
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(
        data=orthogonality_df,
        y="Comparison",
        x="Pearson_r",
        ax=axes[0],
    )
    axes[0].axvline(0, linewidth=0.8)
    axes[0].set_title("Network feature correlation")

    sns.barplot(data=vif_df, y="Feature", x="VIF", ax=axes[1])
    axes[1].axvline(5, linestyle="--", linewidth=1)
    axes[1].set_title("Variance inflation factor")

    _save(fig, out_path)


def plot_structural_breaks_causal_drift(
    drift_series: pd.Series,
    breaks_df: pd.DataFrame,
    threshold,
    out_path: Union[str, Path],
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(drift_series.index, drift_series.values, label="network drift")

    if np.isscalar(threshold) and np.isfinite(threshold):
        ax.axhline(threshold, linestyle="--", label="threshold")

    if not breaks_df.empty:
        ax.scatter(
            pd.to_datetime(breaks_df["date"]),
            breaks_df["causal_drift"],
            label="flagged change",
        )

    ax.set_title("Change in network structure")
    ax.set_ylabel("Frobenius norm")
    ax.legend()
    _save(fig, out_path)


def plot_catboost_ablation_and_importance(
    ablation_res: Dict[str, Any],
    out_path: Union[str, Path],
):
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    baseline = ablation_res["baseline"]["test_rmse"]
    augmented = ablation_res["augmented"]["test_rmse"]
    axes[0].bar(["baseline", "with network"], [baseline, augmented])
    axes[0].set_title("Out-of-sample RMSE")
    axes[0].set_ylabel("RMSE")

    importance = ablation_res["feature_importance_df"].head(10)
    sns.barplot(
        data=importance,
        y="feature",
        x="importance",
        ax=axes[1],
    )
    axes[1].set_title("CatBoost feature importance")

    _save(fig, out_path)


def plot_contagion_strategy_performance(
    strategy_res: Dict[str, Any],
    out_path: Union[str, Path],
):
    set_publication_style()
    equity = strategy_res.get("df_equity_evaluation", strategy_res["df_equity"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for column in equity:
        axes[0].plot(equity.index, equity[column], label=column)
        drawdown = 100 * (equity[column] / equity[column].cummax() - 1)
        axes[1].plot(equity.index, drawdown, label=column)

    axes[0].set_title("Portfolio value")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("Date")

    _save(fig, out_path)
