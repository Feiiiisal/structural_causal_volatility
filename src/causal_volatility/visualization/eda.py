"""
Rigorous Exploratory Data Analysis (EDA) Visualization Suite for Causal Volatility.

Implements high-fidelity, publication-grade statistical and econometric plots
following the visual and analytical standards of Graph_Alpha_Project.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def set_publication_style():
    """Enforce clean, reproducible visual style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8



def plot_correlation_matrix(
    df: pd.DataFrame,
    out_path: Union[str, Path],
    title: str = "Precision Matrix (Inverse Covariance) Heatmap",
):
    """
    Generate annotated partial correlation / precision matrix heatmap across stationary features.
    Matches instructor feedback to benchmark against precision matrix, not plain correlation.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    # Calculate Covariance and Precision Matrix
    cov = df.cov()
    try:
        inv_cov = np.linalg.inv(cov.values)
        precision_df = pd.DataFrame(inv_cov, index=cov.index, columns=cov.columns)
        
        # Convert Precision Matrix to Partial Correlation Matrix
        D = np.diag(1.0 / np.sqrt(np.diag(inv_cov)))
        partial_corr = -1.0 * D.dot(inv_cov).dot(D)
        np.fill_diagonal(partial_corr, 1.0)
        
        plot_df = pd.DataFrame(partial_corr, index=cov.index, columns=cov.columns)
        cbar_label = "Partial Correlation Coefficient"
    except np.linalg.LinAlgError:
        print("WARNING: Covariance matrix singular, falling back to correlation.")
        plot_df = df.corr()
        cbar_label = "Pearson Correlation Coefficient"

    mask = np.triu(np.ones_like(plot_df, dtype=bool), k=1)
    
    sns.heatmap(
        plot_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.75,
        cbar_kws={"shrink": 0.8, "label": cbar_label},
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
        square=True,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_return_and_vol_distributions(
    returns: pd.Series,
    vol_diff: pd.Series,
    out_path: Union[str, Path],
):
    """
    Dual-panel distribution analysis:
    Left: Asset return distribution with KDE and Gaussian fit (skewness & kurtosis).
    Right: Quantile-Quantile (Q-Q) plot testing empirical fat tails vs standard normal.
    Matches eda_2_return_distribution_qq.png style.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    clean_ret = returns.dropna()
    skew = stats.skew(clean_ret)
    kurt = stats.kurtosis(clean_ret)

    # Subplot 1: Return Distribution & Fitted Normal
    sns.histplot(
        clean_ret,
        bins=90,
        kde=True,
        color="#1f77b4",
        alpha=0.45,
        stat="density",
        ax=ax1,
        label="Empirical Density",
    )
    x = np.linspace(clean_ret.min(), clean_ret.max(), 300)
    norm_pdf = stats.norm.pdf(x, loc=clean_ret.mean(), scale=clean_ret.std())
    ax1.plot(x, norm_pdf, "r--", linewidth=2.0, label="Theoretical Gaussian")
    ax1.set_title(
        f"S&P 100 Return Distribution\n(Skewness: {skew:.2f}, Excess Kurtosis: {kurt:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Daily Log Return", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Q-Q Plot (Fat Tail Check)
    stats.probplot(clean_ret, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax2.get_lines()[0].set_markeredgecolor("#1f77b4")
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[0].set_markersize(4.0)
    ax2.get_lines()[1].set_color("#d62728")
    ax2.get_lines()[1].set_linewidth(2.0)
    ax2.set_title("Normal Q-Q Plot (Empirical Fat Tail Check)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Theoretical Normal Quantiles", fontsize=11)
    ax2.set_ylabel("Empirical Ordered Values", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_macro_overlay(
    df_raw: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    Multi-axis macroeconomic overlay: S&P 100 Cumulative Growth against VIX and Credit Spread.
    Matches eda_3_macro_overlay.png style.
    """
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    prices = df_raw["SP100_Close"].dropna()
    cum_growth = prices / prices.iloc[0]

    # Primary axis: Cumulative equity growth
    line1 = ax1.plot(
        cum_growth.index,
        cum_growth.values,
        color="#111111",
        linewidth=1.8,
        label="S&P 100 (OEF) Cumulative Wealth",
    )
    ax1.set_ylabel("Cumulative Wealth ($1 Base)", color="#111111", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#111111")
    ax1.set_xlabel("Date", fontsize=11)

    # Secondary axis: VIX
    ax2 = ax1.twinx()
    vix = df_raw["VIX_Close"].reindex(cum_growth.index).ffill()
    line2 = ax2.plot(
        vix.index,
        vix.values,
        color="#d62728",
        linewidth=1.2,
        alpha=0.75,
        label="CBOE VIX Index (Right)",
    )
    ax2.set_ylabel("VIX Index Level", color="#d62728", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)

    # Tertiary axis: Credit Spread
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    credit = df_raw["Credit_Spread"].reindex(cum_growth.index).ffill()
    line3 = ax3.plot(
        credit.index,
        credit.values,
        color="#ff7f0e",
        linewidth=1.2,
        alpha=0.75,
        linestyle="--",
        label="Moody's Baa Spread (%) (Far Right)",
    )
    ax3.set_ylabel("Baa-10Y Spread (%)", color="#ff7f0e", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#ff7f0e")
    ax3.grid(False)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9)

    plt.title(
        "Macroeconomic Risk Features (VIX & Credit Spread) vs S&P 100 Equity Trajectory",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volatility_estimators_comparison(
    df_raw: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    Comparative analysis of realized volatility estimators:
    Garman-Klass vs Parkinson vs Rogers-Satchell vs Close-to-Close.
    Emits eda_4_volatility_estimators_comparison.png.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    from causal_volatility.estimators.garman_klass import GarmanKlassEstimator
    from causal_volatility.estimators.parkinson import ParkinsonEstimator
    from causal_volatility.estimators.close_to_close import CloseToCloseEstimator
    from causal_volatility.estimators.rogers_satchell import RogersSatchellEstimator

    h, l, c, o = df_raw["SP100_High"], df_raw["SP100_Low"], df_raw["SP100_Close"], df_raw["SP100_Open"]

    gk = GarmanKlassEstimator().estimate(h, l, c, o) * np.sqrt(252) * 100.0
    pk = ParkinsonEstimator().estimate(h, l, c, o) * np.sqrt(252) * 100.0
    rs = RogersSatchellEstimator().estimate(h, l, c, o) * np.sqrt(252) * 100.0
    c2c = CloseToCloseEstimator().estimate(h, l, c, o) * np.sqrt(252) * 100.0

    ax1.plot(gk.index, gk.rolling(5).mean(), label="Garman-Klass (5d MA)", color="#1f77b4", linewidth=1.5)
    ax1.plot(pk.index, pk.rolling(5).mean(), label="Parkinson (5d MA)", color="#2ca02c", linewidth=1.2, alpha=0.8)
    ax1.plot(rs.index, rs.rolling(5).mean(), label="Rogers-Satchell (5d MA)", color="#ff7f0e", linewidth=1.2, alpha=0.8)
    ax1.plot(c2c.index, c2c.rolling(5).mean(), label="Close-to-Close (5d MA)", color="#7f7f7f", linewidth=1.0, linestyle="--", alpha=0.7)
    ax1.set_ylabel("Annualized Volatility (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Multi-Estimator Realized Volatility Comparison (Annualized %)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Estimation spread relative to Garman-Klass
    spread_pk = pk - gk
    spread_c2c = c2c - gk
    ax2.plot(gk.index, spread_c2c.rolling(10).mean(), label="Close-to-Close Spread vs GK", color="#d62728", linewidth=1.2)
    ax2.plot(gk.index, spread_pk.rolling(10).mean(), label="Parkinson Spread vs GK", color="#9467bd", linewidth=1.0)
    ax2.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Diff (% pts)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_title("Efficiency Variance Differentials Relative to Garman-Klass Benchmark", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stationarity_transformation(
    raw_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    4-panel before-and-after stationarity visual inspection showing stabilization of series:
    1. S&P 100 raw price vs Daily Log Returns.
    2. VIX index level vs VIX Log First-Difference.
    3. Credit Spread level vs Credit Spread First-Difference.
    4. Volume/Liquidity level vs Liquidity Differential.
    Emits eda_5_stationarity_transformation.png.
    """
    set_publication_style()
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=True)

    series_pairs = [
        ("S&P 100 Level ($)", raw_df["SP100_Close"], "Log Returns (Δ ln P)", stat_df["SP100_Returns"], "#1f77b4"),
        ("VIX Index Level", raw_df["VIX_Close"], "VIX Log Diff (Δ ln VIX)", stat_df["VIX_Diff"], "#d62728"),
        ("Credit Spread (%)", raw_df["Credit_Spread"], "Spread Diff (Δ S)", stat_df["Credit_Spread_Diff"], "#ff7f0e"),
        ("Liquidity Proxy", raw_df["Liquidity_Proxy"], "Liquidity Diff (Δ L)", stat_df["Liquidity_Diff"], "#2ca02c"),
    ]

    for i, (name_raw, s_raw, name_stat, s_stat, col) in enumerate(series_pairs):
        # Left: Non-stationary raw series I(1)
        axes[i, 0].plot(s_raw.index, s_raw.values, color=col, linewidth=1.2)
        axes[i, 0].set_ylabel(name_raw, fontsize=10, fontweight="bold")
        if i == 0:
            axes[i, 0].set_title("Raw Time-Series Levels: Non-Stationary I(1)", fontsize=12, fontweight="bold")

        # Right: Stationary transformed series I(0)
        axes[i, 1].plot(s_stat.index, s_stat.values, color=col, linewidth=0.8, alpha=0.85)
        axes[i, 1].axhline(0, color="black", linestyle=":", linewidth=0.6)
        axes[i, 1].set_ylabel(name_stat, fontsize=10, fontweight="bold")
        if i == 0:
            axes[i, 1].set_title("Stationarity Transformed Series: Stationary I(0)", fontsize=12, fontweight="bold")

    axes[3, 0].set_xlabel("Date", fontsize=11)
    axes[3, 1].set_xlabel("Date", fontsize=11)

    fig.suptitle(
        "Dual Stationarity Transformation Diagnostics (ADF / KPSS Stabilization)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
