"""
Publication-Grade Visual Diagnostics for CatBoost + DAGMA Version 2.
Emits charts visualizing:
1. Fama-French 3-Factor Residualization (Instructor Feedback a)
2. DAGMA Non-Linear Causal Directed Graph
3. Empirical Orthogonality vs Precision Matrix Benchmark (Instructor Feedback b)
4. Structural Break Detection & Frobenius Causal Drift (Flagging SVB & COVID)
5. CatBoost Ablation & Feature Importance
6. Contagion-Pruned Strategy Equity Curves & Drawdown Profile
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_publication_style():
    """Sets consistent academic publication aesthetic."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
    })


def plot_ff3_residualization_audit(
    raw_returns: pd.DataFrame,
    residuals_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    out_path: Union[str, Path],
):
    """Figure 1: Fama-French 3-Factor Residualization Audit."""
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel A: Return distributions before vs after residualization
    sample_col = raw_returns.columns[0]
    sns.kdeplot(raw_returns[sample_col], ax=axes[0, 0], label=f"Raw {sample_col}", color="#d62728", fill=True, alpha=0.3)
    sns.kdeplot(residuals_df[sample_col], ax=axes[0, 0], label=f"Residualized {sample_col}", color="#1f77b4", fill=True, alpha=0.3)
    axes[0, 0].set_title(f"A. Density: Raw vs Residualized Returns ({sample_col})", fontweight="bold")
    axes[0, 0].set_xlabel("Daily Return")
    axes[0, 0].legend()

    # Panel B: Market Beta Distribution across universe
    if "Mkt-RF" in exposure_df.columns:
        sns.histplot(exposure_df["Mkt-RF"], ax=axes[0, 1], kde=True, color="#2ca02c", bins=15)
        axes[0, 1].axvline(1.0, color="red", linestyle="--", label="Market Beta = 1.0")
        axes[0, 1].set_title("B. Cross-Sectional Market Beta Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Market Beta (β_MKT)")
        axes[0, 1].legend()

    # Panel C: R^2 of Fama-French 3-Factor Model
    if "R2" in exposure_df.columns:
        sns.barplot(x=exposure_df.index[:15], y=exposure_df["R2"][:15], ax=axes[1, 0], palette="Blues_r")
        axes[1, 0].set_title("C. Fama-French Factor Explanatory Power (R²)", fontweight="bold")
        axes[1, 0].set_ylabel("R² Variance Explained")
        axes[1, 0].tick_params(axis="x", rotation=45)

    # Panel D: Cross-Asset Correlation Reduction
    raw_corr = raw_returns.iloc[:, :10].corr().values
    np.fill_diagonal(raw_corr, np.nan)
    res_corr = residuals_df.iloc[:, :10].corr().values
    np.fill_diagonal(res_corr, np.nan)

    mean_raw_corr = float(np.nanmean(np.abs(raw_corr)))
    mean_res_corr = float(np.nanmean(np.abs(res_corr)))

    corr_comp = pd.DataFrame({
        "Panel": ["Raw Returns", "FF3 Residuals"],
        "Mean_Absolute_Correlation": [mean_raw_corr, mean_res_corr]
    })
    sns.barplot(data=corr_comp, x="Panel", y="Mean_Absolute_Correlation", ax=axes[1, 1], palette="Purples_r")
    axes[1, 1].set_title("D. Average Inter-Asset Correlation Reduction", fontweight="bold")
    axes[1, 1].set_ylabel("Mean |r|")
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                            ha="center", va="center", color="white", fontweight="bold")

    fig.suptitle("Fama-French 3-Factor Residualization: Eliminating Shared Systemic Market Beta", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_orthogonality_benchmark(
    orthogonality_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    out_path: Union[str, Path],
):
    """Figure 3: Empirical Orthogonality vs Precision Matrix Benchmark."""
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Pairwise Correlations between Causal, Precision, and Correlation metrics
    sns.barplot(data=orthogonality_df, y="Comparison", x="Pearson_r", ax=axes[0], palette="crest")
    axes[0].axvline(0.0, color="black", linestyle="-", linewidth=0.8)
    axes[0].axvline(0.35, color="red", linestyle="--", label="Orthogonality Ceiling (|r| < 0.35)")
    axes[0].axvline(-0.35, color="red", linestyle="--")
    axes[0].set_title("A. Empirical Orthogonality: Pearson Correlation (|r| < 0.35)", fontweight="bold")
    axes[0].set_xlabel("Pearson Correlation Coefficient (r)")
    axes[0].legend(loc="lower right")

    for p in axes[0].patches:
        width = p.get_width()
        axes[0].annotate(f"{width:.3f}", (width + (0.02 if width >= 0 else -0.05), p.get_y() + p.get_height() / 2.),
                         va="center", fontsize=8, fontweight="bold")

    # Panel B: Variance Inflation Factors (VIF)
    sns.barplot(data=vif_df, y="Feature", x="VIF", ax=axes[1], palette="viridis")
    axes[1].axvline(5.0, color="red", linestyle="--", label="Safe Collinearity Bound (VIF < 5.0)")
    axes[1].set_title("B. Multicollinearity Check: Variance Inflation Factor (VIF)", fontweight="bold")
    axes[1].set_xlabel("Variance Inflation Factor (VIF)")
    axes[1].legend(loc="lower right")

    for p in axes[1].patches:
        width = p.get_width()
        axes[1].annotate(f"{width:.2f}", (width + 0.1, p.get_y() + p.get_height() / 2.),
                         va="center", fontsize=8, fontweight="bold")

    fig.suptitle("Empirical Non-Redundancy: DAGMA Causal Topologies vs Precision Matrix Benchmark", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_structural_breaks_causal_drift(
    drift_series: pd.Series,
    breaks_df: pd.DataFrame,
    threshold: float,
    out_path: Union[str, Path],
):
    """Figure 4: Structural Break Detection & Frobenius Causal Drift."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(drift_series.index, drift_series.values, color="#1f77b4", linewidth=1.5, label="DAGMA Causal Drift ||W_t - W_{t-1}||_F")
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label=f"95th Percentile Threshold (τ = {threshold:.3f})")

    # Highlight historical stress events
    colors = {"COVID_Crash": "#ff7f0e", "SVB_Banking_Collapse": "#d62728", "Fed_Hikes_2022": "#9467bd"}
    for _, row in breaks_df.iterrows():
        b_date = row["date"]
        b_val = row["causal_drift"]
        event_label = row["matched_historical_event"]
        c = colors.get(event_label, "#2ca02c")

        ax.scatter(b_date, b_val, color=c, s=90, zorder=5, edgecolors="black")
        if event_label in ["SVB_Banking_Collapse", "COVID_Crash"]:
            ax.annotate(
                f"Break: {event_label.replace('_', ' ')}\n({b_date.strftime('%Y-%m-%d')})",
                xy=(b_date, b_val),
                xytext=(b_date, b_val + (0.15 * threshold)),
                arrowprops=dict(facecolor=c, shrink=0.05, width=1, headwidth=6),
                fontweight="bold",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c, alpha=0.9),
            )

    ax.set_title("Non-Linear Causal Topological Drift & Structural Break Detection (Flagging SVB Collapse & COVID)", fontweight="bold", pad=12)
    ax.set_ylabel("Frobenius Norm Drift ||ΔW||_F", fontweight="bold")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_catboost_ablation_and_importance(
    ablation_res: Dict[str, Any],
    out_path: Union[str, Path],
):
    """Figure 5: CatBoost Ablation & Feature Importance."""
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Test RMSE & R^2 Comparison (Baseline vs Augmented)
    base_metrics = ablation_res["baseline"]
    aug_metrics = ablation_res["augmented"]

    models = ["Baseline Model\n(Corr + Precision)", "Augmented Model\n(Baseline + DAGMA Causal)"]
    rmses = [base_metrics["test_rmse"], aug_metrics["test_rmse"]]
    colors = ["#7f7f7f", "#2ca02c"]

    bars = axes[0].bar(models, rmses, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    axes[0].set_title(f"A. Out-of-Sample Test RMSE Ablation\n({ablation_res['improvement']['rmse_reduction_pct']:+.2f}% Improvement via Causal Features)", fontweight="bold")
    axes[0].set_ylabel("Test RMSE")

    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:.5f}", (bar.get_x() + bar.get_width() / 2., h / 2.),
                         ha="center", va="center", color="white", fontweight="bold")

    # Panel B: Feature Importance Ranking by Category
    fi_df = ablation_res["feature_importance_df"].head(10)
    cat_palette = {
        "DAGMA Causal Topology": "#2ca02c",
        "Precision Matrix (GLASSO)": "#1f77b4",
        "Correlation Graph": "#9467bd",
        "Baseline Market/Momentum": "#ff7f0e",
    }

    sns.barplot(
        data=fi_df,
        y="feature",
        x="importance",
        hue="category",
        dodge=False,
        palette=cat_palette,
        ax=axes[1],
        edgecolor="black"
    )
    axes[1].set_title("B. CatBoost Feature Importance Attribution by Channel", fontweight="bold")
    axes[1].set_xlabel("Relative Feature Importance (%)")
    axes[1].set_ylabel("Feature Name")
    axes[1].legend(title="Feature Category", loc="lower right", frameon=True)

    fig.suptitle("CatBoost Model Fusion: Quantifying the Incremental Predictive Value of Causal Features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_contagion_strategy_performance(
    strategy_res: Dict[str, Any],
    out_path: Union[str, Path],
):
    """Figure 6: Downstream Contagion-Pruned Strategy Equity Curves & Drawdowns."""
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    df_equity = strategy_res["df_equity"]
    stats = strategy_res["stats_summary"]

    # Normalize to initial capital $10k
    ax1.plot(df_equity.index, df_equity["Buy_And_Hold"], label=f"Buy & Hold (Sharpe: {stats['Buy_And_Hold']['Sharpe_Ratio']}, Ret: {stats['Buy_And_Hold']['Annualized_Return']}%)", color="#7f7f7f", linestyle="--", linewidth=1.3)
    ax1.plot(df_equity.index, df_equity["Standard_Risk_Parity"], label=f"Standard Risk Parity (Sharpe: {stats['Standard_Risk_Parity']['Sharpe_Ratio']}, Ret: {stats['Standard_Risk_Parity']['Annualized_Return']}%)", color="#1f77b4", linewidth=1.5, alpha=0.85)
    ax1.plot(df_equity.index, df_equity["Causal_Contagion_Pruned"], label=f"Causal Contagion-Pruned Strategy (Sharpe: {stats['Causal_Contagion_Pruned']['Sharpe_Ratio']}, Ret: {stats['Causal_Contagion_Pruned']['Annualized_Return']}%)", color="#2ca02c", linewidth=2.0)

    # Monospace summary box
    c_s = stats["Causal_Contagion_Pruned"]
    rp_s = stats["Standard_Risk_Parity"]
    bh_s = stats["Buy_And_Hold"]
    summary_txt = (
        f"STRATEGY PERFORMANCE COMPARISON\n"
        f"─────────────────────────────────────────────────────────────\n"
        f"• Causal Contagion-Pruned: Sharpe = {c_s['Sharpe_Ratio']} | Ret = {c_s['Annualized_Return']:+.1f}% | MaxDD = {c_s['Max_Drawdown']:.1f}%\n"
        f"• Standard Risk Parity:   Sharpe = {rp_s['Sharpe_Ratio']} | Ret = {rp_s['Annualized_Return']:+.1f}% | MaxDD = {rp_s['Max_Drawdown']:.1f}%\n"
        f"• Buy & Hold:             Sharpe = {bh_s['Sharpe_Ratio']} | Ret = {bh_s['Annualized_Return']:+.1f}% | MaxDD = {bh_s['Max_Drawdown']:.1f}%"
    )
    ax1.text(0.02, 0.72, summary_txt, transform=ax1.transAxes, fontsize=8.5, fontfamily="monospace",
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92, edgecolor="#cccccc"))

    ax1.set_title("Downstream Quantitative Application: Causal Contagion-Pruning Equity Curve", fontweight="bold", pad=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Panel 2: Underwater Drawdown
    def get_dd(series):
        peak = series.cummax()
        return ((series - peak) / peak) * 100.0

    dd_bh = get_dd(df_equity["Buy_And_Hold"])
    dd_rp = get_dd(df_equity["Standard_Risk_Parity"])
    dd_causal = get_dd(df_equity["Causal_Contagion_Pruned"])

    ax2.fill_between(df_equity.index, dd_bh, 0, color="#7f7f7f", alpha=0.2, label=f"Buy & Hold Max DD ({stats['Buy_And_Hold']['Max_Drawdown']:.1f}%)")
    ax2.fill_between(df_equity.index, dd_rp, 0, color="#1f77b4", alpha=0.3, label=f"Standard RP Max DD ({stats['Standard_Risk_Parity']['Max_Drawdown']:.1f}%)")
    ax2.fill_between(df_equity.index, dd_causal, 0, color="#2ca02c", alpha=0.45, label=f"Causal Pruned Max DD ({stats['Causal_Contagion_Pruned']['Max_Drawdown']:.1f}%)")

    ax2.plot(df_equity.index, dd_causal, color="#1b5e20", linewidth=1.5)
    ax2.set_title("Underwater Drawdown Profile: Preserving Capital during Structural Breaks", fontweight="bold", pad=8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
