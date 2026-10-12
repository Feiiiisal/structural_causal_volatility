"""
Master End-to-End Execution Pipeline for CatBoost + DAGMA Version 2.
Orchestrates:
Stage 1: Data Ingestion (S&P 100 Constituents + Fama-French 3 Factors)
Stage 2: Fama-French 3-Factor Residualization (Instructor Feedback a)
Stage 3: Rolling Non-Linear DAGMA Causal Discovery with Student-t Loss
Stage 4: Graphical LASSO Precision Matrix Benchmark (Instructor Feedback b)
Stage 5: Empirical Orthogonality & Non-Redundancy Audit
Stage 6: Structural Break Detection & Event Matching (SVB & COVID)
Stage 7: Tabular Feature Engineering
Stage 8: CatBoost Ablation Study & Feature Importance
Stage 9: Downstream Quantitative Contagion-Pruning Backtest
Stage 10: Publication-Grade Visual Diagnostics Emission
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from catboost_dagma.config import (
    DEFAULT_SECTOR_ASSETS,
    DAGMA_CONFIG,
    FIGURES_DIR,
    REPORTS_DIR,
)
from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer
from catboost_dagma.dagma.solver import run_parallel_rolling_dagma
from catboost_dagma.benchmark.precision_matrix import PrecisionMatrixEstimator
from catboost_dagma.benchmark.orthogonality import EmpiricalOrthogonalityAuditor
from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector
from catboost_dagma.ml.feature_engineering import FusionFeatureEngineer
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel
from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy
from catboost_dagma.visualization.diagnostics import (
    plot_ff3_residualization_audit,
    plot_empirical_orthogonality_benchmark,
    plot_structural_breaks_causal_drift,
    plot_catboost_ablation_and_importance,
    plot_contagion_strategy_performance,
)


class CatBoostDagmaPipeline:
    """
    Executes the comprehensive CatBoost + DAGMA Version 2 research pipeline.
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2026-01-01",
        window_size: int = 60,
        step_size: int = 5,
        n_workers: int = 8,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.assets = assets or DEFAULT_SECTOR_ASSETS
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.step_size = step_size
        self.n_workers = n_workers
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Executes full pipeline stages sequentially."""
        start_time = time.time()
        if verbose:
            print("=" * 85)
            print("🚀 LAUNCHING VERSION 2: CATBOOST + DAGMA CAUSAL FUSION PIPELINE")
            print("=" * 85)

        # ----------------------------------------------------------------------
        # Stage 1: Data Ingestion
        # ----------------------------------------------------------------------
        if verbose:
            print(f"\n[Stage 1] Ingesting S&P 100 ({len(self.assets)} assets) and Fama-French factors...")
        loader = UnifiedDataLoader(offline=True)
        returns_df, factors_df = loader.get_aligned_panel(
            assets=self.assets,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        if verbose:
            print(f"   ✅ Synchronized Panel Shape: {returns_df.shape} ({returns_df.index[0].date()} to {returns_df.index[-1].date()})")

        # ----------------------------------------------------------------------
        # Stage 2: Fama-French 3-Factor Residualization
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 2] Residualizing Asset Returns on Fama-French 3 Factors (Shared Beta Removal)...")
        residualizer = FamaFrenchResidualizer()
        residuals_df = residualizer.fit_transform(returns_df, factors_df)
        exposure_df = residualizer.get_factor_exposure_report()
        mean_r2 = float(exposure_df["R2"].mean())
        if verbose:
            print(f"   ✅ Residualization Complete. Mean Factor R²: {mean_r2:.3f}")
            print(f"   🔹 Top Factor Betas:\n{exposure_df[['Mkt-RF', 'SMB', 'HML', 'R2']].head(4).to_string()}")

        # ----------------------------------------------------------------------
        # Stage 3: Rolling Non-Linear DAGMA Causal Discovery
        # ----------------------------------------------------------------------
        if verbose:
            print(f"\n[Stage 3] Executing Parallel Rolling Non-Linear DAGMA (Student-t Loss, Window={self.window_size}, Step={self.step_size})...")
        X_resid = residuals_df.values
        dates = residuals_df.index
        dagma_kwargs = {
            "loss_type": "student-t",
            "learnable_nu": True,
            "warm_iter": 75,
            "max_iter": 150,
            "lambda1": 0.02,
            "lr": 0.005,
            "verbose": False,
        }
        dagma_results = run_parallel_rolling_dagma(
            X_data=X_resid,
            dates=dates,
            window_size=self.window_size,
            step_size=self.step_size,
            dagma_kwargs=dagma_kwargs,
            n_workers=self.n_workers,
        )
        if verbose:
            print(f"   ✅ DAGMA Optimization Complete across {len(dagma_results)} rolling windows.")

        # ----------------------------------------------------------------------
        # Stage 4: Graphical LASSO Precision Matrix Estimation
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 4] Estimating Graphical LASSO Precision Matrix Benchmark (Conditional Independence)...")
        prec_estimator = PrecisionMatrixEstimator()
        precision_results = []
        for i, res in enumerate(dagma_results):
            end_idx = res["end_idx"]
            X_w = X_resid[end_idx - self.window_size : end_idx]
            p_mat, alpha = prec_estimator.fit_precision_matrix(X_w)
            p_topos = prec_estimator.extract_topologies(p_mat)
            precision_results.append(p_topos)
        if verbose:
            print(f"   ✅ Precision Matrix Estimation Complete across {len(precision_results)} windows.")

        # ----------------------------------------------------------------------
        # Stage 5: Structural Break Detection & Causal Drift
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 5] Computing Frobenius Norm Causal Drift & Structural Break Detection...")
        break_detector = StructuralBreakDetector(percentile_threshold=95.0)
        drift_series = break_detector.compute_causal_drift(dagma_results)
        breaks_df = break_detector.fit_detect(drift_series)
        break_features_df = break_detector.generate_break_features(drift_series)
        if verbose:
            print(f"   ✅ Detected {len(breaks_df)} Major Structural Breaks (Threshold τ = {break_detector.threshold_:.4f}):")
            for _, b in breaks_df.iterrows():
                print(f"      - {b['date'].strftime('%Y-%m-%d')}: Drift={b['causal_drift']:.4f} [{b['matched_historical_event']}]")

        # ----------------------------------------------------------------------
        # Stage 6: Tabular ML Feature Engineering
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 6] Engineering Unified Tabular Multi-Source Dataset...")
        engineer = FusionFeatureEngineer(forward_horizon=5)
        df_ml = engineer.build_ml_dataset(
            returns_df=returns_df,
            rolling_dagma_results=dagma_results,
            rolling_precision_results=precision_results,
            drift_df=break_features_df,
        )
        if verbose:
            print(f"   ✅ ML Dataset Assembled: {df_ml.shape[0]} samples x {df_ml.shape[1]} features.")

        # ----------------------------------------------------------------------
        # Stage 7: Empirical Orthogonality & Collinearity Audit
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 7] Performing Empirical Orthogonality Audit (DAGMA vs Precision vs Correlation)...")
        auditor = EmpiricalOrthogonalityAuditor()
        orthogonality_df = auditor.audit_topological_correlations(df_ml)
        vif_df = auditor.audit_variance_inflation_factors(df_ml)
        if verbose:
            print(f"   🔹 Centrality Cross-Correlations:\n{orthogonality_df[['Comparison', 'Pearson_r', 'Empirical_Verdict']].to_string(index=False)}")
            print(f"   🔹 Variance Inflation Factors:\n{vif_df.to_string(index=False)}")

        # ----------------------------------------------------------------------
        # Stage 8: CatBoost Training & Ablation Study
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 8] Training CatBoost Regressor & Conducting Ablation Study...")
        fusion_model = CatBoostFusionModel()
        ablation_res = fusion_model.run_ablation_study(df_ml, split_date="2023-01-01")
        if verbose:
            b_rmse = ablation_res["baseline"]["test_rmse"]
            a_rmse = ablation_res["augmented"]["test_rmse"]
            pct_imp = ablation_res["improvement"]["rmse_reduction_pct"]
            print(f"   ✅ CatBoost Ablation Results (Out-of-Sample 2023–2026):")
            print(f"      - Baseline Model Test RMSE:  {b_rmse:.6f}")
            print(f"      - Augmented Model Test RMSE: {a_rmse:.6f}")
            print(f"      - Error Reduction via Causal Features: {pct_imp:+.3f}%")
            print(f"   🔹 Top Feature Importances:\n{ablation_res['feature_importance_df'].head(6).to_string(index=False)}")

        # ----------------------------------------------------------------------
        # Stage 9: Downstream Strategy Backtest
        # ----------------------------------------------------------------------
        if verbose:
            print("\n[Stage 9] Executing Causal Contagion-Pruning Strategy Backtest...")
        strategy = CausalContagionStrategy()
        strategy_res = strategy.backtest(
            returns_df=returns_df,
            rolling_dagma_results=dagma_results,
            structural_breaks_df=breaks_df,
            drift_series=drift_series,
        )
        if verbose:
            stats = strategy_res["stats_summary"]
            print(f"   ✅ Strategy Comparison Results:")
            for strat_name, s in stats.items():
                print(f"      - {strat_name:<25}: Return={s['Annualized_Return']:>5.1f}%, Sharpe={s['Sharpe_Ratio']:>5.3f}, MaxDD={s['Max_Drawdown']:>5.1f}%")

        # ----------------------------------------------------------------------
        # Stage 10: Emit High-Resolution Diagnostic Figures
        # ----------------------------------------------------------------------
        if verbose:
            print(f"\n[Stage 10] Emitting Publication-Grade Diagnostic Charts to {self.output_dir}...")

        fig1_path = self.output_dir / "v2_fig1_ff3_residualization_audit.png"
        plot_ff3_residualization_audit(returns_df, residuals_df, exposure_df, fig1_path)

        fig3_path = self.output_dir / "v2_fig3_empirical_orthogonality_precision_matrix.png"
        plot_empirical_orthogonality_benchmark(orthogonality_df, vif_df, fig3_path)

        fig4_path = self.output_dir / "v2_fig4_structural_break_causal_drift.png"
        plot_structural_breaks_causal_drift(drift_series, breaks_df, break_detector.threshold_, fig4_path)

        fig5_path = self.output_dir / "v2_fig5_catboost_ablation_and_feature_importance.png"
        plot_catboost_ablation_and_importance(ablation_res, fig5_path)

        fig6_path = self.output_dir / "v2_fig6_contagion_pruning_strategy_equity_drawdown.png"
        plot_contagion_strategy_performance(strategy_res, fig6_path)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n🎉 VERSION 2 PIPELINE EXECUTION COMPLETED IN {elapsed:.1f}s!")
            print("=" * 85)

        return {
            "returns_df": returns_df,
            "residuals_df": residuals_df,
            "exposure_df": exposure_df,
            "dagma_results": dagma_results,
            "precision_results": precision_results,
            "breaks_df": breaks_df,
            "drift_series": drift_series,
            "df_ml": df_ml,
            "orthogonality_df": orthogonality_df,
            "vif_df": vif_df,
            "ablation_results": ablation_res,
            "strategy_results": strategy_res,
            "elapsed_seconds": elapsed,
        }
