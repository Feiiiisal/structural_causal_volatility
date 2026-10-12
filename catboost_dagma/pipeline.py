"""End-to-end research pipeline with chronological safeguards."""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from catboost_dagma.config import DEFAULT_SECTOR_ASSETS, DAGMA_CONFIG, FIGURES_DIR
from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer
from catboost_dagma.dagma.solver import run_parallel_rolling_dagma
from catboost_dagma.benchmark.precision_matrix import PrecisionMatrixEstimator
from catboost_dagma.benchmark.orthogonality import EmpiricalOrthogonalityAuditor
from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector
from catboost_dagma.ml.feature_engineering import FusionFeatureEngineer
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel
from catboost_dagma.strategy.contagion_pruning import CausalContagionStrategy


class CatBoostDagmaPipeline:
    def __init__(
        self,
        assets: Optional[List[str]] = None,
        start_date: str = "2016-01-01",
        end_date: str = "2026-01-01",
        window_size: int = DAGMA_CONFIG["window_size"],
        step_size: int = DAGMA_CONFIG["step_size"],
        n_workers: int = DAGMA_CONFIG["n_workers"],
        sp100_path: Optional[Union[str, Path]] = None,
        offline: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.assets = assets or DEFAULT_SECTOR_ASSETS
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.step_size = step_size
        self.n_workers = n_workers
        self.sp100_path = Path(sp100_path) if sp100_path else None
        self.offline = offline
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        started = time.time()

        # 1. Data
        loader = UnifiedDataLoader(sp100_path=self.sp100_path, offline=self.offline)
        returns_df, factors_df = loader.get_aligned_panel(self.assets, self.start_date, self.end_date)
        if verbose:
            print(f"[1/9] aligned data: {returns_df.shape}, {returns_df.index.min().date()} to {returns_df.index.max().date()}")

        # 2. Factor adjustment. Initial window is historical; later betas are refit without future data.
        residualizer = FamaFrenchResidualizer()
        residuals_df = residualizer.fit_transform_walk_forward(
            returns_df, factors_df,
            min_train_size=min(252, max(60, self.window_size)),
            refit_every=self.step_size,
        )
        exposure_df = residualizer.get_factor_exposure_report()

        # Project title and research question concern volatility transmission, not raw residual-return edges.
        residual_vol_df = residuals_df.abs()
        if verbose:
            print(f"[2/9] factor-adjusted volatility panel: {residual_vol_df.shape}")

        # 3. Directed network
        dagma_kwargs = {
            "loss_type": "student-t", "learnable_nu": True,
            "warm_iter": 75, "max_iter": 150, "lambda1": 0.02,
            "lr": 0.005, "verbose": False,
        }
        dagma_results = run_parallel_rolling_dagma(
            residual_vol_df.values,
            list(residual_vol_df.index),
            window_size=self.window_size,
            step_size=self.step_size,
            dagma_kwargs=dagma_kwargs,
            n_workers=self.n_workers,
        )
        if not dagma_results:
            raise RuntimeError("No rolling DAGMA estimates were produced. Increase sample length or reduce window_size.")
        if verbose:
            print(f"[3/9] DAGMA windows: {len(dagma_results)}")

        # 4. Precision benchmark on exactly the same volatility windows.
        precision_estimator = PrecisionMatrixEstimator()
        precision_results = []
        for res in dagma_results:
            end_idx = res["end_idx"]
            Xw = residual_vol_df.iloc[end_idx-self.window_size:end_idx].values
            precision, alpha = precision_estimator.fit_precision_matrix(Xw)
            topo = precision_estimator.extract_topologies(precision)
            topo["alpha"] = alpha
            precision_results.append(topo)
        if verbose:
            print(f"[4/9] Graphical LASSO windows: {len(precision_results)}")

        # 5. Structural drift with thresholds calibrated from prior drift only.
        break_detector = StructuralBreakDetector(percentile_threshold=95.0, min_history=20, history_window=252)
        drift_series = break_detector.compute_causal_drift(dagma_results)
        breaks_df = break_detector.fit_detect(drift_series)
        break_features_df = break_detector.generate_break_features(drift_series)
        if verbose:
            print(f"[5/9] structural-change flags: {len(breaks_df)}")

        # 6. ML dataset and empirical network comparison.
        engineer = FusionFeatureEngineer(forward_horizon=5)
        df_ml = engineer.build_ml_dataset(returns_df, dagma_results, precision_results, break_features_df)
        auditor = EmpiricalOrthogonalityAuditor()
        network_comparison_df = auditor.audit_topological_correlations(df_ml)
        vif_df = auditor.audit_variance_inflation_factors(df_ml)
        if verbose:
            print(f"[6/9] ML observations: {len(df_ml)}")

        # 7. Volatility-forecast ablation. Test data are not used for early stopping.
        fusion_model = CatBoostFusionModel(target_col="forward_volatility")
        ablation_res = fusion_model.run_ablation_study(df_ml, split_date="2021-01-01")
        if verbose:
            print(f"[7/9] OOS volatility RMSE reduction: {ablation_res['improvement']['rmse_reduction_pct']:.3f}%")

        # 8. Downstream portfolio application, reported primarily for the OOS period.
        strategy = CausalContagionStrategy()
        strategy_res = strategy.backtest(
            returns_df, dagma_results, breaks_df,
            drift_series=drift_series,
            evaluation_start="2021-01-01",
        )
        if verbose:
            print("[8/9] portfolio evaluation complete")

        # 9. Return all research objects. Figure generation remains a separate presentation step.
        elapsed = time.time() - started
        if verbose:
            print(f"[9/9] complete in {elapsed:.1f}s")
        return {
            "returns_df": returns_df,
            "factors_df": factors_df,
            "residuals_df": residuals_df,
            "residual_volatility_df": residual_vol_df,
            "exposure_df": exposure_df,
            "dagma_results": dagma_results,
            "precision_results": precision_results,
            "breaks_df": breaks_df,
            "break_features_df": break_features_df,
            "drift_series": drift_series,
            "df_ml": df_ml,
            "network_comparison_df": network_comparison_df,
            "vif_df": vif_df,
            "ablation_results": ablation_res,
            "strategy_results": strategy_res,
            "elapsed_seconds": elapsed,
        }
