"""Run the main capstone pipeline."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from catboost_dagma.benchmark.orthogonality import NetworkFeatureAudit
from catboost_dagma.benchmark.precision_matrix import PrecisionMatrixEstimator
from catboost_dagma.breaks.structural_breaks import StructuralBreakDetector
from catboost_dagma.config import DAGMA_CONFIG, DEFAULT_SECTOR_ASSETS, FIGURES_DIR
from catboost_dagma.dagma.solver import run_parallel_rolling_dagma
from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer
from catboost_dagma.ml.catboost_fusion import CatBoostFusionModel
from catboost_dagma.ml.feature_engineering import FusionFeatureEngineer
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
        factor_path: Optional[Union[str, Path]] = None,
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
        self.factor_path = Path(factor_path) if factor_path else None
        self.offline = offline
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        started = time.time()

        loader = UnifiedDataLoader(self.sp100_path, self.factor_path, self.offline)
        returns, factors = loader.get_aligned_panel(
            self.assets, self.start_date, self.end_date
        )
        self._log(verbose, f"data: {returns.shape}")

        residualizer = FamaFrenchResidualizer()
        residuals = residualizer.fit_transform_walk_forward(
            returns,
            factors,
            min_train_size=min(252, max(60, self.window_size)),
            refit_every=self.step_size,
        )
        residual_vol = residuals.abs()
        self._log(verbose, "factor adjustment done")

        dagma = run_parallel_rolling_dagma(
            residual_vol.values,
            list(residual_vol.index),
            window_size=self.window_size,
            step_size=self.step_size,
            dagma_kwargs={
                "loss_type": "student-t",
                "learnable_nu": True,
                "warm_iter": 75,
                "max_iter": 150,
                "lambda1": 0.02,
                "lr": 0.005,
                "verbose": False,
            },
            n_workers=self.n_workers,
        )
        if not dagma:
            raise RuntimeError("No DAGMA windows were fitted.")
        self._log(verbose, f"DAGMA windows: {len(dagma)}")

        precision = self._precision_results(residual_vol, dagma)

        detector = StructuralBreakDetector(
            percentile_threshold=95.0,
            min_history=20,
            history_window=252,
        )
        drift = detector.compute_causal_drift(dagma)
        breaks = detector.fit_detect(drift)
        break_features = detector.generate_break_features(drift)

        features = FusionFeatureEngineer(forward_horizon=5).build_ml_dataset(
            returns, dagma, precision, break_features
        )

        audit = NetworkFeatureAudit()
        network_comparison = audit.audit_topological_correlations(features)
        vif = audit.audit_variance_inflation_factors(features)

        catboost = CatBoostFusionModel(target_col="forward_volatility")
        ablation = catboost.run_ablation_study(features, split_date="2021-01-01")

        strategy = CausalContagionStrategy()
        strategy_results = strategy.backtest(
            returns,
            dagma,
            breaks,
            drift_series=drift,
            evaluation_start="2021-01-01",
        )

        elapsed = time.time() - started
        self._log(verbose, f"finished in {elapsed:.1f}s")

        return {
            "returns_df": returns,
            "factors_df": factors,
            "residuals_df": residuals,
            "residual_volatility_df": residual_vol,
            "exposure_df": residualizer.get_factor_exposure_report(),
            "dagma_results": dagma,
            "precision_results": precision,
            "breaks_df": breaks,
            "break_features_df": break_features,
            "drift_series": drift,
            "df_ml": features,
            "network_comparison_df": network_comparison,
            "vif_df": vif,
            "ablation_results": ablation,
            "strategy_results": strategy_results,
            "elapsed_seconds": elapsed,
        }

    def _precision_results(self, residual_vol, dagma):
        estimator = PrecisionMatrixEstimator()
        results = []

        for item in dagma:
            end = item["end_idx"]
            window = residual_vol.iloc[end - self.window_size:end].values
            matrix, alpha = estimator.fit_precision_matrix(window)
            topology = estimator.extract_topologies(matrix)
            topology["alpha"] = alpha
            results.append(topology)

        return results

    @staticmethod
    def _log(verbose, message):
        if verbose:
            print(message)
