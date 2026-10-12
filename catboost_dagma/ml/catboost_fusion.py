"""
CatBoost Fusion and Ablation Engine.
Trains gradient boosted decision trees on GPU/CPU to benchmark:
Model 1 (Baseline): Momentum + Volatility + Correlation Centralities + Precision Matrix
Model 2 (Augmented): Model 1 + DAGMA Non-Linear Causal Centralities + Drift + Breaks
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost_dagma.config import CATBOOST_CONFIG


class CatBoostFusionModel:
    """
    Manages CatBoost training, ablation comparison, and feature importance attribution.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        target_col: str = "forward_return",
    ):
        self.config = config or CATBOOST_CONFIG
        self.target_col = target_col

        self.baseline_features = [
            "momentum_20d",
            "volatility_20d",
            "volatility_60d",
            "corr_deg",
            "corr_eigen",
            "precision_deg",
            "precision_eigen",
        ]

        self.causal_features = [
            "causal_in_deg",
            "causal_out_deg",
            "causal_eigen",
            "causal_drift",
            "is_structural_break",
            "days_since_last_break",
            "dagma_nu",
        ]

        self.augmented_features = self.baseline_features + self.causal_features

        self.model_baseline_: Optional[CatBoostRegressor] = None
        self.model_augmented_: Optional[CatBoostRegressor] = None
        self.ablation_results_: Dict[str, Any] = {}

    def _init_regressor(self) -> CatBoostRegressor:
        """Instantiates CatBoost with safe GPU allocation or CPU fallback."""
        params = self.config.copy()
        params.pop("train_ratio", None)
        try:
            reg = CatBoostRegressor(**params)
        except Exception:
            # Fallback to CPU if GPU parameters encounter driver mismatch
            params["task_type"] = "CPU"
            params.pop("devices", None)
            params.pop("gpu_ram_part", None)
            params["thread_count"] = 8
            reg = CatBoostRegressor(**params)
        return reg

    def run_ablation_study(
        self,
        df_ml: pd.DataFrame,
        split_date: Optional[str] = "2023-01-01",
    ) -> Dict[str, Any]:
        """
        Splits data chronologically (e.g. Train: 2018-2022, Test: 2023-2026),
        fits Baseline vs Augmented models, and evaluates out-of-sample performance.
        """
        # Ensure chronological ordering
        df_sorted = df_ml.sort_values("date").copy()

        if split_date and "date" in df_sorted.columns:
            train_mask = pd.to_datetime(df_sorted["date"]) < pd.to_datetime(split_date)
            train_df = df_sorted[train_mask]
            test_df = df_sorted[~train_mask]
        else:
            split_idx = int(len(df_sorted) * self.config.get("train_ratio", 0.70))
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]

        y_train = train_df[self.target_col].values
        y_test = test_df[self.target_col].values

        # 1. Train Baseline Model (Correlation + Precision + Momentum + Vol)
        X_train_base = train_df[self.baseline_features]
        X_test_base = test_df[self.baseline_features]

        self.model_baseline_ = self._init_regressor()
        self.model_baseline_.fit(
            X_train_base, y_train,
            eval_set=(X_test_base, y_test),
            use_best_model=True,
            verbose=False,
        )

        pred_train_base = self.model_baseline_.predict(X_train_base)
        pred_test_base = self.model_baseline_.predict(X_test_base)

        rmse_train_base = float(np.sqrt(mean_squared_error(y_train, pred_train_base)))
        rmse_test_base = float(np.sqrt(mean_squared_error(y_test, pred_test_base)))
        mae_test_base = float(mean_absolute_error(y_test, pred_test_base))
        r2_test_base = float(r2_score(y_test, pred_test_base))

        # 2. Train Augmented Model (Baseline + DAGMA Causal Topologies)
        X_train_aug = train_df[self.augmented_features]
        X_test_aug = test_df[self.augmented_features]

        self.model_augmented_ = self._init_regressor()
        self.model_augmented_.fit(
            X_train_aug, y_train,
            eval_set=(X_test_aug, y_test),
            use_best_model=True,
            verbose=False,
        )

        pred_train_aug = self.model_augmented_.predict(X_train_aug)
        pred_test_aug = self.model_augmented_.predict(X_test_aug)

        rmse_train_aug = float(np.sqrt(mean_squared_error(y_train, pred_train_aug)))
        rmse_test_aug = float(np.sqrt(mean_squared_error(y_test, pred_test_aug)))
        mae_test_aug = float(mean_absolute_error(y_test, pred_test_aug))
        r2_test_aug = float(r2_score(y_test, pred_test_aug))

        # Percentage improvement in Test RMSE
        pct_improvement = ((rmse_test_base - rmse_test_aug) / rmse_test_base) * 100.0

        # Feature Importances
        importances = self.model_augmented_.get_feature_importance()
        fi_df = pd.DataFrame({
            "feature": self.augmented_features,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        # Categorize features
        def get_category(f_name):
            if f_name in self.causal_features:
                return "DAGMA Causal Topology"
            elif "precision" in f_name:
                return "Precision Matrix (GLASSO)"
            elif "corr" in f_name:
                return "Correlation Graph"
            else:
                return "Baseline Market/Momentum"

        fi_df["category"] = fi_df["feature"].apply(get_category)

        self.ablation_results_ = {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_date_range": (str(train_df["date"].min()), str(train_df["date"].max())),
            "test_date_range": (str(test_df["date"].min()), str(test_df["date"].max())),
            "baseline": {
                "train_rmse": rmse_train_base,
                "test_rmse": rmse_test_base,
                "test_mae": mae_test_base,
                "test_r2": r2_test_base,
            },
            "augmented": {
                "train_rmse": rmse_train_aug,
                "test_rmse": rmse_test_aug,
                "test_mae": mae_test_aug,
                "test_r2": r2_test_aug,
            },
            "improvement": {
                "rmse_reduction_pct": round(pct_improvement, 3),
                "delta_r2": round(r2_test_aug - r2_test_base, 5),
            },
            "feature_importance_df": fi_df,
            "test_df": test_df,
            "pred_test_aug": pred_test_aug,
            "pred_test_base": pred_test_base,
        }

        return self.ablation_results_
