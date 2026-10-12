"""CatBoost feature ablation."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost_dagma.config import CATBOOST_CONFIG


class CatBoostFusionModel:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        target_col: str = "forward_volatility",
    ):
        self.config = (config or CATBOOST_CONFIG).copy()
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

        self.model_baseline_ = None
        self.model_augmented_ = None
        self.ablation_results_: Dict[str, Any] = {}

    def _model(self) -> CatBoostRegressor:
        params = self.config.copy()
        params.pop("train_ratio", None)
        params.pop("validation_ratio", None)
        params = {key: value for key, value in params.items() if value is not None}

        try:
            return CatBoostRegressor(**params)
        except Exception:
            params["task_type"] = "CPU"
            params.pop("devices", None)
            params.pop("gpu_ram_part", None)
            params["thread_count"] = 8
            return CatBoostRegressor(**params)

    @staticmethod
    def _metrics(y, pred):
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }

    def _fit(self, train, validation, test, features):
        model = self._model()
        kwargs = {"verbose": False}

        if not validation.empty:
            kwargs["eval_set"] = (
                validation[features],
                validation[self.target_col].values,
            )
            kwargs["use_best_model"] = True

        model.fit(train[features], train[self.target_col].values, **kwargs)
        return (
            model,
            model.predict(train[features]),
            model.predict(test[features]),
        )

    def run_ablation_study(
        self,
        df_ml: pd.DataFrame,
        split_date: Optional[str] = "2021-01-01",
    ) -> Dict[str, Any]:
        needed = self.augmented_features + [self.target_col, "date"]
        missing = [col for col in needed if col not in df_ml]
        if missing:
            raise ValueError(f"Missing ML columns: {missing}")

        df = df_ml.copy()
        df["date"] = pd.to_datetime(df["date"])
        sort_cols = ["date", "asset"] if "asset" in df else ["date"]
        df = df.sort_values(sort_cols).replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[self.target_col])

        pretest, test = self._split_test(df, split_date)
        train, validation = self._split_validation(pretest)

        if train.empty or test.empty:
            raise ValueError("Train or test data is empty.")

        self.model_baseline_, pred_train_base, pred_test_base = self._fit(
            train, validation, test, self.baseline_features
        )
        self.model_augmented_, pred_train_aug, pred_test_aug = self._fit(
            train, validation, test, self.augmented_features
        )

        y_train = train[self.target_col].values
        y_test = test[self.target_col].values

        base_train = self._metrics(y_train, pred_train_base)
        base_test = self._metrics(y_test, pred_test_base)
        aug_train = self._metrics(y_train, pred_train_aug)
        aug_test = self._metrics(y_test, pred_test_aug)

        improvement = np.nan
        if base_test["rmse"] > 0:
            improvement = 100 * (base_test["rmse"] - aug_test["rmse"]) / base_test["rmse"]

        importance = pd.DataFrame({
            "feature": self.augmented_features,
            "importance": self.model_augmented_.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        importance["category"] = importance["feature"].map(self._category)

        self.ablation_results_ = {
            "target": self.target_col,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "test_samples": len(test),
            "train_date_range": self._date_range(train),
            "validation_date_range": self._date_range(validation),
            "test_date_range": self._date_range(test),
            "baseline": {
                "train_rmse": base_train["rmse"],
                "test_rmse": base_test["rmse"],
                "test_mae": base_test["mae"],
                "test_r2": base_test["r2"],
            },
            "augmented": {
                "train_rmse": aug_train["rmse"],
                "test_rmse": aug_test["rmse"],
                "test_mae": aug_test["mae"],
                "test_r2": aug_test["r2"],
            },
            "improvement": {
                "rmse_reduction_pct": float(improvement),
                "delta_r2": float(aug_test["r2"] - base_test["r2"]),
            },
            "feature_importance_df": importance,
            "test_df": test,
            "pred_test_aug": pred_test_aug,
            "pred_test_base": pred_test_base,
        }
        return self.ablation_results_

    def _split_test(self, df, split_date):
        if split_date:
            split = pd.Timestamp(split_date)
        else:
            dates = np.array(sorted(df["date"].unique()))
            cut = int(len(dates) * self.config.get("train_ratio", 0.70))
            split = pd.Timestamp(dates[cut])

        return df[df["date"] < split].copy(), df[df["date"] >= split].copy()

    def _split_validation(self, pretest):
        dates = np.array(sorted(pretest["date"].unique()))
        if len(dates) < 5:
            return pretest, pretest.iloc[0:0].copy()

        ratio = float(self.config.get("validation_ratio", 0.20))
        count = max(1, int(len(dates) * ratio))
        split = pd.Timestamp(dates[-count])

        return (
            pretest[pretest["date"] < split].copy(),
            pretest[pretest["date"] >= split].copy(),
        )

    @staticmethod
    def _date_range(df):
        if df.empty:
            return None
        return str(df["date"].min()), str(df["date"].max())

    def _category(self, feature):
        if feature in self.causal_features:
            return "Network"
        if "precision" in feature:
            return "Precision"
        if "corr" in feature:
            return "Correlation"
        return "Market"
