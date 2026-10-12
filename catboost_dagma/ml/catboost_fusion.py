"""CatBoost ablation: conventional features versus conventional + network features."""
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost_dagma.config import CATBOOST_CONFIG


class CatBoostFusionModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None, target_col: str = "forward_volatility"):
        self.config = (config or CATBOOST_CONFIG).copy()
        self.target_col = target_col
        self.baseline_features = [
            "momentum_20d", "volatility_20d", "volatility_60d",
            "corr_deg", "corr_eigen", "precision_deg", "precision_eigen",
        ]
        self.causal_features = [
            "causal_in_deg", "causal_out_deg", "causal_eigen", "causal_drift",
            "is_structural_break", "days_since_last_break", "dagma_nu",
        ]
        self.augmented_features = self.baseline_features + self.causal_features
        self.model_baseline_ = None
        self.model_augmented_ = None
        self.ablation_results_: Dict[str, Any] = {}

    def _init_regressor(self) -> CatBoostRegressor:
        params = self.config.copy()
        params.pop("train_ratio", None)
        params.pop("validation_ratio", None)
        # None-valued GPU args are not valid on every CatBoost build.
        params = {k: v for k, v in params.items() if v is not None}
        try:
            return CatBoostRegressor(**params)
        except Exception:
            params["task_type"] = "CPU"
            params.pop("devices", None)
            params.pop("gpu_ram_part", None)
            params["thread_count"] = min(8, params.get("thread_count", 8))
            return CatBoostRegressor(**params)

    @staticmethod
    def _metrics(y_true, pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        return {
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)),
        }

    def _fit_one(self, train_df, val_df, test_df, features):
        model = self._init_regressor()
        fit_kwargs = {"verbose": False}
        if len(val_df):
            fit_kwargs["eval_set"] = (val_df[features], val_df[self.target_col].values)
            fit_kwargs["use_best_model"] = True
        model.fit(train_df[features], train_df[self.target_col].values, **fit_kwargs)
        pred_train = model.predict(train_df[features])
        pred_test = model.predict(test_df[features])
        return model, pred_train, pred_test

    def run_ablation_study(self, df_ml: pd.DataFrame, split_date: Optional[str] = "2021-01-01") -> Dict[str, Any]:
        """Evaluate on a test period that is never used for early stopping/model selection."""
        required = set(self.augmented_features + [self.target_col, "date"])
        missing = sorted(required - set(df_ml.columns))
        if missing:
            raise ValueError(f"ML dataset missing required columns: {missing}")

        df = df_ml.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "asset"] if "asset" in df.columns else ["date"])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[self.target_col])

        if split_date:
            split_ts = pd.Timestamp(split_date)
            pretest = df[df["date"] < split_ts].copy()
            test_df = df[df["date"] >= split_ts].copy()
        else:
            unique_dates = np.array(sorted(df["date"].unique()))
            cut = int(len(unique_dates) * self.config.get("train_ratio", 0.70))
            split_ts = pd.Timestamp(unique_dates[cut])
            pretest = df[df["date"] < split_ts].copy()
            test_df = df[df["date"] >= split_ts].copy()

        if pretest.empty or test_df.empty:
            raise ValueError("Chronological split produced an empty training or test sample.")

        # Validation comes only from the pre-test period. Split by date so all assets on a date stay together.
        train_dates = np.array(sorted(pretest["date"].unique()))
        val_ratio = float(self.config.get("validation_ratio", 0.20))
        val_n = max(1, int(len(train_dates) * val_ratio)) if len(train_dates) >= 5 else 0
        if val_n:
            val_start = pd.Timestamp(train_dates[-val_n])
            train_df = pretest[pretest["date"] < val_start].copy()
            val_df = pretest[pretest["date"] >= val_start].copy()
        else:
            train_df, val_df = pretest, pretest.iloc[0:0].copy()

        self.model_baseline_, pred_train_base, pred_test_base = self._fit_one(train_df, val_df, test_df, self.baseline_features)
        self.model_augmented_, pred_train_aug, pred_test_aug = self._fit_one(train_df, val_df, test_df, self.augmented_features)

        y_train = train_df[self.target_col].values
        y_test = test_df[self.target_col].values
        b_test = self._metrics(y_test, pred_test_base)
        a_test = self._metrics(y_test, pred_test_aug)
        b_train = self._metrics(y_train, pred_train_base)
        a_train = self._metrics(y_train, pred_train_aug)

        pct_improvement = ((b_test["rmse"] - a_test["rmse"]) / b_test["rmse"] * 100.0) if b_test["rmse"] > 0 else np.nan
        fi = pd.DataFrame({"feature": self.augmented_features, "importance": self.model_augmented_.get_feature_importance()}).sort_values("importance", ascending=False)
        fi["category"] = fi["feature"].map(lambda f: "Causal/network" if f in self.causal_features else ("Precision matrix" if "precision" in f else ("Correlation" if "corr" in f else "Market/volatility")))

        self.ablation_results_ = {
            "target": self.target_col,
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "train_date_range": (str(train_df["date"].min()), str(train_df["date"].max())),
            "validation_date_range": (str(val_df["date"].min()), str(val_df["date"].max())) if len(val_df) else None,
            "test_date_range": (str(test_df["date"].min()), str(test_df["date"].max())),
            "baseline": {"train_rmse": b_train["rmse"], "test_rmse": b_test["rmse"], "test_mae": b_test["mae"], "test_r2": b_test["r2"]},
            "augmented": {"train_rmse": a_train["rmse"], "test_rmse": a_test["rmse"], "test_mae": a_test["mae"], "test_r2": a_test["r2"]},
            "improvement": {"rmse_reduction_pct": float(pct_improvement), "delta_r2": float(a_test["r2"] - b_test["r2"])},
            "feature_importance_df": fi,
            "test_df": test_df,
            "pred_test_aug": pred_test_aug,
            "pred_test_base": pred_test_base,
        }
        return self.ablation_results_
