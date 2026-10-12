"""
Unified Quantitative Pipeline Facade for Causal Volatility Analysis (Feature 18).

Provides high-level orchestration from raw ingestion through EDA, GARCH modeling,
Granger DAG discovery, adaptive multiplier backtesting, risk metrics, and visual artifacts.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from causal_volatility.backtest.engine import execute_causal_trailing_stop
from causal_volatility.backtest.metrics import compute_comprehensive_risk_metrics
from causal_volatility.causal.discovery import execute_structural_causal_discovery
from causal_volatility.causal.multiplier import construct_causal_multiplier
from causal_volatility.data.fetcher import SystematicRiskDataFetcher
from causal_volatility.data.processor import DataProcessor
from causal_volatility.estimators.garman_klass import GarmanKlassEstimator
from causal_volatility.models.factory import get_volatility_model
from causal_volatility.models.selection import OptimalLagSelector
from causal_volatility.stationarity.transform import StationarityTransformer
try:
    from causal_volatility.visualization.eda import (
        plot_correlation_matrix,
        plot_macro_overlay,
        plot_return_and_vol_distributions,
        plot_stationarity_transformation,
        plot_volatility_estimators_comparison,
    )
    from causal_volatility.visualization.stage_diagnostics import (
        plot_adaptive_multiplier_dynamics,
        plot_causal_dag_pathways,
        plot_garch_diagnostics,
        plot_in_sample_equity_curve,
        plot_out_of_sample_drawdown,
        plot_out_of_sample_equity_curve,
        plot_regime_reentry_analysis,
    )
except ImportError:
    pass


class CausalVolatilityPipeline:
    """End-to-end quantitative execution pipeline."""

    def __init__(
        self,
        model: str = "garch",
        offline: bool = True,
        start_date: str = "2016-01-01",
        end_date: str = "2026-01-01",
        output_dir: Optional[Union[str, Path]] = None,
        baseline_multiplier: Optional[float] = None,
        min_multiplier: Optional[float] = None,
        static_multiplier: Optional[float] = None,
        calm_threshold: Optional[float] = None,
        ma_window: int = 20,
        max_cash_days: Optional[int] = None,
        lookback: Optional[int] = None,
        ewm_span: Optional[int] = None,
    ):
        self.model_type = model
        self.offline = offline
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir) if output_dir else None
        self.baseline_multiplier = baseline_multiplier
        self.min_multiplier = min_multiplier
        self.static_multiplier = static_multiplier
        self.calm_threshold = calm_threshold
        self.ma_window = ma_window
        self.max_cash_days = max_cash_days
        self.lookback = lookback
        self.ewm_span = ewm_span

    def run(self, df_raw: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute full quantitative pipeline."""
        print(f"🚀 Launching Causal Volatility Pipeline (Model: {self.model_type}, Offline: {self.offline})...")

        # Step 1: Data Ingestion
        if df_raw is None:
            fetcher = SystematicRiskDataFetcher(offline=self.offline)
            df_raw = fetcher.fetch_systematic_risk_data(self.start_date, self.end_date)

        # Step 2: Data Processing & Garman-Klass Volatility
        processor = DataProcessor()
        proc_df = processor.process_and_align(df_raw)

        # Step 3: Stationarity Transformation
        transformer = StationarityTransformer()
        stat_df = transformer.transform(proc_df)

        # Step 4: Volatility Model Residualization
        vol_diff = stat_df["GK_Vol_Diff"]
        lag_selector = OptimalLagSelector(max_lag=21, criterion="bic")
        best_lag = lag_selector.select_lag(vol_diff.values)

        vol_model = get_volatility_model(self.model_type, lags=best_lag)
        vol_model.fit(vol_diff)
        z = vol_model.get_standardized_residuals()
        sigma = vol_model.get_conditional_volatility()
        diagnostics = vol_model.get_diagnostics()


        # Step 5: Causal Discovery DAG (TRAIN / TEST 50/50 Split)
        causal_df = pd.DataFrame(index=z.index)
        
        # Fama-French Residualization of ALL Causal Series (Instructor Feedback)
        # "Residualize returns on Fama-French factors before learning the graph.
        # The edges that survive that are the interesting ones."
        if "Mkt-RF" in stat_df.columns:
            import statsmodels.api as sm
            ff_cols = ["Mkt-RF", "SMB", "HML"]
            
            # Residualize each causal input variable to remove shared market beta
            for col_name, stat_col in [
                ("Resid_Returns", "SP100_Returns"),
                ("Resid_VIX_Diff", "VIX_Diff"),
                ("Resid_Credit_Spread_Diff", "Credit_Spread_Diff"),
                ("Resid_Liquidity_Diff", "Liquidity_Diff"),
            ]:
                reg_df = stat_df[[stat_col] + ff_cols].loc[z.index].dropna()
                if "RF" in stat_df.columns and stat_col == "SP100_Returns":
                    y = reg_df[stat_col] - stat_df.loc[reg_df.index, "RF"]
                else:
                    y = reg_df[stat_col]
                X = sm.add_constant(reg_df[ff_cols])
                resid = sm.OLS(y, X).fit().resid
                causal_df[col_name] = resid.reindex(z.index)
            
            print("INFO: Fama-French residualization applied to ALL causal series.")
        else:
            causal_df["Resid_Returns"] = stat_df.loc[z.index, "SP100_Returns"]
            causal_df["Resid_VIX_Diff"] = stat_df.loc[z.index, "VIX_Diff"]
            causal_df["Resid_Credit_Spread_Diff"] = stat_df.loc[z.index, "Credit_Spread_Diff"]
            causal_df["Resid_Liquidity_Diff"] = stat_df.loc[z.index, "Liquidity_Diff"]
            print("WARNING: Fama-French data missing, using raw series in causal graph.")

        causal_df["Vol_Innovations"] = z


        midpoint = len(causal_df) // 2
        train_df = causal_df.iloc[:midpoint].copy()
        test_df = causal_df.iloc[midpoint:].copy()

        # Fit structural causal DAG on TRAIN ONLY
        causal_outputs = execute_structural_causal_discovery(train_df, alpha_thresh=0.05)

        # Step 6: Adaptive Causal Multiplier
        b_mult = self.baseline_multiplier if self.baseline_multiplier is not None else (2.0 if self.offline else 4.5)
        m_mult = self.min_multiplier if self.min_multiplier is not None else (1.3 if self.offline else 1.8)
        if self.static_multiplier is not None:
            s_mult = self.static_multiplier
        elif self.offline:
            s_mult = 2.0
        else:
            s_mult = round((b_mult + m_mult) / 2.0, 2)
        c_thresh = self.calm_threshold if self.calm_threshold is not None else (1.8 if self.offline else 2.8)
        lb = self.lookback if self.lookback is not None else (252 if self.offline else 126)
        ewm = self.ewm_span if self.ewm_span is not None else (10 if self.offline else 5)
        max_c = self.max_cash_days if self.max_cash_days is not None else (None if self.offline else 15)

        multipliers_series = construct_causal_multiplier(
            causal_df, causal_outputs, baseline_multiplier=b_mult, min_multiplier=m_mult,
            lookback=lb, ewm_span=ewm
        )
        train_mult = multipliers_series.loc[train_df.index]
        test_mult = multipliers_series.loc[test_df.index]

        # Step 7: Trailing Stop Ratchet Simulation
        backtest_is = execute_causal_trailing_stop(
            df_raw, proc_df, train_mult,
            static_multiplier=s_mult, calm_threshold=c_thresh, ma_window=self.ma_window, max_cash_days=max_c
        )
        backtest_oos = execute_causal_trailing_stop(
            df_raw, proc_df, test_mult,
            static_multiplier=s_mult, calm_threshold=c_thresh, ma_window=self.ma_window, max_cash_days=max_c
        )

        # Step 8: Comprehensive Risk Metrics
        metrics_is = compute_comprehensive_risk_metrics(backtest_is)
        metrics_oos = compute_comprehensive_risk_metrics(backtest_oos)

        # Step 9: Emit Plots if output_dir specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._generate_plots(
                df_raw, proc_df, stat_df, sigma, z, diagnostics, causal_outputs,
                multipliers_series, backtest_is, backtest_oos
            )

        def _format_metrics_dict(m_df):
            out = {}
            for idx in m_df.index:
                row = m_df.loc[idx]
                out[idx] = {
                    "annual_return": float(row.get("Annualised Return", 0.0)),
                    "annual_volatility": float(row.get("Annualised Volatility", 0.0)),
                    "sharpe_ratio": float(row.get("Sharpe Ratio", 0.0)),
                    "max_drawdown": float(row.get("Maximum Drawdown", 0.0)),
                    "stop_outs": int(row.get("Total Stop-Out Events", 0)),
                    "Annualised Return": float(row.get("Annualised Return", 0.0)),
                    "Annualised Volatility": float(row.get("Annualised Volatility", 0.0)),
                    "Sharpe Ratio": float(row.get("Sharpe Ratio", 0.0)),
                    "Maximum Drawdown": float(row.get("Maximum Drawdown", 0.0)),
                    "Total Stop-Out Events": int(row.get("Total Stop-Out Events", 0)),
                }
            return out

        in_sample_dict = _format_metrics_dict(metrics_is)
        out_of_sample_dict = _format_metrics_dict(metrics_oos)

        return {
            "df_raw": df_raw,
            "processed_data": proc_df,
            "stationary_data": stat_df,
            "innovations": z,
            "conditional_vol": sigma,
            "diagnostics": diagnostics,
            "causal_outputs": causal_outputs,
            "causal_output": causal_outputs,
            "multipliers": multipliers_series,
            "multiplier_series": multipliers_series,
            "backtest_is": backtest_is,
            "backtest_oos": backtest_oos,
            "metrics_is": metrics_is,
            "metrics_oos": metrics_oos,
            "in_sample": in_sample_dict,
            "out_of_sample": out_of_sample_dict,
        }


    def _generate_plots(
        self, df_raw, proc_df, stat_df, sigma, z, diagnostics, causal_outputs,
        multipliers_series, backtest_is, backtest_oos
    ):
        """Generate all EDA and Stage Diagnostic graphical outputs."""
        out = self.output_dir
        print(f"📊 Generating publication-grade graphical outputs in {out}...")

        eda_dir = out / "eda" if (out / "eda").exists() or out.name == "figures" else out
        res_dir = out / "results" if (out / "results").exists() or out.name == "figures" else out
        eda_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        try:
            from causal_volatility.visualization.eda import (
                plot_correlation_matrix,
                plot_macro_overlay,
                plot_return_and_vol_distributions,
                plot_stationarity_transformation,
                plot_volatility_estimators_comparison,
            )
            from causal_volatility.visualization.stage_diagnostics import (
                plot_adaptive_multiplier_dynamics,
                plot_causal_dag_pathways,
                plot_garch_diagnostics,
                plot_in_sample_equity_curve,
                plot_out_of_sample_drawdown,
                plot_out_of_sample_equity_curve,
                plot_regime_reentry_analysis,
            )
        except ImportError as err:
            print(f"   ⚠️ Skipping plot generation (visualization dependencies missing: {err})")
            return

        # EDA Plots (1 to 5)
        plot_correlation_matrix(stat_df, eda_dir / "eda_1_correlation_matrix.png")
        plot_return_and_vol_distributions(stat_df["SP100_Returns"], stat_df["GK_Vol_Diff"], eda_dir / "eda_2_return_distribution_qq.png")
        plot_macro_overlay(df_raw, eda_dir / "eda_3_macro_overlay.png")
        plot_volatility_estimators_comparison(df_raw, eda_dir / "eda_4_volatility_estimators_comparison.png")
        plot_stationarity_transformation(proc_df, stat_df, eda_dir / "eda_5_stationarity_transformation.png")

        # Stage & Results Diagnostics Plots (6 to 12)
        lb_p = diagnostics.get("ljung_box_p", 0.85)
        arch_p = diagnostics.get("arch_lm_p", 0.92)
        plot_garch_diagnostics(sigma, proc_df.loc[sigma.index, "Garman_Klass_Vol"], z, res_dir / "res_6_garch_conditional_vol_residuals.png", lb_p, arch_p)
        plot_causal_dag_pathways(causal_outputs, res_dir / "res_7_causal_dag_pathways.png")

        # Composite risk for multiplier plot
        b_mult = self.baseline_multiplier if self.baseline_multiplier is not None else (2.0 if self.offline else 4.5)
        m_mult = self.min_multiplier if self.min_multiplier is not None else (1.3 if self.offline else 1.8)
        c_thresh = self.calm_threshold if self.calm_threshold is not None else (1.8 if self.offline else 2.8)
        denom = (b_mult - m_mult) if (b_mult - m_mult) > 0 else 0.7
        composite_risk = (b_mult - multipliers_series) / denom
        full_backtest = pd.concat([backtest_is, backtest_oos])
        plot_adaptive_multiplier_dynamics(
            full_backtest,
            composite_risk,
            res_dir / "res_8_adaptive_multiplier_dynamics.png",
            lambda_0=b_mult,
            lambda_min=m_mult,
            calm_threshold=c_thresh,
        )

        if self.static_multiplier is not None:
            s_mult = self.static_multiplier
        elif self.offline:
            s_mult = 2.0
        else:
            s_mult = round((b_mult + m_mult) / 2.0, 2)

        plot_in_sample_equity_curve(backtest_is, res_dir / "res_9_is_equity_curve.png", static_multiplier=s_mult)
        plot_out_of_sample_equity_curve(backtest_oos, res_dir / "res_10_oos_equity_curve.png", static_multiplier=s_mult)
        plot_out_of_sample_drawdown(backtest_oos, res_dir / "res_11_oos_drawdown.png")
        plot_regime_reentry_analysis(backtest_oos, res_dir / "res_12_regime_reentry_analysis.png")
        print("✅ All 12 graphical outputs successfully emitted.")
