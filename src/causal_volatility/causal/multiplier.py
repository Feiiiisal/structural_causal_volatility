"""
Dynamic Adaptive Risk Multiplier Engine.

Translates discovered causal drivers and time-lagged shock coefficients into an
adaptive risk multiplier lambda_t. Dynamically contracts position bandwidth during
systemic macro stress and widens it during calm volatility regimes.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def construct_causal_multiplier(
    df_causal: pd.DataFrame,
    causal_outputs: Optional[Dict[str, Any]] = None,
    baseline_multiplier: float = 2.0,
    min_multiplier: float = 1.3,
    lookback: int = 252,
    ewm_span: int = 10,
    min_periods: int = 20,
    alpha_thresh: float = 0.05,
    lambda_0: Optional[float] = None,
    lambda_min: Optional[float] = None,
    outputs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> pd.Series:
    """
    Construct dynamic adaptive risk multiplier series lambda_t bounded in [lambda_min, lambda_0].

    Methodology:
        1. Identifies active causal driver channels where p_val < alpha_thresh.
        2. Computes composite risk shock:
           C_t = (1 / K) * sum_{(i, tau) in Edges} |beta_{i, tau}| * |z_{i, t-tau}|
           where z_{i, t-tau} is the standardized shock series shifted by tau.
        3. Applies exponential moving average smoothing:
           S_t = EWM(C_t, span=ewm_span)
        4. Calculates rolling empirical percentile rank:
           pct_t = percentile_rank(S_t, lookback=lookback, min_periods=min_periods)
        5. Scales to adaptive risk multiplier:
           lambda_t = lambda_0 - (lambda_0 - lambda_min) * pct_t
           clipped strictly to [lambda_min, lambda_0].

    Args:
        df_causal: Clean DataFrame containing all driver series.
        causal_outputs: Causal discovery result dictionary (with p_matrix, val_matrix, etc.).
        baseline_multiplier: Multiplier under zero systemic shock (lambda_0, default: 4.5).
        min_multiplier: Multiplier under maximum crisis stress (lambda_min, default: 1.8).
        lookback: Rolling lookback window for percentile ranking (default: 126).
        ewm_span: Exponential moving average smoothing span (default: 5).
        min_periods: Minimum observations before percentile evaluation starts (default: 20).
        alpha_thresh: Significance threshold for active edges (default: 0.05).
        lambda_0: Optional alias for baseline_multiplier.
        lambda_min: Optional alias for min_multiplier.
        outputs: Optional alias for causal_outputs.
        **kwargs: Additional parameters for backward compatibility (e.g. causal_output).

    Returns:
        pd.Series of dynamic multipliers indexed to df_causal.index, strictly in [1.8, 4.5].
    """
    # Harmonize parameter aliases
    if causal_outputs is None:
        if outputs is not None:
            causal_outputs = outputs
        elif "causal_output" in kwargs:
            causal_outputs = kwargs["causal_output"]
        else:
            raise ValueError("causal_outputs must be provided to construct_causal_multiplier.")

    if lambda_0 is not None:
        baseline_multiplier = lambda_0
    if lambda_min is not None:
        min_multiplier = lambda_min

    var_names = causal_outputs.get("var_names", list(df_causal.columns))
    p_matrix = causal_outputs.get("p_matrix")
    val_matrix = causal_outputs.get("val_matrix")
    tau_max = causal_outputs.get("tau_max", 5)

    target_var = causal_outputs.get("target_var", "Vol_Innovations")
    target_idx = var_names.index(target_var) if target_var in var_names else 0

    n = len(df_causal)
    composite_risk = np.zeros(n, dtype=float)
    active_channels = 0

    if p_matrix is not None and val_matrix is not None:
        for source_idx, source_name in enumerate(var_names):
            if source_name == target_var:
                continue

            for tau in range(1, tau_max + 1):
                p_val = p_matrix[source_idx, target_idx, tau]
                beta = val_matrix[source_idx, target_idx, tau]

                if p_val < alpha_thresh:
                    shock_series = df_causal[source_name].shift(tau).fillna(0.0).values
                    shock_std = shock_series.std()
                    z_shock = shock_series / shock_std if shock_std > 1e-12 else np.zeros(n)
                    composite_risk += np.abs(beta) * np.abs(z_shock)
                    active_channels += 1

    if active_channels > 0:
        composite_risk /= active_channels

    risk_series = pd.Series(composite_risk, index=df_causal.index)
    smoothed_risk = risk_series.ewm(span=ewm_span, min_periods=1).mean()

    # Empirical percentile rank over rolling lookback window
    danger_pct = smoothed_risk.rolling(lookback, min_periods=min_periods).apply(
        lambda w: (w.iloc[-1] > w).mean(), raw=False
    ).fillna(0.0).values

    dynamic_multipliers = baseline_multiplier - (baseline_multiplier - min_multiplier) * danger_pct
    dynamic_multipliers = np.clip(dynamic_multipliers, min_multiplier, baseline_multiplier)

    return pd.Series(dynamic_multipliers, index=df_causal.index, name="Causal_Multiplier")


class CausalMultiplier:
    """
    Object-oriented Dynamic Adaptive Risk Multiplier calculator.
    """

    def __init__(
        self,
        baseline_multiplier: float = 4.5,
        min_multiplier: float = 1.8,
        lookback: int = 126,
        ewm_span: int = 5,
        min_periods: int = 20,
        alpha_thresh: float = 0.05,
    ):
        self.baseline_multiplier = baseline_multiplier
        self.min_multiplier = min_multiplier
        self.lookback = lookback
        self.ewm_span = ewm_span
        self.min_periods = min_periods
        self.alpha_thresh = alpha_thresh

    def compute(self, df_causal: pd.DataFrame, causal_output: Dict[str, Any]) -> pd.Series:
        """Compute multiplier series for given causal discovery output."""
        return construct_causal_multiplier(
            df_causal=df_causal,
            causal_outputs=causal_output,
            baseline_multiplier=self.baseline_multiplier,
            min_multiplier=self.min_multiplier,
            lookback=self.lookback,
            ewm_span=self.ewm_span,
            min_periods=self.min_periods,
            alpha_thresh=self.alpha_thresh,
        )
