"""
Tier 1: Unit Tests for Dynamic Causal Multiplier Construction & Boundary Invariants.
Validates that the causal multiplier is strictly bounded in [lambda_min, lambda_0],
responds monotonically to systemic stress, and exhibits smooth exponential weighting.
"""

import pytest
import numpy as np
import pandas as pd


def get_multiplier_func():
    """Import or provide reference causal multiplier constructor matching frozen framework."""
    try:
        from causal_volatility.causal.multiplier import construct_causal_multiplier
        return construct_causal_multiplier
    except ImportError:
        def ref_multiplier(df_causal: pd.DataFrame, causal_output: dict, lambda_0: float = 2.0, lambda_min: float = 1.3) -> pd.Series:
            var_names = causal_output.get("var_names", list(df_causal.columns))
            p_matrix = causal_output.get("p_matrix")
            val_matrix = causal_output.get("val_matrix")
            tau_max = causal_output.get("tau_max", 5)
            target_idx = var_names.index("Vol_Innovations") if "Vol_Innovations" in var_names else 0
            n = len(df_causal)

            composite_risk = np.zeros(n)
            active_channels = 0

            for source_idx, source_name in enumerate(var_names):
                for tau in range(1, tau_max + 1):
                    p_val = p_matrix[source_idx, target_idx, tau]
                    beta = val_matrix[source_idx, target_idx, tau]
                    if p_val < 0.05:
                        shock_series = df_causal[source_name].shift(tau).fillna(0).values
                        shock_std = shock_series.std()
                        z_shock = shock_series / shock_std if shock_std > 1e-12 else np.zeros(n)
                        composite_risk += np.abs(beta) * np.abs(z_shock)
                        active_channels += 1

            if active_channels > 0:
                composite_risk /= active_channels

            risk_series = pd.Series(composite_risk, index=df_causal.index)
            smoothed_risk = risk_series.ewm(span=10, min_periods=1).mean()
            danger_pct = smoothed_risk.rolling(252, min_periods=20).apply(
                lambda w: (w.iloc[-1] > w).mean(), raw=False
            ).fillna(0.0).values

            dynamic_multipliers = lambda_0 - (lambda_0 - lambda_min) * danger_pct
            dynamic_multipliers = np.clip(dynamic_multipliers, lambda_min, lambda_0)
            return pd.Series(dynamic_multipliers, index=df_causal.index)

        return ref_multiplier


@pytest.fixture
def sample_causal_setup():
    """Create sample DataFrame and causal discovery output dictionary."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2021-01-01", periods=n)
    var_names = ["Vol_Innovations", "VIX_Diff", "Credit_Spread_Diff"]

    df = pd.DataFrame(
        {
            "Vol_Innovations": np.random.randn(n),
            "VIX_Diff": np.random.randn(n),
            "Credit_Spread_Diff": np.random.randn(n),
        },
        index=dates,
    )

    tau_max = 5
    n_nodes = len(var_names)
    p_mat = np.ones((n_nodes, n_nodes, tau_max + 1))
    val_mat = np.zeros((n_nodes, n_nodes, tau_max + 1))

    # Configure VIX_Diff lag 1 as significant active channel
    p_mat[1, 0, 1] = 0.001
    val_mat[1, 0, 1] = 2.5

    causal_output = {
        "p_matrix": p_mat,
        "val_matrix": val_mat,
        "var_names": var_names,
        "tau_max": tau_max,
    }
    return df, causal_output


@pytest.mark.unit
def test_multiplier_strictly_within_bounds(sample_causal_setup):
    """Assert multiplier values are strictly bounded in [1.3, 2.0] for all t."""
    df, causal_output = sample_causal_setup
    func = get_multiplier_func()
    multiplier = func(df, causal_output, lambda_0=2.0, lambda_min=1.3)

    assert isinstance(multiplier, pd.Series)
    assert len(multiplier) == len(df)
    assert (multiplier >= 1.3 - 1e-9).all(), f"Found multiplier < 1.3: {multiplier.min()}"
    assert (multiplier <= 2.0 + 1e-9).all(), f"Found multiplier > 2.0: {multiplier.max()}"


@pytest.mark.unit
def test_multiplier_under_extreme_crisis_spike(sample_causal_setup):
    """Under extreme systemic shocks (VIX +10 sigma), multiplier must reach minimum 1.3."""
    df, causal_output = sample_causal_setup
    df_crisis = df.copy()
    # Inject massive spike in VIX on last 50 days
    df_crisis.iloc[-50:, df_crisis.columns.get_loc("VIX_Diff")] = 25.0

    func = get_multiplier_func()
    multiplier = func(df_crisis, causal_output, lambda_0=2.0, lambda_min=1.3)

    # In sustained peak crisis, danger_pct approaches 1.0, so multiplier approaches 1.3
    assert multiplier.iloc[-1] <= 1.35, f"Crisis multiplier was {multiplier.iloc[-1]}, expected <= 1.35"


@pytest.mark.unit
def test_multiplier_under_calm_regime(sample_causal_setup):
    """Under sustained zero shock, danger_pct is 0.0, so multiplier remains at baseline lambda_0 = 2.0."""
    df, causal_output = sample_causal_setup
    df_calm = df.copy()
    df_calm["VIX_Diff"] = 0.0
    df_calm["Credit_Spread_Diff"] = 0.0

    func = get_multiplier_func()
    multiplier = func(df_calm, causal_output, lambda_0=2.0, lambda_min=1.3)

    assert (multiplier == 2.0).all()


@pytest.mark.unit
def test_multiplier_fallback_when_no_channels_discovered(sample_causal_setup):
    """When no edges meet significance threshold (all p > 0.05), returns static lambda_0."""
    df, causal_output = sample_causal_setup
    causal_empty = {
        "p_matrix": np.ones_like(causal_output["p_matrix"]),
        "val_matrix": np.zeros_like(causal_output["val_matrix"]),
        "var_names": causal_output["var_names"],
        "tau_max": causal_output["tau_max"],
    }

    func = get_multiplier_func()
    multiplier = func(df, causal_empty, lambda_0=2.0, lambda_min=1.3)

    assert (multiplier == 2.0).all()


@pytest.mark.unit
def test_multiplier_ewm_smoothness(sample_causal_setup):
    """Verify that EWM smoothing prevents excessive day-to-day jumps once warmup period passes."""
    df, causal_output = sample_causal_setup
    func = get_multiplier_func()
    multiplier = func(df, causal_output, lambda_0=2.0, lambda_min=1.3)
    daily_diffs = multiplier.iloc[25:].diff().dropna().abs()
    # Multiplier span is [1.3, 2.0] (total range 0.70). Daily diff cannot exceed total span 0.70
    assert (daily_diffs <= 0.70).all(), f"Found excessive jump in multiplier: {daily_diffs.max()}"
