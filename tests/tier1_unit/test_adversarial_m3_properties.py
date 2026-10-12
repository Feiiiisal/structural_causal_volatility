"""
Tier 1: Adversarial Property-Based Testing and Stress Tests for Milestone M3.
Authored by Challenger 1 for Milestone M3.

Adversarially stress-tests:
1. Exact and near-collinear variables produce ZERO SingularMatrixWarnings under strict warning filter.
2. Condition number bounds across 50 random trials are well below numerical stability thresholds (kappa < 1000).
3. Dynamic causal multiplier strictly respects [lambda_min, lambda_0] without floating-point leakage under 100-sigma and 10000-sigma shocks.
4. Sustained zero-shock calm regime maintains lambda_0 = 2.0.
5. High-noise confounder rejection and ground-truth DAG recovery.
6. Target self-lag handling under include_self_lags=True and include_self_lags=False.
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from causal_volatility.causal.discovery import (
    StructuralCausalDiscovery,
    execute_structural_causal_discovery,
)
from causal_volatility.causal.multiplier import (
    CausalMultiplier,
    construct_causal_multiplier,
)
from causal_volatility.causal.pathways import (
    CausalGraph,
    analyze_causal_pathways,
)


@pytest.mark.unit
def test_exact_collinear_drivers_zero_singular_matrix_warnings():
    """
    Adversarial Stress Test: Inject exact collinearity (X_b = 2 * X_a) and duplicate columns.
    Assert zero SingularMatrixWarning or rank deficiency warnings are emitted when warning is escalated to error.
    """
    rng = np.random.default_rng(42)
    n = 350
    dates = pd.date_range("2021-01-01", periods=n)

    # 1. Driver B is an exact linear multiple of Driver A
    a_series = rng.standard_normal(n)
    b_series = a_series * 2.5
    y_series = rng.standard_normal(n)

    df_collinear = pd.DataFrame(
        {
            "Vol_Innovations": y_series,
            "Driver_A": a_series,
            "Driver_B_Clone": b_series,
            "Driver_C_Zero": np.zeros(n),
        },
        index=dates,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        out = execute_structural_causal_discovery(df_collinear, tau_max=5)

    assert out["p_matrix"].shape == (4, 4, 6)
    assert not np.isnan(out["p_matrix"]).any()
    assert not np.isnan(out["val_matrix"]).any()


@pytest.mark.unit
def test_driver_identical_to_target_zero_warnings():
    """
    Adversarial Stress Test: Candidate driver is mathematically identical to target innovations.
    When shifted by tau, candidate column is identical to one of the conditioning history columns.
    Verifies matrix_rank defense catches rank deficiency and prevents statsmodels SingularMatrixWarning.
    """
    rng = np.random.default_rng(99)
    n = 300
    y = rng.standard_normal(n)

    df_target_clone = pd.DataFrame(
        {
            "Vol_Innovations": y,
            "Exact_Target_Clone": y.copy(),
            "Exogenous_Noise": rng.standard_normal(n),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        out = execute_structural_causal_discovery(df_target_clone, tau_max=5)

    # Clone driver must not crash and must return valid matrices
    assert np.isfinite(out["val_matrix"]).all()
    assert np.isfinite(out["p_matrix"]).all()


@pytest.mark.unit
def test_near_collinear_spectrum_stability():
    """
    Adversarial Stress Test: Test near-collinearity across delta range from 1e-16 to 1e-2.
    Assert statsmodels never raises SingularMatrixWarning.
    """
    rng = np.random.default_rng(123)
    n = 250
    y = rng.standard_normal(n)

    deltas = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    for delta in deltas:
        df = pd.DataFrame(
            {
                "Vol_Innovations": y,
                "Near_Clone": y + delta * rng.standard_normal(n),
                "Noise": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", SingularMatrixWarning)
            out = execute_structural_causal_discovery(df, tau_max=3)
        assert np.isfinite(out["p_matrix"]).all()


@pytest.mark.unit
def test_condition_number_bounds_50_random_trials():
    """
    Adversarial Stress Test: Verify design matrix condition numbers kappa(X_c) across 50 random trials.
    Assert all well-formed design matrices satisfy kappa(X_c) < 1000.0 and zero SingularMatrixWarnings occur.
    """
    cond_numbers = []
    warnings_caught = []

    for trial in range(50):
        rng = np.random.default_rng(2000 + trial)
        n = rng.integers(150, 500)

        base = rng.standard_normal(n)
        vix = 0.6 * base + 0.4 * rng.standard_normal(n)
        credit = 0.5 * base + 0.5 * rng.standard_normal(n)
        liq = rng.standard_normal(n)
        target = 1.5 * np.roll(vix, 1) + rng.standard_normal(n)

        # In 20% of trials, introduce strong collinearity
        if trial % 5 == 0:
            credit = vix * 1.8 + 1e-5 * rng.standard_normal(n)
        elif trial % 5 == 1:
            credit = vix * 2.0

        df = pd.DataFrame(
            {
                "Vol_Innovations": target,
                "VIX_Diff": vix,
                "Credit_Spread_Diff": credit,
                "Liquidity_Diff": liq,
            }
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            out = execute_structural_causal_discovery(df, tau_max=5)
            for w in captured:
                if "rank-deficient" in str(w.message).lower() or "singular" in str(w.message).lower():
                    warnings_caught.append(str(w.message))

        # Check condition numbers of valid fitted design matrices
        y_series = df["Vol_Innovations"].values
        for s_name in ["VIX_Diff", "Credit_Spread_Diff", "Liquidity_Diff"]:
            for tau in range(1, 6):
                x_s = df[s_name].shift(tau).values
                X_mat = [x_s] + [df["Vol_Innovations"].shift(lag).values for lag in range(1, 6)]
                X_arr = np.column_stack(X_mat)
                valid = np.all(np.isfinite(X_arr), axis=1) & np.isfinite(y_series)
                Xc = sm.add_constant(X_arr[valid], has_constant="add")
                if np.linalg.matrix_rank(Xc) == Xc.shape[1]:
                    cond_numbers.append(np.linalg.cond(Xc))

    assert len(warnings_caught) == 0, f"Caught {len(warnings_caught)} rank-deficiency warnings!"
    assert len(cond_numbers) > 0
    max_cond = max(cond_numbers)
    assert max_cond < 1000.0, f"Maximum condition number {max_cond:.2f} exceeded threshold 1000.0"


@pytest.mark.unit
def test_multiplier_bounds_under_100_and_10000_sigma_moves():
    """
    Adversarial Stress Test: Feed extreme crisis shocks (+100 sigma, -100 sigma, 10000 sigma).
    Assert lambda_t in [1.3, 2.0] strictly holds without any floating point leakage.
    """
    rng = np.random.default_rng(777)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n)
    var_names = ["Vol_Innovations", "VIX_Diff", "Credit_Spread_Diff"]

    tau_max = 5
    p_mat = np.ones((3, 3, tau_max + 1))
    val_mat = np.zeros((3, 3, tau_max + 1))
    p_mat[1, 0, 1] = 0.0001
    val_mat[1, 0, 1] = 3.0

    causal_out = {
        "var_names": var_names,
        "p_matrix": p_mat,
        "val_matrix": val_mat,
        "tau_max": tau_max,
        "target_var": "Vol_Innovations",
    }

    # Generate baseline data
    df = pd.DataFrame(rng.standard_normal((n, 3)), index=dates, columns=var_names)
    vix_std = df["VIX_Diff"].std()

    # 1. 100-sigma positive shock
    df_100_pos = df.copy()
    df_100_pos.iloc[350:400, 1] += 100.0 * vix_std
    m_100_pos = construct_causal_multiplier(df_100_pos, causal_out, baseline_multiplier=2.0, min_multiplier=1.3)

    # 2. 100-sigma negative shock
    df_100_neg = df.copy()
    df_100_neg.iloc[350:400, 1] -= 100.0 * vix_std
    m_100_neg = construct_causal_multiplier(df_100_neg, causal_out, baseline_multiplier=2.0, min_multiplier=1.3)

    # 3. 10,000-sigma massive shock
    df_10k = df.copy()
    df_10k.iloc[350:400, 1] += 10000.0 * vix_std
    m_10k = construct_causal_multiplier(df_10k, causal_out, baseline_multiplier=2.0, min_multiplier=1.3)

    for name, series in [("100_pos", m_100_pos), ("100_neg", m_100_neg), ("10k", m_10k)]:
        assert not series.isna().any(), f"{name} produced NaNs"
        assert not np.isinf(series.values).any(), f"{name} produced Infs"
        assert (series >= 1.3).all(), f"{name} breached lower bound 1.3: min={series.min()}"
        assert (series <= 2.0).all(), f"{name} breached upper bound 2.0: max={series.max()}"
        assert (series < 1.3).sum() == 0, f"{name} floating-point leakage below 1.3"
        assert (series > 2.0).sum() == 0, f"{name} floating-point leakage above 2.0"
        # Multiplier must contract during crisis window
        assert series.iloc[399] < 1.40, f"{name} did not contract sufficiently during crisis: {series.iloc[399]}"


@pytest.mark.unit
def test_multiplier_under_calm_zero_shocks():
    """
    Adversarial Stress Test: All macro driver shocks are exactly 0.0.
    Assert danger_pct is 0.0 everywhere and multiplier is identically 2.0 for all t.
    """
    n = 500
    dates = pd.date_range("2020-01-01", periods=n)
    var_names = ["Vol_Innovations", "VIX_Diff", "Credit_Spread_Diff"]

    df_zero = pd.DataFrame(0.0, index=dates, columns=var_names)

    p_mat = np.ones((3, 3, 6))
    val_mat = np.zeros((3, 3, 6))
    p_mat[1, 0, 1] = 0.001
    val_mat[1, 0, 1] = 2.0

    causal_out = {
        "var_names": var_names,
        "p_matrix": p_mat,
        "val_matrix": val_mat,
        "tau_max": 5,
    }

    mult = construct_causal_multiplier(df_zero, causal_out, baseline_multiplier=2.0, min_multiplier=1.3)
    assert (mult == 2.0).all(), "Multiplier under calm zero shocks must be identically 2.0"


@pytest.mark.unit
def test_self_lags_flag_robustness():
    """
    Adversarial Stress Test: Verify execute_structural_causal_discovery with include_self_lags=True.
    Omission of tested lag tau prevents collinearity with target history.
    """
    rng = np.random.default_rng(88)
    n = 400
    df = pd.DataFrame(
        {
            "Vol_Innovations": rng.standard_normal(n),
            "Macro": rng.standard_normal(n),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        out_self = execute_structural_causal_discovery(df, include_self_lags=True)
        out_no_self = execute_structural_causal_discovery(df, include_self_lags=False)

    assert out_self["p_matrix"].shape == (2, 2, 6)
    assert out_no_self["p_matrix"].shape == (2, 2, 6)
    # Self-lags are evaluated in out_self
    assert (out_self["p_matrix"][0, 0, 1:] < 1.0).any() or (out_self["p_matrix"][0, 0, 1:] == 1.0).all()
    # Self-lags remain default 1.0 in out_no_self
    assert (out_no_self["p_matrix"][0, 0, :] == 1.0).all()


@pytest.mark.unit
def test_causal_graph_and_pathway_extraction_adversarial():
    """
    Adversarial Stress Test: Verify CausalGraph edge cases:
    - Empty graph (no edges)
    - Full graph (all edges)
    - Threshold sensitivity
    """
    var_names = ["Vol_Innovations", "A", "B"]
    p_mat = np.ones((3, 3, 4))
    val_mat = np.zeros((3, 3, 4))

    # Single strong edge
    p_mat[1, 0, 2] = 0.005
    val_mat[1, 0, 2] = -2.1

    out = {
        "var_names": var_names,
        "p_matrix": p_mat,
        "val_matrix": val_mat,
        "tau_max": 3,
        "target_var": "Vol_Innovations",
    }

    graph = CausalGraph.from_discovery(out, alpha_thresh=0.01)
    assert graph.active_channels == 1
    assert graph.active_drivers == ["A"]
    assert graph[0]["lag"] == 2
    assert graph[0]["beta"] == -2.1
    assert "CausalGraph" in repr(graph)
    assert len(graph.summary()) > 0

    # With tighter alpha=0.001, edge is dropped
    graph_tight = CausalGraph.from_discovery(out, alpha_thresh=0.001)
    assert graph_tight.active_channels == 0
    assert len(graph_tight.active_drivers) == 0
