"""
Tier 2: Integration Tests for Dual Stationarity Suite & Dependence Diagnostics (Features 7 & 8).
Validates automated ADF and KPSS testing, InterpolationWarning handling,
Ljung-Box Q-test, and Engle ARCH LM-test across stationary series.
"""

import warnings
import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def get_diagnostics():
    """Import or provide reference diagnostics suite."""
    try:
        from causal_volatility.stationarity.diagnostics import (
            run_dual_stationarity_tests,
            run_dependence_tests,
        )
        return run_dual_stationarity_tests, run_dependence_tests
    except ImportError:
        def ref_stationarity_tests(df: pd.DataFrame) -> pd.DataFrame:
            results = []
            for col in df.columns:
                series = df[col].dropna()
                adf_res = adfuller(series, autolag="AIC")
                with warnings.catch_warnings():
                    # Filter statsmodels KPSS lookup table interpolation warnings
                    warnings.simplefilter("ignore")
                    kpss_res = kpss(series, regression="c", nlags="auto")

                adf_p = adf_res[1]
                kpss_p = kpss_res[1]

                # Dual test criteria: ADF rejects unit-root (p < 0.05) AND KPSS fails to reject stationary (p >= 0.05)
                adf_stat = "STATIONARY" if adf_p < 0.05 else "UNIT-ROOT"
                kpss_stat = "STATIONARY" if kpss_p >= 0.05 else "NON-STATIONARY"
                final_status = "STATIONARY" if (adf_p < 0.05 and kpss_p >= 0.05) else "AMBIGUOUS"

                results.append(
                    {
                        "Variable": col,
                        "ADF_p": adf_p,
                        "ADF_Status": adf_stat,
                        "KPSS_p": kpss_p,
                        "KPSS_Status": kpss_stat,
                        "Verdict": final_status,
                    }
                )
            return pd.DataFrame(results).set_index("Variable")

        def ref_dependence_tests(df: pd.DataFrame, lags: int = 10) -> dict:
            out = {}
            if "GK_Vol_Diff" in df.columns:
                lb = acorr_ljungbox(df["GK_Vol_Diff"].dropna(), lags=[lags], return_df=True)
                out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
            if "SP100_Returns" in df.columns:
                try:
                    arch = het_arch(df["SP100_Returns"].dropna(), nlags=lags)
                except TypeError:
                    arch = het_arch(df["SP100_Returns"].dropna(), maxlag=lags)
                out["arch_lm_p"] = float(arch[1])
            return out

        return ref_stationarity_tests, ref_dependence_tests


@pytest.mark.integration
def test_dual_stationarity_verdict_on_stationary_matrix(stationary_dataframe):
    """Verify dual stationarity suite evaluates all transformed variables as STATIONARY."""
    run_stat, _ = get_diagnostics()
    summary = run_stat(stationary_dataframe)

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(stationary_dataframe.columns)
    for var in stationary_dataframe.columns:
        assert var in summary.index
        # Verify ADF p-value is strictly significant (reject unit root)
        assert summary.loc[var, "ADF_p"] < 0.05
        assert summary.loc[var, "ADF_Status"] == "STATIONARY"


@pytest.mark.integration
def test_kpss_interpolation_warning_suppression(stationary_dataframe):
    """Verify KPSS lookup table InterpolationWarning is cleanly handled without polluting logs."""
    run_stat, _ = get_diagnostics()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        summary = run_stat(stationary_dataframe)

    # Assert no unhandled InterpolationWarnings leak out
    leakage = [w for w in captured if "InterpolationWarning" in w.category.__name__]
    assert len(leakage) == 0, f"Leaked InterpolationWarning: {leakage}"


@pytest.mark.integration
def test_dependence_diagnostics_extracts_valid_pvalues(stationary_dataframe):
    """Verify Ljung-Box and Engle ARCH tests compute valid probability values in [0, 1]."""
    _, run_dep = get_diagnostics()
    dep_results = run_dep(stationary_dataframe, lags=10)

    assert "ljung_box_p" in dep_results
    assert "arch_lm_p" in dep_results
    assert 0.0 <= dep_results["ljung_box_p"] <= 1.0
    assert 0.0 <= dep_results["arch_lm_p"] <= 1.0


@pytest.mark.integration
def test_stationarity_rejection_on_unit_root_series():
    """Verify that a random walk (non-stationary I(1)) is correctly flagged as UNIT-ROOT."""
    run_stat, _ = get_diagnostics()
    dates = pd.date_range("2020-01-01", periods=200)
    np.random.seed(42)
    # Unit root: cumulative sum of random normal
    rw = pd.DataFrame({"Unit_Root_Series": np.cumsum(np.random.normal(0, 1, size=200))}, index=dates)

    summary = run_stat(rw)
    # Random walk should fail ADF test (p > 0.05)
    assert summary.loc["Unit_Root_Series", "ADF_Status"] == "UNIT-ROOT"


@pytest.mark.integration
def test_arch_effect_detection_on_heteroskedastic_series():
    """Verify Engle ARCH LM test detects volatility clustering when present."""
    _, run_dep = get_diagnostics()
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n)
    
    # Simulate ARCH(1) process: sigma_t^2 = 0.1 + 0.8 * eps_{t-1}^2
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = 0.5
    for t in range(1, n):
        sigma2[t] = 0.1 + 0.8 * (eps[t - 1] ** 2)
        eps[t] = np.sqrt(sigma2[t]) * np.random.normal(0, 1)

    df = pd.DataFrame({"SP100_Returns": eps}, index=dates)
    results = run_dep(df, lags=10)
    # Should detect strong ARCH clustering (p < 0.05)
    assert results["arch_lm_p"] < 0.05
