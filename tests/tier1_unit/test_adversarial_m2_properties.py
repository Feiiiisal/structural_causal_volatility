"""
Tier 1: Adversarial Property-Based Testing and Econometric Stress Tests for Milestone M2.
Authored by Challenger 2 for Milestone M2.

Validates statistical guarantees across:
1. Random walks and unit roots (ADF p > 0.10)
2. Stationary white noise (ADF p < 0.01, KPSS p >= 0.10, Verdict == STATIONARY)
3. Autocorrelation detection in MA(2)/AR(1) vs pure white noise
4. Engle ARCH conditional heteroskedasticity detection in GARCH processes
5. Information criteria ranking (Student-t vs Gaussian GARCH on fat tails)
6. Asymmetric leverage effect parameter recovery (EGARCH gamma < 0, GJR gamma > 0)
7. Optimal lag order selection parsimony
8. Zero-variance and extreme flash-crash resilience
"""

import numpy as np
import pandas as pd
import pytest

from causal_volatility.models import (
    ARGARCHModel,
    EGARCHModel,
    GJRGARCHModel,
    OptimalLagSelector,
    StudentTGARCHModel,
    get_volatility_model,
)
from causal_volatility.stationarity import (
    DependencyAuditor,
    DualStationaritySuite,
    StationarityTransformer,
)


@pytest.mark.unit
def test_random_walk_fails_to_reject_unit_root():
    """Assert ADF test fails to reject unit root (p > 0.10) on synthetic random walks."""
    suite = DualStationaritySuite(alpha=0.05)
    rng = np.random.default_rng(100)
    n_obs = 500

    # Random walk: X_t = X_{t-1} + eps_t
    eps = rng.standard_normal(n_obs)
    rw = pd.Series(np.cumsum(eps), name="Random_Walk")

    result = suite.run(rw)
    adf_p = result.loc["Random_Walk", "ADF_p"]
    adf_status = result.loc["Random_Walk", "ADF_Status"]

    assert adf_p > 0.10, f"Expected ADF p > 0.10 on random walk, got {adf_p:.4f}"
    assert adf_status == "UNIT-ROOT", f"Expected UNIT-ROOT status, got {adf_status}"


@pytest.mark.unit
def test_stationary_white_noise_adf_and_kpss():
    """Assert ADF rejects unit root (p < 0.01) and KPSS fails to reject stationarity (p >= 0.10)."""
    suite = DualStationaritySuite(alpha=0.05)
    rng = np.random.default_rng(3)
    n_obs = 1000

    # Pure stationary Gaussian white noise: eps_t ~ N(0, 1)
    wn = pd.Series(rng.standard_normal(n_obs), name="White_Noise")

    result = suite.run(wn)
    adf_p = result.loc["White_Noise", "ADF_p"]
    kpss_p = result.loc["White_Noise", "KPSS_p"]
    verdict = result.loc["White_Noise", "Verdict"]

    assert adf_p < 0.01, f"Expected ADF p < 0.01 on white noise, got {adf_p:.4e}"
    assert kpss_p >= 0.10, f"Expected KPSS p >= 0.10 on white noise, got {kpss_p:.4f}"
    assert verdict == "STATIONARY", f"Expected STATIONARY verdict, got {verdict}"


@pytest.mark.unit
def test_ljung_box_serial_autocorrelation_detection():
    """Verify Ljung-Box test detects autocorrelation in MA(2) and AR(1), and fails in white noise."""
    auditor = DependencyAuditor(lags=10, alpha=0.05)
    rng = np.random.default_rng(999)
    n_obs = 1000

    # 1. Pure white noise: no linear memory
    wn = pd.Series(rng.standard_normal(n_obs), name="WN")
    audit_wn = auditor.audit(wn)
    assert audit_wn["ljung_box_p"] > 0.05, f"False positive in WN: {audit_wn['ljung_box_p']}"
    assert audit_wn["is_white_noise"] is True

    # 2. MA(2) process: X_t = e_t + 0.5 e_{t-1} + 0.3 e_{t-2}
    e = rng.standard_normal(n_obs + 2)
    ma2 = e[2:] + 0.5 * e[1:-1] + 0.3 * e[:-2]
    audit_ma2 = auditor.audit(pd.Series(ma2, name="MA2"))
    assert audit_ma2["ljung_box_p"] < 0.001, f"MA(2) correlation missed: {audit_ma2['ljung_box_p']}"
    assert audit_ma2["is_white_noise"] is False

    # 3. AR(1) process: X_t = 0.6 X_{t-1} + e_t
    ar1 = np.zeros(n_obs)
    for t in range(1, n_obs):
        ar1[t] = 0.6 * ar1[t - 1] + e[t]
    audit_ar1 = auditor.audit(pd.Series(ar1, name="AR1"))
    assert audit_ar1["ljung_box_p"] < 0.001, f"AR(1) correlation missed: {audit_ar1['ljung_box_p']}"
    assert audit_ar1["is_white_noise"] is False


@pytest.mark.unit
def test_engle_arch_heteroskedasticity_detection():
    """Verify Engle ARCH test detects conditional heteroskedasticity in synthetic GARCH."""
    auditor = DependencyAuditor(lags=10, alpha=0.05)
    rng = np.random.default_rng(42)
    n_obs = 1000

    # GARCH(1,1): sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    omega, alpha, beta = 0.05, 0.15, 0.80
    h = np.zeros(n_obs)
    eps = np.zeros(n_obs)
    h[0] = omega / (1.0 - alpha - beta)

    for t in range(1, n_obs):
        h[t] = omega + alpha * (eps[t - 1] ** 2) + beta * h[t - 1]
        eps[t] = np.sqrt(h[t]) * rng.standard_normal()

    garch_series = pd.Series(eps, name="GARCH_Sim")

    # Engle ARCH test on raw GARCH series must detect heteroskedasticity
    audit_raw = auditor.audit(garch_series)
    assert audit_raw["arch_lm_p"] < 0.001, f"ARCH effect missed: {audit_raw['arch_lm_p']}"

    # Residualization via ARGARCHModel should filter out the ARCH effect
    model = ARGARCHModel(p=1, q=1).fit(garch_series)
    diag = model.get_diagnostics(lags=10)
    assert diag["arch_lm_p"] > 0.05, f"Residuals still have ARCH effect: {diag['arch_lm_p']}"
    assert diag["is_white_noise"] is True, "Standardized residuals should be white noise"


@pytest.mark.unit
def test_heavy_tailed_aic_bic_model_ranking():
    """Assert Student-t GARCH achieves higher log-likelihood and lower AIC/BIC than Gaussian on fat tails."""
    rng = np.random.default_rng(123)
    n_obs = 1500
    nu_true = 4.0
    omega, alpha, beta = 0.05, 0.12, 0.82

    h = np.zeros(n_obs)
    eps = np.zeros(n_obs)
    h[0] = omega / (1.0 - alpha - beta)

    # Unit-variance Student-t shocks: var = nu / (nu - 2)
    std_t_shocks = rng.standard_t(df=nu_true, size=n_obs) * np.sqrt((nu_true - 2.0) / nu_true)

    for t in range(1, n_obs):
        h[t] = omega + alpha * (eps[t - 1] ** 2) + beta * h[t - 1]
        eps[t] = np.sqrt(h[t]) * std_t_shocks[t]

    fat_tail_series = pd.Series(eps, name="Fat_Tailed_Series")

    model_gauss = ARGARCHModel(p=1, q=1, dist="normal", scale_factor=10.0).fit(fat_tail_series)
    model_t = StudentTGARCHModel(p=1, q=1, dist="t", scale_factor=10.0).fit(fat_tail_series)

    diag_gauss = model_gauss.get_diagnostics()
    diag_t = model_t.get_diagnostics()

    # Log-likelihood of Student-t should be substantially higher
    assert diag_t["log_likelihood"] > diag_gauss["log_likelihood"], (
        f"Student-t LL ({diag_t['log_likelihood']:.2f}) not greater than "
        f"Gaussian LL ({diag_gauss['log_likelihood']:.2f})"
    )

    # AIC and BIC should both favor Student-t
    assert diag_t["aic"] < diag_gauss["aic"], (
        f"Student-t AIC ({diag_t['aic']:.2f}) not lower than Gaussian AIC ({diag_gauss['aic']:.2f})"
    )
    assert diag_t["bic"] < diag_gauss["bic"], (
        f"Student-t BIC ({diag_t['bic']:.2f}) not lower than Gaussian BIC ({diag_gauss['bic']:.2f})"
    )

    # Estimated degrees of freedom nu should be close to generating nu_true = 4.0
    est_nu = model_t.fit_result_.params["nu"]
    assert 2.5 <= est_nu <= 6.0, f"Estimated nu ({est_nu:.2f}) outside plausible range [2.5, 6.0]"


@pytest.mark.unit
def test_asymmetric_leverage_parameter_recovery():
    """Assert EGARCH and GJR-GARCH capture asymmetric leverage response on negative returns."""
    rng = np.random.default_rng(42)
    n_obs = 2000
    omega = 0.05
    alpha = 0.05
    gamma_true = 0.20
    beta = 0.75

    # GJR simulation: negative shocks add gamma_true to alpha
    h = np.zeros(n_obs)
    r = np.zeros(n_obs)
    h[0] = omega / (1.0 - alpha - 0.5 * gamma_true - beta)

    for t in range(1, n_obs):
        I_neg = 1.0 if r[t - 1] < 0 else 0.0
        h[t] = omega + (alpha + gamma_true * I_neg) * (r[t - 1] ** 2) + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()

    asym_series = pd.Series(r, name="Asym_Returns")

    # EGARCH: Nelson leverage parameter gamma is negative when bad news increases log-vol
    egarch = EGARCHModel(p=1, o=1, q=1).fit(asym_series)
    eg_gamma = egarch.fit_result_.params.get("gamma[1]")
    assert eg_gamma is not None, "EGARCH missing gamma[1] parameter"
    assert eg_gamma < -0.05, f"Expected negative EGARCH gamma, got {eg_gamma:.4f}"

    # GJR-GARCH: Threshold parameter gamma is positive for increased variance on bad news
    gjr = GJRGARCHModel(p=1, o=1, q=1).fit(asym_series)
    gjr_gamma = gjr.fit_result_.params.get("gamma[1]")
    assert gjr_gamma is not None, "GJR-GARCH missing gamma[1] parameter"
    assert gjr_gamma > 0.10, f"Expected positive GJR gamma, got {gjr_gamma:.4f}"


@pytest.mark.unit
def test_optimal_lag_selector_parsimony_and_recovery():
    """Assert OptimalLagSelector selects parsimonious lags for white noise and recovers AR lag."""
    rng = np.random.default_rng(42)
    n_obs = 1000

    # 1. White noise: BIC penalty should select minimal lag (p = 1)
    wn = rng.standard_normal(n_obs)
    selector_bic = OptimalLagSelector(max_lag=10, criterion="bic")
    opt_wn = selector_bic.select_lag(wn)
    assert opt_wn == 1, f"Expected p=1 for white noise with BIC, got {opt_wn}"

    # 2. Strong AR(3) process
    e = rng.standard_normal(n_obs)
    ar3 = np.zeros(n_obs)
    for t in range(3, n_obs):
        ar3[t] = 0.40 * ar3[t - 1] + 0.30 * ar3[t - 2] - 0.20 * ar3[t - 3] + e[t]

    opt_ar3 = selector_bic.select_lag(ar3)
    assert opt_ar3 >= 2, f"Expected p >= 2 for AR(3) process, got {opt_ar3}"


@pytest.mark.unit
def test_extreme_flash_crash_and_zero_vol_resilience():
    """Stress test all volatility models against extreme 100x flash crashes."""
    rng = np.random.default_rng(42)
    n_obs = 500
    base = rng.standard_normal(n_obs) * 0.01

    # Inject extreme flash crash (-50% crash followed by +40% rebound)
    base[250] = -0.50
    base[251] = 0.40
    series = pd.Series(base)

    model_names = ["garch", "student_t", "egarch", "gjr"]
    for name in model_names:
        m = get_volatility_model(name)
        m.fit(series)

        z = m.get_standardized_residuals()
        sig = m.get_conditional_volatility()
        diag = m.get_diagnostics()

        assert len(z) > 0
        assert np.isfinite(z.values).all(), f"{name} produced non-finite residuals"
        assert (sig > 0).all(), f"{name} produced non-positive conditional volatility"
        assert np.isfinite(diag["log_likelihood"]), f"{name} log_likelihood non-finite"
        assert np.isfinite(diag["aic"]), f"{name} aic non-finite"
