import numpy as np
import pandas as pd

from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer


def _write_sample_data(tmp_path):
    rng = np.random.default_rng(12)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")

    factors = pd.DataFrame({
        "Mkt-RF": rng.normal(0, 0.01, len(dates)),
        "SMB": rng.normal(0, 0.005, len(dates)),
        "HML": rng.normal(0, 0.005, len(dates)),
        "RF": 0.0,
    }, index=dates)

    returns = pd.DataFrame({
        "AAPL": 1.1 * factors["Mkt-RF"] + rng.normal(0, 0.01, len(dates)),
        "MSFT": 0.9 * factors["Mkt-RF"] + rng.normal(0, 0.01, len(dates)),
        "JPM": 1.2 * factors["Mkt-RF"] + rng.normal(0, 0.012, len(dates)),
    }, index=dates)

    returns_path = tmp_path / "returns.csv"
    factors_path = tmp_path / "factors.csv"
    returns.to_csv(returns_path)
    factors.to_csv(factors_path)
    return returns_path, factors_path


def test_loader_aligns_returns_and_factors(tmp_path):
    returns_path, factors_path = _write_sample_data(tmp_path)
    loader = UnifiedDataLoader(returns_path, factors_path, offline=True)

    returns, factors = loader.get_aligned_panel(
        assets=["AAPL", "MSFT", "JPM"],
        start_date="2020-01-01",
        end_date="2022-01-01",
    )

    assert returns.index.equals(factors.index)
    assert list(returns.columns) == ["AAPL", "MSFT", "JPM"]
    assert list(factors.columns) == ["Mkt-RF", "SMB", "HML", "RF"]


def test_residuals_have_low_factor_covariance(tmp_path):
    returns_path, factors_path = _write_sample_data(tmp_path)
    loader = UnifiedDataLoader(returns_path, factors_path, offline=True)
    returns, factors = loader.get_aligned_panel(assets=["AAPL", "JPM"])

    residualizer = FamaFrenchResidualizer()
    residuals = residualizer.fit_transform(returns, factors)

    for asset in residuals:
        covariance = np.cov(residuals[asset], factors["Mkt-RF"])[0, 1]
        assert abs(covariance) < 1e-4

    report = residualizer.get_factor_exposure_report()
    assert set(report.index) == {"AAPL", "JPM"}
    assert {"Mkt-RF", "R2"}.issubset(report.columns)
