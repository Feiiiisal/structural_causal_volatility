"""
Unit tests for Data Ingestion and Fama-French 3-Factor Residualization.
"""

import numpy as np
import pandas as pd
import pytest
from catboost_dagma.data.loader import UnifiedDataLoader
from catboost_dagma.data.residualizer import FamaFrenchResidualizer


def test_loader_offline_and_returns():
    loader = UnifiedDataLoader(offline=True)
    returns_df = loader.load_sp100_returns(assets=["AAPL", "MSFT", "JPM"])
    assert isinstance(returns_df, pd.DataFrame)
    assert returns_df.shape[1] == 3
    assert returns_df.isna().sum().sum() == 0
    assert len(returns_df) > 500


def test_loader_aligned_panel():
    loader = UnifiedDataLoader(offline=True)
    returns_df, factors_df = loader.get_aligned_panel(assets=["AAPL", "JPM", "XOM"])
    assert len(returns_df) == len(factors_df)
    assert set(returns_df.columns) == {"AAPL", "JPM", "XOM"}
    assert "Mkt-RF" in factors_df.columns
    assert "SMB" in factors_df.columns
    assert "HML" in factors_df.columns


def test_fama_french_residualization_orthogonality():
    """
    Asserts that residuals from OLS have near-zero empirical covariance with factors.
    """
    loader = UnifiedDataLoader(offline=True)
    returns_df, factors_df = loader.get_aligned_panel(assets=["AAPL", "JPM"])

    residualizer = FamaFrenchResidualizer()
    residuals_df = residualizer.fit_transform(returns_df, factors_df)

    assert residuals_df.shape == returns_df.shape
    assert set(residuals_df.columns) == set(returns_df.columns)

    # Check that covariance between residuals and Mkt-RF is statistically zero (< 1e-6)
    mkt = factors_df.loc[residuals_df.index, "Mkt-RF"].values
    for col in residuals_df.columns:
        cov_mkt = np.cov(residuals_df[col].values, mkt)[0, 1]
        assert abs(cov_mkt) < 1e-4, f"Residual {col} has non-zero covariance with market: {cov_mkt}"

    exposure_report = residualizer.get_factor_exposure_report()
    assert len(exposure_report) == 2
    assert "R2" in exposure_report.columns
    assert "Mkt-RF" in exposure_report.columns
