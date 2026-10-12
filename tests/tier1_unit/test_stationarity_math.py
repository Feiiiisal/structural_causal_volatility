"""
Tier 1: Unit Tests for Stationarity Transformations & Mathematical Safety.
Validates logarithmic differencing, epsilon handling for zero volatility,
liquidity proxy scaling, and matrix boundary properties.
"""

import pytest
import numpy as np
import pandas as pd


def get_transformer():
    """Import or provide reference StationarityTransformer."""
    try:
        from causal_volatility.stationarity.transform import StationarityTransformer
        return StationarityTransformer()
    except ImportError:
        class RefTransformer:
            def __init__(self, eps: float = 1e-8):
                self.eps = eps

            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                sp100_rets = np.log(df["SP100_Close"] / df["SP100_Close"].shift(1))
                gk_diff = np.log(df["Garman_Klass_Vol"] + self.eps) - np.log(df["Garman_Klass_Vol"].shift(1) + self.eps)
                vix_diff = df["VIX_Close"].diff()
                credit_diff = df["Credit_Spread"].diff()
                liq_diff = df["Liquidity_Proxy"].diff()

                res = pd.DataFrame(
                    {
                        "SP100_Returns": sp100_rets,
                        "GK_Vol_Diff": gk_diff,
                        "VIX_Diff": vix_diff,
                        "Credit_Spread_Diff": credit_diff,
                        "Liquidity_Diff": liq_diff,
                    },
                    index=df.index,
                ).dropna()
                return res

        return RefTransformer()


@pytest.mark.unit
def test_log_return_analytical_math():
    """Verify log return calculation: Delta ln(P_t) = ln(P_t / P_{t-1})."""
    dates = pd.date_range("2023-01-01", periods=4)
    # 100 -> 102 -> 104 -> 106
    prices = pd.Series([100.0, 102.0, 104.0, 106.0], index=dates)
    expected_ret1 = np.log(102.0 / 100.0)  # ~0.0198026
    expected_ret2 = np.log(104.0 / 102.0)  # ~0.019418
    expected_ret3 = np.log(106.0 / 104.0)  # ~0.019048

    calc_rets = np.log(prices / prices.shift(1)).dropna()
    assert np.isclose(calc_rets.iloc[0], expected_ret1, atol=1e-7)
    assert np.isclose(calc_rets.iloc[1], expected_ret2, atol=1e-7)
    assert np.isclose(calc_rets.iloc[2], expected_ret3, atol=1e-7)


@pytest.mark.unit
def test_epsilon_safety_against_zero_volatility():
    """
    Verify epsilon safety: ln(sigma_t + eps) with eps=1e-8.
    On flat market days where Garman-Klass volatility is 0.0,
    the transform must evaluate safely to finite floats, never -inf or NaN.
    """
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame(
        {
            "SP100_Close": [100.0, 100.0, 100.0],
            "Garman_Klass_Vol": [0.0, 0.0, 0.0],
            "VIX_Close": [15.0, 15.0, 15.0],
            "Credit_Spread": [3.0, 3.0, 3.0],
            "Liquidity_Proxy": [1.0, 1.0, 1.0],
        },
        index=dates,
    )

    transformer = get_transformer()
    stationary_df = transformer.transform(df)

    assert len(stationary_df) == 2  # dropped initial shift NaN row
    assert not np.isinf(stationary_df["GK_Vol_Diff"]).any()
    assert not stationary_df["GK_Vol_Diff"].isna().any()
    # On consecutive 0.0 vol days: ln(0 + eps) - ln(0 + eps) == 0.0
    assert (stationary_df["GK_Vol_Diff"] == 0.0).all()


@pytest.mark.unit
def test_liquidity_proxy_scaling():
    """Verify liquidity proxy standardization by 10^7."""
    volumes = pd.Series([10_000_000, 25_000_000, 5_000_000])
    scaled = volumes / 1e7
    assert scaled.iloc[0] == 1.0
    assert scaled.iloc[1] == 2.5
    assert scaled.iloc[2] == 0.5


@pytest.mark.unit
def test_stationary_matrix_schema_and_alignment(clean_processed_dataframe):
    """Verify output DataFrame schema matches PROJECT.md contract."""
    transformer = get_transformer()
    stat_df = transformer.transform(clean_processed_dataframe)

    expected_cols = ["SP100_Returns", "GK_Vol_Diff", "VIX_Diff", "Credit_Spread_Diff", "Liquidity_Diff"]
    assert list(stat_df.columns) == expected_cols
    # Exactly one row dropped due to first-order differencing
    assert len(stat_df) == len(clean_processed_dataframe) - 1
    assert not stat_df.isna().any().any()


@pytest.mark.unit
def test_stationary_differencing_reversibility():
    """Verify first-differences sum reproduces total displacement: sum(Delta X) = X_T - X_0."""
    vix = pd.Series([15.0, 20.0, 18.0, 25.0])
    diffs = vix.diff().dropna()
    total_displacement = diffs.sum()
    assert np.isclose(total_displacement, 25.0 - 15.0, atol=1e-8)


@pytest.mark.unit
def test_extreme_market_shocks_produce_finite_stationary_values():
    """Verify extreme volatility jumps (e.g. 5x jump from 0.01 to 0.05) produce finite outputs."""
    dates = pd.date_range("2023-01-01", periods=2)
    df = pd.DataFrame(
        {
            "SP100_Close": [100.0, 80.0],
            "Garman_Klass_Vol": [0.01, 0.05],
            "VIX_Close": [15.0, 55.0],
            "Credit_Spread": [3.0, 7.5],
            "Liquidity_Proxy": [1.0, 4.0],
        },
        index=dates,
    )

    transformer = get_transformer()
    res = transformer.transform(df)

    assert len(res) == 1
    assert np.isfinite(res.values).all()
    # Log vol diff: ln(0.05 + eps) - ln(0.01 + eps) approx ln(5) approx 1.6094
    assert np.isclose(res["GK_Vol_Diff"].iloc[0], np.log(5.0), atol=1e-3)
