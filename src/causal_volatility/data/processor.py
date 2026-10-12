"""Data alignment, cleaning, and feature engineering processor."""

from typing import Optional
import numpy as np
import pandas as pd

from causal_volatility.estimators.base import BaseVolatilityEstimator
from causal_volatility.estimators.garman_klass import GarmanKlassEstimator


class DataProcessor:
    """Cleans market data, aligns calendars, forward/back-fills macro series,

    and engineers realized volatility and liquidity proxy features.
    """

    def __init__(
        self,
        estimator: Optional[BaseVolatilityEstimator] = None,
        volume_scale: float = 1e7,
    ):
        """Initialize the DataProcessor.

        Parameters
        ----------
        estimator : BaseVolatilityEstimator, optional
            Realized volatility estimator. Defaults to GarmanKlassEstimator.
        volume_scale : float, default 1e7
            Scaling divisor for the volume-to-float liquidity proxy.
        """
        self.estimator = estimator if estimator is not None else GarmanKlassEstimator()
        self.volume_scale = volume_scale

    def process(self, df: pd.DataFrame, keep_raw: bool = False) -> pd.DataFrame:
        """Process and align raw systematic risk dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw market data containing OHLCV, VIX, and credit spread columns.
        keep_raw : bool, default False
            If True, retains raw columns alongside engineered features.

        Returns
        -------
        pd.DataFrame
            Cleaned and aligned DataFrame indexed by trading dates.
            Default columns: ['SP100_Close', 'Garman_Klass_Vol', 'VIX_Close', 'Credit_Spread', 'Liquidity_Proxy']
        """
        cleaned_df = df.copy()

        # Check required columns
        required_cols = ["SP100_Close", "SP100_Volume", "VIX_Close", "Credit_Spread"]
        for col in required_cols:
            if col not in cleaned_df.columns:
                raise ValueError(f"Missing required column in input dataframe: '{col}'")

        # 1. Eliminate non-trading calendar frames (weekends and national market closures)
        # Drop days where SP100_Close is NaN or Volume <= 0
        valid_mask = cleaned_df["SP100_Close"].notna() & (cleaned_df["SP100_Volume"] > 0)
        cleaned_df = cleaned_df.loc[valid_mask]

        # 2. Forward-fill and back-fill macro series to match reporting lags without lookahead
        cleaned_df["Credit_Spread"] = cleaned_df["Credit_Spread"].ffill().bfill()
        cleaned_df["VIX_Close"] = cleaned_df["VIX_Close"].ffill().bfill()

        # 3. Engineer Liquidity Proxy: Volume standardized by volume_scale
        cleaned_df["Liquidity_Proxy"] = cleaned_df["SP100_Volume"] / self.volume_scale

        # 4. Compute Realized Volatility
        vol_col_name = "Garman_Klass_Vol"
        vol_series = self.estimator.estimate(
            high=cleaned_df["SP100_High"],
            low=cleaned_df["SP100_Low"],
            close=cleaned_df["SP100_Close"],
            open=cleaned_df["SP100_Open"],
        )
        cleaned_df[vol_col_name] = vol_series

        # 5. Compute Average True Range (ATR) for trailing stop band
        high = cleaned_df["SP100_High"]
        low = cleaned_df["SP100_Low"]
        close_prev = cleaned_df["SP100_Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ], axis=1).max(axis=1)
        cleaned_df["ATR_14"] = tr.rolling(14, min_periods=1).mean()

        # 6. Remove any remaining NA rows from edge computations
        cleaned_df = cleaned_df.dropna()


        if keep_raw:
            return cleaned_df

        target_columns = [
            "SP100_Close",
            "SP100_High",
            "SP100_Low",
            "SP100_Open",
            vol_col_name,
            "ATR_14",
            "VIX_Close",
            "Credit_Spread",
            "Liquidity_Proxy",
        ]
        
        for ff_col in ["Mkt-RF", "SMB", "HML", "RF"]:
            if ff_col in cleaned_df.columns:
                target_columns.append(ff_col)
                # Forward-fill and back-fill FF3 factors just in case of missing days
                cleaned_df[ff_col] = cleaned_df[ff_col].ffill().bfill()
                
        return cleaned_df[target_columns]


    process_and_align = process

