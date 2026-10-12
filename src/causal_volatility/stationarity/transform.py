"""Stationarity transformations for systemic risk variables and realized volatility."""

from typing import Optional
import numpy as np
import pandas as pd


class StationarityTransformer:
    """Transforms raw market and macro series into stationary return and difference series.

    Applies first-order logarithmic differencing to price series and realized volatility
    (with an epsilon regularization factor to prevent infinite or NaN values on zero-volatility days),
    and absolute first differencing to interest rate spreads and liquidity metrics.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        vol_col: str = "Garman_Klass_Vol",
    ):
        """Initialize StationarityTransformer.

        Parameters
        ----------
        eps : float, default 1e-8
            Small positive constant inside logarithmic transforms to protect against
            flat trading days where realized volatility is zero: ln(sigma + eps).
        vol_col : str, default 'Garman_Klass_Vol'
            Column name containing realized volatility.
        """
        self.eps = eps
        self.vol_col = vol_col

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "StationarityTransformer":
        """Fit transformer (stateless pass-through).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with raw/processed features.
        y : Optional[pd.Series]
            Ignored.

        Returns
        -------
        StationarityTransformer
            Self.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform aligned market series into stationary differenced representation.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned market dataframe containing at minimum:
            - 'SP100_Close': Close prices of market proxy
            - vol_col (default 'Garman_Klass_Vol'): Realized volatility
            - 'VIX_Close': Market volatility index
            - 'Credit_Spread': Corporate credit spread
            - 'Liquidity_Proxy': Volume-to-float liquidity proxy

        Returns
        -------
        pd.DataFrame
            Stationary dataframe with columns:
            ['SP100_Returns', 'GK_Vol_Diff', 'VIX_Diff', 'Credit_Spread_Diff', 'Liquidity_Diff']
            with initial NaN row dropped.
        """
        # Determine volatility column
        vol_col = self.vol_col
        if vol_col not in df.columns:
            candidates = [c for c in df.columns if "vol" in c.lower()]
            if candidates:
                vol_col = candidates[0]
            else:
                raise ValueError(
                    f"Volatility column '{self.vol_col}' not found in dataframe. "
                    f"Available columns: {list(df.columns)}"
                )

        # Validate required columns
        required_cols = ["SP100_Close", "VIX_Close", "Credit_Spread", "Liquidity_Proxy"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for stationarity transformation.")

        # 1. Log return of S&P 100 close: ln(P_t / P_{t-1})
        sp100_close = df["SP100_Close"]
        sp100_rets = np.log(sp100_close / sp100_close.shift(1))

        # 2. Log difference of realized volatility with epsilon safety:
        # Delta ln(sigma_t + eps) = ln(sigma_t + eps) - ln(sigma_{t-1} + eps)
        vol = df[vol_col]
        gk_diff = np.log(vol + self.eps) - np.log(vol.shift(1) + self.eps)

        # 3. Absolute first difference of VIX close
        vix_diff = df["VIX_Close"].diff()

        # 4. Absolute first difference of Credit Spread
        credit_diff = df["Credit_Spread"].diff()

        # 5. Absolute first difference of Liquidity Proxy
        liq_diff = df["Liquidity_Proxy"].diff()


        stationary_df = pd.DataFrame(
            {
                "SP100_Returns": sp100_rets,
                "GK_Vol_Diff": gk_diff,
                "VIX_Diff": vix_diff,
                "Credit_Spread_Diff": credit_diff,
                "Liquidity_Diff": liq_diff,
            },
            index=df.index,
        )
        
        for ff_col in ["Mkt-RF", "SMB", "HML", "RF"]:
            if ff_col in df.columns:
                stationary_df[ff_col] = df[ff_col]
                
        return stationary_df.dropna()

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform market dataframe in a single call.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned market dataframe.
        y : Optional[pd.Series]
            Ignored.

        Returns
        -------
        pd.DataFrame
            Stationary transformed dataframe.
        """
        return self.fit(df, y).transform(df)
