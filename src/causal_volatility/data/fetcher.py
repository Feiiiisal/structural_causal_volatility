"""Systematic risk data ingestion fetcher supporting Yahoo Finance and FRED with 3-tier fallback."""

import os
from pathlib import Path
from typing import Optional, Union
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_datareader.data as web
import datetime

from causal_volatility.config import DataConfig
from causal_volatility.data.storage import DataStorage


class SystematicRiskDataFetcher:
    """Ingests high-fidelity equity index parameters and macroeconomic risk features

    from Yahoo Finance and St. Louis FRED with a resilient 3-tier fallback hierarchy.
    """

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        fred_api_key: Optional[str] = None,
        offline: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initialize data fetcher with configuration or explicit keyword arguments."""
        if config is not None:
            self.config = config
        else:
            merged_kwargs = dict(kwargs)
            if fred_api_key is not None:
                merged_kwargs["fred_api_key"] = fred_api_key
            if offline:
                merged_kwargs["offline"] = offline
            if cache_dir is not None:
                merged_kwargs["cache_dir"] = cache_dir
            self.config = DataConfig(**merged_kwargs)

    @property
    def fred_api_key(self) -> str:
        """Get FRED API key from config or environment."""
        return self.config.fred_api_key or os.environ.get("FRED_API_KEY", "")

    @property
    def offline(self) -> bool:
        """Whether fetcher operates in offline mode."""
        return self.config.offline

    @property
    def cache_dir(self) -> Union[str, Path]:
        """Directory for cached fixtures."""
        return self.config.cache_dir or ".cache"

    def fetch_fred_credit_spread(self, start_date: str, end_date: str) -> pd.Series:
        """Fetch FRED BAA10Y corporate bond credit spread using the 3-tier fallback hierarchy.

        Mode 1: Authenticated official FRED JSON API.
        Mode 2: Direct unauthenticated FRED CSV stream parsing.
        Mode 3: Deterministic synthetic fallback proxy emitting a clear warning.

        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD).
        end_date : str
            End date (YYYY-MM-DD).

        Returns
        -------
        pd.Series
            Credit spread series indexed by business dates.
        """
        dates = pd.date_range(start_date, end_date, freq="B")

        # Mode 1: Authenticated Official JSON API
        if self.fred_api_key and not self.offline:
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={self.config.fred_series_id}"
                    f"&api_key={self.fred_api_key}&file_type=json"
                )
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json().get("observations", [])
                    if data:
                        df_obs = pd.DataFrame(data)
                        df_obs["date"] = pd.to_datetime(df_obs["date"])
                        df_obs["value"] = pd.to_numeric(df_obs["value"], errors="coerce")
                        s = df_obs.set_index("date")["value"].dropna()
                        return s.reindex(dates).ffill().bfill()
            except Exception:
                pass  # Fall through to Mode 2

        # Mode 2: Unauthenticated Direct CSV Stream
        if not self.offline:
            try:
                csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={self.config.fred_series_id}"
                resp = requests.get(csv_url, timeout=15)
                if resp.status_code == 200:
                    df_csv = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True)
                    # Support multiple column name formats ('BAA10Y' or 'Credit_Spread')
                    target_col = "Credit_Spread"
                    if self.config.fred_series_id in df_csv.columns:
                        df_csv[target_col] = pd.to_numeric(df_csv[self.config.fred_series_id], errors="coerce")
                    elif len(df_csv.columns) > 0:
                        df_csv[target_col] = pd.to_numeric(df_csv.iloc[:, 0], errors="coerce")
                    return df_csv[target_col].dropna().reindex(dates).ffill().bfill()
            except Exception:
                pass  # Fall through to Mode 3

        # Mode 3: Deterministic Synthetic Fallback Proxy
        warnings.warn(
            "FRED connection unavailable. Generating synthetic BAA10Y credit spread proxy.",
            UserWarning,
        )
        np.random.seed(42)
        n = len(dates)
        walk = np.cumsum(np.random.normal(0, 0.02, size=n))
        synth = np.clip(3.2 + walk, 1.5, 8.0)
        return pd.Series(synth, index=dates, name="Credit_Spread")

    def fetch(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        offline: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Fetch combined systematic risk dataset.

        Parameters
        ----------
        start_date : str, optional
            Start date (YYYY-MM-DD). Defaults to config.start_date.
        end_date : str, optional
            End date (YYYY-MM-DD). Defaults to config.end_date.
        offline : bool, optional
            Whether to strictly use offline cached fixture. Defaults to config.offline.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame indexed by DatetimeIndex with columns:
            ['SP100_Open', 'SP100_High', 'SP100_Low', 'SP100_Close', 'SP100_Volume', 'VIX_Close', 'Credit_Spread']
        """
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        is_offline = offline if offline is not None else self.config.offline

        # If offline mode requested, load directly from storage fixture
        if is_offline:
            return self._fetch_offline(start, end)

        # Attempt live ingestion with graceful fallback to offline fixture on network failure
        try:
            return self._fetch_live(start, end)
        except Exception as e:
            warnings.warn(
                f"Live data ingestion failed ({e}). Falling back to cached offline fixture.",
                RuntimeWarning,
            )
            return self._fetch_offline(start, end)

    def _fetch_offline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load from cached local fixture and crop to date range."""
        fixture_path = self.config.fixture_path
        df = DataStorage.load_fixture(fixture_path)

        # Filter date range if applicable
        if len(df) > 0:
            df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        return df

    def _fetch_live(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Execute live data extraction from Yahoo Finance, FRED, and Fama-French."""
        equity_sym = self.config.equity_ticker
        vix_sym = self.config.vix_ticker
        tickers = [equity_sym, vix_sym]

        market_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if market_data.empty:
            raise RuntimeError("Yahoo Finance returned an empty dataset.")

        df_market = pd.DataFrame(index=market_data.index)
        df_market["SP100_Close"] = market_data["Close"][equity_sym]
        df_market["SP100_Open"] = market_data["Open"][equity_sym]
        df_market["SP100_High"] = market_data["High"][equity_sym]
        df_market["SP100_Low"] = market_data["Low"][equity_sym]
        df_market["SP100_Volume"] = market_data["Volume"][equity_sym]
        df_market["VIX_Close"] = market_data["Close"][vix_sym]

        credit_spread_series = self.fetch_fred_credit_spread(start_date, end_date)
        df_fred = pd.DataFrame({"Credit_Spread": credit_spread_series})

        raw_df = df_market.join(df_fred, how="left")
        
        # Fetch Fama-French 3 Factors
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            ff3 = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start_dt, end_dt)[0]
            # Convert PeriodIndex to DatetimeIndex
            if hasattr(ff3.index, "to_timestamp"):
                ff3.index = ff3.index.to_timestamp()
            elif not isinstance(ff3.index, pd.DatetimeIndex):
                ff3.index = pd.to_datetime(ff3.index.astype(str))
            
            # Align timezone if raw_df index is tz-aware or naive
            if raw_df.index.tz is not None:
                raw_df.index = raw_df.index.tz_localize(None)
            if ff3.index.tz is not None:
                ff3.index = ff3.index.tz_localize(None)

            # FF3 data is in percentages, divide by 100
            ff3 = ff3 / 100.0
            raw_df = raw_df.join(ff3, how="left")
            raw_df[["Mkt-RF", "SMB", "HML", "RF"]] = raw_df[["Mkt-RF", "SMB", "HML", "RF"]].ffill().bfill()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to fetch Fama-French factors: {e}")
            raw_df["Mkt-RF"] = 0.0
            raw_df["SMB"] = 0.0
            raw_df["HML"] = 0.0
            raw_df["RF"] = 0.0

        # Standardize column order matching interface contract
        standard_cols = [
            "SP100_Open",
            "SP100_High",
            "SP100_Low",
            "SP100_Close",
            "SP100_Volume",
            "VIX_Close",
            "Credit_Spread",
            "Mkt-RF",
            "SMB",
            "HML",
            "RF"
        ]
        # Only return columns that exist (in case of offline mode missing FF3)
        cols_to_return = [c for c in standard_cols if c in raw_df.columns]
        return raw_df[cols_to_return]


    fetch_systematic_risk_data = fetch

