"""
Data Loader for Version 2: S&P 100 Constituents and Fama-French 3-Factor Panel.
Handles ingestion, alignment, date normalization, and market proxy extraction.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

# Add parent directories to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = MODULE_ROOT.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import causal_volatility as cv
from catboost_dagma.config import (
    SP100_PICKLE_PATH,
    DEFAULT_SECTOR_ASSETS,
    FF_FACTORS,
    FF_RISK_FREE,
)


class UnifiedDataLoader:
    """
    Ingests and aligns S&P 100 equity constituent panel with Fama-French 3-Factor series.
    Ensures zero forward lookahead and complete chronological integrity.
    """

    def __init__(
        self,
        sp100_path: Optional[Path] = None,
        offline: bool = True,
    ):
        self.sp100_path = Path(sp100_path) if sp100_path else SP100_PICKLE_PATH
        self.offline = offline

    def load_sp100_returns(
        self,
        assets: Optional[List[str]] = None,
        min_history_pct: float = 0.95,
    ) -> pd.DataFrame:
        """
        Loads S&P 100 daily returns panel from pickle file.
        Filters for requested assets or top liquid constituents.
        """
        if not self.sp100_path.exists():
            raise FileNotFoundError(
                f"S&P 100 pickle dataset not found at {self.sp100_path}. "
                "Please verify path or data ingestion script."
            )

        df = pd.read_pickle(self.sp100_path)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Filter out assets with too many NaNs
        thresh = int(len(df) * min_history_pct)
        df_clean = df.dropna(axis=1, thresh=thresh).ffill().bfill().fillna(0.0)

        if assets:
            available_assets = [a for a in assets if a in df_clean.columns]
            if len(available_assets) == 0:
                raise ValueError(f"None of requested assets {assets} found in data columns.")
            return df_clean[available_assets]

        return df_clean

    def load_fama_french_factors(
        self,
        start_date: str = "2018-01-01",
        end_date: str = "2026-01-01",
    ) -> pd.DataFrame:
        """
        Retrieves daily Fama-French 3 Factors (Mkt-RF, SMB, HML) and Risk-Free Rate (RF).
        Uses systematic data fetcher with robust offline/online handling.
        """
        fetcher = cv.SystematicRiskDataFetcher(offline=self.offline)
        raw_ff = fetcher.fetch_systematic_risk_data(start_date, end_date)

        cols_needed = FF_FACTORS + [FF_RISK_FREE]
        available_cols = [c for c in cols_needed if c in raw_ff.columns]

        ff_df = raw_ff[available_cols].copy()
        ff_df.index = pd.to_datetime(ff_df.index)
        if ff_df.index.tz is not None:
            ff_df.index = ff_df.index.tz_localize(None)

        return ff_df.ffill().bfill()

    def get_aligned_panel(
        self,
        assets: Optional[List[str]] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2026-01-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns synchronized (returns_panel, factors_panel) over the common date index.
        """
        target_assets = assets or DEFAULT_SECTOR_ASSETS
        returns_df = self.load_sp100_returns(assets=target_assets)
        factors_df = self.load_fama_french_factors(start_date=start_date, end_date=end_date)

        # Common date intersection
        common_dates = returns_df.index.intersection(factors_df.index).sort_values()
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between S&P 100 returns and Fama-French factors.")

        returns_aligned = returns_df.loc[common_dates].copy()
        factors_aligned = factors_df.loc[common_dates].copy()

        return returns_aligned, factors_aligned
