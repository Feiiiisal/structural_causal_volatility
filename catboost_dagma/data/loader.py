"""Load and align returns and Fama-French factors."""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from catboost_dagma.config import (
    DEFAULT_SECTOR_ASSETS,
    FF_DATA_PATH,
    FF_FACTORS,
    FF_RISK_FREE,
    SP100_PICKLE_PATH,
)


class UnifiedDataLoader:
    def __init__(
        self,
        sp100_path: Optional[Path] = None,
        factor_path: Optional[Path] = None,
        offline: bool = True,
    ):
        self.sp100_path = Path(sp100_path).expanduser() if sp100_path else SP100_PICKLE_PATH
        self.factor_path = Path(factor_path).expanduser() if factor_path else FF_DATA_PATH
        self.offline = offline

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(path)
        elif suffix == ".csv":
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported data file: {path.name}")

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df.sort_index()[~df.index.duplicated(keep="last")]

    def load_sp100_returns(
        self,
        assets: Optional[List[str]] = None,
        min_history_pct: float = 0.95,
    ) -> pd.DataFrame:
        if not self.sp100_path.exists():
            raise FileNotFoundError(
                f"Return data not found at {self.sp100_path}. Set SP100_DATA_PATH."
            )

        df = self._read_frame(self.sp100_path)
        df = df.dropna(axis=1, thresh=int(len(df) * min_history_pct))

        requested = assets or DEFAULT_SECTOR_ASSETS
        available = [name for name in requested if name in df.columns]
        if not available:
            raise ValueError("None of the requested assets are in the return data.")

        missing = sorted(set(requested) - set(available))
        if missing:
            print("[data] skipped:", ", ".join(missing))

        return df[available].dropna()

    def load_fama_french_factors(
        self,
        start_date: str = "2016-01-01",
        end_date: str = "2026-01-01",
    ) -> pd.DataFrame:
        if self.factor_path:
            if not self.factor_path.exists():
                raise FileNotFoundError(f"Factor data not found at {self.factor_path}.")
            ff = self._read_frame(self.factor_path)
        elif self.offline:
            raise FileNotFoundError(
                "No factor file configured. Set FF_DATA_PATH or run with offline=False."
            )
        else:
            ff = self._download_factors(start_date, end_date)

        cols = FF_FACTORS + [FF_RISK_FREE]
        missing = [col for col in cols if col not in ff]
        if missing:
            raise ValueError(f"Missing factor columns: {missing}")

        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        ff = ff.loc[(ff.index >= start) & (ff.index < end), cols]
        return ff.dropna()

    @staticmethod
    def _download_factors(start_date: str, end_date: str) -> pd.DataFrame:
        from pandas_datareader import data as web

        ff = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )[0]

        if hasattr(ff.index, "to_timestamp"):
            ff.index = ff.index.to_timestamp()
        else:
            ff.index = pd.to_datetime(ff.index)

        if ff.index.tz is not None:
            ff.index = ff.index.tz_localize(None)

        return ff.astype(float) / 100.0

    def get_aligned_panel(
        self,
        assets: Optional[List[str]] = None,
        start_date: str = "2016-01-01",
        end_date: str = "2026-01-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        returns = self.load_sp100_returns(assets)
        factors = self.load_fama_french_factors(start_date, end_date)

        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        returns = returns.loc[(returns.index >= start) & (returns.index < end)]

        dates = returns.index.intersection(factors.index).sort_values()
        if dates.empty:
            raise ValueError("Return and factor data do not overlap.")

        return returns.loc[dates].copy(), factors.loc[dates].copy()
