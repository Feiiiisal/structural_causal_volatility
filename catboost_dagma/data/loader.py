"""Load and align market data."""
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from catboost_dagma.config import (
    DEFAULT_SECTOR_ASSETS,
    FF_FACTORS,
    FF_RISK_FREE,
    SP100_PICKLE_PATH,
)


class UnifiedDataLoader:
    def __init__(self, sp100_path: Optional[Path] = None, offline: bool = True):
        self.sp100_path = Path(sp100_path).expanduser() if sp100_path else SP100_PICKLE_PATH
        self.offline = offline

    def load_sp100_returns(
        self,
        assets: Optional[List[str]] = None,
        min_history_pct: float = 0.95,
    ) -> pd.DataFrame:
        if not self.sp100_path.exists():
            raise FileNotFoundError(
                f"Return data not found at {self.sp100_path}. Set SP100_DATA_PATH first."
            )

        suffix = self.sp100_path.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(self.sp100_path)
        elif suffix == ".csv":
            df = pd.read_csv(self.sp100_path, index_col=0, parse_dates=True)
        else:
            raise ValueError("Return data must be a pickle or CSV file.")

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
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
        import causal_volatility as cv

        raw = cv.SystematicRiskDataFetcher(offline=self.offline).fetch_systematic_risk_data(
            start_date, end_date
        )
        cols = FF_FACTORS + [FF_RISK_FREE]
        missing = [col for col in cols if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing factor columns: {missing}")

        ff = raw[cols].copy()
        ff.index = pd.to_datetime(ff.index)
        if ff.index.tz is not None:
            ff.index = ff.index.tz_localize(None)

        ff = ff.sort_index()
        return ff[~ff.index.duplicated(keep="last")].dropna()

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
        factors = factors.loc[(factors.index >= start) & (factors.index < end)]

        dates = returns.index.intersection(factors.index).sort_values()
        if dates.empty:
            raise ValueError("Return and factor data do not overlap.")

        return returns.loc[dates].copy(), factors.loc[dates].copy()
