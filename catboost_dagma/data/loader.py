"""Data loading and chronological alignment for the capstone pipeline.

The loader deliberately avoids backward filling because using a future value to
fill an earlier observation creates look-ahead leakage.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from catboost_dagma.config import SP100_PICKLE_PATH, DEFAULT_SECTOR_ASSETS, FF_FACTORS, FF_RISK_FREE


class UnifiedDataLoader:
    def __init__(self, sp100_path: Optional[Path] = None, offline: bool = True):
        self.sp100_path = Path(sp100_path).expanduser() if sp100_path else SP100_PICKLE_PATH
        self.offline = offline

    def load_sp100_returns(self, assets: Optional[List[str]] = None, min_history_pct: float = 0.95) -> pd.DataFrame:
        if not self.sp100_path.exists():
            raise FileNotFoundError(
                f"Equity return panel not found at {self.sp100_path}. "
                "Set SP100_DATA_PATH to the research dataset. Synthetic fixtures should be used only for tests."
            )
        if self.sp100_path.suffix.lower() in {".pkl", ".pickle"}:
            df = pd.read_pickle(self.sp100_path)
        elif self.sp100_path.suffix.lower() == ".csv":
            df = pd.read_csv(self.sp100_path, index_col=0, parse_dates=True)
        else:
            raise ValueError("SP100_DATA_PATH must point to a .pkl/.pickle or .csv return panel.")

        df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(pd.to_datetime(df.index), "tz", None) is not None else pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # First remove assets that are mostly missing. Do not backfill or replace missing returns by zero.
        min_non_missing = int(len(df) * min_history_pct)
        df = df.dropna(axis=1, thresh=min_non_missing)

        requested = assets or DEFAULT_SECTOR_ASSETS
        if requested:
            available = [a for a in requested if a in df.columns]
            missing = sorted(set(requested) - set(available))
            if not available:
                raise ValueError(f"None of the requested assets are present in the return panel: {requested}")
            if missing:
                # Missing constituents are allowed, but the final panel is explicit and reproducible.
                print(f"[data] Excluding unavailable assets: {', '.join(missing)}")
            df = df[available]

        # Complete-case dates avoid inventing returns. The report should disclose the resulting sample.
        return df.dropna(axis=0, how="any")

    def load_fama_french_factors(self, start_date: str = "2016-01-01", end_date: str = "2026-01-01") -> pd.DataFrame:
        # Reuse the project's systematic-risk fetcher; do not use backward fill afterwards.
        import causal_volatility as cv
        fetcher = cv.SystematicRiskDataFetcher(offline=self.offline)
        raw_ff = fetcher.fetch_systematic_risk_data(start_date, end_date)

        cols_needed = FF_FACTORS + [FF_RISK_FREE]
        missing = [c for c in cols_needed if c not in raw_ff.columns]
        if missing:
            raise ValueError(f"Fama-French factor panel is missing required columns: {missing}")

        ff = raw_ff[cols_needed].copy()
        ff.index = pd.to_datetime(ff.index)
        if ff.index.tz is not None:
            ff.index = ff.index.tz_localize(None)
        ff = ff.sort_index()
        ff = ff[~ff.index.duplicated(keep="last")]
        return ff.dropna(axis=0, how="any")

    def get_aligned_panel(
        self,
        assets: Optional[List[str]] = None,
        start_date: str = "2016-01-01",
        end_date: str = "2026-01-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        returns_df = self.load_sp100_returns(assets=assets)
        factors_df = self.load_fama_french_factors(start_date=start_date, end_date=end_date)

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        returns_df = returns_df.loc[(returns_df.index >= start) & (returns_df.index < end)]
        factors_df = factors_df.loc[(factors_df.index >= start) & (factors_df.index < end)]

        common = returns_df.index.intersection(factors_df.index).sort_values()
        if len(common) == 0:
            raise ValueError("No overlapping trading dates between equity returns and factor data.")
        return returns_df.loc[common].copy(), factors_df.loc[common].copy()
