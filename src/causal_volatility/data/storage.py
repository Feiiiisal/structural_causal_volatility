"""Data storage and fixture caching utilities."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


class DataStorage:
    """Handles persistence and retrieval of historical market datasets and analytical fixtures."""

    @staticmethod
    def get_default_fixture_path() -> Path:
        """Return default path to market data fixture."""
        # Find project root
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists() or (parent / "PROJECT.md").exists():
                return parent / "tests" / "fixtures" / "market_data_2016_2026.csv"
        return Path("tests/fixtures/market_data_2016_2026.csv")

    @classmethod
    def save_fixture(
        cls,
        df: pd.DataFrame,
        path: Optional[Union[str, Path]] = None,
        format: Optional[str] = None,
    ) -> Path:
        """Save market data DataFrame to disk as CSV or Parquet.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame indexed by DatetimeIndex.
        path : Union[str, Path], optional
            Target file path. If None, uses default fixture path.
        format : str, optional
            'csv' or 'parquet'. Inferred from file extension if None.

        Returns
        -------
        Path
            Path to the saved file.
        """
        dest_path = Path(path) if path is not None else cls.get_default_fixture_path()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = format.lower() if format else dest_path.suffix.lstrip(".").lower()
        if fmt == "parquet":
            try:
                df.to_parquet(dest_path)
            except Exception as e:
                # If parquet engine is not available, save as csv
                csv_path = dest_path.with_suffix(".csv")
                df.to_csv(csv_path)
                return csv_path
        else:
            df.to_csv(dest_path)

        return dest_path

    @classmethod
    def load_fixture(cls, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load market data DataFrame from disk (CSV or Parquet).

        Parameters
        ----------
        path : Union[str, Path], optional
            Source file path. If None, searches default fixture locations.

        Returns
        -------
        pd.DataFrame
            Parsed DataFrame with DatetimeIndex.
        """
        source_path = Path(path) if path is not None else cls.get_default_fixture_path()

        if not source_path.exists():
            # Check alternatives: .parquet or .csv
            if source_path.suffix == ".csv" and source_path.with_suffix(".parquet").exists():
                source_path = source_path.with_suffix(".parquet")
            elif source_path.suffix == ".parquet" and source_path.with_suffix(".csv").exists():
                source_path = source_path.with_suffix(".csv")
            else:
                raise FileNotFoundError(f"Fixture file not found at: {source_path}")

        fmt = source_path.suffix.lstrip(".").lower()
        if fmt == "parquet":
            df = pd.read_parquet(source_path)
        else:
            df = pd.read_csv(source_path, index_col=0, parse_dates=True)

        # Ensure index is pd.DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        return df
