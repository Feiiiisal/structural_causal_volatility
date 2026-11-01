#!/usr/bin/env python
"""Download and freeze the market inputs used by the capstone experiments."""
import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


START = "2016-01-01"
END = "2026-01-01"

UNIVERSES = {
    "us": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "JPM", "WMT", "GS", "XOM", "CVX",
        "JNJ", "UNH", "AMZN", "HD", "PG",
        "CAT", "BA", "NEE", "VZ", "COST",
    ],
    "india": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "BHARTIARTL.NS",
        "ITC.NS", "HINDUNILVR.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "ASIANPAINT.NS",
    ],
}

US_FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
INDIA_FACTOR_URL = (
    "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/"
    "archive/2025-12.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_bytes(url: str, attempts: int = 4) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (research; MScFE capstone)",
        "Accept": "*/*",
    }
    error = None

    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=90, headers=headers)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Could not download {url}: {error}")


def extract_close(data: pd.DataFrame, tickers):
    if data.empty:
        raise RuntimeError("Price download returned no rows.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Downloaded data do not contain a Close field.")
    else:
        if "Close" not in data.columns:
            raise RuntimeError("Downloaded data do not contain a Close field.")
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    return close.sort_index()


def download_prices(tickers, start=START, end=END):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    close = extract_close(data, tickers)

    close = close.reindex(columns=tickers)
    first_coverage = close.notna().mean()

    retry = [
        ticker for ticker in tickers
        if first_coverage.get(ticker, 0.0) < 0.98
    ]

    for ticker in retry:
        for attempt in range(5):
            single = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if not single.empty:
                series = extract_close(single, [ticker]).iloc[:, 0]
                if series.notna().mean() >= 0.98:
                    close[ticker] = series.reindex(close.index)
                    break
            time.sleep(2 ** attempt)

    coverage = close.notna().mean()
    bad = coverage[coverage < 0.98]

    if not bad.empty:
        details = ", ".join(f"{ticker}={value:.3f}" for ticker, value in bad.items())
        raise RuntimeError(
            "The frozen universe is incomplete. "
            f"Assets below the 98% price-coverage rule: {details}"
        )

    close = close[tickers]
    returns = close.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    return close, returns, coverage.to_dict()


def parse_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")

    dates = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if compact.any():
        dates.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    if (~compact).any():
        dates.loc[~compact] = pd.to_datetime(
            text.loc[~compact], errors="coerce"
        )

    return dates


def normal_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def factor_frame_from_text(text: str) -> pd.DataFrame:
    lines = text.replace("\ufeff", "").splitlines()
    header_idx = None

    for i, line in enumerate(lines):
        key = normal_name(line)
        if "smb" in key and "hml" in key:
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError("Could not locate factor-data header.")

    frame = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    frame = frame.dropna(how="all")
    first = frame.columns[0]
    frame = frame.rename(columns={first: "Date"})
    frame["Date"] = parse_date(frame["Date"])
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()

    names = {normal_name(c): c for c in frame.columns}

    def pick(candidates, contains=None):
        for candidate in candidates:
            if candidate in names:
                return names[candidate]
        if contains:
            for norm, original in names.items():
                if all(part in norm for part in contains):
                    return original
        return None

    market = pick(
        ["mktrf", "rmrf", "mf", "marketpremium", "marketprem"],
        contains=["market", "premium"],
    )
    smb = pick(["smb"])
    hml = pick(["hml"])
    rf = pick(["rf", "riskfree", "riskfreerate"])

    if market is None:
        numeric_candidates = [
            c for c in frame.columns
            if pd.to_numeric(frame[c], errors="coerce").notna().mean() > 0.9
        ]
        numeric_candidates = [
            c for c in numeric_candidates if c not in {smb, hml, rf}
        ]
        if numeric_candidates:
            market = numeric_candidates[0]

    missing = [
        label for label, col in
        [("Mkt-RF", market), ("SMB", smb), ("HML", hml), ("RF", rf)]
        if col is None
    ]
    if missing:
        raise RuntimeError(
            f"Factor columns not found: {missing}. Available={list(frame.columns)}"
        )

    out = frame[[market, smb, hml, rf]].copy()
    out.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    out = out.apply(pd.to_numeric, errors="coerce").dropna()

    q99 = out.abs().quantile(0.99).max()
    if q99 > 0.5:
        out = out / 100.0

    return out


def read_factor_csv_from_zip(payload: bytes, market: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("Factor archive contains no CSV files.")

        if market == "india":
            preferred = [
                n for n in csv_names
                if "daily" in n.lower()
                and "surviv" in n.lower()
                and ("factor" in n.lower() or "market" in n.lower())
            ]
            names = preferred or [
                n for n in csv_names if "daily" in n.lower()
            ] or csv_names
        else:
            names = csv_names

        errors = []
        for name in names:
            raw = archive.read(name)
            for encoding in ("utf-8-sig", "latin1"):
                try:
                    text = raw.decode(encoding)
                    frame = factor_frame_from_text(text)
                    if len(frame) > 1000:
                        return frame
                except Exception as exc:
                    errors.append(f"{name}: {exc}")

    raise RuntimeError("Could not parse factor archive. " + " | ".join(errors[:8]))


def download_factors(market: str) -> pd.DataFrame:
    url = US_FACTOR_URL if market == "us" else INDIA_FACTOR_URL
    return read_factor_csv_from_zip(request_bytes(url), market)


def save_market(market: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    tickers = UNIVERSES[market]

    close, returns, coverage = download_prices(tickers)
    factors = download_factors(market)

    start = pd.Timestamp(START)
    end = pd.Timestamp(END)
    factors = factors.loc[(factors.index >= start) & (factors.index < end)]

    print(
        f"{market}: price returns {returns.index.min().date()} to {returns.index.max().date()} "
        f"({len(returns)} rows, {returns.shape[1]} assets)"
    )
    print(
        f"{market}: factors {factors.index.min().date()} to {factors.index.max().date()} "
        f"({len(factors)} rows)"
    )

    common = returns.index.intersection(factors.index)
    returns = returns.loc[common]
    factors = factors.loc[common]

    if len(common) < 1800:
        raise RuntimeError(
            f"Only {len(common)} aligned daily observations for {market}."
        )

    returns_path = output / "returns.csv"
    factors_path = output / "factors.csv"
    prices_path = output / "adjusted_close.csv"

    returns.to_csv(returns_path)
    factors.to_csv(factors_path)
    close.loc[returns.index].to_csv(prices_path)

    manifest = {
        "market": market,
        "sample_start": str(returns.index.min().date()),
        "sample_end": str(returns.index.max().date()),
        "observations": int(len(returns)),
        "assets": list(returns.columns),
        "asset_count": int(returns.shape[1]),
        "requested_assets": tickers,
        "price_source": "Yahoo Finance via yfinance",
        "factor_source": (
            "Kenneth R. French Data Library"
            if market == "us"
            else "IIM Ahmedabad Indian Fama-French and Momentum Data Library, 2025-12 release"
        ),
        "factor_url": US_FACTOR_URL if market == "us" else INDIA_FACTOR_URL,
        "coverage_before_balancing": coverage,
        "files": {},
    }

    for path in (returns_path, factors_path, prices_path):
        manifest["files"][path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps({
        "market": market,
        "rows": len(returns),
        "assets": list(returns.columns),
        "factor_rows": len(factors),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "india"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    save_market(args.market, Path(args.output))


if __name__ == "__main__":
    main()
