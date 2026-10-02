"""Daily OHLCV sources. yfinance is unofficial and for prototyping only."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ecsa.data.cache import DiskCache
from ecsa.data.ratelimit import TokenBucket
from ecsa.events.calendar import TradingCalendar
from ecsa.settings import Settings
from ecsa.storage import utc, write_parquet


class PriceSource(Protocol):
    """Daily price table contract; date is a session label, prices in currency/share."""

    def daily_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...


def daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute simple adjusted-close decimal returns; never forward-fill missing prices."""
    result = frame.sort_values(["ticker", "date"]).copy()
    result["return"] = result.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    return result


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate units, timestamp awareness, uniqueness and nonnegative share volume."""
    result = frame.copy()
    required = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "available_at",
    }
    if not required <= set(result):
        raise ValueError(f"Missing bar fields: {required - set(result)}")
    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy()).all():
        raise ValueError("Non-finite price or volume")
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result["available_at"] = pd.to_datetime([utc(t) for t in result["available_at"]], utc=True)
    if result.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/session bars")
    if (result[["open", "high", "low", "close", "adj_close"]] <= 0).any().any() or (
        result["volume"] < 0
    ).any():
        raise ValueError("Invalid price or volume")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any() or (
        result["high"] < result[["open", "close", "low"]].max(axis=1)
    ).any():
        raise ValueError("Invalid OHLC range")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


class LocalPriceSource:
    """Licensed or synthetic local Parquet input; no network operations."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def daily_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return inclusive session-date bounds from the local table."""
        frame = validate_bars(pd.read_parquet(self.path))
        return frame[(frame.ticker == ticker) & (frame.date >= start) & (frame.date <= end)].copy()


class YFinancePriceSource:
    """Prototype adapter: cached download calls, bounded retries and rate limiting.

    Adjusted prices are the provider's current vintage; snapshot_at records retrieval.
    Such data cannot establish point-in-time corporate-action/revision provenance.
    """

    def __init__(self, settings: Settings, downloader: Any = None) -> None:
        self.settings, self.downloader = settings, downloader
        self.limiter = TokenBucket(1)

    def daily_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch inclusive dates; persist normalized bars to a TTL-controlled Parquet cache."""
        params = {
            "ticker": ticker,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "auto_adjust": False,
        }
        path = (
            self.settings.cache_dir / "prices" / (DiskCache.key("yfinance:v1", params) + ".parquet")
        )
        if path.exists() and time.time() - path.stat().st_mtime < self.settings.cache_ttl_seconds:
            return validate_bars(pd.read_parquet(path))
        if self.downloader is None:
            import yfinance as yf

            cache_directory = self.settings.cache_dir / "yfinance"
            cache_directory.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_directory))
            self.downloader = yf.download
        for attempt in range(self.settings.request_max_retries + 1):
            self.limiter.acquire()
            try:
                frame = self.downloader(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=self.settings.request_timeout_seconds,
                )
                if frame.empty:
                    raise ValueError(
                        f"No prices returned for {ticker}; no fabricated/forward-filled bars"
                    )
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                frame = frame.reset_index().rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                    }
                )
                frame["date"] = pd.to_datetime(frame["date"]).dt.date
                calendar = TradingCalendar(list(frame["date"]))
                frame["ticker"] = ticker
                frame["available_at"] = pd.to_datetime(
                    [calendar.close(day) for day in frame["date"]], utc=True
                )
                frame["snapshot_at"] = datetime.now(UTC)
                frame = validate_bars(frame)
                write_parquet(frame, path)
                return frame
            except Exception:
                if attempt == self.settings.request_max_retries:
                    raise
                time.sleep(self.settings.retry_backoff_seconds * 2**attempt)
        raise RuntimeError("Price retries exhausted")
