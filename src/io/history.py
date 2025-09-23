"""Utilities for managing local historical candle storage."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..utils.logging import get_logger
from ..utils.timeframes import interval_to_seconds
from .binance_ws import fetch_klines

LOGGER = get_logger(__name__)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _klines_to_frame(payload: Iterable[Iterable[object]]) -> pd.DataFrame:
    data = list(payload or [])
    if not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(data, columns=KLINE_COLUMNS[: len(data[0])])
    if "open_time" not in frame.columns:
        raise ValueError("Binance payload missing open_time column")
    frame = frame.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "number_of_trades" in frame.columns:
        frame["number_of_trades"] = pd.to_numeric(frame["number_of_trades"], errors="coerce", downcast="integer")
    frame.set_index("open_time", inplace=True)
    frame.sort_index(inplace=True)
    if "ignore" in frame.columns:
        frame = frame.drop(columns=["ignore"])
    return frame


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return incoming.sort_index()
    if incoming.empty:
        return existing.sort_index()
    combined = pd.concat([existing, incoming])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined


def _expected_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    interval_seconds = max(1.0, interval_to_seconds(interval))
    step = pd.to_timedelta(interval_seconds, unit="s")
    if end < start:
        end = start
    periods = int(((end - start).total_seconds() / interval_seconds)) + 1
    periods = max(periods, 1)
    return pd.date_range(start=start, periods=periods, freq=step, tz="UTC")


def _range_complete(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> bool:
    if frame.empty:
        return False
    subset = frame.loc[(frame.index >= start) & (frame.index <= end)]
    if subset.empty:
        return False
    expected = _expected_grid(start, end, interval)
    missing = expected.difference(subset.index)
    return missing.empty


@dataclass(slots=True)
class BackfillResult:
    candles: pd.DataFrame
    downloaded: int = 0
    batches: int = 0


class BinanceBackfiller:
    """Handle Binance REST pagination and local parquet storage."""

    def __init__(
        self,
        data_root: Path | str = Path("data") / "raw",
        batch_limit: int = 1000,
        cooldown: float = 0.2,
    ) -> None:
        self.data_root = Path(data_root)
        self.batch_limit = max(1, int(batch_limit))
        self.cooldown = max(0.0, float(cooldown))
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        safe_symbol = (symbol or "").upper()
        safe_interval = interval or "1m"
        return self.data_root / f"{safe_symbol}_{safe_interval}.parquet"

    async def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int],
        end_ms: Optional[int],
    ) -> pd.DataFrame:
        try:
            payload = await fetch_klines(
                symbol,
                interval,
                limit=self.batch_limit,
                start_time=start_ms,
                end_time=end_ms,
            )
        except Exception as exc:  # pragma: no cover - network safeguards
            LOGGER.warning("Failed to fetch klines for %s %s: %s", symbol, interval, exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return _klines_to_frame(payload)

    async def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = await asyncio.to_thread(pd.read_parquet, path)
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[~frame.index.isna()].sort_index()
        return frame

    async def _save(self, path: Path, frame: pd.DataFrame) -> None:
        await asyncio.to_thread(frame.to_parquet, path)

    async def ensure_history(
        self,
        symbol: str,
        interval: str,
        mode: str,
        target: str | datetime,
        minimum_rows: int = 0,
    ) -> BackfillResult:
        """Ensure local storage has candles covering the requested range."""

        target_ts = pd.Timestamp(target, tz="UTC")
        path = self._path(symbol, interval)
        candles = await self._load(path)
        batches = 0
        downloaded = 0

        interval_seconds = max(1.0, interval_to_seconds(interval))
        interval_ms = int(interval_seconds * 1000)

        def _coverage_ok(frame: pd.DataFrame) -> bool:
            if frame.empty:
                return False
            if mode == "backward":
                return frame.index.min() <= target_ts
            return frame.index.max() >= target_ts

        max_attempts = 200  # hard safety guard ~200k bars
        attempts = 0

        while attempts < max_attempts and (not _coverage_ok(candles) or len(candles) < minimum_rows):
            attempts += 1
            if mode == "backward":
                end_ms: Optional[int]
                if candles.empty:
                    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                else:
                    end_ms = int(candles.index.min().timestamp() * 1000) - 1
                start_ms = None
            else:
                if candles.empty:
                    start_ms = int(target_ts.timestamp() * 1000)
                else:
                    start_ms = int(candles.index.max().timestamp() * 1000) + interval_ms
                end_ms = None

            frame = await self._fetch_klines(symbol, interval, start_ms, end_ms)
            if frame.empty:
                LOGGER.info(
                    "Binance backfill exhausted for %s %s after %s batches",
                    symbol,
                    interval,
                    batches,
                )
                break

            batches += 1
            downloaded += len(frame)
            candles = _merge_frames(candles, frame)
            await self._save(path, candles)

            if len(frame) < self.batch_limit:
                # API returned less than requested, likely reached boundary
                break
            if self.cooldown:
                await asyncio.sleep(self.cooldown)

        return BackfillResult(candles=candles, downloaded=downloaded, batches=batches)

    async def ensure_range(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        max_attempts: int = 5,
    ) -> BackfillResult:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        path = self._path(symbol, interval)
        candles = await self._load(path)
        batches = 0
        downloaded = 0
        interval_seconds = max(1.0, interval_to_seconds(interval))
        extension = pd.to_timedelta(interval_seconds, unit="s")
        attempts = 0

        while attempts < max_attempts and not _range_complete(candles, start_ts, end_ts, interval):
            attempts += 1
            start_ms = int(start_ts.timestamp() * 1000)
            end_boundary = end_ts + extension
            end_ms = int(end_boundary.timestamp() * 1000)
            frame = await self._fetch_klines(symbol, interval, start_ms, end_ms)
            if frame.empty:
                break
            batches += 1
            downloaded += len(frame)
            candles = _merge_frames(candles, frame)
            await self._save(path, candles)
            if len(frame) < self.batch_limit:
                break
            if self.cooldown:
                await asyncio.sleep(self.cooldown)

        return BackfillResult(candles=candles, downloaded=downloaded, batches=batches)
