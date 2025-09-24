"""Lightweight persistence for OHLC candles."""
from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DEFAULT_MAX_CANDLES = 10_000


class CandleStore:
    """Persist candle history snapshots with async-safe helpers."""

    def __init__(self, path: str | Path, *, max_bars: int = DEFAULT_MAX_CANDLES) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._max_bars = max(1, int(max_bars))
        self._async_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def read(self, symbol: str, interval: str) -> List[Dict[str, float]]:
        """Return stored candles ordered by timestamp ascending."""

        norm_symbol = symbol.upper()
        async with self._async_lock:
            await self._ensure_initialized()
            return await asyncio.to_thread(self._read_sync, norm_symbol, interval)

    async def write(self, symbol: str, interval: str, bars: Sequence[Dict[str, float]]) -> None:
        """Persist candles for the (symbol, interval) pair with deduplication."""

        norm_symbol = symbol.upper()
        async with self._async_lock:
            await self._ensure_initialized()
            await asyncio.to_thread(self._write_sync, norm_symbol, interval, list(bars))

    async def clear(self, symbol: str, interval: str) -> None:
        """Remove stored candles for the given key."""

        await self.write(symbol, interval, [])

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize)
            self._initialized = True

    def _initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            conn = sqlite3.connect(self._path, timeout=30, isolation_level=None, check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candles (
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        ts_ms_utc INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL DEFAULT 0.0,
                        close_time_ms_utc INTEGER DEFAULT 0,
                        timestamp TEXT,
                        PRIMARY KEY (symbol, interval, ts_ms_utc)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_ts ON candles(symbol, interval, ts_ms_utc)"
                )
            finally:
                conn.close()

    def _read_sync(self, symbol: str, interval: str) -> List[Dict[str, float]]:
        with self._file_lock():
            conn = sqlite3.connect(self._path, timeout=30, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(
                    """
                    SELECT ts_ms_utc, open, high, low, close, volume, close_time_ms_utc, timestamp
                    FROM candles
                    WHERE symbol = ? AND interval = ?
                    ORDER BY ts_ms_utc ASC
                    LIMIT ?
                    """,
                    (symbol, interval, self._max_bars),
                )
                rows = cursor.fetchall()
            finally:
                conn.close()
        candles: List[Dict[str, float]] = []
        for row in rows:
            ts_ms = int(row["ts_ms_utc"])
            timestamp = self._parse_timestamp(row["timestamp"], ts_ms)
            candle = {
                "ts_ms_utc": ts_ms,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "close_time_ms_utc": int(row["close_time_ms_utc"]),
                "timestamp": timestamp,
            }
            candles.append(candle)
        return candles[-self._max_bars :]

    def _write_sync(self, symbol: str, interval: str, bars: List[Dict[str, float]]) -> None:
        payload = self._prepare_payload(symbol, interval, bars)
        with self._file_lock():
            conn = sqlite3.connect(self._path, timeout=30, isolation_level=None, check_same_thread=False)
            try:
                conn.execute("BEGIN IMMEDIATE")
                if not payload:
                    conn.execute("DELETE FROM candles WHERE symbol = ? AND interval = ?", (symbol, interval))
                    conn.commit()
                    return
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO candles (
                        symbol,
                        interval,
                        ts_ms_utc,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        close_time_ms_utc,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                total = conn.execute(
                    "SELECT COUNT(*) FROM candles WHERE symbol = ? AND interval = ?",
                    (symbol, interval),
                ).fetchone()[0]
                overflow = max(0, int(total) - self._max_bars)
                if overflow > 0:
                    cutoff_row = conn.execute(
                        """
                        SELECT ts_ms_utc FROM candles
                        WHERE symbol = ? AND interval = ?
                        ORDER BY ts_ms_utc DESC
                        LIMIT 1 OFFSET ?
                        """,
                        (symbol, interval, self._max_bars - 1),
                    ).fetchone()
                    if cutoff_row:
                        cutoff = int(cutoff_row[0])
                        conn.execute(
                            "DELETE FROM candles WHERE symbol = ? AND interval = ? AND ts_ms_utc < ?",
                            (symbol, interval, cutoff),
                        )
                conn.commit()
            finally:
                conn.close()

    def _prepare_payload(
        self, symbol: str, interval: str, bars: Iterable[Dict[str, float]]
    ) -> List[tuple]:
        prepared: List[tuple] = []
        for bar in bars:
            try:
                ts_ms = int(bar["ts_ms_utc"])
                open_price = float(bar["open"])
                high_price = float(bar["high"])
                low_price = float(bar["low"])
                close_price = float(bar["close"])
            except (KeyError, TypeError, ValueError):
                continue
            volume = float(bar.get("volume", 0.0) or 0.0)
            close_time = bar.get("close_time_ms_utc", ts_ms)
            try:
                close_time = int(close_time)
            except (TypeError, ValueError):
                close_time = ts_ms
            timestamp_iso = self._to_iso_timestamp(bar.get("timestamp"), ts_ms)
            prepared.append(
                (
                    symbol,
                    interval,
                    ts_ms,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    close_time,
                    timestamp_iso,
                )
            )
        prepared.sort(key=lambda entry: entry[2])
        return prepared[-self._max_bars :]

    def _parse_timestamp(self, value: str | None, ts_ms: int) -> datetime:
        if not value:
            return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        try:
            text = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (TypeError, ValueError):
                parsed = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _to_iso_timestamp(self, value, ts_ms: int) -> str:
        timestamp = value
        if timestamp is None:
            timestamp = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc).isoformat()
        if isinstance(timestamp, str):
            return timestamp
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()

    @contextmanager
    def _file_lock(self):
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    os.write(fd, b"0")
                except OSError:
                    pass
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
