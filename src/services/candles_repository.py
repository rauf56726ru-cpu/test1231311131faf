"""Persistence helpers for summary candle collection coverage."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, List, Mapping, MutableMapping, Sequence

from .ohlc_sanitizer import sanitize_candles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "var"
DB_PATH = DB_DIR / "candles.sqlite"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
    PRIMARY KEY(symbol, interval, open_ms)
);
CREATE TRIGGER IF NOT EXISTS candles_touch_updated
AFTER UPDATE ON candles
FOR EACH ROW BEGIN
    UPDATE candles SET updated_at = strftime('%s','now') * 1000
    WHERE rowid = NEW.rowid;
END;
CREATE TABLE IF NOT EXISTS gap_progress (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    gap_start_ms INTEGER NOT NULL,
    last_open_ms INTEGER NOT NULL,
    PRIMARY KEY(symbol, interval, gap_start_ms)
);
"""

_LOCK = RLock()
_DEFAULT_REPOSITORY: "CandleRepository | None" = None

_TIMEFRAME_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _resolve_interval_ms(interval: str) -> int:
    key = (interval or "").strip().lower()
    return _TIMEFRAME_TO_MS.get(key, 60_000)


def _bucket_timestamp(timestamp_ms: int, interval_ms: int) -> int:
    if interval_ms <= 0:
        return timestamp_ms
    return (timestamp_ms // interval_ms) * interval_ms


def _normalise_ts_fields(entry: MutableMapping[str, object], aligned_ms: int) -> None:
    for key in ("t", "time", "openTime", "open_time", "timestamp", "ts"):
        if key in entry:
            entry[key] = aligned_ms


@dataclass(slots=True)
class UpsertStats:
    """Statistics returned by :meth:`CandleRepository.upsert_candles`."""

    written: int
    dropped_ts: int
    dropped_ohlc: int

    @property
    def dropped(self) -> int:
        return self.dropped_ts + self.dropped_ohlc


class CandleRepository:
    """Wrapper around a SQLite database storing normalised candles."""

    def __init__(self, path: Path | str = DB_PATH):
        self._path = Path(path)
        self._schema_applied = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._schema_applied:
            return
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._schema_applied = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_open_times(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> List[int]:
        """Return sorted open timestamps for the requested window."""

        self._ensure_schema()
        interval_clean = interval.lower()
        interval_ms = _resolve_interval_ms(interval_clean)
        end_bound = end_ms + max(interval_ms - 1, 0)
        query = """
            SELECT open_ms
            FROM candles
            WHERE symbol = ? AND interval = ?
              AND open_ms BETWEEN ? AND ?
            ORDER BY open_ms ASC
        """
        with self._connect() as conn:
            rows = conn.execute(
                query,
                (symbol.upper(), interval_clean, start_ms, end_bound),
            ).fetchall()
        buckets = {
            _bucket_timestamp(int(row["open_ms"]), interval_ms)
            for row in rows
            if row["open_ms"] is not None
        }
        return sorted(buckets)

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Dict[str, float | int]]:
        """Return normalised candles for the requested window."""

        self._ensure_schema()
        interval_clean = interval.lower()
        interval_ms = _resolve_interval_ms(interval_clean)
        end_bound = end_ms + max(interval_ms - 1, 0)
        query = """
            SELECT open_ms, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ?
              AND open_ms BETWEEN ? AND ?
            ORDER BY open_ms ASC
        """
        with self._connect() as conn:
            rows = conn.execute(
                query,
                (symbol.upper(), interval_clean, start_ms, end_bound),
            ).fetchall()

        candles: List[Dict[str, float | int]] = []
        for row in rows:
            raw_ts = int(row["open_ms"])
            aligned_ts = _bucket_timestamp(raw_ts, interval_ms)
            candles.append(
                {
                    "t": aligned_ts,
                    "o": float(row["open"]),
                    "h": float(row["high"]),
                    "l": float(row["low"]),
                    "c": float(row["close"]),
                    "v": float(row["volume"]),
                }
            )
        return candles

    def upsert_candles(
        self,
        symbol: str,
        interval: str,
        candles: Sequence[Mapping[str, object]] | Sequence[Sequence[object]],
        *,
        stage: str,
    ) -> UpsertStats:
        """Insert or update candles and return sanitisation stats."""

        self._ensure_schema()
        sanitized = sanitize_candles(candles, stage=stage)
        if not sanitized.candles:
            return UpsertStats(written=0, dropped_ts=sanitized.invalid_ts, dropped_ohlc=sanitized.invalid_ohlc)

        interval_clean = interval.lower()
        interval_ms = _resolve_interval_ms(interval_clean)
        symbol_clean = symbol.upper()

        payload: List[tuple[object, ...]] = []
        legacy_ts: List[int] = []
        for item in sanitized.candles:
            if not isinstance(item, MutableMapping):
                continue
            try:
                raw_ts = int(item.get("t"))
            except (TypeError, ValueError):
                sanitized.invalid_ts += 1
                continue
            aligned_ts = _bucket_timestamp(raw_ts, interval_ms)
            if aligned_ts != raw_ts:
                legacy_ts.append(raw_ts)
            _normalise_ts_fields(item, aligned_ts)
            try:
                open_price = float(item.get("o"))
                high_price = float(item.get("h"))
                low_price = float(item.get("l"))
                close_price = float(item.get("c"))
            except (TypeError, ValueError):
                sanitized.invalid_ohlc += 1
                continue
            volume_value = item.get("v", 0.0)
            try:
                volume = float(volume_value)
            except (TypeError, ValueError):
                volume = 0.0
            payload.append(
                (
                    symbol_clean,
                    interval_clean,
                    aligned_ts,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                )
            )

        statement = """
            INSERT INTO candles (symbol, interval, open_ms, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_ms) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
        """
        with self._connect() as conn:
            if legacy_ts:
                deletions = [
                    (symbol_clean, interval_clean, ts)
                    for ts in {ts for ts in legacy_ts if ts >= 0}
                ]
                if deletions:
                    conn.executemany(
                        "DELETE FROM candles WHERE symbol = ? AND interval = ? AND open_ms = ?",
                        deletions,
                    )
            if payload:
                conn.executemany(statement, payload)

        return UpsertStats(
            written=len(payload),
            dropped_ts=sanitized.invalid_ts,
            dropped_ohlc=sanitized.invalid_ohlc,
        )

    def load_gap_progress(self, symbol: str, interval: str, gap_start_ms: int) -> int | None:
        """Return the last filled timestamp for the tracked gap if present."""

        self._ensure_schema()
        interval_clean = interval.lower()
        interval_ms = _resolve_interval_ms(interval_clean)
        query = """
            SELECT last_open_ms
            FROM gap_progress
            WHERE symbol = ? AND interval = ? AND gap_start_ms = ?
        """
        with self._connect() as conn:
            row = conn.execute(query, (symbol.upper(), interval_clean, gap_start_ms)).fetchone()
            if not row:
                return None
            stored_ms = int(row["last_open_ms"])
            aligned_ms = _bucket_timestamp(stored_ms, interval_ms)
            if aligned_ms != stored_ms:
                conn.execute(
                    "UPDATE gap_progress SET last_open_ms = ? WHERE symbol = ? AND interval = ? AND gap_start_ms = ?",
                    (aligned_ms, symbol.upper(), interval_clean, gap_start_ms),
                )
            return aligned_ms

    def update_gap_progress(
        self, symbol: str, interval: str, gap_start_ms: int, last_open_ms: int
    ) -> None:
        """Persist progress for an active gap."""

        self._ensure_schema()
        interval_clean = interval.lower()
        aligned_last = _bucket_timestamp(last_open_ms, _resolve_interval_ms(interval_clean))
        statement = """
            INSERT INTO gap_progress(symbol, interval, gap_start_ms, last_open_ms)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(symbol, interval, gap_start_ms) DO UPDATE SET last_open_ms = excluded.last_open_ms
        """
        with self._connect() as conn:
            conn.execute(statement, (symbol.upper(), interval_clean, gap_start_ms, aligned_last))

    def clear_gap_progress(self, symbol: str, interval: str, gap_start_ms: int) -> None:
        """Remove persisted progress for a completed gap."""

        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM gap_progress WHERE symbol = ? AND interval = ? AND gap_start_ms = ?",
                (symbol.upper(), interval.lower(), gap_start_ms),
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_repository() -> CandleRepository:
    """Return the process-wide candle repository singleton."""

    global _DEFAULT_REPOSITORY
    with _LOCK:
        if _DEFAULT_REPOSITORY is None:
            _DEFAULT_REPOSITORY = CandleRepository(DB_PATH)
        return _DEFAULT_REPOSITORY


def set_repository(repository: CandleRepository | None) -> None:
    """Override the global repository singleton (useful for tests)."""

    global _DEFAULT_REPOSITORY
    with _LOCK:
        _DEFAULT_REPOSITORY = repository


__all__ = [
    "CandleRepository",
    "UpsertStats",
    "get_repository",
    "set_repository",
]
