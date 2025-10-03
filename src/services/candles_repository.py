"""Persistence helpers for summary candle collection coverage."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import List, Mapping, Sequence

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
                (symbol.upper(), interval.lower(), start_ms, end_ms),
            ).fetchall()
        return [int(row["open_ms"]) for row in rows]

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

        payload = [
            (
                symbol.upper(),
                interval.lower(),
                int(item["t"]),
                float(item["o"]),
                float(item["h"]),
                float(item["l"]),
                float(item["c"]),
                float(item.get("v", 0.0)),
            )
            for item in sanitized.candles
        ]

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
            cursor = 0
            batch_size = 1000
            while cursor < len(payload):
                batch = payload[cursor : cursor + batch_size]
                conn.executemany(statement, batch)
                cursor += batch_size

        return UpsertStats(
            written=len(payload),
            dropped_ts=sanitized.invalid_ts,
            dropped_ohlc=sanitized.invalid_ohlc,
        )

    def load_gap_progress(self, symbol: str, interval: str, gap_start_ms: int) -> int | None:
        """Return the last filled timestamp for the tracked gap if present."""

        self._ensure_schema()
        query = """
            SELECT last_open_ms
            FROM gap_progress
            WHERE symbol = ? AND interval = ? AND gap_start_ms = ?
        """
        with self._connect() as conn:
            row = conn.execute(query, (symbol.upper(), interval.lower(), gap_start_ms)).fetchone()
        return int(row["last_open_ms"]) if row else None

    def update_gap_progress(
        self, symbol: str, interval: str, gap_start_ms: int, last_open_ms: int
    ) -> None:
        """Persist progress for an active gap."""

        self._ensure_schema()
        statement = """
            INSERT INTO gap_progress(symbol, interval, gap_start_ms, last_open_ms)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(symbol, interval, gap_start_ms) DO UPDATE SET last_open_ms = excluded.last_open_ms
        """
        with self._connect() as conn:
            conn.execute(statement, (symbol.upper(), interval.lower(), gap_start_ms, last_open_ms))

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
