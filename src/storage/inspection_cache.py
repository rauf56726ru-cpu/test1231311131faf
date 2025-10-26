"""DuckDB-backed storage for daily inspection payloads."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, Optional

import duckdb

LOGGER_NAME = "storage.inspection_cache"


@dataclass(slots=True)
class InspectionDayRecord:
    symbol: str
    day: date
    payload: str
    window_start_ms: int
    window_end_ms: int
    minutes_expected: int
    minutes_found: int
    updated_at: datetime


class InspectionCacheStore:
    """Persist inspection daily payloads inside DuckDB."""

    def __init__(self, index_path: Path | str) -> None:
        self._index_path = Path(index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._index_path))

    def _ensure_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inspection_day_cache (
                        symbol TEXT,
                        day DATE,
                        window_start_ms BIGINT,
                        window_end_ms BIGINT,
                        minutes_expected INTEGER,
                        minutes_found INTEGER,
                        payload TEXT,
                        updated_at TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_inspection_day_cache
                    ON inspection_day_cache(symbol, day)
                    """
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_days(
        self,
        symbol: str,
        days: Iterable[date],
    ) -> Dict[date, InspectionDayRecord]:
        day_list = list(days)
        if not day_list:
            return {}
        min_day = min(day_list)
        max_day = max(day_list)
        records: Dict[date, InspectionDayRecord] = {}
        with self._lock:
            with self._connect() as conn:
                result = conn.execute(
                    """
                    SELECT symbol, day, payload, window_start_ms, window_end_ms,
                           minutes_expected, minutes_found, updated_at
                    FROM inspection_day_cache
                    WHERE symbol = ?
                      AND day BETWEEN ? AND ?
                    """,
                    [symbol.upper(), min_day.isoformat(), max_day.isoformat()],
                )
                for row in result.fetchall():
                    record_day_raw = row[1]
                    if isinstance(record_day_raw, date):
                        record_day = record_day_raw
                    elif isinstance(record_day_raw, str):
                        record_day = date.fromisoformat(record_day_raw)
                    else:
                        continue
                    records[record_day] = InspectionDayRecord(
                        symbol=row[0],
                        day=record_day,
                        payload=row[2],
                        window_start_ms=int(row[3] or 0),
                        window_end_ms=int(row[4] or 0),
                        minutes_expected=int(row[5] or 0),
                        minutes_found=int(row[6] or 0),
                        updated_at=row[7]
                        if isinstance(row[7], datetime)
                        else datetime.now(UTC),
                    )
        return records

    def upsert_day(
        self,
        symbol: str,
        day: date,
        *,
        payload: dict,
        window_start_ms: int,
        window_end_ms: int,
        minutes_expected: int,
        minutes_found: int,
    ) -> None:
        symbol_clean = symbol.strip().upper()
        payload_text = json.dumps(payload, ensure_ascii=False)
        iso_day = day.isoformat()
        updated_at = datetime.now(UTC)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM inspection_day_cache WHERE symbol = ? AND day = ?",
                    [symbol_clean, iso_day],
                )
                conn.execute(
                    """
                    INSERT INTO inspection_day_cache (
                        symbol,
                        day,
                        window_start_ms,
                        window_end_ms,
                        minutes_expected,
                        minutes_found,
                        payload,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        symbol_clean,
                        iso_day,
                        int(window_start_ms),
                        int(window_end_ms),
                        int(minutes_expected),
                        int(minutes_found),
                        payload_text,
                        updated_at,
                    ],
                )

    def load_payload(self, symbol: str, day: date) -> Optional[dict]:
        with self._lock:
            with self._connect() as conn:
                result = conn.execute(
                    """
                    SELECT payload
                    FROM inspection_day_cache
                    WHERE symbol = ? AND day = ?
                    """,
                    [symbol.upper(), day.isoformat()],
                )
                row = result.fetchone()
                if not row:
                    return None
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return None
