"""Parquet-backed storage implementation with DuckDB indexing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Iterable, Iterator, Mapping, Sequence

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.common.ts import ensure_epoch_ms

from .base import Storage, StorageError, StorageWriteStats

LOGGER = logging.getLogger(__name__)
UTC = timezone.utc

BAR_COLUMNS: tuple[str, ...] = (
    "ts_open",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_vol",
    "taker_buy_quote",
    "trades",
)

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("ts_open", pa.int64(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("taker_buy_vol", pa.float64(), nullable=False),
        pa.field("taker_buy_quote", pa.float64(), nullable=False),
        pa.field("trades", pa.int32(), nullable=False),
    ]
)

MIN_ROW_GROUP = 100_000
MAX_ROW_GROUP = 1_000_000
DEFAULT_ROW_GROUP = 250_000


def _coerce_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise StorageError("symbol is required")
    return cleaned


def _coerce_interval(interval: str) -> str:
    cleaned = (interval or "").strip().lower()
    if not cleaned:
        raise StorageError("interval is required")
    return cleaned


def _clamp_row_group(value: int | float | None) -> int:
    candidate = int(value) if value else DEFAULT_ROW_GROUP
    if candidate < MIN_ROW_GROUP:
        return MIN_ROW_GROUP
    if candidate > MAX_ROW_GROUP:
        return MAX_ROW_GROUP
    return candidate


def _pd_date_range(start_ts: int, end_ts: int) -> Iterator[pd.Timestamp]:
    start = pd.to_datetime(start_ts, unit="ms", utc=True).normalize()
    end = pd.to_datetime(end_ts, unit="ms", utc=True).normalize()
    current = start
    while current <= end:
        yield current
        current += pd.Timedelta(days=1)


@dataclass(slots=True)
class _ParquetConfig:
    root: Path
    market: str
    index: Path
    row_group_size: int


class ParquetStorage(Storage):
    """Persist OHLCV bars as partitioned Parquet files indexed by DuckDB."""

    def __init__(
        self,
        root: Path | str = "data",
        *,
        market: str = "default",
        index_path: Path | str | None = None,
        row_group_size: int | float | None = None,
    ) -> None:
        root_path = Path(root)
        if index_path is None:
            index_path = root_path.parent / "meta" / "index.duckdb"
        self._config = _ParquetConfig(
            root=root_path,
            market=market,
            index=Path(index_path),
            row_group_size=_clamp_row_group(row_group_size),
        )
        self._lock = RLock()
        self._ensure_directories()
        self._ensure_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_window(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        symbol_clean = _coerce_symbol(symbol)
        interval_clean = _coerce_interval(interval)
        if end_ts < start_ts:
            raise StorageError("end_ts must be >= start_ts")

        started = time.perf_counter()
        candidates = self._lookup_files(symbol_clean, interval_clean, start_ts, end_ts)
        if not candidates:
            LOGGER.info(
                "storage.load_window",
                extra={
                    "backend": "parquet",
                    "symbol": symbol_clean,
                    "interval": interval_clean,
                    "start_ts": int(start_ts),
                    "end_ts": int(end_ts),
                    "read_window_ms": 0,
                    "rows_read": 0,
                    "files_touched": 0,
                    "columns": list(columns) if columns else list(BAR_COLUMNS),
                },
            )
            return pd.DataFrame(columns=list(columns) if columns else list(BAR_COLUMNS))

        paths = [self._absolute_from_relative(path) for path in candidates]
        columns_request = list(columns) if columns else list(BAR_COLUMNS)
        if "ts_open" not in columns_request:
            columns_request = ["ts_open"] + [col for col in columns_request if col != "ts_open"]

        filter_expression = (ds.field("ts_open") >= start_ts) & (ds.field("ts_open") <= end_ts)
        dataset = ds.dataset(paths, format="parquet")
        table = dataset.to_table(columns=columns_request, filter=filter_expression)
        df = table.to_pandas()
        df.sort_values("ts_open", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if columns and "ts_open" not in columns:
            df = df.loc[:, list(columns)]

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "storage.load_window",
            extra={
                "backend": "parquet",
                "symbol": symbol_clean,
                "interval": interval_clean,
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "read_window_ms": elapsed_ms,
                "rows_read": int(len(df)),
                "files_touched": len(paths),
                "columns": list(columns) if columns else list(BAR_COLUMNS),
            },
        )
        return df

    def write_rows(
        self,
        symbol: str,
        interval: str,
        rows: Iterable[Mapping[str, object]],
    ) -> Sequence[StorageWriteStats]:
        symbol_clean = _coerce_symbol(symbol)
        interval_clean = _coerce_interval(interval)

        dataframe = self._normalise_rows(rows)
        if dataframe.empty:
            return []

        dataframe.sort_values("ts_open", inplace=True)
        dataframe.drop_duplicates("ts_open", keep="last", inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

        dataframe["day"] = pd.to_datetime(dataframe["ts_open"], unit="ms", utc=True).dt.normalize()
        write_stats: list[StorageWriteStats] = []

        for day_value, subset in dataframe.groupby("day", sort=True):
            stats = self._write_partition(
                symbol=symbol_clean,
                interval=interval_clean,
                day=day_value.to_pydatetime(),
                frame=subset.drop(columns="day"),
            )
            write_stats.append(stats)
        return write_stats

    def rebuild_index(self) -> None:
        with self._lock:
            self._ensure_index()
            with self._connect() as conn:
                conn.execute("DELETE FROM bars")
            files = sorted(self._iter_existing_files())
            entries: list[tuple[str, str, int, int, str, int]] = []
            for relative in files:
                absolute = self._absolute_from_relative(relative)
                symbol, interval, start_ts, end_ts = self._describe_file(relative, absolute)
                metadata = pq.ParquetFile(absolute).metadata
                rows = metadata.num_rows if metadata else 0
                entries.append((symbol, interval, start_ts, end_ts, relative, rows))

            if entries:
                with self._connect() as conn:
                    conn.executemany(
                        """
                        INSERT INTO bars(symbol, interval, start_ts, end_ts, path, rows)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        entries,
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_directories(self) -> None:
        self._config.root.mkdir(parents=True, exist_ok=True)
        (self._config.root / self._config.market).mkdir(parents=True, exist_ok=True)
        self._config.index.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._config.index))

    def _ensure_index(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bars(
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        start_ts BIGINT NOT NULL,
                        end_ts BIGINT NOT NULL,
                        path TEXT NOT NULL,
                        rows BIGINT NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol_interval ON bars(symbol, interval)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_window ON bars(symbol, interval, start_ts, end_ts)")

    def _normalise_rows(self, rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
        normalised: list[dict[str, object]] = []
        for entry in rows:
            if not isinstance(entry, Mapping):
                continue
            ts_open = self._coerce_int(
                entry.get("ts_open")
                or entry.get("ts")
                or entry.get("open_time")
                or entry.get("openTime")
                or entry.get("time")
            )
            if ts_open is None:
                continue
            try:
                ts_open = ensure_epoch_ms(ts_open)
            except ValueError:
                LOGGER.warning("storage.parquet.invalid_ts", extra={"value": entry.get("ts_open")})
                continue
            open_price = self._coerce_float(entry.get("open") or entry.get("o"))
            high_price = self._coerce_float(entry.get("high") or entry.get("h"))
            low_price = self._coerce_float(entry.get("low") or entry.get("l"))
            close_price = self._coerce_float(entry.get("close") or entry.get("c"))
            volume = self._coerce_float(entry.get("volume") or entry.get("v"))
            if None in (open_price, high_price, low_price, close_price, volume):
                continue

            taker_buy_vol = self._coerce_float(entry.get("taker_buy_vol") or entry.get("takerBuyBase")) or 0.0
            taker_buy_quote = self._coerce_float(entry.get("taker_buy_quote") or entry.get("takerBuyQuote")) or 0.0
            trades = self._coerce_int(entry.get("trades") or entry.get("trade_count") or entry.get("trades_cnt")) or 0
            normalised.append(
                {
                    "ts_open": ts_open,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "taker_buy_vol": taker_buy_vol,
                    "taker_buy_quote": taker_buy_quote,
                    "trades": trades,
                }
            )
        return pd.DataFrame.from_records(normalised, columns=list(BAR_COLUMNS))

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _write_partition(
        self,
        *,
        symbol: str,
        interval: str,
        day: datetime,
        frame: pd.DataFrame,
    ) -> StorageWriteStats:
        frame = frame.loc[:, list(BAR_COLUMNS)].copy()
        frame = frame.astype(
            {
                "ts_open": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
                "taker_buy_vol": "float64",
                "taker_buy_quote": "float64",
                "trades": "int32",
            }
        )
        relative = self._relative_path(symbol, interval, day)
        absolute = self._absolute_from_relative(relative)
        absolute.parent.mkdir(parents=True, exist_ok=True)

        min_ts = int(frame["ts_open"].min())
        max_ts = int(frame["ts_open"].max())
        rows = len(frame)

        table = pa.Table.from_pandas(frame, schema=PARQUET_SCHEMA, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"min_ts": str(min_ts).encode("utf-8"),
                b"max_ts": str(max_ts).encode("utf-8"),
                b"rows": str(rows).encode("utf-8"),
            }
        )
        table = table.replace_schema_metadata(metadata)
        pq.write_table(
            table,
            absolute,
            compression="snappy",
            row_group_size=self._config.row_group_size,
        )

        self._update_index(
            symbol=symbol,
            interval=interval,
            start_ts=min_ts,
            end_ts=max_ts,
            relative_path=relative,
            rows=rows,
        )
        return StorageWriteStats(
            symbol=symbol,
            interval=interval,
            rows=rows,
            path=relative,
            start_ts=min_ts,
            end_ts=max_ts,
        )

    def _relative_path(self, symbol: str, interval: str, day: datetime) -> str:
        day = day.astimezone(UTC)
        relative = Path(self._config.market) / symbol / interval / f"{day.year:04d}" / f"{day.month:02d}"
        return (relative / f"{day.day:02d}.parquet").as_posix()

    def _absolute_from_relative(self, relative: str) -> Path:
        return self._config.root / Path(relative)

    def _update_index(
        self,
        *,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int,
        relative_path: str,
        rows: int,
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM bars WHERE path = ?", (relative_path,))
                conn.execute(
                    """
                    INSERT INTO bars(symbol, interval, start_ts, end_ts, path, rows)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (symbol, interval, start_ts, end_ts, relative_path, rows),
                )

    def _lookup_files(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int,
    ) -> list[str]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT path, start_ts, end_ts
                    FROM bars
                    WHERE symbol = ?
                      AND interval = ?
                      AND end_ts >= ?
                      AND start_ts <= ?
                    ORDER BY start_ts ASC
                    """,
                    (symbol, interval, start_ts, end_ts),
                )
                indexed = [row[0] for row in cursor.fetchall()]

        fallback = [
            self._relative_path(symbol, interval, day.to_pydatetime())
            for day in _pd_date_range(start_ts, end_ts)
        ]
        files = {path for path in indexed}
        for candidate in fallback:
            if candidate in files:
                continue
            if self._absolute_from_relative(candidate).exists():
                files.add(candidate)
        return sorted(files)

    def _iter_existing_files(self) -> Iterator[str]:
        base = self._config.root / self._config.market
        if not base.exists():
            return iter(())
        for path in base.rglob("*.parquet"):
            try:
                relative = path.relative_to(self._config.root).as_posix()
            except ValueError:
                relative = path.as_posix()
            yield relative

    def _describe_file(self, relative: str, absolute: Path) -> tuple[str, str, int, int]:
        parts = Path(relative).parts
        if len(parts) < 6:
            raise StorageError(f"Unexpected partition layout for {relative}")
        symbol = parts[-5]
        interval = parts[-4]
        arrow_file = pq.ParquetFile(absolute)
        metadata = arrow_file.schema_arrow.metadata or {}
        min_ts = metadata.get(b"min_ts")
        max_ts = metadata.get(b"max_ts")
        if min_ts is None or max_ts is None:
            column_index = arrow_file.schema.names.index("ts_open")
            min_candidates: list[int] = []
            max_candidates: list[int] = []
            for index in range(arrow_file.metadata.num_row_groups):
                stats = arrow_file.metadata.row_group(index).column(column_index).statistics
                if stats is None:
                    continue
                if stats.has_min:
                    min_candidates.append(int(stats.min))
                if stats.has_max:
                    max_candidates.append(int(stats.max))
            if min_candidates and max_candidates:
                start_ts = min(min_candidates)
                end_ts = max(max_candidates)
            else:
                column = arrow_file.read(columns=["ts_open"]).to_pandas()["ts_open"]
                start_ts = int(column.min()) if not column.empty else 0
                end_ts = int(column.max()) if not column.empty else 0
        else:
            start_ts = int(min_ts.decode("utf-8"))
            end_ts = int(max_ts.decode("utf-8"))
        return (symbol.upper(), interval.lower(), start_ts, end_ts)
