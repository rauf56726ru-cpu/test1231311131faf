"""Materialise Binance Vision archives into Parquet storage."""

from __future__ import annotations

import logging
import zipfile
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.common.ts import ensure_epoch_ms
from src.storage.parquet import ParquetStorage

LOGGER = logging.getLogger(__name__)

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _default_index_path(data_dir: Path) -> Path:
    return data_dir.parent / "meta" / "index.duckdb"


def _normalise_rows(frame: pd.DataFrame) -> Iterable[Dict[str, object]]:
    for row in frame.itertuples(index=False):
        try:
            ts_open = ensure_epoch_ms(row.open_time)
        except ValueError:
            LOGGER.warning(
                "vision.materialize.invalid_ts",
                extra={"value": row.open_time},
            )
            continue
        if ts_open == 0:
            LOGGER.warning("vision.materialize.zero_ts", extra={"value": row.open_time})
            continue
        try:
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)
            close_price = float(row.close)
            volume = float(row.volume)
            taker_buy_base = float(row.taker_buy_base)
            taker_buy_quote = float(row.taker_buy_quote)
            trades = int(row.trades)
        except (TypeError, ValueError):
            LOGGER.warning("vision.materialize.invalid_row", extra={"ts_open": ts_open})
            continue
        numeric_values = (
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            taker_buy_base,
            taker_buy_quote,
        )
        if any(math.isnan(value) or math.isinf(value) for value in numeric_values):
            LOGGER.warning("vision.materialize.nan_row", extra={"ts_open": ts_open})
            continue
        yield {
            "ts_open": ts_open,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "taker_buy_vol": taker_buy_base,
            "taker_buy_quote": taker_buy_quote,
            "trades": trades,
        }


def materialize_klines_from_vision_zip(
    zip_path: str,
    symbol: str,
    interval: str,
    market: str,
    data_dir: str,
) -> List[str]:
    """Materialise a Binance Vision daily archive into Parquet storage."""

    archive_path = Path(zip_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Vision archive not found: {archive_path}")

    data_root = Path(data_dir)
    storage = ParquetStorage(
        root=data_root,
        market=market,
        index_path=_default_index_path(data_root),
    )

    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members:
            LOGGER.warning("vision.materialize.empty_archive", extra={"path": str(archive_path)})
            return []

        created_files: List[str] = []
        for member in members:
            with archive.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    header=0,
                    names=COLUMNS,
                    dtype={
                        "open_time": "int64",
                        "open": "float64",
                        "high": "float64",
                        "low": "float64",
                        "close": "float64",
                        "volume": "float64",
                        "close_time": "int64",
                        "quote_volume": "float64",
                        "trades": "int64",
                        "taker_buy_base": "float64",
                        "taker_buy_quote": "float64",
                    },
                )

            if frame.empty:
                continue

            stats_by_day: Dict[str, List[Dict[str, object]]] = {}
            for row in _normalise_rows(frame):
                try:
                    ts_open = ensure_epoch_ms(row["ts_open"])
                except ValueError:
                    LOGGER.warning(
                        "vision.materialize.invalid_normalized_ts",
                        extra={"value": row.get("ts_open")},
                    )
                    continue
                day = datetime.fromtimestamp(ts_open / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                stats_by_day.setdefault(day, []).append(row)

            for day, records in stats_by_day.items():
                records.sort(key=lambda item: item["ts_open"])
                monotonic = all(records[i]["ts_open"] >= records[i - 1]["ts_open"] for i in range(1, len(records)))
                if not monotonic:
                    LOGGER.warning(
                        "vision.materialize.non_monotonic",
                        extra={"path": str(archive_path), "day": day},
                    )
                write_stats = storage.write_rows(symbol, interval, records)
                for stat in write_stats:
                    created_files.append(str((data_root / stat.path).resolve()))
                LOGGER.info(
                    "vision.materialize.day",
                    extra={
                        "path": str(archive_path),
                        "symbol": symbol.upper(),
                        "interval": interval.lower(),
                        "day": day,
                        "rows": len(records),
                    },
                )

    return created_files


__all__ = ["materialize_klines_from_vision_zip"]
