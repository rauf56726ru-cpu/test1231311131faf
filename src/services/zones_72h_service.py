"""Service helpers to collect 72h zones and export JSONL snapshots."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.analysis.zones_72h import (
    Zone,
    ZoneDetectionConfig,
    detect_zones_72h,
    export_open_zones_jsonl,
    infer_tick_size,
)
from src.common.config import AppConfig
from src.storage.parquet import ParquetStorage

LOGGER = logging.getLogger(__name__)

DEFAULT_HOURS = 72


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ensure_storage(storage: ParquetStorage | None) -> ParquetStorage:
    if storage is not None:
        return storage
    app_config = AppConfig.load()
    return ParquetStorage(
        root=app_config.data_dir,
        market=app_config.market,
        index_path=app_config.duckdb_path,
    )


def collect_open_zones(
    symbols: Sequence[str],
    *,
    end_ms: int | None = None,
    hours: int = DEFAULT_HOURS,
    storage: ParquetStorage | None = None,
    tick_size_overrides: Mapping[str, float] | None = None,
    config: ZoneDetectionConfig | None = None,
    export_path: str | Path | None = None,
) -> list[Zone]:
    """Collect open/mitigated zones for the supplied symbols.

    When ``export_path`` is provided, the results are exported as JSON Lines.
    """

    if not symbols:
        return []

    end_ms = int(end_ms if end_ms is not None else _now_ms())
    hours = max(1, int(hours))
    start_ms = end_ms - hours * 60 * 60_000

    storage_instance = _ensure_storage(storage)
    overrides = dict(tick_size_overrides or {})
    zones: list[Zone] = []

    start_time = time.perf_counter()
    for raw_symbol in symbols:
        symbol = (raw_symbol or "").strip().upper()
        if not symbol:
            continue

        symbol_started = time.perf_counter()
        frame = storage_instance.load_window(symbol, "1m", start_ms, end_ms)
        if frame.empty:
            LOGGER.info(
                "zones72.collect.no_data",
                extra={"symbol": symbol, "start_ms": start_ms, "end_ms": end_ms},
            )
            continue

        frame = _normalise_frame(frame, symbol)
        tick_size = overrides.get(symbol) or infer_tick_size(frame, default=(config.tick_size if config else 0.1))
        base_cfg = config or ZoneDetectionConfig()
        cfg = replace(base_cfg, tick_size=tick_size)

        detected = detect_zones_72h(frame, config=cfg)
        zones.extend(detected)
        elapsed_ms = int((time.perf_counter() - symbol_started) * 1000)
        LOGGER.info(
            "zones72.collect.symbol",
            extra={
                "symbol": symbol,
                "zones_detected": len(detected),
                "tick_size": tick_size,
                "elapsed_ms": elapsed_ms,
            },
        )

    if export_path:
        export_open_zones_jsonl(zones, export_path)

    total_elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    LOGGER.info(
        "zones72.collect.summary",
        extra={
            "symbols": [s for s in (symbol.strip().upper() for symbol in symbols) if s],
            "zones": len(zones),
            "pipeline_total_ms": total_elapsed_ms,
        },
    )

    return zones


def _normalise_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    normalised = frame.copy()
    normalised["symbol"] = symbol
    if "taker_buy_vol" not in normalised:
        normalised["taker_buy_vol"] = 0.0
    columns = ["ts_open", "open", "high", "low", "close", "volume", "symbol", "taker_buy_vol"]
    for column in columns:
        if column not in normalised:
            if column == "volume":
                normalised[column] = 0.0
            else:
                raise KeyError(f"input frame missing required column '{column}'")
    return normalised


__all__ = ["collect_open_zones"]
