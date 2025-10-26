"""Service helpers for computing last UTC session metrics."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.analysis.session_last import (
    SessionMetrics,
    compute_session_metrics,
    export_sessions_jsonl,
    last_closed_session_bounds,
)
from src.analysis.zones_72h import Zone
from src.common.config import AppConfig
from src.storage.parquet import ParquetStorage

LOGGER = logging.getLogger(__name__)


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


def collect_last_sessions(
    symbols: Sequence[str],
    *,
    end_ms: int | None = None,
    storage: ParquetStorage | None = None,
    zones: Sequence[Zone] | None = None,
    export_path: str | Path | None = None,
) -> list[SessionMetrics]:
    if not symbols:
        return []

    reference_ms = end_ms if end_ms is not None else _now_ms()
    session_start_ms, session_end_ms = last_closed_session_bounds(reference_ms)
    storage_instance = _ensure_storage(storage)

    zone_lookup: dict[str, list[Zone]] = {}
    if zones:
        for zone in zones:
            if zone.filled_pct >= 0.8:
                continue
            zone_lookup.setdefault(zone.symbol, []).append(zone)

    metrics: list[SessionMetrics] = []
    start_time = time.perf_counter()

    for raw_symbol in symbols:
        symbol = (raw_symbol or "").strip().upper()
        if not symbol:
            continue
        frame = storage_instance.load_window(symbol, "1m", session_start_ms, session_end_ms)
        if frame.empty:
            LOGGER.info(
                "session_last.no_data",
                extra={"symbol": symbol, "session_start_ms": session_start_ms, "session_end_ms": session_end_ms},
            )
            continue
        frame = _normalise_frame(frame)
        metric = compute_session_metrics(
            frame,
            symbol=symbol,
            session_start_ms=session_start_ms,
            session_end_ms=session_end_ms,
            zones=zone_lookup.get(symbol, []),
        )
        if metric is not None:
            metrics.append(metric)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    LOGGER.info(
        "session.compute",
        extra={
            "session_compute_ms": elapsed_ms,
            "symbols": [symbol for symbol in symbols if symbol],
            "records": len(metrics),
        },
    )

    if export_path:
        export_sessions_jsonl(metrics, export_path)

    return metrics


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    required = ["ts_open", "open", "high", "low", "close", "volume", "taker_buy_vol"]
    for column in required:
        if column not in normalised:
            if column in {"volume", "taker_buy_vol"}:
                normalised[column] = 0.0
            else:
                raise KeyError(f"input frame missing required column '{column}'")
    return normalised


__all__ = ["collect_last_sessions"]
