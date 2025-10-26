"""Service helpers for warming and reading inspection daily caches."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, time as dtime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.common.config import AppConfig
from src.storage.inspection_cache import InspectionCacheStore
from src.services.check_all_datas import build_check_all_datas_async


LOGGER = logging.getLogger(__name__)
DEFAULT_DAYS = 3


def _cache_store() -> InspectionCacheStore:
    config = AppConfig.load()
    return InspectionCacheStore(config.duckdb_path)


def _make_snapshot(symbol: str) -> Dict[str, object]:
    captured_iso = datetime.now(UTC).isoformat()
    return {
        "symbol": symbol.upper(),
        "tf": "1m",
        "frames": {},
        "meta": {
            "symbol": symbol.upper(),
            "source": {"kind": "cache_warmup", "captured_at": captured_iso},
            "requested": {"frames": ["1m"], "lookback_days": 1},
        },
    }


def _target_days(now_utc: datetime, days: int) -> List[date]:
    base_date = now_utc.date()
    return [base_date - timedelta(days=offset + 1) for offset in range(days)]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_candle(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    ts = _safe_int(entry.get("t"))
    if ts is None:
        return None
    record = dict(entry)
    record["t"] = ts
    for key in ("o", "h", "l", "c", "v"):
        val = entry.get(key)
        try:
            record[key] = float(val) if val is not None else None
        except (TypeError, ValueError):
            record[key] = None
        if record[key] is None:
            return None
    return record


def build_cache_seed_from_payloads(
    payloads: Mapping[date, Mapping[str, Any]],
    *,
    days: int,
    now_utc: Optional[datetime] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Materialise per-timeframe frames from stored payloads."""

    if not payloads:
        return {}, {"minutes_found": 0, "minutes_expected": days * 24 * 60}

    timeframe_indexes: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for day, payload in payloads.items():
        data_section = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data_section, Mapping):
            continue
        ohlcv_section = data_section.get("ohlcv")
        if not isinstance(ohlcv_section, Mapping):
            continue
        for tf_key, frame_payload in ohlcv_section.items():
            if not isinstance(frame_payload, Mapping):
                continue
            raw_candles = frame_payload.get("candles")
            if not isinstance(raw_candles, Sequence):
                continue
            tf_index = timeframe_indexes.setdefault(str(tf_key), {})
            for candle in raw_candles:
                if not isinstance(candle, Mapping):
                    continue
                record = _coerce_candle(candle)
                if record is None:
                    continue
                ts = record["t"]
                tf_index[ts] = record

    if not timeframe_indexes:
        return {}, {"minutes_found": 0, "minutes_expected": days * 24 * 60}

    end_ms: int
    if now_utc is not None:
        end_ms = int(now_utc.replace(tzinfo=UTC).timestamp() * 1000)
    else:
        max_ts = max(
            (ts for index in timeframe_indexes.values() for ts in index),
            default=0,
        )
        end_ms = max(max_ts, 0)
    minute_span_ms = max(1, days) * 24 * 60 * 60 * 1000
    start_ms = max(0, end_ms - minute_span_ms)

    frames: Dict[str, List[Dict[str, Any]]] = {}
    for tf_key, tf_index in timeframe_indexes.items():
        ordered_ts = sorted(ts for ts in tf_index if start_ms <= ts <= end_ms)
        frames[tf_key] = [tf_index[ts] for ts in ordered_ts]

    minutes_expected = max(1, days * 24 * 60)
    minutes_found = len(frames.get("1m", []))
    coverage_pct = 0.0
    if minutes_expected:
        coverage_pct = round(min(minutes_found / minutes_expected, 1.0) * 100.0, 2)

    stats = {
        "minutes_found": minutes_found,
        "minutes_expected": minutes_expected,
        "coverage_pct": coverage_pct,
        "window_start_ms": start_ms if minutes_found else None,
        "window_end_ms": end_ms if minutes_found else None,
        "timeframes": sorted(frames),
    }
    LOGGER.info(
        "inspection.cache.seed.built",
        extra={
            "timeframes": stats["timeframes"],
            "minutes_found": minutes_found,
            "minutes_expected": minutes_expected,
            "coverage_pct": coverage_pct,
        },
    )
    return frames, stats


async def ensure_inspection_daily_cache(
    symbol: str,
    *,
    days: int = DEFAULT_DAYS,
    now_utc: Optional[datetime] = None,
    network_backfill: bool = True,
    collection_timeout: float | None = None,
) -> Dict[date, dict]:
    """Ensure the inspection cache contains daily payloads for the requested span."""

    if not symbol:
        raise ValueError("symbol is required")
    symbol_clean = symbol.strip().upper()
    now_dt = now_utc.astimezone(UTC) if now_utc else datetime.now(UTC)
    target_days = _target_days(now_dt, max(1, days))
    store = _cache_store()
    LOGGER.info(
        "inspection.cache.ensure.start",
        extra={
            "symbol": symbol_clean,
            "days": days,
            "target_days": [day.isoformat() for day in target_days],
            "network_backfill": bool(network_backfill),
        },
    )
    existing = store.list_days(symbol_clean, target_days)
    collected: Dict[date, dict] = {}
    for cached_day, record in existing.items():
        try:
            payload = store.load_payload(symbol_clean, cached_day)
        except Exception:
            payload = None
        if payload is not None:
            collected[cached_day] = payload
            LOGGER.info(
                "inspection.cache.ensure.hit",
                extra={
                    "symbol": symbol_clean,
                    "day": cached_day.isoformat(),
                    "minutes_found": record.minutes_found,
                    "minutes_expected": record.minutes_expected,
                },
            )

    missing = [day for day in target_days if day not in collected]
    if not missing:
        LOGGER.info(
            "inspection.cache.ensure.complete",
            extra={
                "symbol": symbol_clean,
                "status": "warm",
                "days": len(collected),
            },
        )
        return collected

    for day in sorted(missing):
        day_end = datetime.combine(day + timedelta(days=1), dtime.min, tzinfo=UTC)
        day_start = day_end - timedelta(days=1)
        snapshot = _make_snapshot(symbol_clean)
        LOGGER.info(
            "inspection.cache.ensure.collect",
            extra={
                "symbol": symbol_clean,
                "day": day.isoformat(),
                "window_start_ms": int(day_start.timestamp() * 1000),
                "window_end_ms": int(day_end.timestamp() * 1000),
            },
        )
        payload = await build_check_all_datas_async(
            snapshot,
            now_utc=day_end,
            window_hours=24,
            strict_window=True,
            network_backfill=network_backfill,
            timeout=collection_timeout,
        )
        if payload is None:
            LOGGER.warning(
                "inspection.cache.ensure.skipped",
                extra={"symbol": symbol_clean, "day": day.isoformat(), "reason": "pipeline_returned_none"},
            )
            continue
        minutes_found = 0
        minutes_expected = 24 * 60
        orderflow_section = payload.get("orderflow") if isinstance(payload, dict) else None
        if isinstance(orderflow_section, dict):
            per_bar = orderflow_section.get("per_bar")
            if isinstance(per_bar, dict):
                one_minute = per_bar.get("1m")
                if isinstance(one_minute, list):
                    minutes_found = len(one_minute)
        store.upsert_day(
            symbol_clean,
            day,
            payload=payload,
            window_start_ms=int(day_start.timestamp() * 1000),
            window_end_ms=int(day_end.timestamp() * 1000),
            minutes_expected=minutes_expected,
            minutes_found=minutes_found,
        )
        LOGGER.info(
            "inspection.cache.ensure.stored",
            extra={
                "symbol": symbol_clean,
                "day": day.isoformat(),
                "minutes_found": minutes_found,
                "minutes_expected": minutes_expected,
            },
        )
        collected[day] = payload

    LOGGER.info(
        "inspection.cache.ensure.complete",
        extra={
            "symbol": symbol_clean,
            "status": "updated",
            "days": len(collected),
            "missing_filled": [day.isoformat() for day in missing],
        },
    )
    return collected


async def schedule_cache_backfill(
    symbol: str,
    *,
    days: int = DEFAULT_DAYS,
    now_utc: Optional[datetime] = None,
    collection_timeout: float | None = None,
) -> None:
    """Fire-and-forget cache warmup without blocking the caller."""

    async def _runner() -> None:
        try:
            await ensure_inspection_daily_cache(
                symbol,
                days=days,
                now_utc=now_utc,
                collection_timeout=collection_timeout,
            )
        except Exception:
            # Background warmup errors are logged inside the pipeline;
            # we swallow them here to avoid affecting the caller.
            return

    LOGGER.info(
        "inspection.cache.ensure.scheduled",
        extra={"symbol": symbol.strip().upper() if symbol else symbol, "days": days},
    )
    asyncio.create_task(_runner())
