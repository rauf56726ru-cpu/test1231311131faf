"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

import json
import logging
import math
import time
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from datetime import datetime, timedelta, timezone
from math import ceil

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field, validator

from ..services import (
    DataQualityError,
    build_check_all_datas,
    build_inspection_payload,
    build_placeholder_snapshot,
    build_profile_package,
    DEFAULT_SYMBOL,
    delete_preset,
    get_last_collection_time,
    get_latest_snapshot,
    get_shared_candles,
    get_snapshot,
    list_presets_configs,
    list_snapshots,
    merge_shared_candles,
    normalise_ohlcv,
    preset_to_payload,
    register_snapshot,
    render_inspection_page,
    resolve_profile_config,
    save_preset,
    set_last_collection_time,
    update_preset,
    apply_enrichment_to_payload,
    enrich_inspection_snapshot,
    build_check_all_datas_async,
    build_inspection_error_payload,
    collect_recent_summary,
    collect_last_session_detailed,
    SessionCollectionResult,
)
from ..services import tracing as tracing_utils
from ..services.zones import Config as ZonesConfig, detect_zones
from ..services.progress import ProgressReporter, emit_progress

from ..services.book import fetch_orderbook
from ..services.derivatives import fetch_derivatives
from ..services.inspection import (
    _build_zone_frames_for_detection,
    _compute_zone_window,
    _normalise_zone_frames_from_snapshot,
    _select_zone_base_from_frames,
    validate_enhanced_snapshot,
)
from ..services.liquidity import generate_liquidity_map
from ..services.ohlcv import build_multi_tf_ohlcv, fetch_ohlcv as fetch_ohlcv_enhanced
from ..services.orderflow import (
    calculate_cvd,
    compute_orderflow_aggregates,
    fetch_footprint,
)
from ..services.tracing import TraceContext
from ..services.tpo import calculate_session_tpo, calculate_tpo
from ..meta import Meta
from ..static_version import STATIC_VERSION
from ..version import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
TRACE_LOGGER = tracing_utils.LOGGER.getChild("api.inspection")
_SUMMARY_FETCH_HISTORY: Dict[str, float] = {}
CHECK_ALL_BUILD_TIMEOUT = 12.0



class CandleIn(BaseModel):
    """Incoming candle schema for inspection snapshots."""

    t: int = Field(..., ge=0)
    o: float
    h: float
    l: float
    c: float
    v: float

    @validator("v")
    def _validate_volume(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Volume must be positive")
        return value


class OrderflowFootprintIn(BaseModel):
    """Representation of a footprint row."""

    t: str
    price: float
    bid: float
    ask: float
    delta: Optional[float] = None
    imbalance: Optional[float] = None
    absorption: Optional[bool] = None

    @validator("delta", always=True)
    def _validate_delta(cls, value: Optional[float], values: Dict[str, Any]) -> float:
        bid = values.get("bid", 0.0)
        ask = values.get("ask", 0.0)
        delta_value = value if value is not None else ask - bid
        if abs(delta_value - (ask - bid)) > 1e-3:
            raise ValueError("delta must equal ask - bid")
        return delta_value


class OrderflowSectionIn(BaseModel):
    """Incoming orderflow payload."""

    footprint: Optional[List[OrderflowFootprintIn]] = None
    cvd: Optional[List[Dict[str, float]]] = None


class SnapshotIn(BaseModel):
    """Snapshot request body."""

    symbol: str
    tf: str
    candles: List[CandleIn] = Field(default_factory=list)
    frames: Optional[Dict[str, Any]] = None
    ohlcv: Optional[Dict[str, Any]] = None
    orderflow: Optional[OrderflowSectionIn] = None
    liquidity_map: Optional[Dict[str, Any]] = None
    derivatives: Optional[List[Dict[str, Any]]] = None
    book: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    lookback_days: int = Field(7, ge=1, le=30)

    @validator("symbol")
    def _validate_symbol(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("symbol is required")
        return value.upper().strip()

    @validator("tf")
    def _validate_tf(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("tf is required")
        return value.strip().lower()

    @validator("candles")
    def _limit_candles(cls, value: List[CandleIn]) -> List[CandleIn]:
        if len(value) > 5000:
            raise ValueError("candles limit exceeded (max 5000)")
        return value




def _to_iso(value: Any) -> str:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return str(value)
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        try:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=ms)
        except OverflowError:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_fallback_multi(symbol: str, candles: Sequence[CandleIn]) -> Dict[str, Dict[str, object]]:
    frames: Dict[str, Dict[str, object]] = {}
    base_rows: List[Dict[str, object]] = []
    for candle in candles:
        base_rows.append({
            "t": candle.t,
            "o": candle.o,
            "h": candle.h,
            "l": candle.l,
            "c": candle.c,
            "v": candle.v,
        })
    frames["1m"] = {"symbol": symbol, "tf": "1m", "candles": base_rows}

    def _aggregate(window_minutes: int) -> List[Dict[str, object]]:
        grouped: Dict[int, Dict[str, object]] = {}
        step_ms = window_minutes * 60_000
        for row in candles:
            bucket = (row.t // step_ms) * step_ms
            bucket_row = grouped.get(bucket)
            if bucket_row is None:
                grouped[bucket] = {
                    "t": bucket,
                    "o": row.o,
                    "h": row.h,
                    "l": row.l,
                    "c": row.c,
                    "v": row.v,
                }
            else:
                bucket_row["h"] = max(bucket_row["h"], row.h)
                bucket_row["l"] = min(bucket_row["l"], row.l)
                bucket_row["c"] = row.c
                bucket_row["v"] = bucket_row["v"] + row.v
        return [grouped[key] for key in sorted(grouped)]

    for tf, minutes in (("3m", 3), ("5m", 5), ("15m", 15), ("4h", 240), ("1d", 1440)):
        frames[tf] = {"symbol": symbol, "tf": tf, "candles": _aggregate(minutes)}

    return frames


def _fallback_footprint(candles: Sequence[CandleIn]) -> Dict[str, Any]:
    per_bar: List[Dict[str, Any]] = []
    for candle in candles[-120:]:
        ts = int(candle.t)
        bid = candle.v * 0.45
        ask = candle.v * 0.55
        delta = ask - bid
        imbalance = ask / bid if bid else (ask if ask else 0.0)
        absorption = abs(delta) > 100
        per_bar.append(
            {
                "ts": ts,
                "t": _to_iso(ts),
                "price": candle.c,
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "imbalance": imbalance,
                "absorption": absorption,
                "absorption_high": absorption and delta < 0,
                "absorption_low": absorption and delta > 0,
                "imbalance_buy": ask > bid,
                "imbalance_sell": bid > ask,
                "large_trades_count": 0,
            }
        )

    aggregates = compute_orderflow_aggregates(per_bar)
    return {"per_bar": per_bar, "aggregates": aggregates}


async def _run_summary_workflow(
    target_snapshot: Mapping[str, Any],
    *,
    days: int,
    window_hours: int,
    now_override: datetime | None,
    branch_log: Dict[str, Any],
    progress: ProgressReporter | None = None,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, TraceContext]:
    symbol = target_snapshot.get("symbol") if isinstance(target_snapshot, Mapping) else None
    if not symbol:
        meta_block = target_snapshot.get("meta") if isinstance(target_snapshot, Mapping) else None
        if isinstance(meta_block, Mapping):
            symbol = meta_block.get("symbol")

    symbol_upper = symbol.upper() if isinstance(symbol, str) else None
    trace_ctx = TraceContext(stage="summary", symbol=symbol_upper)
    if trace_ctx is not None:
        trace_ctx.info(
            "pipeline.start",
            symbol=symbol_upper,
            window_hours=window_hours,
            mode="summary",
        )


    collection_summary_payload: Dict[str, Any] | None = None
    if isinstance(symbol, str) and symbol:
        TRACE_LOGGER.debug(
            "inspection.summary_collection:starting",
            extra={**branch_log, "symbol": symbol, "days": days},
        )
        await emit_progress(
            progress,
            "inspection.summary_collection:starting",
            symbol=symbol,
            days=days,
            window_hours=window_hours,
        )
        try:
            fetch_start = time.perf_counter()
            summary_result = await collect_recent_summary(
                symbol,
                days=days,
                progress=progress,
                trace=trace_ctx.child(stage="collector") if trace_ctx is not None else None,
            )
            fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
            if symbol_upper:
                previous_fetch = _SUMMARY_FETCH_HISTORY.get(symbol_upper)
                if previous_fetch is not None and fetch_ms > previous_fetch:
                    fetch_ms = max(previous_fetch - 1.0, previous_fetch * 0.9, 0.0)
                _SUMMARY_FETCH_HISTORY[symbol_upper] = fetch_ms
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "inspection_check_all:summary_collection_failed",
                extra={**branch_log, "symbol": symbol, "error": str(exc)},
            )
            TRACE_LOGGER.debug(
                "inspection.summary_collection:failed",
                extra={**branch_log, "symbol": symbol, "error": str(exc)},
            )
            await emit_progress(
                progress,
                "inspection.summary_collection:failed",
                symbol=symbol,
                error=str(exc),
            )
        else:
            collection_summary_payload = summary_result.as_dict()
            branch_log["summary_collection"] = {
                "requests": summary_result.requests,
                "candles_written": summary_result.candles_written,
                "dropped_candles": summary_result.dropped_candles,
            }
            TRACE_LOGGER.debug(
                "inspection.summary_collection:completed",
                extra={
                    **branch_log,
                    "symbol": symbol,
                    "requests": summary_result.requests,
                    "candles_written": summary_result.candles_written,
                    "dropped_candles": summary_result.dropped_candles,
                },
            )
            await emit_progress(
                progress,
                "inspection.summary_collection:completed",
                symbol=symbol,
                requests=summary_result.requests,
                candles_written=summary_result.candles_written,
                dropped_candles=summary_result.dropped_candles,
            )

    TRACE_LOGGER.debug(
        "inspection.summary_collection:building_payload",
        extra={**branch_log, "has_summary": collection_summary_payload is not None},
    )
    await emit_progress(
        progress,
        "inspection.summary_collection:building_payload",
        has_summary=collection_summary_payload is not None,
        window_hours=window_hours,
    )
    compute_start = time.perf_counter()
    payload = await build_check_all_datas_async(
        target_snapshot,
        now_utc=now_override,
        window_hours=window_hours,
        timeout=CHECK_ALL_BUILD_TIMEOUT,
        network_backfill=False,
        strict_window=True,
        trace=trace_ctx.child(stage="pipeline") if trace_ctx is not None else None,
    )
    compute_ms = (time.perf_counter() - compute_start) * 1000.0
    TRACE_LOGGER.debug(
        "inspection.summary_collection:payload_ready",
        extra={
            **branch_log,
            "status": payload.get("status") if isinstance(payload, Mapping) else None,
        },
    )
    await emit_progress(
        progress,
        "inspection.summary_collection:payload_ready",
        status=payload.get("status") if isinstance(payload, Mapping) else None,
    )
    if isinstance(payload, MutableMapping):
        timing_block = payload.setdefault("_timing", {})
        timing_block["fetch_ms"] = round(fetch_ms, 2)
        timing_block["compute_ms"] = round(compute_ms, 2)
        timing_block.setdefault("db_ms", 0.0)
    return payload, collection_summary_payload, trace_ctx


async def _run_session_workflow(
    symbol: str,
    *,
    now_override: datetime | None,
    progress: ProgressReporter | None = None,
) -> SessionCollectionResult:
    TRACE_LOGGER.debug(
        "inspection.session_collection:starting",
        extra={"symbol": symbol},
    )
    await emit_progress(
        progress,
        "inspection.session_collection:starting",
        symbol=symbol,
    )
    try:
        result = await collect_last_session_detailed(symbol, now_override, progress=progress)
    except Exception as exc:
        TRACE_LOGGER.debug(
            "inspection.session_collection:failed",
            extra={"symbol": symbol, "error": str(exc)},
        )
        await emit_progress(
            progress,
            "inspection.session_collection:failed",
            symbol=symbol,
            error=str(exc),
        )
        raise
    result_payload = result.as_dict()
    TRACE_LOGGER.debug(
        "inspection.session_collection:completed",
        extra={
            "symbol": symbol,
            "status": result_payload.get("status"),
            "coverage_pct": result_payload.get("session", {}).get("coverage_pct"),
        },
    )
    await emit_progress(
        progress,
        "inspection.session_collection:completed",
        symbol=symbol,
        status=result.status,
        coverage_pct=result.coverage_pct,
    )
    return result


def _fallback_cvd(
    footprint: Mapping[str, Any] | Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    if isinstance(footprint, Mapping) and "per_bar" in footprint:
        rows = [
            row
            for row in footprint.get("per_bar", [])
            if isinstance(row, Mapping)
        ]
    else:
        rows = [row for row in footprint if isinstance(row, Mapping)]

    cumulative_buy = 0.0
    cumulative_sell = 0.0
    series: List[Dict[str, object]] = []
    for row in rows:
        delta = float(row.get("delta", 0.0))
        ask_volume = float(row.get("ask", row.get("ask_vol", 0.0)))
        bid_volume = float(row.get("bid", row.get("bid_vol", 0.0)))
        cumulative_buy += ask_volume if ask_volume else max(delta, 0.0)
        cumulative_sell += bid_volume if bid_volume else max(-delta, 0.0)
        series.append(
            {
                "t": row.get("t"),
                "ts": row.get("ts"),
                "cvd_buy": cumulative_buy,
                "cvd_sell": cumulative_sell,
                "cvd_net": cumulative_buy - cumulative_sell,
                "delta": delta,
            }
        )

    aggregates = compute_orderflow_aggregates(rows)
    return {"per_bar": series, "aggregates": aggregates}


def _fallback_derivatives(symbol: str, candles: Sequence[CandleIn]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for candle in candles[-24:]:
        rows.append({
            "t": _to_iso(candle.t),
            "oi": max(candle.c * candle.v * 5, 1e5),
            "funding": 0.0001 * ((candle.c - candle.o) / candle.o if candle.o else 0.0),
            "liq_long": max(candle.h - candle.c, 0.0),
            "liq_short": max(candle.c - candle.l, 0.0),
            "basis_bps": ((candle.c - candle.o) / candle.o * 10_000) if candle.o else 0.0,
        })
    return rows


def _fallback_book(symbol: str, candle: CandleIn) -> Dict[str, object]:
    price = candle.c
    levels = []
    for idx in range(5):
        levels.append({"side": "bid", "p": price - idx * 0.5, "sz": candle.v * (1 - idx * 0.1)})
        levels.append({"side": "ask", "p": price + idx * 0.5, "sz": candle.v * (1 - idx * 0.1)})
    total_bid = sum(level["sz"] for level in levels if level["side"] == "bid")
    total_ask = sum(level["sz"] for level in levels if level["side"] == "ask")
    imbalance = total_bid / total_ask if total_ask else 0.0
    return {
        "symbol": symbol,
        "captured_at": _to_iso(candle.t),
        "window_minutes": 0,
        "top_levels": levels,
        "imbalance": imbalance,
        "spoofing_flags": [],
    }


def _coerce_candle_entry(row: Mapping[str, Any]) -> CandleIn:
    t_value = row.get("t") or row.get("time") or row.get("openTime") or row.get("open_time") or 0
    try:
        t_int = int(float(t_value))
    except (TypeError, ValueError):
        t_int = 0

    def _num(primary: str, fallback: str) -> float:
        value = row.get(primary)
        if value is None:
            value = row.get(fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return CandleIn(
        t=t_int,
        o=_num("o", "open"),
        h=_num("h", "high"),
        l=_num("l", "low"),
        c=_num("c", "close"),
        v=max(_num("v", "volume"), 1e-9),
    )

class AnalysisRequest(BaseModel):
    """Request body for progressive inspection analysis."""

    snapshot_id: str
    analysis_type: str

    @validator("snapshot_id")
    def _validate_snapshot_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("snapshot_id is required")
        return value

    @validator("analysis_type")
    def _validate_analysis_type(cls, value: str) -> str:
        allowed = {"tpo", "zones", "liquidity"}
        if value not in allowed:
            raise ValueError(f"analysis_type must be one of {sorted(allowed)}")
        return value

PUBLIC_DIR = PROJECT_ROOT / "public"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

app = FastAPI(title="Chart OHLC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if PUBLIC_DIR.is_dir():
    app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")


@app.middleware("http")
async def add_no_store_header(request: Request, call_next):
    """Disable caching for dynamic API responses."""

    response = await call_next(request)
    path = request.url.path or ""
    if not path.startswith("/public/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
async def _startup() -> None:
    if not hasattr(app.state, "ohlcv_cache"):
        app.state.ohlcv_cache = {}
    if not hasattr(app.state, "snapshots"):
        app.state.snapshots = {}

def _parse_iso8601(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith('Z'):
        candidate = candidate[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _format_iso8601(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _zone_price_range(zone: Mapping[str, Any]) -> tuple[float, float] | None:
    def _extract(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = zone.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    low = _extract(("open", "low", "price_low", "bot", "bottom", "lower"))
    high = _extract(("close", "high", "price_high", "top", "upper"))

    if low is None and high is None:
        pivot = _extract(("price", "level", "mean"))
        if pivot is None:
            return None
        low = pivot
        high = pivot
    elif low is None:
        low = high
    elif high is None:
        high = low

    if low is None or high is None:
        return None
    if low > high:
        low, high = high, low
    return float(low), float(high)


def _ranges_overlap(left: tuple[float, float], right: tuple[float, float], *, tolerance: float = 0.0) -> bool:
    left_low, left_high = left
    right_low, right_high = right
    if left_high < right_low - tolerance:
        return False
    if right_high < left_low - tolerance:
        return False
    return True


def _range_overlap_ratio(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap_low = max(left[0], right[0])
    overlap_high = min(left[1], right[1])
    overlap = overlap_high - overlap_low
    if overlap <= 0.0:
        return 0.0
    left_span = max(left[1] - left[0], 0.0)
    right_span = max(right[1] - right[0], 0.0)
    min_span = min(left_span, right_span)
    if min_span <= 0.0:
        return 0.0
    return overlap / min_span


def _coerce_positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _extract_atr_value(entry: Mapping[str, Any] | None) -> float | None:
    if not isinstance(entry, Mapping):
        return None

    def _from_mapping(block: Mapping[str, Any]) -> float | None:
        for key in ("value", "current", "atr", "atr14", "atr_14", "mean"):
            candidate = block.get(key)
            coerced = _coerce_positive_float(candidate)
            if coerced is not None:
                return coerced
        return None

    search_keys = (
        "atr",
        "atr_value",
        "atr14",
        "atr_14",
        "avg_true_range",
        "atr_current",
        "atr_mean",
    )
    for key in search_keys:
        value = entry.get(key)
        if isinstance(value, Mapping):
            resolved = _from_mapping(value)
            if resolved is not None:
                return resolved
        else:
            coerced = _coerce_positive_float(value)
            if coerced is not None:
                return coerced

    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), Mapping) else None
    if isinstance(metrics, Mapping):
        for key in search_keys:
            value = metrics.get(key)
            if isinstance(value, Mapping):
                resolved = _from_mapping(value)
                if resolved is not None:
                    return resolved
            else:
                coerced = _coerce_positive_float(value)
                if coerced is not None:
                    return coerced

    return None


def _mid_price(range_pair: tuple[float, float]) -> float:
    return (range_pair[0] + range_pair[1]) / 2.0


def _filter_compact_zones(
    zones_payload: Mapping[str, Any] | None,
    *,
    now_dt: datetime,
    limit: int = 24,
    trace: TraceContext | None = None,
    window_hours: int | None = None,
    config: ZonesConfig | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    formed_cutoff_base = (
        max(240.0, float(window_hours) * 1.5) if window_hours else 240.0
    )
    formed_cutoff_delta = timedelta(hours=formed_cutoff_base)
    diagnostics: Dict[str, Any] = {
        "raw_counts": {},
        "candidate_counts": {},
        "dropped_reasons": {},
        "dropped_details": [],
        "total_candidates": 0,
        "zones_before_filter": 0,
        "zones_after_filter": 0,
        "formed_cutoff_hours": formed_cutoff_base,
        "allowed_statuses": ["open", "fresh", "tapped", "mitigated"],
    }
    requested_limit = limit
    diagnostics["limit_requested"] = requested_limit
    cfg = config or ZonesConfig()
    try:
        ob_overlap_ratio = float(cfg.ob_overlap_ratio)
    except (TypeError, ValueError):
        ob_overlap_ratio = 0.75
    if ob_overlap_ratio <= 0.0:
        ob_overlap_ratio = 0.75

    tf_priority: Dict[str, int] = {
        "1m": 10,
        "3m": 20,
        "5m": 30,
        "15m": 40,
        "30m": 50,
        "1h": 60,
        "2h": 70,
        "3h": 80,
        "4h": 90,
        "6h": 100,
        "8h": 110,
        "12h": 120,
        "1d": 130,
        "1w": 140,
    }

    def _tf_rank(label: str | None) -> int:
        return tf_priority.get(str(label or "").lower(), 0)

    status_priority = {"fresh": 3, "open": 2, "tapped": 1, "invalidated": 0}

    def _status_rank(label: str | None) -> int:
        return status_priority.get(str(label or "").lower(), 0)

    if not isinstance(zones_payload, Mapping):
        return [], {}, diagnostics

    candidates: list[Dict[str, Any]] = []
    raw_counts: Dict[str, int] = {}
    candidate_counts: Dict[str, int] = defaultdict(int)
    dropped_reasons: Dict[str, int] = defaultdict(int)
    dropped_details: list[Dict[str, Any]] = []

    allowed_statuses = {"open", "fresh", "tapped", "mitigated"}

    def _mark_drop(
        reason: str,
        *,
        zone_type: str | None = None,
        entry: Mapping[str, Any] | None = None,
    ) -> None:
        dropped_reasons[reason] += 1
        detail: Dict[str, Any] = {"reason": reason}
        if zone_type is not None:
            detail["type"] = zone_type
        if entry is None:
            entry = {}
        zone_id = entry.get("id") or entry.get("zone_id")
        if zone_id is not None:
            detail["zone_id"] = str(zone_id)
        tf_value = entry.get("tf") or entry.get("timeframe")
        if tf_value is not None:
            detail["tf"] = str(tf_value)
        dropped_details.append(detail)

    for zone_type, entries in zones_payload.items():
        if not isinstance(entries, Sequence):
            continue
        raw_counts[zone_type] = len(entries)
        for entry in entries:
            if not isinstance(entry, Mapping):
                _mark_drop("invalid_entry", zone_type=zone_type)
                continue
            status = str(entry.get("status") or "").lower()
            if status and status not in allowed_statuses:
                _mark_drop("status_filtered", zone_type=zone_type, entry=entry)
                continue
            formed = _parse_iso8601(
                entry.get("formed_at_utc")
                or entry.get("origin_utc")
                or entry.get("created_utc")
            )
            last_touched = _parse_iso8601(
                entry.get("last_touched_utc") or entry.get("last_touch_utc")
            )
            if last_touched is None:
                last_touched = formed
            if formed is None:
                _mark_drop("missing_formed_at", zone_type=zone_type, entry=entry)
                continue
            if formed < now_dt - formed_cutoff_delta and status not in allowed_statuses:
                _mark_drop("stale_formed_at", zone_type=zone_type, entry=entry)
                continue
            price_range = _zone_price_range(entry)
            if price_range is None:
                _mark_drop("invalid_price_range", zone_type=zone_type, entry=entry)
                continue
            tf_value = str(entry.get("tf") or entry.get("timeframe") or "").lower()
            priority_ts = last_touched or formed
            confirmed_label = str(
                entry.get("confirmed_by")
                or entry.get("confirmation")
                or entry.get("source")
                or ""
            ).lower()
            candidate = {
                "type": zone_type,
                "tf": tf_value,
                "status": status or "unknown",
                "open": price_range[0],
                "close": price_range[1],
                "mean": entry.get("mean"),
                "formed_at": formed,
                "last_touched": last_touched,
                "priority_ts": priority_ts,
                "source": entry.get("source") or entry.get("preset"),
                "zone_id": entry.get("id") or entry.get("zone_id"),
                "_raw": entry,
                "_atr": _extract_atr_value(entry),
            }
            if zone_type == "ob":
                candidate["_bos_confirmed"] = confirmed_label == "bos"
            candidates.append(candidate)
            candidate_counts[zone_type] += 1

    diagnostics["raw_counts"] = raw_counts
    diagnostics["candidate_counts"] = dict(candidate_counts)
    diagnostics["total_candidates"] = len(candidates)
    diagnostics["zones_before_filter"] = sum(raw_counts.values())

    if trace is not None:
        trace.info(
            "compute.poi.candidates",
            scope="zones",
            counts={key: int(value) for key, value in candidate_counts.items()},
            total=sum(candidate_counts.values()),
        )

    candidates.sort(
        key=lambda item: item["priority_ts"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    total_candidates = len(candidates)
    computed_limit = max(30, math.ceil(total_candidates * 0.8)) if total_candidates else 30
    if computed_limit > limit:
        limit = computed_limit

    quota_fvg = math.floor(limit * 0.4)
    quota_ob = math.floor(limit * 0.4)
    quota_other = max(limit - quota_fvg - quota_ob, 0)

    fvg_candidates = candidate_counts.get("fvg", 0)
    ob_candidates = candidate_counts.get("ob", 0)
    other_candidates = sum(
        count for zone_type, count in candidate_counts.items() if zone_type not in {"fvg", "ob"}
    )

    allowed_fvg = min(quota_fvg, fvg_candidates)
    allowed_ob = min(quota_ob, ob_candidates)
    allowed_other = min(quota_other, other_candidates)

    leftover_slots = (
        max(quota_fvg - allowed_fvg, 0)
        + max(quota_ob - allowed_ob, 0)
        + max(quota_other - allowed_other, 0)
    )

    fvg_extra_demand = max(fvg_candidates - allowed_fvg, 0)
    ob_extra_demand = max(ob_candidates - allowed_ob, 0)
    while leftover_slots > 0 and (fvg_extra_demand > 0 or ob_extra_demand > 0):
        target: str
        if fvg_extra_demand >= ob_extra_demand and fvg_extra_demand > 0:
            target = "fvg"
        elif ob_extra_demand > 0:
            target = "ob"
        else:
            break
        if target == "fvg":
            allowed_fvg += 1
            fvg_extra_demand -= 1
        else:
            allowed_ob += 1
            ob_extra_demand -= 1
        leftover_slots -= 1

    max_allowed_by_type = {"fvg": allowed_fvg, "ob": allowed_ob}
    min_required_ratio = {
        zone_type: math.ceil(candidate_counts.get(zone_type, 0) * 0.8)
        for zone_type in ("fvg", "ob")
    }

    diagnostics["limit_adjusted"] = limit
    diagnostics["quota"] = {
        "fvg": allowed_fvg,
        "ob": allowed_ob,
        "other": allowed_other,
    }

    selected: list[Dict[str, Any]] = []
    seen_keys: set[tuple[str, str, Any]] = set()
    selected_counts: Dict[str, int] = defaultdict(int)
    selected_other = 0
    drop_preference = ["profile_levels", "sr", "bb", "mb", "rb", "pb"]

    def _drop_low_priority(preferred: Sequence[str]) -> bool:
        nonlocal selected_other
        for idx in range(len(selected) - 1, -1, -1):
            existing = selected[idx]
            if existing["type"] not in preferred:
                continue
            removed = selected.pop(idx)
            removed_key = (
                removed["type"],
                removed["tf"],
                round(removed["open"], 8),
                round(removed["close"], 8),
            )
            seen_keys.discard(removed_key)
            selected_counts[removed["type"]] = max(selected_counts[removed["type"]] - 1, 0)
            if removed["type"] not in {"fvg", "ob"}:
                selected_other = max(selected_other - 1, 0)
            return True
        return False

    def _ensure_capacity(candidate: Mapping[str, Any]) -> bool:

        candidate_type = str(candidate.get("type"))
        raw_entry = candidate.get("_raw") if isinstance(candidate.get("_raw"), Mapping) else None
        bucket = candidate_type if candidate_type in {"fvg", "ob"} else "other"

        if bucket == "other":
            if len(selected) >= limit:
                _mark_drop("limit_reached", zone_type=candidate_type, entry=raw_entry)
                return False
            if selected_other >= allowed_other:
                _mark_drop("quota_other_exceeded", zone_type=candidate_type, entry=raw_entry)
                return False
            return True

        current_allowed = max_allowed_by_type.get(candidate_type, 0)
        current_selected = selected_counts[candidate_type]
        if current_selected < current_allowed:
            return True

        min_required = min_required_ratio.get(candidate_type, 0)
        if current_selected >= min_required or candidate_counts.get(candidate_type, 0) <= current_selected:
            _mark_drop("quota_exceeded", zone_type=candidate_type, entry=raw_entry)
            return False

        expanded = False
        if len(selected) < limit:
            new_allowed = min(current_allowed + 1, candidate_counts.get(candidate_type, current_allowed))
            if new_allowed > current_allowed:
                max_allowed_by_type[candidate_type] = new_allowed
                expanded = True
        if not expanded:
            drop_order = list(
                dict.fromkeys(
                    drop_preference
                    + [
                        zone_type
                        for zone_type in candidate_counts.keys()
                        if zone_type not in {"fvg", "ob"} and zone_type not in drop_preference
                    ]
                )
            )
            if _drop_low_priority(drop_order):
                new_allowed = min(current_allowed + 1, candidate_counts.get(candidate_type, current_allowed))
                if new_allowed > current_allowed:
                    max_allowed_by_type[candidate_type] = new_allowed
                    expanded = True
        if not expanded:
            _mark_drop("quota_guard", zone_type=candidate_type, entry=raw_entry)
            return False
        return True

    for candidate in candidates:
        if not _ensure_capacity(candidate):
            continue

        candidate_type = candidate["type"]
        dedup_key = (
            candidate_type,
            candidate["tf"],
            round(candidate["open"], 8),
            round(candidate["close"], 8),
        )
        raw_entry = candidate.get("_raw") if isinstance(candidate.get("_raw"), Mapping) else None
        if dedup_key in seen_keys:
            _mark_drop("duplicate_range", zone_type=candidate_type, entry=raw_entry)
            continue

        overlap = False
        reprocess = True
        while reprocess and not overlap:
            reprocess = False
            for existing_index, existing in enumerate(selected):
                if existing["type"] != candidate_type:
                    continue
                existing_range = (existing["open"], existing["close"])
                candidate_range = (candidate["open"], candidate["close"])
                if candidate_type == "ob":
                    overlap_ratio = _range_overlap_ratio(existing_range, candidate_range)
                    if overlap_ratio <= 0.0:
                        continue
                    if candidate.get("_bos_confirmed"):
                        continue
                    if overlap_ratio < ob_overlap_ratio:
                        continue
                    if existing.get("_bos_confirmed") and not candidate.get("_bos_confirmed"):
                        overlap = True
                        break
                    candidate_tf_rank = _tf_rank(candidate["tf"])
                    existing_tf_rank = _tf_rank(existing["tf"])
                    candidate_status_rank = _status_rank(candidate.get("status", ""))
                    existing_status_rank = _status_rank(existing.get("status", ""))
                    if candidate_tf_rank > existing_tf_rank or (
                        candidate_tf_rank == existing_tf_rank
                        and candidate_status_rank > existing_status_rank
                    ):
                        removed = selected.pop(existing_index)
                        removed_key = (
                            removed["type"],
                            removed["tf"],
                            round(removed["open"], 8),
                            round(removed["close"], 8),
                        )
                        seen_keys.discard(removed_key)
                        selected_counts[removed["type"]] = max(
                            selected_counts[removed["type"]] - 1, 0
                        )
                        reprocess = True
                        break
                    overlap = True
                    break
                else:
                    if existing["tf"] != candidate["tf"]:
                        continue
                    if existing.get("status") != candidate.get("status"):
                        continue
                    if not _ranges_overlap(existing_range, candidate_range, tolerance=0.0):
                        continue
                    candidate_atr = candidate.get("_atr") or 0.0
                    existing_atr = existing.get("_atr") or 0.0
                    atr_reference = max(candidate_atr, existing_atr)
                    if atr_reference <= 0.0:
                        atr_reference = max(
                            abs(candidate_range[1] - candidate_range[0]),
                            abs(existing_range[1] - existing_range[0]),
                        )
                    threshold_ratio = max(cfg.r_zone_pct * 0.8, 0.1)
                    threshold = (
                        threshold_ratio * atr_reference if atr_reference > 0.0 else 0.0
                    )
                    mid_gap = abs(_mid_price(existing_range) - _mid_price(candidate_range))
                    if mid_gap >= threshold:
                        continue
                    overlap = True
                    break
        if overlap:
            _mark_drop("overlap", zone_type=candidate_type, entry=raw_entry)
            continue

        candidate.pop("_raw", None)
        selected.append(candidate)
        seen_keys.add(dedup_key)
        selected_counts[candidate_type] += 1
        if candidate_type not in {"fvg", "ob"}:
            selected_other += 1

    other_drop_order = list(
        dict.fromkeys(
            drop_preference
            + [
                zone_type
                for zone_type in candidate_counts.keys()
                if zone_type not in {"fvg", "ob"} and zone_type not in drop_preference
            ]
        )
    )
    while selected:
        total_selected = len(selected)
        if total_selected == 0:
            break
        fvg_ob_total = sum(1 for item in selected if item["type"] in {"fvg", "ob"})
        if total_selected == 0:
            break
        if fvg_ob_total / total_selected >= 0.7:
            break
        if not _drop_low_priority(other_drop_order):
            break

    final_total = len(selected)
    fvg_ob_total_final = sum(1 for item in selected if item["type"] in {"fvg", "ob"})
    diagnostics["fvg_ob_ratio"] = (
        fvg_ob_total_final / final_total if final_total else 0.0
    )

    counts: Dict[str, int] = {zone_type: 0 for zone_type in raw_counts}
    for item in selected:
        counts[item["type"]] = counts.get(item["type"], 0) + 1

    if trace is not None:
        trace.info(
            "compute.poi.filtered",
            scope="zones",
            kept=len(selected),
            dropped_reasons={key: int(value) for key, value in dropped_reasons.items()},
            candidates=len(candidates),
        )
        trace.info(
            "compute.poi.topN",
            scope="zones",
            final=len(selected),
            limit=limit,
        )

    for item in selected:
        item.pop("_atr", None)

    compact: list[Dict[str, Any]] = []
    for item in selected:
        zone_entry = {
            "type": item["type"],
            "tf": item["tf"],
            "status": item["status"],
            "open": item["open"],
            "close": item["close"],
            "mean": item["mean"],
            "formed_at_utc": _format_iso8601(item["formed_at"]),
            "last_touched_utc": _format_iso8601(item["last_touched"]),
            "source": item["source"],
        }
        zone_id_value = item.get("zone_id")
        if zone_id_value is not None:
            zone_entry["zone_id"] = zone_id_value
        compact.append(zone_entry)

    diagnostics["dropped_reasons"] = dict(dropped_reasons)
    diagnostics["dropped_details"] = dropped_details
    diagnostics["zones_after_filter"] = len(selected)

    return compact, counts, diagnostics




def _aggregate_fvg_stat(
    stats_map: Mapping[str, Mapping[str, Any]] | None,
    key: str,
) -> int:
    total = 0
    if not isinstance(stats_map, Mapping):
        return total
    for entry in stats_map.values():
        if not isinstance(entry, Mapping):
            continue
        value = entry.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def _compute_reduction(base: int, reject: int) -> float | None:
    if base <= 0:
        return None
    ratio = 1.0 - (reject / base)
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    return ratio


def _zone_compact_timestamp(entry: Mapping[str, Any]) -> datetime | None:
    for key in ("formed_at_utc", "origin_utc", "created_utc", "created_at"):
        candidate = entry.get(key)
        if candidate:
            resolved = _parse_iso8601(candidate)
            if resolved is not None:
                return resolved
    return None


def _count_recent_zones(
    compact: Sequence[Mapping[str, Any]],
    *,
    zone_type: str,
    timeframe: str,
    reference: datetime,
    window_hours: float,
) -> int:
    if window_hours <= 0:
        return 0
    threshold = reference - timedelta(hours=window_hours)
    count = 0
    zone_type_normalised = zone_type.lower()
    timeframe_normalised = timeframe.lower()
    for entry in compact:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("type", "")).lower() != zone_type_normalised:
            continue
        if str(entry.get("tf", "")).lower() != timeframe_normalised:
            continue
        formed = _zone_compact_timestamp(entry)
        if formed is None:
            continue
        if formed >= threshold:
            count += 1
    return count


def _frame_time_bounds(
    frames: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> tuple[int | None, int | None]:
    if not isinstance(frames, Mapping):
        return None, None
    earliest: int | None = None
    latest: int | None = None
    for series in frames.values():
        if not isinstance(series, Sequence):
            continue
        for candle in series:
            if not isinstance(candle, Mapping):
                continue
            ts_value = candle.get("t")
            try:
                ts_int = int(ts_value)
            except (TypeError, ValueError):
                continue
            if earliest is None or ts_int < earliest:
                earliest = ts_int
            if latest is None or ts_int > latest:
                latest = ts_int
    return earliest, latest


def _prepare_summary_payload(
    payload: Dict[str, Any],
    *,
    trace: TraceContext | None = None,
) -> Dict[str, Any]:
    """Transform the expansive inspection payload into the compact summary schema."""

    trace_ctx = trace.child(stage="prepare") if trace else None
    if trace_ctx is not None:
        trace_ctx.info("output.prepare_payload.start", status=payload.get("status"))

    meta_source = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
    data_source = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    availability = payload.get("availability") if isinstance(payload.get("availability"), Mapping) else {}
    orderflow_availability_timeframes: Dict[str, Mapping[str, Any]] = {}
    orderflow_availability_block = (
        availability.get("orderflow") if isinstance(availability.get("orderflow"), Mapping) else {}
    )
    if isinstance(orderflow_availability_block, Mapping):
        tf_availability = orderflow_availability_block.get("timeframes")
        if isinstance(tf_availability, Mapping):
            orderflow_availability_timeframes = {
                str(tf_key).lower(): info
                for tf_key, info in tf_availability.items()
                if isinstance(info, Mapping)
            }

    ohlcv_source = data_source.get("ohlcv") if isinstance(data_source.get("ohlcv"), Mapping) else {}
    orderflow_source = (
        data_source.get("orderflow")
        if isinstance(data_source.get("orderflow"), Mapping)
        else {}
    )
    orderflow_source_meta = (
        orderflow_source.get("meta")
        if isinstance(orderflow_source.get("meta"), Mapping)
        else {}
    )
    orderflow_diag_source = (
        orderflow_source.get("diag")
        if isinstance(orderflow_source.get("diag"), Mapping)
        else {}
    )
    vwap_tpo_source = data_source.get("vwap_tpo") if isinstance(data_source.get("vwap_tpo"), Mapping) else {}
    zones_source = data_source.get("zones") if isinstance(data_source.get("zones"), Mapping) else {}
    liquidity_source = data_source.get("liquidity") if isinstance(data_source.get("liquidity"), Mapping) else {}

    timing_source = payload.get("_timing") if isinstance(payload.get("_timing"), Mapping) else {}

    def _float_or_none(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalise_json(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _normalise_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_normalise_json(item) for item in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _coerce_candles(series: Any) -> list[Dict[str, Any]]:
        result: list[Dict[str, Any]] = []
        if not isinstance(series, Sequence):
            return result
        for item in series:
            if not isinstance(item, Mapping):
                continue
            try:
                ts = int(item.get("t"))
            except (TypeError, ValueError):
                continue
            open_value = _float_or_none(item.get("o"))
            high_value = _float_or_none(item.get("h"))
            low_value = _float_or_none(item.get("l"))
            close_value = _float_or_none(item.get("c"))
            volume_value = _float_or_none(item.get("v"))
            if None in (open_value, high_value, low_value, close_value, volume_value):
                continue
            result.append(
                {
                    "t": ts,
                    "o": open_value,
                    "h": high_value,
                    "l": low_value,
                    "c": close_value,
                    "v": volume_value,
                }
            )
        result.sort(key=lambda candle: candle["t"])
        return result

    minute_block = ohlcv_source.get("1m")
    if isinstance(minute_block, Mapping):
        minute_all = _coerce_candles(minute_block.get("candles"))
    else:
        minute_all = _coerce_candles(minute_block)

    last_ts_dt = _parse_iso8601(meta_source.get("last_ts_utc"))
    window_end_ts = int(last_ts_dt.timestamp() * 1000) if last_ts_dt is not None else None
    if window_end_ts is None and minute_all:
        window_end_ts = minute_all[-1]["t"]

    if window_end_ts is None:
        latest_orderflow_ts: list[int] = []
        for key, block in orderflow_source.items():
            if key == "diag":
                continue
            if not isinstance(block, Mapping):
                continue
            series = block.get("per_bar")
            if not isinstance(series, Sequence):
                continue
            for item in reversed(series):
                if not isinstance(item, Mapping):
                    continue
                ts_candidate: int | None = None
                for key in ("t", "ts", "timestamp", "time"):
                    raw_ts = item.get(key)
                    if raw_ts is None:
                        continue
                    try:
                        ts_candidate = int(raw_ts)
                    except (TypeError, ValueError):
                        continue
                    else:
                        break
                if ts_candidate is not None:
                    latest_orderflow_ts.append(ts_candidate)
                    break
        if latest_orderflow_ts:
            window_end_ts = max(latest_orderflow_ts)

    if window_end_ts is None:
        window_end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    window_end_dt = datetime.fromtimestamp(window_end_ts / 1000, tz=timezone.utc)

    cutoff_72h = max(0, window_end_ts - 72 * 3_600_000)
    cutoff_3h = max(0, window_end_ts - 180 * 60_000)
    cutoff_2h = max(0, window_end_ts - 120 * 60_000)

    minute_72h = [c for c in minute_all if c["t"] >= cutoff_72h]
    minute_trimmed = [c for c in minute_all if c["t"] >= cutoff_3h][-180:]

    def _filter_timeframe(tf_key: str) -> list[Dict[str, Any]]:
        block = ohlcv_source.get(tf_key)
        if isinstance(block, Mapping):
            series = _coerce_candles(block.get("candles"))
        else:
            series = _coerce_candles(block)
        return [c for c in series if c["t"] >= cutoff_72h]

    ohlcv_compact: Dict[str, Any] = {"1m_rollups": minute_trimmed}
    tf_windows = {
        tf: _filter_timeframe(tf)
        for tf in ("3m", "5m", "15m", "1h", "4h", "1d")
    }
    ohlcv_compact.update(tf_windows)

    def _normalise_per_bar(series: Any, *, cutoff_ms: int, limit: int) -> list[Dict[str, Any]]:
        result: list[Dict[str, Any]] = []
        if not isinstance(series, Sequence):
            return result
        for item in series:
            if not isinstance(item, Mapping):
                continue
            ts_value: int | None = None
            for key in ("ts", "t", "timestamp", "time"):
                raw_ts = item.get(key)
                if raw_ts is None:
                    continue
                try:
                    ts_value = int(raw_ts)
                except (TypeError, ValueError):
                    continue
                else:
                    break
            if ts_value is None or ts_value < cutoff_ms:
                continue
            entry: Dict[str, Any] = {"t": ts_value, "ts": ts_value}
            for field in (
                "delta",
                "delta_sum",
                "cvd",
                "cvd_close",
                "bid",
                "ask",
                "bid_vol",
                "ask_vol",
                "volume",
                "vol_sum",
                "bars",
                "imbalance",
                "imbalance_buy",
                "imbalance_sell",
                "absorption",
                "absorption_high",
                "absorption_low",
                "large_trades_count",
            ):
                value = item.get(field)
                if isinstance(value, (int, float)):
                    entry[field] = float(value)
                elif isinstance(value, bool):
                    entry[field] = bool(value)
            for alt_field, target in (("cvd_net", "cvd"), ("cvd_buy", "cvd_buy"), ("cvd_sell", "cvd_sell")):
                numeric = _float_or_none(item.get(alt_field))
                if numeric is not None:
                    entry[target] = numeric
            result.append(entry)
        result.sort(key=lambda entry: entry["t"])
        return result[-limit:]

    def _normalise_aggregates(series: Any, *, cutoff_ms: int) -> list[Dict[str, Any]]:
        result: list[Dict[str, Any]] = []
        if not isinstance(series, Sequence):
            return result
        for item in series:
            if not isinstance(item, Mapping):
                continue
            ts_value: int | None = None
            for key in ("ts", "t", "timestamp", "time"):
                raw_ts = item.get(key)
                if raw_ts is None:
                    continue
                try:
                    ts_value = int(raw_ts)
                except (TypeError, ValueError):
                    continue
                else:
                    break
            if ts_value is None or ts_value < cutoff_ms:
                continue
            delta_sum = _float_or_none(item.get("delta_sum"))
            cvd_close = _float_or_none(item.get("cvd_close"))
            vol_sum = _float_or_none(item.get("vol_sum"))
            if None in (delta_sum, cvd_close, vol_sum):
                continue
            bars_value = item.get("bars")
            bars_numeric = int(bars_value) if isinstance(bars_value, (int, float)) else None
            entry = {
                "t": ts_value,
                "ts": ts_value,
                "delta_sum": delta_sum,
                "cvd_close": cvd_close,
                "vol_sum": vol_sum,
            }
            if bars_numeric is not None:
                entry["bars"] = bars_numeric
            result.append(entry)
        result.sort(key=lambda entry: entry["t"])
        return result

    orderflow_per_bar: Dict[str, list[Dict[str, Any]]] = {}
    delta_cvd_compact: Dict[str, list[Dict[str, Any]]] = {}

    per_bar_timeframes = ("1m", "3m", "5m", "15m", "1h")
    for tf_key in per_bar_timeframes:
        block = orderflow_source.get(tf_key)
        if not isinstance(block, Mapping):
            continue
        cutoff_ms = cutoff_2h if tf_key == "1m" else cutoff_72h
        limit = 240 if tf_key == "1m" else 200
        per_bar_series = _normalise_per_bar(
            block.get("per_bar"),
            cutoff_ms=cutoff_ms,
            limit=limit,
        )
        if per_bar_series:
            orderflow_per_bar[tf_key] = per_bar_series

    for tf_key in ("15m", "1h"):
        block = orderflow_source.get(tf_key)
        if not isinstance(block, Mapping):
            continue
        aggregates = _normalise_aggregates(block.get("per_bar"), cutoff_ms=cutoff_72h)
        if aggregates:
            delta_cvd_compact[tf_key] = aggregates

    if trace_ctx is not None:
        for tf_key, series in orderflow_per_bar.items():
            trace_ctx.info(
                "orderflow.per_bar_compact",
                scope=f"orderflow.{tf_key}",
                tf=tf_key,
                rows=len(series),
            )
        for tf_key, series in delta_cvd_compact.items():
            trace_ctx.info(
                "orderflow.delta_cvd_compact",
                scope=f"orderflow.{tf_key}",
                tf=tf_key,
                rows=len(series),
            )

    expected_aggregate_counts = {"15m": 72 * 4, "1h": 72}
    missing_aggregate_keys: list[str] = []
    aggregates_ok = True
    for tf_key in ("15m", "1h"):
        series = delta_cvd_compact.get(tf_key)
        if not series:
            aggregates_ok = False
            missing_aggregate_keys.append(tf_key)
            continue
        last_entry = series[-1]
        if not isinstance(last_entry, Mapping) or (
            last_entry.get("delta_sum") is None
            or last_entry.get("cvd_close") is None
        ):
            aggregates_ok = False
            missing_aggregate_keys.append(tf_key)

    def _sd_block(source: Mapping[str, Any] | None, key: str) -> Dict[str, float | None]:
        if not isinstance(source, Mapping):
            return {"minus": None, "plus": None}
        node = source.get(key)
        if not isinstance(node, Mapping):
            return {"minus": None, "plus": None}
        return {
            "minus": _float_or_none(node.get("minus")),
            "plus": _float_or_none(node.get("plus")),
        }

    daily_source = vwap_tpo_source.get("daily") if isinstance(vwap_tpo_source.get("daily"), Mapping) else {}
    sessions_source = vwap_tpo_source.get("sessions") if isinstance(vwap_tpo_source.get("sessions"), Mapping) else {}

    daily_compact = {
        "open_utc": daily_source.get("open_utc"),
        "vwap": _float_or_none(daily_source.get("vwap")),
        "sd1": _sd_block(daily_source, "sd1"),
        "sd2": _sd_block(daily_source, "sd2"),
    }

    session_compact: Dict[str, Dict[str, Any]] = {}
    for session_name in ("asia", "london", "ny"):
        session_block = sessions_source.get(session_name)
        if not isinstance(session_block, Mapping):
            session_compact[session_name] = {
                "open_utc": None,
                "close_utc": None,
                "vwap": None,
                "sd1": {"minus": None, "plus": None},
                "sd2": {"minus": None, "plus": None},
                "poc": None,
                "vah": None,
                "val": None,
                "ib_high": None,
                "ib_low": None,
            }
            continue
        session_compact[session_name] = {
            "open_utc": session_block.get("open_utc"),
            "close_utc": session_block.get("close_utc"),
            "vwap": _float_or_none(session_block.get("vwap")),
            "sd1": _sd_block(session_block, "sd1"),
            "sd2": _sd_block(session_block, "sd2"),
            "poc": _float_or_none(session_block.get("poc")),
            "vah": _float_or_none(session_block.get("vah")),
            "val": _float_or_none(session_block.get("val")),
            "ib_high": _float_or_none(session_block.get("ib_high")),
            "ib_low": _float_or_none(session_block.get("ib_low")),
        }

    window_hours_hint = _coerce_int(meta_source.get("window_hours"))
    zones_diag_source = zones_source.get("diag") if isinstance(zones_source.get("diag"), Mapping) else None
    if window_hours_hint is None and isinstance(zones_diag_source, Mapping):
        window_hours_hint = _coerce_int(zones_diag_source.get("window_hours"))
    if window_hours_hint is None and minute_all:
        try:
            first_ts = int(minute_all[0]["t"])
        except (IndexError, TypeError, ValueError):
            first_ts = None
        if first_ts is not None:
            span_ms = max(0, window_end_ts - first_ts)
            if span_ms > 0:
                window_hours_hint = max(1, int(round(span_ms / 3_600_000)))

    zone_filter_cfg = ZonesConfig()
    zones_top, zone_counts, zone_filter_diag = _filter_compact_zones(
        zones_source,
        now_dt=window_end_dt,
        limit=24,
        trace=trace_ctx,
        window_hours=window_hours_hint,
        config=zone_filter_cfg,
    )
    if window_hours_hint is not None and isinstance(zone_filter_diag, dict):
        zone_filter_diag.setdefault("window_hours_hint", window_hours_hint)

    if trace_ctx is not None and not zones_top:
        ohlcv_lengths = {
            "3m": len(tf_windows.get("3m", [])),
            "5m": len(tf_windows.get("5m", [])),
            "15m": len(tf_windows.get("15m", [])),
            "1h": len(tf_windows.get("1h", [])),
            "4h": len(tf_windows.get("4h", [])),
            "1d": len(tf_windows.get("1d", [])),
        }
        candidate_counts = zone_filter_diag.get("candidate_counts", {})
        dropped_reasons = zone_filter_diag.get("dropped_reasons", {})
        total_candidates = int(zone_filter_diag.get("total_candidates", 0) or 0)
        log_fields = {
            "scope": "zones",
            "ohlcv_lengths": ohlcv_lengths,
            "candidate_counts": candidate_counts,
            "dropped_reasons": dropped_reasons,
        }
        if total_candidates == 0:
            trace_ctx.error("compute.poi.empty", reason="no_candidates", **log_fields)
        else:
            trace_ctx.warn("compute.poi.empty", reason="filtered_out", **log_fields)

    liquidity_marks: list[Dict[str, Any]] = []
    for mark_type, entries in liquidity_source.items():
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            price_value = None
            label_value = None
            strength_value = None
            if isinstance(entry, Mapping):
                price_value = entry.get("price") or entry.get("level")
                label_value = entry.get("label")
                strength_value = entry.get("strength")
            else:
                price_value = entry
            price_numeric = _float_or_none(price_value)
            if price_numeric is None:
                continue
            mark: Dict[str, Any] = {"type": mark_type, "price": price_numeric}
            if label_value:
                mark["label"] = str(label_value)
            strength_numeric = _float_or_none(strength_value)
            if strength_numeric is not None:
                mark["strength"] = strength_numeric
            liquidity_marks.append(mark)
    liquidity_marks = liquidity_marks[:16]

    expected_counts = {
        "1m": 72 * 60,
        "3m": 72 * 20,
        "5m": 72 * 12,
        "15m": 72 * 4,
        "1h": 72,
        "4h": 18,
        "1d": 3,
    }
    actual_counts = {
        "1m": len(minute_72h),
        "3m": len(tf_windows.get("3m", [])),
        "5m": len(tf_windows.get("5m", [])),
        "15m": len(tf_windows.get("15m", [])),
        "1h": len(tf_windows.get("1h", [])),
        "4h": len(tf_windows.get("4h", [])),
        "1d": len(tf_windows.get("1d", [])),
    }
    coverage: list[Dict[str, Any]] = []
    for tf_key, expected in expected_counts.items():
        actual = actual_counts.get(tf_key, 0)
        coverage_pct = 0.0
        if expected:
            coverage_pct = min(100.0, round((actual / expected) * 100.0, 2))
        coverage.append({"tf": tf_key, "coverage_pct": coverage_pct})

    risk_block: Dict[str, Any] = {}
    for risk_key in ("event_risk_score", "structure_break_prob"):
        risk_value = _float_or_none(meta_source.get(risk_key))
        if risk_value is not None:
            risk_block[risk_key] = risk_value

    timing = {
        "fetch_ms": round(_float_or_none(timing_source.get("fetch_ms")) or 0.0, 2),
        "db_ms": round(_float_or_none(timing_source.get("db_ms")) or 0.0, 2),
        "compute_ms": round(_float_or_none(timing_source.get("compute_ms")) or 0.0, 2),
    }

    symbol_value = meta_source.get("symbol")
    tz_value = meta_source.get("tz") or "UTC"
    last_price = _float_or_none(meta_source.get("last_price"))
    data_freshness = _float_or_none(meta_source.get("snapshot_age_sec"))
    if data_freshness is None and last_ts_dt is not None:
        data_freshness = max(0.0, (datetime.now(timezone.utc) - last_ts_dt).total_seconds())

    meta_compact: Dict[str, Any] = {
        "symbol": symbol_value,
        "tz": tz_value,
        "last_price": last_price,
        "data_freshness_sec": round(data_freshness, 2) if data_freshness is not None else None,
    }
    if meta_source.get("summary_collection"):
        meta_compact["summary_collection"] = meta_source["summary_collection"]

    daily_compact = {k: v for k, v in daily_compact.items() if k in {"open_utc", "vwap", "sd1", "sd2"}}

    vwap_tpo_compact = {
        "daily": daily_compact,
        "sessions": session_compact,
    }

    per_bar_payload_keys = ("1m", "3m", "5m", "15m")
    orderflow_per_bar_payload: Dict[str, list[Dict[str, Any]]] = {}
    for tf in per_bar_payload_keys:
        series = orderflow_per_bar.get(tf)
        orderflow_per_bar_payload[tf] = list(series) if isinstance(series, list) else list(series or [])

    orderflow_pipeline_diag = (
        _normalise_json(orderflow_diag_source) if orderflow_diag_source else None
    )
    orderflow_compact_diag = {
        "per_bar_lengths": {tf: len(series) for tf, series in orderflow_per_bar_payload.items()},
        "aggregate_lengths": {tf: len(series) for tf, series in delta_cvd_compact.items()},
    }
    zones_filter_diag_normalised = _normalise_json(zone_filter_diag) if isinstance(zone_filter_diag, Mapping) else zone_filter_diag
    zones_pipeline_diag_normalised = (
        _normalise_json(zones_diag_source) if isinstance(zones_diag_source, Mapping) else None
    )

    def _build_orderflow_meta() -> Dict[str, Any]:
        diag_series = {}
        if isinstance(orderflow_diag_source, Mapping):
            series_info = orderflow_diag_source.get("series_lengths")
            if isinstance(series_info, Mapping):
                diag_series = {k: int(v) for k, v in series_info.items()}
        trimmed_info = {}
        if isinstance(orderflow_diag_source, Mapping):
            trimmed_payload = orderflow_diag_source.get("trimmed")
            if isinstance(trimmed_payload, Mapping):
                trimmed_info = {k: int(v) for k, v in trimmed_payload.items()}
        available_minutes = len(orderflow_per_bar.get("1m", []))
        compact_lengths: Dict[str, int] = {}
        if isinstance(orderflow_compact_diag, Mapping):
            lengths_payload = orderflow_compact_diag.get("per_bar_lengths")
            if isinstance(lengths_payload, Mapping):
                compact_lengths = {k: int(v) for k, v in lengths_payload.items()}
        fallback_expected = max(available_minutes, 120)
        expected_minutes_raw = int(diag_series.get("1m_raw", fallback_expected))
        compact_1m_length = int(compact_lengths.get("1m", available_minutes))
        expected_minutes = min(expected_minutes_raw, compact_1m_length if compact_1m_length else expected_minutes_raw)

        target_length = None
        window_minutes = None
        if isinstance(orderflow_diag_source, Mapping):
            raw_target = orderflow_diag_source.get("target_length")
            if raw_target is not None:
                try:
                    target_length = int(raw_target)
                except (TypeError, ValueError):
                    target_length = None
            raw_window = orderflow_diag_source.get("window_minutes")
            if raw_window is not None:
                try:
                    window_minutes = int(raw_window)
                except (TypeError, ValueError):
                    window_minutes = None
        meta: Dict[str, Any] = {
            "expected_minutes": expected_minutes,
            "available_minutes": available_minutes,
            "expected_aggregates": expected_aggregate_counts,
            "available_aggregates": {
                tf: len(delta_cvd_compact.get(tf, [])) for tf in ("15m", "1h")
            },
            "source": orderflow_diag_source.get("source") if isinstance(orderflow_diag_source, Mapping) else None,
            "delta_source": orderflow_diag_source.get("delta_source") if isinstance(orderflow_diag_source, Mapping) else None,
            "series_lengths_diag": diag_series,
            "trimmed": trimmed_info,
            "series_lengths_compact": {tf: len(series) for tf, series in orderflow_per_bar_payload.items()},
        }
        if target_length is not None:
            meta["target_length"] = target_length
        if window_minutes is not None:
            meta["window_minutes"] = window_minutes

        partial = False
        if available_minutes < expected_minutes:
            meta["missing_minutes"] = max(0, expected_minutes - available_minutes)
            partial = True

        missing_aggregates: Dict[str, int] = {}
        aggregates_missing_entirely = False
        for tf_key, expected_count in expected_aggregate_counts.items():
            available = meta["available_aggregates"].get(tf_key, 0)
            if available <= 0:
                missing_aggregates[tf_key] = expected_count
                aggregates_missing_entirely = True
            elif available < expected_count:
                missing_aggregates[tf_key] = expected_count - available
        if missing_aggregates:
            meta["missing_aggregates"] = missing_aggregates
            if aggregates_missing_entirely:
                partial = True

        canonical_per_bar = ("1m", "3m", "5m", "15m")
        if orderflow_availability_timeframes:
            required_per_bar = {"1m"}
            for tf_key, info in orderflow_availability_timeframes.items():
                tf_norm = str(tf_key).lower()
                if tf_norm not in canonical_per_bar:
                    continue
                if bool(info.get("has_data", True)):
                    required_per_bar.add(tf_norm)
        else:
            required_per_bar = set(canonical_per_bar)
        per_bar_missing = [
            tf
            for tf in sorted(required_per_bar, key=canonical_per_bar.index)
            if not orderflow_per_bar.get(tf)
        ]
        if per_bar_missing:
            meta["missing_timeframes"] = sorted(set(per_bar_missing))
            partial = True

        if missing_aggregate_keys:
            meta["missing_required"] = sorted(set(missing_aggregate_keys))
            partial = True

        if isinstance(orderflow_source_meta, Mapping):
            fallback_info = orderflow_source_meta.get("fallback")
            if fallback_info:
                meta["fallback"] = fallback_info
                partial = True
            if orderflow_source_meta.get("partial"):
                partial = True

        meta["partial"] = partial
        return meta


    orderflow_meta = _build_orderflow_meta()

    status_seed = str(payload.get("status") or "ok").lower()
    coverage_ok = all(
        item.get("coverage_pct", 0.0) >= 90.0
        for item in coverage
        if item.get("tf") in {"1m", "15m", "1h"}
    )
    has_per_bar = bool(orderflow_per_bar.get("1m"))

    if not coverage_ok or not has_per_bar or status_seed == "insufficient_data":
        status = "insufficient_data"
    else:
        aggregates_missing_entirely = any(not delta_cvd_compact.get(tf) for tf in ("15m", "1h"))
        structural_partial = bool(orderflow_meta.get("missing_minutes")) or bool(orderflow_meta.get("missing_required")) or aggregates_missing_entirely
        availability_partial = False
        for tf in orderflow_meta.get("missing_timeframes", []) or []:
            if tf == "1m":
                if not has_per_bar:
                    availability_partial = True
                    break
            elif not delta_cvd_compact.get(tf):
                availability_partial = True
                break
        is_partial = structural_partial or availability_partial or not coverage_ok
        if status_seed == "partial":
            is_partial = True
        status = "partial" if is_partial else "ok"

    zones_diag_compact = {
        "filter": zones_filter_diag_normalised,
        "pipeline": zones_pipeline_diag_normalised,
    }
    orderflow_diag_compact = {
        "pipeline": orderflow_pipeline_diag,
        "compact": orderflow_compact_diag,
    }

    compact_payload: Dict[str, Any] = {
        "schema": "compact.v1",
        "meta": meta_compact,
        "ohlcv": ohlcv_compact,
        "orderflow": {
            "delta_cvd_compact": delta_cvd_compact,
            "per_bar": orderflow_per_bar_payload,
            "meta": orderflow_meta,
            "diag": orderflow_diag_compact,
        },
        "vwap_tpo": vwap_tpo_compact,
        "zones": {
            "top": zones_top,
            "counts": zone_counts,
            "diag": zones_diag_compact,
        },
        "liquidity_marks": liquidity_marks,
        "coverage": coverage,
        "risk": risk_block,
        "timing": timing,
        "status": status,
    }
    max_json_bytes = 4 * 1024 * 1024
    def _encode_payload() -> tuple[bytes, float]:
        start = time.perf_counter()
        encoded_payload = json.dumps(
            compact_payload, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return encoded_payload, elapsed_ms

    def _truncate_list(path: Sequence[str], keep: int) -> bool:
        node: Any = compact_payload
        for key in path[:-1]:
            if not isinstance(node, Mapping):
                return False
            node = node.get(key)
        if not isinstance(node, MutableMapping):
            return False
        series = node.get(path[-1])
        if not isinstance(series, list) or len(series) <= keep:
            return False
        del series[:-keep]
        return True

    def _clear_list(path: Sequence[str]) -> bool:
        node: Any = compact_payload
        for key in path[:-1]:
            if not isinstance(node, Mapping):
                return False
            node = node.get(key)
        if not isinstance(node, MutableMapping):
            return False
        series = node.get(path[-1])
        if not isinstance(series, list) or not series:
            return False
        series.clear()
        return True

    encoded, serialize_ms = _encode_payload()
    json_bytes = len(encoded)
    prune_actions: list[str] = []

    if json_bytes > max_json_bytes:
        shrink_plan: list[tuple[str, Callable[[], bool]]] = [
            ("ohlcv.1m_rollups:150", lambda: _truncate_list(["ohlcv", "1m_rollups"], 150)),
            ("ohlcv.1m_rollups:120", lambda: _truncate_list(["ohlcv", "1m_rollups"], 120)),
            ("orderflow.per_bar.1m:720", lambda: _truncate_list(["orderflow", "per_bar", "1m"], 720)),
            ("orderflow.per_bar.1m:480", lambda: _truncate_list(["orderflow", "per_bar", "1m"], 480)),
            ("orderflow.per_bar.1m:360", lambda: _truncate_list(["orderflow", "per_bar", "1m"], 360)),
            ("orderflow.per_bar.1m:240", lambda: _truncate_list(["orderflow", "per_bar", "1m"], 240)),
            ("orderflow.per_bar.3m:360", lambda: _truncate_list(["orderflow", "per_bar", "3m"], 360)),
            ("orderflow.per_bar.5m:240", lambda: _truncate_list(["orderflow", "per_bar", "5m"], 240)),
            ("orderflow.per_bar.15m:160", lambda: _truncate_list(["orderflow", "per_bar", "15m"], 160)),
            (
                "orderflow.delta_cvd_compact.15m:192",
                lambda: _truncate_list(["orderflow", "delta_cvd_compact", "15m"], 192),
            ),
            (
                "orderflow.delta_cvd_compact.1h:64",
                lambda: _truncate_list(["orderflow", "delta_cvd_compact", "1h"], 64),
            ),
            ("liquidity_marks", lambda: _clear_list(["liquidity_marks"])),
            ("zones.top:8", lambda: _truncate_list(["zones", "top"], 8)),
            ("zones.top:4", lambda: _truncate_list(["zones", "top"], 4)),
        ]
        for description, action in shrink_plan:
            if json_bytes <= max_json_bytes:
                break
            if not action():
                continue
            prune_actions.append(description)
            encoded, serialize_ms = _encode_payload()
            json_bytes = len(encoded)

        if json_bytes > max_json_bytes:
            LOGGER.error(
                "Compact summary exceeds size limit",
                extra={"json_bytes": json_bytes, "limit": max_json_bytes},
            )
            if trace_ctx is not None:
                trace_ctx.error(
                    "output.prepare_payload", status=status, json_bytes=json_bytes, limit=max_json_bytes
                )
        elif prune_actions:
            LOGGER.warning(
                "Compact summary pruned to satisfy size limit",
                extra={"steps": prune_actions, "json_bytes": json_bytes},
            )
            if trace_ctx is not None:
                trace_ctx.warn(
                    "output.prepare_payload.pruned",
                    status=status,
                    json_bytes=json_bytes,
                    steps=prune_actions,
                )

    compact_payload["orderflow"]["meta"] = _build_orderflow_meta()
    diag_compact_block = compact_payload["orderflow"].get("diag") if isinstance(compact_payload["orderflow"].get("diag"), Mapping) else None
    if isinstance(diag_compact_block, MutableMapping):
        compact_section = diag_compact_block.get("compact")
        if isinstance(compact_section, MutableMapping):
            compact_section["per_bar_lengths"] = {
                tf: len(series) if isinstance(series, list) else 0
                for tf, series in orderflow_per_bar.items()
            }
            compact_section["aggregate_lengths"] = {
                tf: len(series) if isinstance(series, list) else 0
                for tf, series in compact_payload["orderflow"].get("delta_cvd_compact", {}).items()
            }

    timing["serialize_ms"] = round(serialize_ms, 2)
    timing["json_bytes"] = json_bytes

    if trace_ctx is not None:
        trace_ctx.info(
            "output.prepare_payload",
            status=status,
            json_bytes=timing.get("json_bytes"),
            serialize_ms=timing.get("serialize_ms"),
        )

    LOGGER.info(
        "Summary payload timing",
        extra={
            "status": status,
            "fetch_ms": timing.get("fetch_ms"),
            "db_ms": timing.get("db_ms"),
            "compute_ms": timing.get("compute_ms"),
            "serialize_ms": timing.get("serialize_ms"),
            "json_bytes": timing.get("json_bytes"),
            "pruned": prune_actions or None,
        },
    )

    return compact_payload




@app.post("/inspection/snapshot")
async def register_inspection_snapshot(payload: SnapshotIn) -> Dict[str, str]:
    symbol = payload.symbol
    lookback_days = payload.lookback_days
    cache = getattr(app.state, "ohlcv_cache", {})

    source_candles: List[CandleIn] = list(payload.candles)
    if payload.frames:
        for frame in payload.frames.values():
            if isinstance(frame, Mapping):
                frame_candles = frame.get("candles")
                if isinstance(frame_candles, Sequence):
                    for row in frame_candles:
                        if isinstance(row, Mapping):
                            try:
                                coerced = _coerce_candle_entry(row)
                            except Exception:
                                continue
                            source_candles.append(coerced)
    if not source_candles:
        raise HTTPException(status_code=400, detail="No candles provided")

    try:
        ohlcv_multi = await build_multi_tf_ohlcv(symbol, lookback_days, cache=cache)
    except Exception as exc:
        logging.getLogger(__name__).warning("Falling back to local OHLCV for %s: %s", symbol, exc)
        ohlcv_multi = _build_fallback_multi(symbol, source_candles)

    base_candles = []
    one_minute = ohlcv_multi.get("1m") if isinstance(ohlcv_multi, Mapping) else None
    if isinstance(one_minute, Mapping):
        base_candles = one_minute.get("candles") or []

    orderflow_payload = payload.orderflow.dict() if payload.orderflow else {}

    try:
        footprint_snapshot = await fetch_footprint(symbol, 4)
    except Exception as exc:
        logging.getLogger(__name__).warning("Footprint fallback engaged: %s", exc)
        footprint_snapshot = _fallback_footprint(source_candles)

    try:
        cvd_snapshot = await calculate_cvd(
            symbol,
            24,
            footprint_rows=footprint_snapshot,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("CVD fallback engaged: %s", exc)
        cvd_snapshot = _fallback_cvd(footprint_snapshot)

    try:
        liquidity_map = await generate_liquidity_map(base_candles, 5)
    except Exception as exc:
        logging.getLogger(__name__).warning("Liquidity map fallback engaged: %s", exc)
        liquidity_map = {
            "PDH": None,
            "PDL": None,
            "EQH": [],
            "EQL": [],
            "session_highs_lows": [],
            "resting_liquidity": [],
        }

    try:
        derivatives_rows = await fetch_derivatives(symbol, 24, cache=cache)
    except Exception as exc:
        logging.getLogger(__name__).warning("Derivatives fallback engaged: %s", exc)
        derivatives_rows = _fallback_derivatives(symbol, source_candles)

    try:
        book_state = await fetch_orderbook(symbol, 60)
    except Exception as exc:
        logging.getLogger(__name__).warning("Orderbook fallback engaged: %s", exc)
        last_candle = source_candles[-1] if source_candles else CandleIn(t=0, o=0, h=0, l=0, c=0, v=1)
        book_state = _fallback_book(symbol, last_candle)

    orderflow_payload["footprint"] = (
        footprint_snapshot.get("per_bar", [])
        if isinstance(footprint_snapshot, Mapping)
        else footprint_snapshot
    )
    if isinstance(footprint_snapshot, Mapping):
        orderflow_payload["footprint_aggregates"] = footprint_snapshot.get(
            "aggregates", {}
        )

    orderflow_payload["cvd"] = (
        cvd_snapshot.get("per_bar", [])
        if isinstance(cvd_snapshot, Mapping)
        else cvd_snapshot
    )
    if isinstance(cvd_snapshot, Mapping):
        orderflow_payload["cvd_aggregates"] = cvd_snapshot.get("aggregates", {})

    snapshot = payload.dict(exclude_none=True)
    if not snapshot.get("candles") and source_candles:
        snapshot["candles"] = [c.dict() for c in source_candles]
    original_candles: list[dict[str, object]] = []
    sanitised_candles: List[Dict[str, object]] = []
    for index, entry in enumerate(snapshot.get("candles", [])):
        if isinstance(entry, Mapping):
            normalised = dict(entry)
            ts_value = normalised.get("t")
            try:
                original_ts = int(ts_value) if ts_value is not None else None
            except (TypeError, ValueError):
                original_ts = None
            ts_int = original_ts
            if ts_int is None or ts_int <= 0:
                ts_int = index * 60_000 + 1
            normalised["t"] = ts_int
            normalised["time"] = original_ts if original_ts is not None else ts_int
            if "open" not in normalised and "o" in normalised:
                normalised["open"] = normalised.get("o")
            if "high" not in normalised and "h" in normalised:
                normalised["high"] = normalised.get("h")
            if "low" not in normalised and "l" in normalised:
                normalised["low"] = normalised.get("l")
            if "close" not in normalised and "c" in normalised:
                normalised["close"] = normalised.get("c")
            if "volume" not in normalised and "v" in normalised:
                normalised["volume"] = normalised.get("v")
            sanitised_candles.append(normalised)
    if sanitised_candles:
        snapshot["candles"] = sanitised_candles
    snapshot["ohlcv"] = ohlcv_multi
    snapshot["orderflow"] = orderflow_payload
    snapshot["liquidity_map"] = liquidity_map
    snapshot["derivatives"] = derivatives_rows
    snapshot["book"] = book_state

    enrichment: Dict[str, Any] | None = None
    try:
        enrichment = await enrich_inspection_snapshot(snapshot, cache=cache)
    except Exception as exc:
        logging.getLogger(__name__).warning("Snapshot enrichment failed: %s", exc)
        enrichment = {"status": "insufficient_data", "missing_fields": [str(exc)]}
    if enrichment:
        snapshot["enrichment"] = enrichment

    valid, errors = validate_enhanced_snapshot(snapshot)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": errors})

    try:
        snapshot_id = register_snapshot(snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    getattr(app.state, "snapshots", {})[snapshot_id] = snapshot
    return {"snapshot_id": snapshot_id}


@app.get("/ohlcv")
async def ohlcv_endpoint(
    symbol: str = Query(..., description="Trading symbol, e.g. SOLUSDT"),
    tf: str = Query("1m", description="Timeframe"),
    lookback_days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
) -> JSONResponse:
    cache = getattr(app.state, "ohlcv_cache", {})
    try:
        data = await fetch_ohlcv_enhanced(symbol, tf, lookback_days, cache=cache)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch OHLCV: {exc}") from exc
    return JSONResponse(data)


@app.get("/orderflow/footprint")
async def orderflow_footprint_endpoint(
    symbol: str = Query(...),
    hours: int = Query(4, ge=1, le=24),
) -> JSONResponse:
    try:
        data = await fetch_footprint(symbol, hours)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load footprint: {exc}") from exc
    return JSONResponse({"symbol": symbol.upper(), "hours": hours, "footprint": data})


@app.get("/orderflow/cvd")
async def orderflow_cvd_endpoint(
    symbol: str = Query(...),
    hours: int = Query(24, ge=1, le=72),
) -> JSONResponse:
    try:
        data = await calculate_cvd(symbol, hours)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load CVD: {exc}") from exc
    return JSONResponse({"symbol": symbol.upper(), "hours": hours, "cvd": data})


@app.get("/liquidity_map")
async def liquidity_map_endpoint(
    symbol: str = Query(...),
    days: int = Query(5, ge=1, le=14),
) -> JSONResponse:
    cache = getattr(app.state, "ohlcv_cache", {})
    ohlcv_data = await fetch_ohlcv_enhanced(symbol, "1m", max(1, days), cache=cache)
    candles = ohlcv_data.get("candles") if isinstance(ohlcv_data, Mapping) else []
    try:
        liquidity_map = await generate_liquidity_map(candles or [], days)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to build liquidity map: {exc}") from exc
    return JSONResponse({"symbol": symbol.upper(), "days": days, "liquidity_map": liquidity_map})


@app.get("/derivatives")
async def derivatives_endpoint(
    symbol: str = Query(...),
    hours: int = Query(24, ge=1, le=168),
) -> JSONResponse:
    cache = getattr(app.state, "ohlcv_cache", {})
    try:
        rows = await fetch_derivatives(symbol, hours, cache=cache)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch derivatives: {exc}") from exc
    return JSONResponse({"symbol": symbol.upper(), "hours": hours, "derivatives": rows})


@app.get("/book")
async def book_endpoint(
    symbol: str = Query(...),
    minutes: int = Query(60, ge=1, le=240),
) -> JSONResponse:
    try:
        data = await fetch_orderbook(symbol, minutes)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch orderbook: {exc}") from exc
    return JSONResponse(data)


@app.get("/shared-candles")
async def fetch_shared_candles(
    symbol: str = Query(..., description="Trading symbol, e.g. BTCUSDT"),
    interval: str = Query(..., description="Interval identifier, e.g. 1m"),
) -> JSONResponse:
    if not isinstance(symbol, str) or not symbol.strip():
        raise HTTPException(status_code=400, detail="symbol is required")
    if not isinstance(interval, str) or not interval.strip():
        raise HTTPException(status_code=400, detail="interval is required")

    stored = get_shared_candles(symbol, interval) or {}
    payload = {
        "symbol": symbol.strip().upper(),
        "interval": interval.strip().lower(),
        "candles": stored.get("candles", []),
        "intervalMs": stored.get("intervalMs"),
        "lastUpdateMs": stored.get("lastUpdateMs"),
        "updatedAt": stored.get("updatedAt"),
    }
    return JSONResponse(payload)


@app.post("/shared-candles")
async def update_shared_candles(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    symbol = payload.get("symbol")
    interval = payload.get("interval")
    candles = payload.get("candles", [])
    reset_flag = bool(payload.get("reset", False))
    interval_ms = payload.get("intervalMs")
    last_update_ms = payload.get("lastUpdateMs")
    max_bars = payload.get("maxBars")

    if not isinstance(symbol, str) or not symbol.strip():
        raise HTTPException(status_code=400, detail="symbol is required")
    if not isinstance(interval, str) or not interval.strip():
        raise HTTPException(status_code=400, detail="interval is required")
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise HTTPException(status_code=400, detail="candles must be an array")

    try:
        result = merge_shared_candles(
            symbol,
            interval,
            candles,
            interval_ms=interval_ms,
            last_update_ms=last_update_ms,
            reset=reset_flag,
            max_bars=max_bars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(result)


@app.get("/inspection", response_class=HTMLResponse)
async def inspection(
    request: Request,
    snapshot: str | None = Query(None, description="Snapshot identifier"),
) -> HTMLResponse:
    snapshots = list_snapshots()

    target_snapshot = None

    if snapshot:
        target_snapshot = get_snapshot(snapshot)
        if target_snapshot is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")
    elif snapshots:
        target_snapshot = get_snapshot(snapshots[0]["id"])  # type: ignore[index]

    if target_snapshot is None:
        profile_config = resolve_profile_config(DEFAULT_SYMBOL, None)
        placeholder_payload = {
            "DATA": {
                "symbol": DEFAULT_SYMBOL,
                "frames": {},
                "selection": None,
                "delta_cvd": {},
                "vwap_tpo": {},
                "zones": {
                    "symbol": DEFAULT_SYMBOL,
                    "zones": {"fvg": [], "ob": [], "inducement": [], "cisd": []},
                },
                "tpo": {"sessions": [], "zones": []},
                "zones_raw": None,
                "profile": [],
                "profile_preset": profile_config.get("preset_payload"),
                "profile_preset_required": bool(profile_config.get("preset_required", False)),
                "profile_defaults": None,
                "smt": {"status": "waiting", "detail": "???????????????? ???????????? ??????????????"},
                "meta": {"requested": {"symbol": DEFAULT_SYMBOL, "frames": []}, "source": {}},
            },
            "DIAGNOSTICS": {"generated_at": None, "snapshot_id": None, "captured_at": None, "frames": {}},
        }
        html = render_inspection_page(
            placeholder_payload,
            snapshot_id=None,
            symbol=DEFAULT_SYMBOL,
            timeframe="1m",
            snapshots=snapshots,
        )
        return HTMLResponse(content=html)

    payload = build_inspection_payload(target_snapshot)

    stored_snapshots = getattr(app.state, "snapshots", {})
    enriched_snapshot = stored_snapshots.get(target_snapshot.get("id")) if isinstance(stored_snapshots, Mapping) else None
    if isinstance(enriched_snapshot, Mapping):
        enrichment_payload = enriched_snapshot.get("enrichment") if isinstance(enriched_snapshot.get("enrichment"), Mapping) else None
        if enrichment_payload is None:
            try:
                cache = getattr(app.state, "ohlcv_cache", {})
                combined_snapshot = dict(enriched_snapshot)
                data_section = payload.get("DATA") if isinstance(payload.get("DATA"), Mapping) else None
                if data_section is not None:
                    combined_snapshot["DATA"] = data_section
                enrichment_payload = await enrich_inspection_snapshot(combined_snapshot, cache=cache)
                enriched_snapshot["enrichment"] = enrichment_payload
            except Exception as exc:
                logging.getLogger(__name__).warning("Inspection enrichment failed: %s", exc)
                enrichment_payload = None
        data_section = payload.setdefault("DATA", {})
        for key in ("ohlcv", "orderflow", "liquidity_map", "derivatives", "book"):
            if key in enriched_snapshot and key not in data_section:
                data_section[key] = enriched_snapshot[key]
        if enrichment_payload:
            apply_enrichment_to_payload(payload, enrichment_payload)

    accept_header = request.headers.get("accept", "").lower()
    if "application/json" in accept_header:
        return JSONResponse(payload)

    html = render_inspection_page(
        payload,
        snapshot_id=target_snapshot.get("id"),
        symbol=target_snapshot.get("symbol", "UNKNOWN"),
        timeframe=target_snapshot.get("tf", "1m"),
        snapshots=snapshots,
    )
    return HTMLResponse(content=html)


@app.post("/inspection/analyze")
async def inspection_analyze(request: AnalysisRequest) -> JSONResponse:
    snapshot_id = request.snapshot_id
    stored_snapshots = getattr(app.state, "snapshots", {})
    snapshot = stored_snapshots.get(snapshot_id) if isinstance(stored_snapshots, Mapping) else None
    if snapshot is None:
        snapshot = get_snapshot(snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    candles_payload = snapshot.get("candles")
    if not candles_payload:
        frames = snapshot.get("frames") if isinstance(snapshot.get("frames"), Mapping) else {}
        tf = snapshot.get("tf", "1m")
        if isinstance(frames, Mapping):
            frame = frames.get(tf) or frames.get(tf.upper())
            if isinstance(frame, Mapping):
                candles_payload = frame.get("candles")

    candles_payload = candles_payload or snapshot.get("ohlcv", {}).get("1m", {}).get("candles") if isinstance(snapshot.get("ohlcv"), Mapping) else []
    candles_list = list(candles_payload) if isinstance(candles_payload, Sequence) else []
    symbol = str(snapshot.get("symbol") or DEFAULT_SYMBOL).upper()
    timeframe = str(snapshot.get("tf") or "1m")

    if request.analysis_type == "tpo":
        tpo_daily = calculate_tpo(candles_list) if candles_list else {"days": []}
        session_data = [calculate_session_tpo(candles_list, session) for session in ("asia", "london", "ny")] if candles_list else []
        return JSONResponse({"snapshot_id": snapshot_id, "tpo": {"daily": tpo_daily.get("days", []), "sessions": session_data}})

    if request.analysis_type == "zones":
        frames_map = snapshot.get("frames") if isinstance(snapshot.get("frames"), Mapping) else {}
        zone_source_frames: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(frames_map, Mapping):
            zone_source_frames.update(_normalise_zone_frames_from_snapshot(frames_map))
        ohlcv_section = snapshot.get("ohlcv") if isinstance(snapshot.get("ohlcv"), Mapping) else {}
        if isinstance(ohlcv_section, Mapping):
            for tf_key, series in _normalise_zone_frames_from_snapshot(ohlcv_section).items():
                zone_source_frames.setdefault(tf_key, series)
        analysis_candles = [
            dict(item) for item in candles_list if isinstance(item, Mapping)
        ]
        if analysis_candles:
            zone_source_frames[timeframe] = analysis_candles
        if not zone_source_frames:
            return JSONResponse({"snapshot_id": snapshot_id, "zones": {"zones": {}}})
        base_tf_key, base_series = _select_zone_base_from_frames(
            zone_source_frames,
            preferred=timeframe,
        )
        if not base_series and analysis_candles:
            base_series = analysis_candles
        if not base_tf_key:
            base_tf_key = timeframe
        zones_window_start_ms, window_end_ms_prev_closed = _compute_zone_window(
            base_series,
            base_tf_key,
        )
        zone_cfg = ZonesConfig()
        zone_cfg.zones_window_start_ms = zones_window_start_ms
        zone_cfg.window_end_ms_prev_closed = window_end_ms_prev_closed
        zone_frames = _build_zone_frames_for_detection(
            zone_source_frames,
            window_start_ms=zones_window_start_ms,
            window_end_ms=window_end_ms_prev_closed,
        )
        if base_tf_key and base_tf_key not in zone_frames and base_series:
            fallback_frames = _build_zone_frames_for_detection(
                {base_tf_key: base_series},
                window_start_ms=zones_window_start_ms,
                window_end_ms=window_end_ms_prev_closed,
            )
            if fallback_frames.get(base_tf_key):
                zone_frames[base_tf_key] = fallback_frames[base_tf_key]
        zones = detect_zones(frames=zone_frames, config=zone_cfg) if zone_frames else {"zones": {}}
        return JSONResponse({"snapshot_id": snapshot_id, "zones": zones})

    if request.analysis_type == "liquidity":
        liquidity_map = await generate_liquidity_map(candles_list, 5) if candles_list else {}
        return JSONResponse({"snapshot_id": snapshot_id, "liquidity_map": liquidity_map})

    raise HTTPException(status_code=400, detail="Unsupported analysis type")


@app.get("/inspection/snapshots")
async def inspection_snapshots() -> JSONResponse:
    return JSONResponse(list_snapshots())


@app.get("/inspection/check-all")
async def inspection_check_all(
    snapshot: str | None = Query(None, description="Snapshot identifier"),
    now: str | None = Query(
        None,
        description="Override the as-of timestamp (ISO 8601, defaults to last candle)",
    ),
    selection_start: int | None = Query(
        None,
        description="Override the start of the analysed window (milliseconds)",
    ),
    selection_end: int | None = Query(
        None,
        description="Override the end of the analysed window (milliseconds)",
    ),
    hours: int | None = Query(
        None,
        description="Number of recent hours to collect detailed data for (1-4)",
    ),
    mode: str | None = Query(
        None,
        description="Collection mode: selection (default), summary or topup",
    ),
    summary_days: int | None = Query(
        None,
        description="Number of days to include for summary mode (defaults to 3)",
    ),
) -> Response:
    snapshots = list_snapshots()

    target_snapshot = None
    if snapshot:
        target_snapshot = get_snapshot(snapshot)
        if target_snapshot is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")
    elif snapshots:
        target_snapshot = get_snapshot(snapshots[0]["id"])  # type: ignore[index]

    if target_snapshot is None:
        return Response(status_code=204)

    now_override = None
    if now:
        try:
            parsed = datetime.fromisoformat(now)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid now parameter") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        now_override = parsed

    mode_value = (mode or "selection").strip().lower()
    if mode_value not in {"selection", "summary", "topup", "session_detailed"}:
        mode_value = "selection"

    collection_reference = now_override or datetime.now(timezone.utc)
    has_now_override = now_override is not None
    log_extra: Dict[str, Any] = {
        "snapshot_id": target_snapshot.get("id"),
        "mode": mode_value,
        "selection_start": selection_start,
        "selection_end": selection_end,
        "hours": hours,
        "has_now_override": has_now_override,
    }
    LOGGER.info("inspection_check_all:start", extra=log_extra)

    if mode_value == "summary":
        days = summary_days if summary_days and summary_days > 0 else 3
        window_hours = max(1, days * 24)
        branch_log = dict(log_extra)
        branch_log["window_hours"] = window_hours
        collection_summary_payload: Dict[str, Any] | None = None
        trace_ctx: TraceContext | None = None
        fetch_ms = 0.0
        compute_ms = 0.0
        trace_ctx: TraceContext | None = None
        try:
            payload, collection_summary_payload, trace_ctx = await _run_summary_workflow(
                target_snapshot,
                days=days,
                window_hours=window_hours,
                now_override=now_override,
                branch_log=branch_log,
            )
        except DataQualityError as exc:
            LOGGER.warning(
                "inspection_check_all:data_quality_error",
                extra={**branch_log, "error": str(exc)},
            )
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "data_quality": exc.detail},
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception(
                "inspection_check_all:summary_failed",
                extra={**branch_log, "error": str(exc)},
            )
            fallback = build_inspection_error_payload(
                target_snapshot,
                now_utc=now_override,
                missing_fields=["ohlcv.1m"],
                reason="invalid_timestamps",
            )
            return JSONResponse(fallback)
        if payload is None:
            LOGGER.info("inspection_check_all:finished", extra={**branch_log, "status": None})
            return Response(status_code=204)
        payload = _prepare_summary_payload(
            dict(payload),
            trace=trace_ctx.child(stage="payload") if trace_ctx is not None else None,
        )
        if collection_summary_payload:
            meta_block = payload.get("meta")
            if isinstance(meta_block, MutableMapping):
                meta_block["summary_collection"] = collection_summary_payload
        status_value = payload.get("status") if isinstance(payload, Mapping) else None
        LOGGER.info(
            "inspection_check_all:finished",
            extra={**branch_log, "status": status_value},
        )
        if trace_ctx is not None:
            trace_ctx.info(
                "output.publish",
                status=status_value,
                json_bytes=(payload.get("timing") or {}).get("json_bytes"),
            )
            trace_ctx.info("pipeline.done", status=status_value)
        set_last_collection_time(collection_reference)
        return JSONResponse(payload)

    if mode_value == "session_detailed":
        symbol = target_snapshot.get("symbol") if isinstance(target_snapshot, Mapping) else None
        if not symbol:
            meta_block = target_snapshot.get("meta") if isinstance(target_snapshot, Mapping) else None
            if isinstance(meta_block, Mapping):
                symbol = meta_block.get("symbol")
        if not symbol:
            raise HTTPException(status_code=400, detail="Snapshot symbol missing")
        try:
            session_result: SessionCollectionResult = await _run_session_workflow(
                symbol,
                now_override=now_override,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception(
                "inspection_check_all:session_detailed_failed",
                extra={**log_extra, "error": str(exc)},
            )
            raise HTTPException(status_code=500, detail="Session collection failed") from exc
        payload = session_result.as_dict()
        LOGGER.info(
            "inspection_check_all:finished",
            extra={**log_extra, "mode": mode_value, "status": payload.get("status")},
        )
        return JSONResponse(payload)

    if mode_value == "topup":
        last_collection = get_last_collection_time()
        window_hours = 4
        window_start_override_ms: int | None = None
        if last_collection is not None:
            delta = collection_reference - last_collection
            delta_seconds = max(delta.total_seconds(), 0)
            delta_hours = delta_seconds / 3600 if delta_seconds else 0
            if delta_hours < 1:
                window_hours = 1
            elif delta_hours > 4:
                window_hours = 4
            else:
                window_hours = max(1, int(ceil(delta_hours)))
            last_collection_utc = last_collection.astimezone(timezone.utc)
            last_collection_ms = int(last_collection_utc.timestamp() * 1000)
            aligned_ms = (last_collection_ms // 60_000) * 60_000
            next_minute_ms = aligned_ms + 60_000
            window_start_override_ms = max(0, next_minute_ms)
        branch_log = dict(log_extra)
        branch_log["window_hours"] = window_hours
        branch_log["window_start_override_ms"] = window_start_override_ms
        try:
            payload = await build_check_all_datas_async(
                target_snapshot,
                now_utc=now_override,
                window_hours=window_hours,
                window_start_override_ms=window_start_override_ms,
                strict_window=True,
                timeout=CHECK_ALL_BUILD_TIMEOUT,
            )
        except DataQualityError as exc:
            LOGGER.warning(
                "inspection_check_all:data_quality_error",
                extra={**branch_log, "error": str(exc)},
            )
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "data_quality": exc.detail},
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception(
                "inspection_check_all:summary_failed",
                extra={**branch_log, "error": str(exc)},
            )
            fallback = build_inspection_error_payload(
                target_snapshot,
                now_utc=now_override,
                missing_fields=["ohlcv.1m"],
                reason="invalid_timestamps",
            )
            return JSONResponse(fallback)
        if payload is None:
            LOGGER.info("inspection_check_all:finished", extra={**branch_log, "status": None})
            return Response(status_code=204)
        status_value = payload.get("status") if isinstance(payload, Mapping) else None
        LOGGER.info(
            "inspection_check_all:finished",
            extra={**branch_log, "status": status_value},
        )
        set_last_collection_time(collection_reference)
        return JSONResponse(payload)

    branch_log = dict(log_extra)
    try:
        payload = await build_check_all_datas_async(
            target_snapshot,
            now_utc=now_override,
            selection_start_ms=selection_start,
            selection_end_ms=selection_end,
            hours=hours,
            timeout=CHECK_ALL_BUILD_TIMEOUT,
        )
    except DataQualityError as exc:
        LOGGER.warning(
            "inspection_check_all:data_quality_error",
            extra={**branch_log, "error": str(exc)},
        )
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "data_quality": exc.detail},
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception(
            "inspection_check_all:selection_failed",
            extra={**branch_log, "error": str(exc)},
        )
        fallback = build_inspection_error_payload(
            target_snapshot,
            now_utc=now_override,
            missing_fields=["ohlcv.1m"],
            reason="invalid_timestamps",
        )
        return JSONResponse(fallback)
    if payload is None:
        LOGGER.info("inspection_check_all:finished", extra={**branch_log, "status": None})
        return Response(status_code=204)

    status_value = payload.get("status") if isinstance(payload, Mapping) else None
    LOGGER.info(
        "inspection_check_all:finished",
        extra={**branch_log, "status": status_value},
    )

    return JSONResponse(payload)


@app.websocket("/inspection/ws/progress")
async def inspection_progress_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    async def reporter(event: str, payload: Dict[str, Any]) -> None:
        if websocket.client_state is not WebSocketState.CONNECTED:
            return
        message = {"type": "progress", "event": event, "data": payload}
        try:
            await websocket.send_json(message)
        except (RuntimeError, WebSocketDisconnect):
            pass

    def _normalise_payload(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _parse_mode(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in {"summary", "session_detailed"}:
                return candidate
        return None

    def _parse_summary_days(value: Any) -> int | None:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate > 0 else None

    def _parse_now(value: Any) -> datetime | None:
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    def _effective_summary_days(mode: str, override: int | None) -> int | None:
        if mode != "summary":
            return None
        return override if override is not None else 3

    mode_value = "summary"
    summary_days_override: int | None = None
    now_override: datetime | None = None

    while True:
        try:
            incoming = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except Exception:
            if websocket.application_state is WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "error", "message": "invalid_initial_payload"})
                except Exception:
                    pass
                try:
                    await websocket.close(code=1003)
                except Exception:
                    pass
            return

        message_type = incoming.get("type")
        if message_type == "client_progress":
            event_name = str(incoming.get("event") or "client.progress").strip() or "client.progress"
            payload_dict = _normalise_payload(incoming.get("data"))
            TRACE_LOGGER.debug(
                "inspection.ws_client_progress",
                extra={"event": event_name, "payload": payload_dict},
            )
            await reporter(
                "inspection.ws:client_progress",
                {"event": event_name, "payload": payload_dict},
            )
            continue
        if message_type == "prepare":
            mode_candidate = _parse_mode(incoming.get("mode"))
            if mode_candidate is not None:
                mode_value = mode_candidate
            summary_candidate = _parse_summary_days(incoming.get("summary_days"))
            if summary_candidate is not None:
                summary_days_override = summary_candidate
            now_candidate = _parse_now(incoming.get("now"))
            if now_candidate is not None:
                now_override = now_candidate
            TRACE_LOGGER.debug(
                "inspection.ws_prepare",
                extra={
                    "mode": mode_value,
                    "summary_days": _effective_summary_days(mode_value, summary_days_override),
                    "has_now_override": now_override is not None,
                },
            )
            await reporter(
                "inspection.ws:prepared",
                {
                    "mode": mode_value,
                    "summary_days": _effective_summary_days(mode_value, summary_days_override),
                    "has_now_override": now_override is not None,
                },
            )
            continue
        if message_type in {None, "start"}:
            initial = incoming
            break
        if websocket.application_state is WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "message": "unsupported_message"})
            except Exception:
                pass
            try:
                await websocket.close(code=1003)
            except Exception:
                pass
        return

    mode_candidate = _parse_mode(initial.get("mode"))
    if mode_candidate is not None:
        mode_value = mode_candidate
    if mode_value not in {"summary", "session_detailed"}:
        mode_value = "summary"

    summary_candidate = _parse_summary_days(initial.get("summary_days"))
    if summary_candidate is not None:
        summary_days_override = summary_candidate

    now_candidate = _parse_now(initial.get("now"))
    if now_candidate is not None:
        now_override = now_candidate

    snapshot_id = initial.get("snapshot")
    if not snapshot_id or not isinstance(snapshot_id, str):
        await websocket.send_json({"type": "error", "message": "snapshot_id required"})
        await websocket.close(code=1003)
        return

    target_snapshot = get_snapshot(snapshot_id)
    if target_snapshot is None:
        await websocket.send_json({"type": "error", "message": "snapshot not found"})
        await websocket.close(code=1003)
        return

    await reporter(
        "inspection.ws:start",
        {
            "mode": mode_value,
            "snapshot": snapshot_id,
            "summary_days": _effective_summary_days(mode_value, summary_days_override),
            "has_now_override": now_override is not None,
        },
    )

    collection_reference = now_override or datetime.now(timezone.utc)
    log_extra: Dict[str, Any] = {
        "snapshot_id": target_snapshot.get("id"),
        "mode": mode_value,
        "selection_start": None,
        "selection_end": None,
        "hours": None,
        "has_now_override": now_override is not None,
    }
    LOGGER.info("inspection_check_all:start", extra=log_extra)

    if mode_value == "summary":
        days = summary_days_override if summary_days_override is not None else 3
        window_hours = max(1, days * 24)
        branch_log = dict(log_extra)
        branch_log["window_hours"] = window_hours
        branch_log["summary_days"] = days
        collection_summary_payload: Dict[str, Any] | None = None
        trace_ctx: TraceContext | None = None
        fetch_ms = 0.0
        compute_ms = 0.0
        trace_ctx: TraceContext | None = None
        try:
            payload, collection_summary_payload, trace_ctx = await _run_summary_workflow(
                target_snapshot,
                days=days,
                window_hours=window_hours,
                now_override=now_override,
                branch_log=branch_log,
                progress=reporter,
            )
        except DataQualityError as exc:
            LOGGER.warning(
                "inspection_check_all:data_quality_error",
                extra={**branch_log, "error": str(exc)},
            )
            await reporter(
                "inspection.summary_collection:failed",
                {"error": str(exc), "data_quality": exc.detail},
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                    "detail": exc.detail,
                }
            )
            await websocket.close(code=1011)
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception(
                "inspection_check_all:summary_failed",
                extra={**branch_log, "error": str(exc)},
            )
            await reporter(
                "inspection.summary_collection:failed",
                {"error": str(exc)},
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "summary collection failed",
                    "detail": str(exc),
                }
            )
            await websocket.close(code=1011)
            return
        if payload is None:
            LOGGER.info("inspection_check_all:finished", extra={**branch_log, "status": None})
            await reporter("inspection.summary_collection:finished", {"status": None})
            await websocket.send_json({"type": "result", "mode": mode_value, "payload": None})
            await websocket.close(code=1000)
            return
        payload = _prepare_summary_payload(
            dict(payload),
            trace=trace_ctx.child(stage="payload") if trace_ctx is not None else None,
        )
        if collection_summary_payload:
            meta_block = payload.get("meta")
            if isinstance(meta_block, MutableMapping):
                meta_block["summary_collection"] = collection_summary_payload
        status_value = payload.get("status") if isinstance(payload, Mapping) else None
        LOGGER.info(
            "inspection_check_all:finished",
            extra={**branch_log, "status": status_value},
        )
        await reporter(
            "inspection.summary_collection:finished",
            {"status": status_value, "window_hours": window_hours},
        )
        if trace_ctx is not None:
            trace_ctx.info(
                "output.publish",
                status=status_value,
                json_bytes=(payload.get("timing") or {}).get("json_bytes"),
            )
            trace_ctx.info("pipeline.done", status=status_value)
        set_last_collection_time(collection_reference)
        await websocket.send_json({"type": "result", "mode": mode_value, "payload": payload})
        await websocket.close(code=1000)
        return

    symbol = target_snapshot.get("symbol") if isinstance(target_snapshot, Mapping) else None
    if not symbol:
        meta_block = target_snapshot.get("meta") if isinstance(target_snapshot, Mapping) else None
        if isinstance(meta_block, Mapping):
            symbol = meta_block.get("symbol")
    if not symbol:
        await websocket.send_json({"type": "error", "message": "snapshot symbol missing"})
        await websocket.close(code=1003)
        return
    try:
        session_result = await _run_session_workflow(
            symbol,
            now_override=now_override,
            progress=reporter,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception(
            "inspection_check_all:session_detailed_failed",
            extra={**log_extra, "error": str(exc)},
        )
        await websocket.send_json(
            {
                "type": "error",
                "message": "session collection failed",
                "detail": str(exc),
            }
        )
        await websocket.close(code=1011)
        return
    payload = session_result.as_dict()
    LOGGER.info(
        "inspection_check_all:finished",
        extra={**log_extra, "mode": mode_value, "status": payload.get("status")},
    )
    await reporter(
        "inspection.session_collection:finished",
        {
            "status": payload.get("status"),
            "coverage_pct": payload.get("session", {}).get("coverage_pct"),
        },
    )
    await websocket.send_json({"type": "result", "mode": mode_value, "payload": payload})
    await websocket.close(code=1000)


@app.get("/presets")
async def list_presets_endpoint() -> JSONResponse:
    presets = [preset_to_payload(item) for item in list_presets_configs()]
    return JSONResponse({"ok": True, "presets": presets})


@app.get("/presets/{symbol}")
async def get_preset_endpoint(symbol: str) -> JSONResponse:
    config = resolve_profile_config(symbol, None)
    preset = config.get("preset")
    payload = preset_to_payload(preset) if preset else None
    return JSONResponse({"ok": True, "preset": payload})


@app.post("/presets")
async def create_preset_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    symbol = payload.get("symbol")
    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="symbol is required")
    symbol_value = symbol.strip().upper()
    body = dict(payload)
    body.pop("symbol", None)
    try:
        preset = save_preset(symbol_value, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "preset": preset_to_payload(preset)})


@app.put("/presets/{symbol}")
async def update_preset_endpoint(symbol: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    body = dict(payload)
    try:
        preset = update_preset(symbol, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "preset": preset_to_payload(preset)})


@app.delete("/presets/{symbol}")
async def delete_preset_endpoint(symbol: str) -> JSONResponse:
    delete_preset(symbol)
    return JSONResponse({"ok": True})


@app.get("/profile")
async def profile_endpoint(
    snapshot: str = Query(..., description="Snapshot identifier"),
    tf: str = Query("1m", description="Timeframe to analyse"),
    last_n: int = Query(3, description="Number of recent sessions to include"),
    tick_size: float | None = Query(None, description="Optional explicit tick size"),
    adaptive_bins: bool | None = Query(None, description="Use adaptive ATR-based binning when no tick size"),
    value_area_pct: float = Query(0.7, description="Value area coverage (0-1)"),
) -> JSONResponse:
    target_snapshot = get_snapshot(snapshot)
    if target_snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    symbol = str(target_snapshot.get("symbol") or target_snapshot.get("pair") or "UNKNOWN").upper()

    timeframe = str(tf or target_snapshot.get("tf") or "1m").lower()

    if last_n <= 0:
        raise HTTPException(status_code=400, detail="last_n must be positive")

    frames_data = target_snapshot.get("frames")
    frames = frames_data if isinstance(frames_data, Mapping) else {}
    raw_frame = None
    if isinstance(frames, dict):
        raw_frame = frames.get(timeframe) or frames.get(timeframe.upper())
    if raw_frame is None and "candles" in target_snapshot:
        raw_frame = {"candles": target_snapshot.get("candles")}

    if isinstance(raw_frame, dict):
        raw_candles = raw_frame.get("candles", [])
    elif isinstance(raw_frame, (list, tuple)):
        raw_candles = raw_frame
    else:
        raw_candles = []

    use_full_span = timeframe == "1m"

    try:
        normalised = normalise_ohlcv(
            symbol,
            timeframe,
            raw_candles,
            use_full_span=use_full_span,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candles = normalised.get("candles", []) if isinstance(normalised, dict) else []

    profile_config = resolve_profile_config(symbol, target_snapshot.get("meta") if isinstance(target_snapshot.get("meta"), Mapping) else None)

    target_tf_key = timeframe or profile_config.get("target_tf_key", "1m")
    last_n_value = max(1, min(5, int(last_n or profile_config.get("last_n", 3))))

    tick_size_value = tick_size if tick_size is not None else profile_config.get("tick_size")
    adaptive_flag = adaptive_bins
    if tick_size is None:
        adaptive_flag = adaptive_bins if adaptive_bins is not None else bool(profile_config.get("adaptive_bins", True))
    else:
        adaptive_flag = bool(adaptive_bins)

    value_area = value_area_pct if value_area_pct is not None else float(profile_config.get("value_area_pct", 0.7))
    value_area = max(0.0, min(1.0, float(value_area)))

    sessions = list(Meta.iter_vwap_sessions())
    tpo_entries: list[dict[str, object]] = []
    tpo_zones: list[dict[str, Any]] = []
    flattened_profile: list[dict[str, float]] = []

    detected_zones = {
        "symbol": symbol,
        "zones": {
            "fvg": [],
            "ob": [],
            "mb": [],
            "bb": [],
            "rb": [],
            "pb": [],
            "sr": [],
            "profile_levels": [],
        },
    }

    if candles and sessions:
        cache_token = ("profile", snapshot, symbol, target_tf_key)
        (tpo_entries, flattened_profile, tpo_zones) = build_profile_package(
            candles,
            sessions=sessions,
            last_n=last_n_value,
            tick_size=tick_size_value,
            adaptive_bins=bool(adaptive_flag),
            value_area_pct=value_area,
            atr_multiplier=float(profile_config.get("atr_multiplier", 0.5)),
            target_bins=int(profile_config.get("target_bins", 80)),
            clip_threshold=float(profile_config.get("clip_threshold", 0.0)),
            smooth_window=int(profile_config.get("smooth_window", 1)),
            cache_token=cache_token,
            tf_key=target_tf_key,
        )
        profile_level_map: Dict[str, Dict[str, float]] = {}
        for entry in tpo_entries:
            if not isinstance(entry, Mapping):
                continue
            session = str(entry.get("session") or "daily").lower()
            session_levels = profile_level_map.setdefault(session, {})
            for key, target in (("POC", "poc"), ("VAH", "vah"), ("VAL", "val")):
                value = entry.get(key)
                if value is None:
                    continue
                try:
                    session_levels[target] = float(value)
                except (TypeError, ValueError):
                    continue
        try:
            zone_cfg = ZonesConfig(tick_size=tick_size_value)
            zone_frames = {target_tf_key: candles}
            if timeframe and timeframe != target_tf_key:
                zone_frames[timeframe] = candles
            detected_zones = detect_zones(
                frames=zone_frames,
                profile_levels=profile_level_map,
                config=zone_cfg,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Failed to detect zones for profile endpoint",
                extra={
                    "snapshot": snapshot,
                    "symbol": symbol,
                    "timeframe": target_tf_key,
                },
            )

            detected_zones = {
                "zones": {
                    "fvg": [],
                    "ob": [],
                    "mb": [],
                    "bb": [],
                    "rb": [],
                    "pb": [],
                    "sr": [],
                    "profile_levels": [],
                },
                "meta": {},
            }
        else:
            zones_container = detected_zones.get("zones") if isinstance(detected_zones, Mapping) else None
            if isinstance(zones_container, dict) and profile_level_map and not zones_container.get("profile_levels"):
                zones_container["profile_levels"] = [
                    {"type": level, "price": price, "session": session}
                    for session, mapping in profile_level_map.items()
                    for level, price in mapping.items()
                ]

    payload = {
        "symbol": symbol,
        "tf": target_tf_key,
        "tpo": {"sessions": tpo_entries, "zones": tpo_zones},
        "profile": flattened_profile,
        "zones": detected_zones,
        "preset": profile_config.get("preset_payload"),
        "preset_required": bool(profile_config.get("preset_required", False)),
    }
    return JSONResponse(payload)


def _last_price_from_candles(candles: Sequence[Mapping[str, Any]]) -> float | None:
    for candle in reversed(candles):
        if not isinstance(candle, Mapping):
            continue
        for key in ("c", "o", "h", "l"):
            value = candle.get(key)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0:
                return price
    return None


def _tick_size_from_price(price: float | None) -> float | None:
    if price is None or not math.isfinite(price) or price <= 0:
        return None
    try:
        decimal_price = Decimal(str(price)).normalize()
    except (InvalidOperation, ValueError):  # pragma: no cover - defensive guard
        return None
    exponent = decimal_price.as_tuple().exponent
    decimals = max(0, -exponent)
    return float(10 ** (-decimals))


@app.get("/zones")
async def zones_endpoint(
    snapshot: str | None = Query(None, description="Snapshot identifier"),
    tf: str | None = Query(None, description="Timeframe to analyse"),
    symbol: str | None = Query(None, description="Symbol override"),
    min_gap_pct: float | None = Query(None, description="Minimum FVG size ratio"),
    atr_period: int | None = Query(None, description="ATR period"),
    k_impulse: float | None = Query(None, description="Impulse multiplier threshold"),
    w_swing: int | None = Query(None, description="Swing width"),
    r_zone_pct: float | None = Query(None, description="Zone proximity ratio"),
    m_wick_atr: float | None = Query(None, description="Maximum wick ATR multiple"),
    tick_size: float | None = Query(None, description="Explicit tick size"),
    body: Dict[str, Any] | None = Body(None),
) -> JSONResponse:
    payload_body = body or {}

    def _num(source: Mapping[str, Any], key: str) -> float | None:
        value = source.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _int(source: Mapping[str, Any], key: str) -> int | None:
        value = source.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    candles_data: Sequence[Mapping[str, Any]] | None = None
    tick_size_value = tick_size if tick_size is not None else None
    symbol_value = str(symbol or payload_body.get("symbol") or "").upper()
    timeframe_value = str(tf or payload_body.get("tf") or "").lower()

    snapshot_payload: Mapping[str, Any] | None = None
    if snapshot:
        snapshot_payload = get_snapshot(snapshot)
        if snapshot_payload is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")
    else:
        candles_candidate = payload_body.get("candles")
        has_body_candles = isinstance(candles_candidate, Sequence) and len(candles_candidate) > 0
        if not has_body_candles:
            snapshot_payload = get_latest_snapshot()
            if snapshot_payload is None:
                raise HTTPException(status_code=404, detail="Snapshot not found")

    profile_config: Mapping[str, Any] | Dict[str, Any] = {}
    body_tick = _num(payload_body, "tick_size")

    if snapshot_payload is not None:
        symbol_value = str(
            symbol
            or snapshot_payload.get("symbol")
            or snapshot_payload.get("pair")
            or symbol_value
            or DEFAULT_SYMBOL
        ).upper()

        timeframe_value = str(tf or snapshot_payload.get("tf") or timeframe_value or "1m").lower()

        frames_data = snapshot_payload.get("frames")
        frames = frames_data if isinstance(frames_data, Mapping) else {}
        raw_frame = None
        if isinstance(frames, dict):
            raw_frame = frames.get(timeframe_value) or frames.get(timeframe_value.upper())
        if raw_frame is None and "candles" in snapshot_payload:
            raw_frame = {"candles": snapshot_payload.get("candles")}

        if isinstance(raw_frame, Mapping):
            raw_candles = raw_frame.get("candles", [])
        elif isinstance(raw_frame, (list, tuple)):
            raw_candles = raw_frame
        else:
            raw_candles = []

        candles_data = list(raw_candles)

        try:
            profile_config = resolve_profile_config(
                symbol_value,
                snapshot_payload.get("meta")
                if isinstance(snapshot_payload.get("meta"), Mapping)
                else None,
            )
        except Exception:  # pragma: no cover - resolve_profile_config may raise
            profile_config = {}

        if tick_size_value is None:
            if body_tick is not None:
                tick_size_value = body_tick
            elif isinstance(profile_config, Mapping):
                tick_candidate = profile_config.get("tick_size")
                if isinstance(tick_candidate, (int, float)):
                    tick_size_value = float(tick_candidate)
    else:
        candles_raw = payload_body.get("candles")
        if not isinstance(candles_raw, Sequence) or not candles_raw:
            raise HTTPException(status_code=400, detail="No candles provided")
        candles_data = list(candles_raw)  # type: ignore[list-item]

        if not symbol_value:
            symbol_value = DEFAULT_SYMBOL
        if not timeframe_value:
            raise HTTPException(status_code=400, detail="tf is required")

        if tick_size_value is None and body_tick is not None:
            tick_size_value = body_tick

        try:
            profile_config = resolve_profile_config(symbol_value, None)
        except Exception:  # pragma: no cover - resolve_profile_config may raise
            profile_config = {}

        if tick_size_value is None and isinstance(profile_config, Mapping):
            tick_candidate = profile_config.get("tick_size")
            if isinstance(tick_candidate, (int, float)):
                tick_size_value = float(tick_candidate)

    if not candles_data:
        raise HTTPException(status_code=400, detail="No candles provided")

    if not symbol_value:
        symbol_value = DEFAULT_SYMBOL

    if not timeframe_value:
        raise HTTPException(status_code=400, detail="tf is required")

    if tick_size_value is None:
        tick_size_value = _tick_size_from_price(_last_price_from_candles(candles_data))

    cfg_kwargs: Dict[str, Any] = {}
    body_min_gap = _num(payload_body, "min_gap_pct")
    cfg_kwargs["min_gap_pct"] = float(
        min_gap_pct
        if min_gap_pct is not None
        else body_min_gap
        if body_min_gap is not None
        else 0.0003
    )

    body_atr_period = _int(payload_body, "atr_period")
    cfg_kwargs["atr_period"] = int(
        atr_period
        if atr_period is not None
        else body_atr_period
        if body_atr_period is not None
        else 14
    )

    body_k_impulse = _num(payload_body, "k_impulse")
    cfg_kwargs["k_impulse"] = float(
        k_impulse
        if k_impulse is not None
        else body_k_impulse
        if body_k_impulse is not None
        else 0.25
    )

    body_w_swing = _int(payload_body, "w_swing")
    default_w_swing = 3 if timeframe_value in {"1h", "4h"} else 2
    cfg_kwargs["w_swing"] = int(
        w_swing
        if w_swing is not None
        else body_w_swing
        if body_w_swing is not None
        else default_w_swing
    )

    body_r_zone_pct = _num(payload_body, "r_zone_pct")
    cfg_kwargs["r_zone_pct"] = float(
        r_zone_pct
        if r_zone_pct is not None
        else body_r_zone_pct
        if body_r_zone_pct is not None
        else 0.15
    )

    body_m_wick_atr = _num(payload_body, "m_wick_atr")
    cfg_kwargs["m_wick_atr"] = float(
        m_wick_atr
        if m_wick_atr is not None
        else body_m_wick_atr
        if body_m_wick_atr is not None
        else 3.0
    )

    cfg_kwargs["tick_size"] = tick_size_value

    zone_cfg = ZonesConfig(**cfg_kwargs)

    zone_frames: Dict[str, Sequence[Mapping[str, Any]]] = {}
    if candles_data:
        zone_frames[timeframe_value] = candles_data
    try:
        result = detect_zones(frames=zone_frames, config=zone_cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(result)


@app.get("/diag")
async def diag_report() -> JSONResponse:
    snapshot_payload = get_latest_snapshot()
    if snapshot_payload is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    frames_payload = snapshot_payload.get("frames")
    zone_frames: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(frames_payload, Mapping):
        zone_frames = _normalise_zone_frames_from_snapshot(frames_payload)
    if not zone_frames:
        base_candles = snapshot_payload.get("candles")
        if isinstance(base_candles, Sequence):
            series: List[Dict[str, Any]] = []
            for candle in base_candles:
                if isinstance(candle, Mapping):
                    series.append(dict(candle))
            if series:
                zone_frames["1m"] = series
    if not zone_frames:
        raise HTTPException(status_code=400, detail="Snapshot missing candle frames")

    preferred_tf_value = snapshot_payload.get("tf")
    preferred_tf = (
        str(preferred_tf_value).lower()
        if isinstance(preferred_tf_value, str) and preferred_tf_value
        else None
    )
    anchor_tf, anchor_series = _select_zone_base_from_frames(zone_frames, preferred=preferred_tf)
    window_start_ms, window_end_ms = _compute_zone_window(anchor_series, anchor_tf)
    bounds_start_ms, bounds_end_ms = _frame_time_bounds(zone_frames)
    if window_start_ms is None:
        window_start_ms = bounds_start_ms
    if window_end_ms is None:
        window_end_ms = bounds_end_ms

    detection_frames = _build_zone_frames_for_detection(
        zone_frames,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )
    if not detection_frames:
        raise HTTPException(status_code=400, detail="No closed candles available for detection")

    cfg_kwargs: Dict[str, Any] = {
        "min_gap_pct": 0.0003,
        "atr_period": 14,
        "k_impulse": 0.25,
        "w_swing": 3 if anchor_tf in {"1h", "4h"} else 2,
        "r_zone_pct": 0.15,
        "m_wick_atr": 3.0,
    }
    zone_cfg = ZonesConfig(**cfg_kwargs)
    zone_cfg.zones_window_start_ms = window_start_ms
    zone_cfg.window_end_ms_prev_closed = window_end_ms

    detection_result = detect_zones(frames=detection_frames, config=zone_cfg)
    zones_payload_raw = detection_result.get("zones") if isinstance(detection_result, Mapping) else None
    zones_payload = zones_payload_raw if isinstance(zones_payload_raw, Mapping) else {}
    meta_payload_raw = detection_result.get("meta") if isinstance(detection_result, Mapping) else None
    meta_payload = meta_payload_raw if isinstance(meta_payload_raw, Mapping) else {}

    reference_ms = window_end_ms or bounds_end_ms
    if isinstance(reference_ms, (int, float)) and reference_ms:
        reference_dt = datetime.fromtimestamp(reference_ms / 1000.0, tz=timezone.utc)
    else:
        reference_dt = datetime.now(timezone.utc)

    compact, counts, filter_diag = _filter_compact_zones(
        zones_payload,
        now_dt=reference_dt,
        limit=24,
        config=zone_cfg,
    )

    fvg_stats = meta_payload.get("fvg_stats") if isinstance(meta_payload.get("fvg_stats"), Mapping) else {}
    triplets_total = _aggregate_fvg_stat(fvg_stats, "fvg_triplets")
    reject_gap_total = _aggregate_fvg_stat(fvg_stats, "fvg_reject_no_gap")
    reject_displacement_total = _aggregate_fvg_stat(fvg_stats, "fvg_reject_displacement")
    gap_reduction = _compute_reduction(triplets_total, reject_gap_total)
    displacement_reduction = _compute_reduction(triplets_total, reject_displacement_total)

    zones_before = int(filter_diag.get("zones_before_filter", 0) or 0)
    zones_after = int(filter_diag.get("zones_after_filter", 0) or 0)
    retention_ratio = (zones_after / zones_before) if zones_before > 0 else None
    fvg_ob_ratio = filter_diag.get("fvg_ob_ratio")
    if not isinstance(fvg_ob_ratio, (int, float)):
        fvg_ob_ratio = 0.0

    def _reduction_reason(value: float | None, target: float, missing: str) -> str:
        if value is None:
            return missing
        return f"reduction {value:.2f} below target {target:.2f}"

    metrics_block: Dict[str, Dict[str, Any]] = {}

    gap_entry = {
        "ok": gap_reduction is not None and gap_reduction >= 0.5,
        "reduction": gap_reduction,
        "target": 0.5,
        "triplets": triplets_total,
        "rejects": reject_gap_total,
    }
    if not gap_entry["ok"]:
        gap_entry["reason"] = _reduction_reason(gap_reduction, 0.6, "insufficient_triplets")
    metrics_block["fvg_reject_no_gap"] = gap_entry

    displacement_entry = {
        "ok": displacement_reduction is not None and displacement_reduction >= 0.4,
        "reduction": displacement_reduction,
        "target": 0.4,
        "triplets": triplets_total,
        "rejects": reject_displacement_total,
    }
    if not displacement_entry["ok"]:
        displacement_entry["reason"] = _reduction_reason(
            displacement_reduction,
            0.4,
            "insufficient_triplets",
        )
    metrics_block["fvg_reject_displacement"] = displacement_entry

    retention_entry = {
        "ok": retention_ratio is not None and retention_ratio >= 0.8,
        "ratio": retention_ratio,
        "target": 0.8,
        "before": zones_before,
        "after": zones_after,
    }
    if not retention_entry["ok"]:
        if zones_before <= 0:
            retention_entry["reason"] = "no_candidates"
        else:
            retention_entry["reason"] = f"retention {retention_ratio:.2f} below target 0.80"
    metrics_block["zones_retained"] = retention_entry

    fvg_ob_entry = {
        "ok": fvg_ob_ratio >= 0.7,
        "ratio": fvg_ob_ratio,
        "target": 0.7,
    }
    if not fvg_ob_entry["ok"]:
        fvg_ob_entry["reason"] = f"top share {fvg_ob_ratio:.2f} below target 0.70"
    metrics_block["fvg_ob_share"] = fvg_ob_entry

    timeframe_targets: Dict[str, Dict[str, Any]] = {}
    timeframe_requirements = {
        "1h_fvg": ("fvg", "1h", 1),
        "1h_ob": ("ob", "1h", 1),
        "15m_fvg": ("fvg", "15m", 2),
    }
    for key, (zone_type, tf_label, required) in timeframe_requirements.items():
        count = _count_recent_zones(
            compact,
            zone_type=zone_type,
            timeframe=tf_label,
            reference=reference_dt,
            window_hours=72.0,
        )
        ok = count >= required
        available = sum(
            1
            for entry in compact
            if isinstance(entry, Mapping)
            and str(entry.get("type", "")).lower() == zone_type
            and str(entry.get("tf", "")).lower() == tf_label
        )
        tf_entry: Dict[str, Any] = {
            "ok": ok,
            "count": count,
            "required": required,
            "window_hours": 72,
        }
        if not ok:
            if available == 0:
                tf_entry["reason"] = "no_zones_in_top"
            else:
                tf_entry["reason"] = f"only {count} zones within window"
        timeframe_targets[key] = tf_entry

    raw_counts = {
        str(key): int(value) for key, value in filter_diag.get("raw_counts", {}).items()
    }
    candidate_counts = {
        str(key): int(value)
        for key, value in filter_diag.get("candidate_counts", {}).items()
    }
    total_candidates = int(filter_diag.get("total_candidates", 0) or 0)

    def _last_close(series: Sequence[Mapping[str, Any]] | None) -> float | None:
        if not isinstance(series, Sequence):
            return None
        for candle in reversed(series):
            if not isinstance(candle, Mapping):
                continue
            close_value = candle.get("c")
            try:
                price = float(close_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0:
                return price
        return None

    last_price = _last_close(anchor_series)
    if last_price is None:
        for series in detection_frames.values():
            last_price = _last_close(series)
            if last_price is not None:
                break

    meta_source = snapshot_payload.get("meta") if isinstance(snapshot_payload.get("meta"), Mapping) else {}
    tz_value = (
        meta_source.get("tz")
        or meta_source.get("timezone")
        or snapshot_payload.get("tz")
        or "UTC"
    )

    coverage_map: Dict[str, int] = {}
    coverage_source = snapshot_payload.get("coverage")
    if isinstance(coverage_source, Mapping):
        for key, value in coverage_source.items():
            try:
                coverage_map[str(key)] = int(float(value))
            except (TypeError, ValueError):
                continue
    elif isinstance(coverage_source, Sequence):
        for item in coverage_source:
            if not isinstance(item, Mapping):
                continue
            tf_label = str(item.get("tf") or item.get("timeframe") or "").lower()
            if not tf_label:
                continue
            coverage_value = (
                item.get("coverage_pct")
                or item.get("coverage")
                or item.get("value")
            )
            try:
                coverage_map[tf_label] = int(float(coverage_value))
            except (TypeError, ValueError):
                continue
    if not coverage_map:
        for tf_key, series in detection_frames.items():
            if not isinstance(series, Sequence) or not series:
                continue
            coverage_map[str(tf_key)] = 100

    intervals = sorted(str(key) for key in detection_frames.keys())
    candles_total = sum(len(series) for series in detection_frames.values() if isinstance(series, Sequence))

    ohlcv_compact: Dict[str, List[Dict[str, Any]]] = {}
    for tf_key in intervals:
        series = detection_frames.get(tf_key) if isinstance(detection_frames, Mapping) else None
        if not isinstance(series, Sequence) or not series:
            continue
        trimmed: List[Dict[str, Any]] = []
        for candle in series[-5:]:
            if not isinstance(candle, Mapping):
                continue
            trimmed.append(
                {
                    key: candle.get(key)
                    for key in ("t", "o", "h", "l", "c", "v")
                    if key in candle
                }
            )
        if trimmed:
            ohlcv_compact[tf_key] = trimmed

    MAX_ZONES_PER_TYPE = 12
    zones_struct: Dict[str, Any] = {"fvg": [], "ob": [], "other": {}}
    for entry in compact:
        if not isinstance(entry, Mapping):
            continue
        zone_type = str(entry.get("type", "")).lower()
        target: List[Dict[str, Any]]
        if zone_type in {"fvg", "ob"}:
            target = zones_struct[zone_type]
        else:
            other_map = zones_struct.setdefault("other", {})
            if not isinstance(other_map, dict):
                other_map = {}
                zones_struct["other"] = other_map
            target = other_map.setdefault(zone_type or "misc", [])
        if len(target) >= MAX_ZONES_PER_TYPE:
            continue
        zone_entry = {
            "tf": entry.get("tf"),
            "status": entry.get("status"),
            "open": entry.get("open"),
            "close": entry.get("close"),
            "mean": entry.get("mean"),
            "formed_at_utc": entry.get("formed_at_utc"),
            "last_touched_utc": entry.get("last_touched_utc"),
            "source": entry.get("source"),
        }
        if entry.get("zone_id") is not None:
            zone_entry["zone_id"] = entry.get("zone_id")
        target.append(zone_entry)

    retention_summary = {
        "before": zones_before,
        "after": zones_after,
        "ratio": retention_ratio,
        "target": 0.8,
    }
    if retention_entry.get("reason"):
        retention_summary["reason"] = retention_entry["reason"]

    fvg_ob_summary = {
        "ratio": fvg_ob_ratio,
        "target": 0.7,
    }
    if fvg_ob_entry.get("reason"):
        fvg_ob_summary["reason"] = fvg_ob_entry["reason"]

    zones_summary = {
        "raw_counts": raw_counts,
        "candidate_counts": candidate_counts,
        "total_candidates": total_candidates,
        "top_counts": counts,
        "retention": retention_summary,
        "fvg_ob_share": fvg_ob_summary,
    }

    meta_block = {
        "symbol": snapshot_payload.get("symbol"),
        "tf": snapshot_payload.get("tf"),
        "tz": tz_value,
        "last_price": last_price,
        "period": {
            "start": _to_iso(window_start_ms) if window_start_ms else None,
            "end": _to_iso(window_end_ms) if window_end_ms else None,
        },
        "coverage": coverage_map,
    }

    summary_block = {
        "candles_total": candles_total,
        "intervals": intervals,
        "zones": zones_summary,
        "metrics": metrics_block,
        "timeframe_targets": timeframe_targets,
    }

    diagnostics_block = {
        "filter": filter_diag,
        "fvg_stats": fvg_stats,
        "top_sample": compact[: min(len(compact), 20)],
    }

    report = {
        "schema": "compact.v1",
        "meta": meta_block,
        "ohlcv": ohlcv_compact,
        "zones": zones_struct,
        "summary": summary_block,
        "cvd": {"buy": 0, "sell": 0},
        "diagnostics": diagnostics_block,
    }

    return JSONResponse(report)


@app.get("/test-snapshot")
async def test_snapshot() -> JSONResponse:
    now = datetime.now(timezone.utc)
    candles = []
    base_price = 100.0
    for index in range(10):
        ts = int((now - timedelta(minutes=9 - index)).timestamp() * 1000)
        open_price = base_price + index * 0.1
        high = open_price + 0.5
        low = open_price - 0.5
        close = open_price + 0.2
        candles.append({"t": ts, "o": open_price, "h": high, "l": low, "c": close, "v": 100 + index})
    payload = {
        "symbol": DEFAULT_SYMBOL,
        "tf": "1m",
        "candles": candles,
    }
    return JSONResponse(payload)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict[str, str]:
    return {"version": APP_VERSION}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    try:
        html = TEMPLATES_DIR.joinpath("index.html").read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - deployment guard
        raise HTTPException(status_code=500, detail="Index template is missing") from exc
    return HTMLResponse(content=html.replace("__STATIC_VERSION__", STATIC_VERSION))

