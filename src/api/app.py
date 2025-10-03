"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from datetime import datetime, timedelta, timezone
from math import ceil

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
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
from ..services.zones import Config as ZonesConfig, detect_zones

from ..services.book import fetch_orderbook
from ..services.derivatives import fetch_derivatives
from ..services.inspection import validate_enhanced_snapshot
from ..services.liquidity import generate_liquidity_map
from ..services.news import fetch_news
from ..services.ohlcv import build_multi_tf_ohlcv, fetch_ohlcv as fetch_ohlcv_enhanced
from ..services.orderflow import calculate_cvd, fetch_footprint
from ..services.tpo import calculate_session_tpo, calculate_tpo
from ..meta import Meta
from ..static_version import STATIC_VERSION
from ..version import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
CHECK_ALL_BUILD_TIMEOUT = 5.0



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
    news_events: Optional[List[Dict[str, Any]]] = None
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




def _to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _build_fallback_multi(symbol: str, candles: Sequence[CandleIn]) -> Dict[str, Dict[str, object]]:
    frames: Dict[str, Dict[str, object]] = {}
    base_rows: List[Dict[str, object]] = []
    for candle in candles:
        base_rows.append({
            "t": _to_iso(candle.t),
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
                    "t": _to_iso(bucket),
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


def _fallback_footprint(candles: Sequence[CandleIn]) -> List[Dict[str, object]]:
    footprint: List[Dict[str, object]] = []
    for candle in candles[-120:]:
        bid = candle.v * 0.45
        ask = candle.v * 0.55
        delta = ask - bid
        footprint.append({
            "t": _to_iso(candle.t),
            "price": candle.c,
            "bid": bid,
            "ask": ask,
            "delta": delta,
            "imbalance": ask / bid if bid else 0.0,
            "absorption": abs(delta) > 100,
        })
    return footprint


def _fallback_cvd(footprint: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    cumulative_buy = 0.0
    cumulative_sell = 0.0
    series: List[Dict[str, object]] = []
    for row in footprint:
        delta = float(row.get("delta", 0.0))
        if delta >= 0:
            cumulative_buy += delta
        else:
            cumulative_sell += abs(delta)
        series.append({
            "t": row.get("t"),
            "cvd_buy": cumulative_buy,
            "cvd_sell": cumulative_sell,
            "cvd_net": cumulative_buy - cumulative_sell,
        })
    return series


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


def _fallback_news(symbol: str) -> List[Dict[str, object]]:
    now = datetime.now(timezone.utc)
    return [
        {
            "symbol": symbol,
            "time_utc": _to_iso(int(now.timestamp() * 1000)),
            "title": "System snapshot",
            "impact": "low",
            "tag": "mock",
        }
    ]


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

def _prepare_summary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce payload weight while keeping essential context."""

    schema = payload.get("schema")
    data_section = payload.get("data") if isinstance(payload.get("data"), Mapping) else None

    if schema == "compact.v1" and isinstance(data_section, MutableMapping):
        compact_ohlcv = data_section.get("ohlcv_compact")
        if isinstance(compact_ohlcv, Mapping):
            recent_1m = compact_ohlcv.get("1m_recent")
            if isinstance(recent_1m, Sequence):
                compact_ohlcv["1m_recent"] = list(recent_1m)[-180:]
        orderflow_section = data_section.get("orderflow")
        if isinstance(orderflow_section, Mapping):
            for block in orderflow_section.values():
                if not isinstance(block, MutableMapping):
                    continue
                per_bar = block.get("per_bar")
                if isinstance(per_bar, Sequence):
                    block["per_bar"] = list(per_bar)[-120:]
        data_section.pop("ohlcv", None)
        return payload

    ohlcv_section = payload.get("ohlcv")
    if not isinstance(ohlcv_section, Mapping):
        return payload

    summary_section: Dict[str, Any] = {}

    hour_block = ohlcv_section.get("1h")
    if isinstance(hour_block, Mapping):
        summary_section["1h"] = dict(hour_block)

    four_hour_block = ohlcv_section.get("4h")
    if isinstance(four_hour_block, Mapping):
        summary_section["4h"] = dict(four_hour_block)

    minute_block = ohlcv_section.get("1m")
    if isinstance(minute_block, Mapping):
        candles = minute_block.get("candles")
        if isinstance(candles, Sequence):
            trimmed: list[Dict[str, Any]] = []
            last_ts: int | None = None
            normalised_candles: list[Dict[str, Any]] = []
            for entry in candles:
                if not isinstance(entry, Mapping):
                    continue
                normalised_candles.append(dict(entry))
                candidate_ts = None
                for key in ("t", "time", "ts", "timestamp"):
                    raw_value = entry.get(key)
                    if isinstance(raw_value, (int, float)):
                        candidate_ts = int(raw_value)
                        break
                if candidate_ts is not None:
                    last_ts = candidate_ts if last_ts is None else max(last_ts, candidate_ts)
            if normalised_candles and last_ts is not None:
                cutoff = last_ts - 179 * 60_000
                for item in normalised_candles:
                    ts_value = None
                    for key in ("t", "time", "ts", "timestamp"):
                        raw_value = item.get(key)
                        if isinstance(raw_value, (int, float)):
                            ts_value = int(raw_value)
                            break
                    if ts_value is None or ts_value >= cutoff:
                        trimmed.append(item)
            else:
                trimmed = normalised_candles
            minute_section = dict(minute_block)
            minute_section["candles"] = trimmed[-180:]
            summary_section["1m"] = minute_section

    if summary_section:
        payload["ohlcv"] = summary_section

    return payload


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
        footprint = await fetch_footprint(symbol, 4)
    except Exception as exc:
        logging.getLogger(__name__).warning("Footprint fallback engaged: %s", exc)
        footprint = _fallback_footprint(source_candles)

    try:
        cvd_series = await calculate_cvd(symbol, 24)
    except Exception as exc:
        logging.getLogger(__name__).warning("CVD fallback engaged: %s", exc)
        cvd_series = _fallback_cvd(footprint)

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

    try:
        news_items = await fetch_news(symbol, 72)
    except Exception as exc:
        logging.getLogger(__name__).warning("News fallback engaged: %s", exc)
        news_items = _fallback_news(symbol)

    orderflow_payload["footprint"] = footprint
    orderflow_payload["cvd"] = cvd_series

    snapshot = payload.dict(exclude_none=True)
    if not snapshot.get("candles") and source_candles:
        snapshot["candles"] = [c.dict() for c in source_candles]
    sanitised_candles: List[Dict[str, object]] = []
    for index, entry in enumerate(snapshot.get("candles", [])):
        if isinstance(entry, Mapping):
            normalised = dict(entry)
            ts_value = normalised.get("t")
            try:
                ts_int = int(ts_value) if ts_value is not None else None
            except (TypeError, ValueError):
                ts_int = None
            if ts_int is None or ts_int <= 0:
                ts_int = index * 60_000 + 1
            normalised["t"] = ts_int
            normalised["time"] = ts_int
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
    snapshot["news_events"] = news_items

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


@app.get("/news_events")
async def news_events_endpoint(
    symbol: str = Query(...),
    hours: int = Query(72, ge=1, le=168),
) -> JSONResponse:
    try:
        events = await fetch_news(symbol, hours)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch news: {exc}") from exc
    return JSONResponse({"symbol": symbol.upper(), "hours": hours, "events": events})


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

    response_payload = {
        "symbol": symbol.strip().upper(),
        "interval": interval.strip().lower(),
        "candles": result.get("candles", []),
        "intervalMs": result.get("intervalMs"),
        "lastUpdateMs": result.get("lastUpdateMs"),
        "updatedAt": result.get("updatedAt"),
    }
    return JSONResponse(response_payload)


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
                "smt": {"status": "waiting", "detail": "Создайте первый снэпшот"},
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
        for key in ("ohlcv", "orderflow", "liquidity_map", "derivatives", "book", "news_events"):
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
        zone_frames = {timeframe: candles_list}
        zones = detect_zones(frames=zone_frames, config=ZonesConfig()) if candles_list else {"zones": {}}
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
        symbol = target_snapshot.get("symbol") if isinstance(target_snapshot, Mapping) else None
        if not symbol:
            meta_block = target_snapshot.get("meta") if isinstance(target_snapshot, Mapping) else None
            if isinstance(meta_block, Mapping):
                symbol = meta_block.get("symbol")
        collection_summary_payload: Dict[str, Any] | None = None
        if isinstance(symbol, str) and symbol:
            try:
                summary_result = await collect_recent_summary(symbol, days=days)
                collection_summary_payload = summary_result.as_dict()
                branch_log["summary_collection"] = {
                    "requests": summary_result.requests,
                    "candles_written": summary_result.candles_written,
                    "dropped_candles": summary_result.dropped_candles,
                }
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning(
                    "inspection_check_all:summary_collection_failed",
                    extra={**branch_log, "error": str(exc)},
                )
        try:
            payload = await build_check_all_datas_async(
                target_snapshot,
                now_utc=now_override,
                window_hours=window_hours,
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
        payload = _prepare_summary_payload(dict(payload))
        if collection_summary_payload:
            meta_block = payload.get("meta")
            if isinstance(meta_block, MutableMapping):
                meta_block["summary_collection"] = collection_summary_payload
        status_value = payload.get("status") if isinstance(payload, Mapping) else None
        LOGGER.info(
            "inspection_check_all:finished",
            extra={**branch_log, "status": status_value},
        )
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
            session_result: SessionCollectionResult = await collect_last_session_detailed(symbol, now_override)
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

    if snapshot:
        target_snapshot = get_snapshot(snapshot)
        if target_snapshot is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")

        symbol_value = str(
            symbol
            or target_snapshot.get("symbol")
            or target_snapshot.get("pair")
            or "UNKNOWN"
        ).upper()

        timeframe_value = str(tf or target_snapshot.get("tf") or "1m").lower()

        frames_data = target_snapshot.get("frames")
        frames = frames_data if isinstance(frames_data, Mapping) else {}
        raw_frame = None
        if isinstance(frames, dict):
            raw_frame = frames.get(timeframe_value) or frames.get(timeframe_value.upper())
        if raw_frame is None and "candles" in target_snapshot:
            raw_frame = {"candles": target_snapshot.get("candles")}

        if isinstance(raw_frame, Mapping):
            raw_candles = raw_frame.get("candles", [])
        elif isinstance(raw_frame, (list, tuple)):
            raw_candles = raw_frame
        else:
            raw_candles = []

        candles_data = list(raw_candles)

        profile_config = resolve_profile_config(
            symbol_value, target_snapshot.get("meta") if isinstance(target_snapshot.get("meta"), Mapping) else None
        )
        body_tick = _num(payload_body, "tick_size")
        if tick_size_value is None:
            tick_size_value = body_tick if body_tick is not None else profile_config.get("tick_size")
    else:
        candles_raw = payload_body.get("candles")
        if not symbol_value:
            symbol_value = DEFAULT_SYMBOL
        if not timeframe_value:
            raise HTTPException(status_code=400, detail="tf is required")
        if not isinstance(candles_raw, Sequence):
            raise HTTPException(status_code=400, detail="candles must be a sequence")
        candles_data = list(candles_raw)  # type: ignore[list-item]
        body_tick = _num(payload_body, "tick_size")
        if tick_size_value is None and body_tick is not None:
            tick_size_value = body_tick

        try:
            profile_config = resolve_profile_config(symbol_value, None)
        except Exception:  # pragma: no cover - resolve_profile_config may raise
            profile_config = {}
        if tick_size_value is None:
            tick_size_value = profile_config.get("tick_size") if isinstance(profile_config, Mapping) else None

    if not candles_data:
        raise HTTPException(status_code=400, detail="No candles provided")

    if not symbol_value:
        symbol_value = DEFAULT_SYMBOL

    if not timeframe_value:
        raise HTTPException(status_code=400, detail="tf is required")

    cfg_kwargs: Dict[str, Any] = {}
    body_min_gap = _num(payload_body, "min_gap_pct")
    if min_gap_pct is not None:
        cfg_kwargs["min_gap_pct"] = float(min_gap_pct)
    elif body_min_gap is not None:
        cfg_kwargs["min_gap_pct"] = float(body_min_gap)

    body_atr_period = _int(payload_body, "atr_period")
    if atr_period is not None:
        cfg_kwargs["atr_period"] = int(atr_period)
    elif body_atr_period is not None:
        cfg_kwargs["atr_period"] = int(body_atr_period)

    body_k_impulse = _num(payload_body, "k_impulse")
    if k_impulse is not None:
        cfg_kwargs["k_impulse"] = float(k_impulse)
    elif body_k_impulse is not None:
        cfg_kwargs["k_impulse"] = float(body_k_impulse)

    body_w_swing = _int(payload_body, "w_swing")
    if w_swing is not None:
        cfg_kwargs["w_swing"] = int(w_swing)
    elif body_w_swing is not None:
        cfg_kwargs["w_swing"] = int(body_w_swing)

    body_r_zone_pct = _num(payload_body, "r_zone_pct")
    if r_zone_pct is not None:
        cfg_kwargs["r_zone_pct"] = float(r_zone_pct)
    elif body_r_zone_pct is not None:
        cfg_kwargs["r_zone_pct"] = float(body_r_zone_pct)

    body_m_wick_atr = _num(payload_body, "m_wick_atr")
    if m_wick_atr is not None:
        cfg_kwargs["m_wick_atr"] = float(m_wick_atr)
    elif body_m_wick_atr is not None:
        cfg_kwargs["m_wick_atr"] = float(body_m_wick_atr)

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


