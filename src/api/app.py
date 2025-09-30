"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from datetime import datetime, timezone
from math import ceil

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..services import (
    DataQualityError,
    build_check_all_datas,
    build_inspection_payload,
    build_placeholder_snapshot,
    dispatch_trade_analysis,
    build_profile_package,
    DEFAULT_SYMBOL,
    delete_preset,
    get_snapshot,
    list_presets_configs,
    list_snapshots,
    normalise_ohlcv,
    preset_to_payload,
    register_snapshot,
    render_inspection_page,
    resolve_profile_config,
    save_preset,
    update_preset,
    get_last_collection_time,
    set_last_collection_time,
)
from ..services.zones import Config as ZonesConfig, detect_zones
from ..version import APP_VERSION
from ..meta import Meta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _extract_openai_error(response: httpx.Response) -> Any | None:
    """Return a sanitised representation of an OpenAI error payload."""

    if response is None:
        return None

    try:
        payload = response.json()
    except ValueError:
        text = response.text
        if not text:
            return None
        stripped = text.strip()
        return stripped[:500] if len(stripped) > 500 else stripped

    if isinstance(payload, Mapping):
        error_node = payload.get("error")
        if isinstance(error_node, Mapping):
            cleaned: Dict[str, Any] = {}
            for key in ("message", "type", "code", "param"):
                value = error_node.get(key)
                if isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed:
                        cleaned[key] = trimmed[:500] if len(trimmed) > 500 else trimmed
                elif value is not None:
                    cleaned[key] = value
            if cleaned:
                return cleaned
            return {
                key: error_node[key]
                for key in error_node.keys()
                if key in {"message", "type", "code", "param"} and error_node[key] is not None
            } or str(error_node)[:500]
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            trimmed = message.strip()
            return trimmed[:500] if len(trimmed) > 500 else trimmed
        return str(payload)[:500]

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [payload[index] for index in range(min(len(payload), 5))]

    return payload


def _condense_payload_to_hourly(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Limit OHLCV output to hourly candles for summary mode."""

    ohlcv_section = payload.get("ohlcv")
    if isinstance(ohlcv_section, Mapping):
        hourly_block = ohlcv_section.get("1h")
        if isinstance(hourly_block, Mapping):
            ohlcv_section = {"1h": dict(hourly_block)}
            payload["ohlcv"] = ohlcv_section
    return payload


@app.post("/inspection/snapshot")
async def register_inspection_snapshot(payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    try:
        snapshot_id = register_snapshot(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"snapshot_id": snapshot_id}


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
    if mode_value not in {"selection", "summary", "topup"}:
        mode_value = "selection"

    collection_reference = now_override or datetime.now(timezone.utc)

    if mode_value == "summary":
        days = summary_days if summary_days and summary_days > 0 else 3
        window_hours = max(1, days * 24)
        try:
            payload = build_check_all_datas(
                target_snapshot,
                now_utc=now_override,
                window_hours=window_hours,
            )
        except DataQualityError as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "data_quality": exc.detail},
            ) from exc
        if payload is None:
            return Response(status_code=204)
        payload = _condense_payload_to_hourly(dict(payload))
        set_last_collection_time(collection_reference)
        return JSONResponse(payload)

    if mode_value == "topup":
        last_collection = get_last_collection_time()
        window_hours = 4
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
        try:
            payload = build_check_all_datas(
                target_snapshot,
                now_utc=now_override,
                window_hours=window_hours,
            )
        except DataQualityError as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "data_quality": exc.detail},
            ) from exc
        if payload is None:
            return Response(status_code=204)
        set_last_collection_time(collection_reference)
        return JSONResponse(payload)

    try:
        payload = build_check_all_datas(
            target_snapshot,
            now_utc=now_override,
            selection_start_ms=selection_start,
            selection_end_ms=selection_end,
            hours=hours,
        )
    except DataQualityError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "data_quality": exc.detail},
        ) from exc
    if payload is None:
        return Response(status_code=204)

    return JSONResponse(payload)


@app.post("/api/analyze-from-inspection")
async def analyze_from_inspection(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    snapshot_id = payload.get("snapshot_id")
    if not snapshot_id or not isinstance(snapshot_id, str):
        raise HTTPException(status_code=400, detail="snapshot_id is required")

    selection_start_raw = payload.get("selection_start")
    selection_end_raw = payload.get("selection_end")
    hours_raw = payload.get("hours")
    period_raw = payload.get("period")

    try:
        selection_start = int(selection_start_raw)
        selection_end = int(selection_end_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="selection_start and selection_end must be integers")

    if selection_end < selection_start:
        selection_start, selection_end = selection_end, selection_start

    try:
        hours = int(hours_raw)
    except (TypeError, ValueError):
        hours = 1
    hours = max(1, min(4, hours))

    target_snapshot = get_snapshot(snapshot_id)
    if target_snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    try:
        check_payload = build_check_all_datas(
            target_snapshot,
            selection_start_ms=selection_start,
            selection_end_ms=selection_end,
            hours=hours,
        )
    except DataQualityError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "data_quality": exc.detail},
        ) from exc

    if check_payload is None:
        raise HTTPException(status_code=400, detail="Snapshot does not contain analyzable data")

    if not isinstance(check_payload, Mapping):
        raise HTTPException(status_code=400, detail="Snapshot payload is invalid")

    check_payload = dict(check_payload)
    check_payload.setdefault("symbol", target_snapshot.get("symbol"))
    check_payload["snapshot_id"] = snapshot_id
    check_payload["selection"] = {"start": selection_start, "end": selection_end}

    symbol = str(
        check_payload.get("symbol")
        or target_snapshot.get("symbol")
        or target_snapshot.get("pair")
        or "UNKNOWN"
    )

    period = None
    if isinstance(period_raw, str) and period_raw.strip():
        period = period_raw.strip()
    else:
        meta_source = target_snapshot.get("meta")
        if isinstance(meta_source, Mapping):
            if isinstance(meta_source.get("period"), str) and meta_source.get("period").strip():
                period = str(meta_source.get("period")).strip()
            elif isinstance(meta_source.get("window"), Mapping):
                label = meta_source["window"].get("label")
                if isinstance(label, str) and label.strip():
                    period = label.strip()
    if period is None:
        period = "custom_range"

    def _extract_last_price(payload: Mapping[str, Any]) -> float | None:
        ohlcv_block = payload.get("ohlcv")
        if isinstance(ohlcv_block, Mapping):
            minute_block = ohlcv_block.get("1m")
            if isinstance(minute_block, Mapping):
                candles = minute_block.get("candles")
                if isinstance(candles, Sequence) and candles:
                    tail = candles[-1]
                    if isinstance(tail, Mapping):
                        for key in ("c", "close", "price"):
                            value = tail.get(key)
                            if isinstance(value, (int, float)):
                                return float(value)
        return None

    last_price_value = _extract_last_price(check_payload)
    last_price = last_price_value if last_price_value is not None else 0.0

    api_key: str | None = None
    api_key_raw = payload.get("api_key")
    if isinstance(api_key_raw, str):
        candidate = api_key_raw.strip()
        if candidate:
            api_key = candidate

    if not api_key:
        env_key = os.environ.get("OPENAI_API_KEY")
        if isinstance(env_key, str):
            candidate = env_key.strip()
            if candidate:
                api_key = candidate

    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required")

    model_candidate = payload.get("model")
    model_id = "gpt-5"
    if isinstance(model_candidate, str) and model_candidate.strip().lower() == "gpt-5":
        model_id = "gpt-5"
    else:
        env_model = os.environ.get("OPENAI_MODEL_ID")
        if isinstance(env_model, str) and env_model.strip().lower() == "gpt-5":
            model_id = "gpt-5"

    upload_dir = Path(
        os.environ.get(
            "ANALYSIS_UPLOAD_DIR",
            str(PROJECT_ROOT / "var" / "analysis_uploads"),
        )
    ).expanduser()

    api_base = os.environ.get("OPENAI_API_BASE")

    try:
        analysis_result = await dispatch_trade_analysis(
            check_payload,
            symbol=symbol,
            period=period,
            last_price=last_price,
            upload_dir=upload_dir,
            api_key=api_key,
            model=model_id,
            api_base=api_base,
        )
    except httpx.HTTPStatusError as exc:
        openai_error_details = _extract_openai_error(exc.response)
        logging.getLogger(__name__).exception(
            "OpenAI request returned an error",
            extra={
                "snapshot_id": snapshot_id,
                "status_code": exc.response.status_code,
                "openai_error": openai_error_details,
            },
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "OpenAI returned an error",
                "status_code": exc.response.status_code,
                "openai_error": openai_error_details,
            },
        ) from exc
    except httpx.RequestError as exc:
        logging.getLogger(__name__).exception(
            "Failed to reach OpenAI",
            extra={"snapshot_id": snapshot_id},
        )
        raise HTTPException(
            status_code=502,
            detail="Failed to reach OpenAI",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.getLogger(__name__).exception(
            "Unexpected error during trade analysis",
            extra={"snapshot_id": snapshot_id},
        )
        raise HTTPException(status_code=502, detail="Trade analysis failed") from exc

    debug_block: Dict[str, Any] = {
        "symbol": symbol,
        "period": period,
        "last_price": last_price,
        "model": model_id,
        "attachment_file": analysis_result.file_path.name,
        "attachment_size_bytes": analysis_result.attachment_size,
        "attachment_sha256": analysis_result.attachment_sha256,
        "latency_ms": analysis_result.latency_ms,
    }
    if analysis_result.raw_text and analysis_result.status != "ok":
        debug_block["raw_text"] = analysis_result.raw_text

    response_payload = {
        "status": analysis_result.status,
        "request_id": analysis_result.request_id,
        "trade_json": analysis_result.trade_json,
        "debug": debug_block,
    }

    return JSONResponse(response_payload)


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
    return HTMLResponse(content=html)


