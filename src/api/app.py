"""FastAPI application powering the trading copilot UI."""
from __future__ import annotations

import html
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Sequence

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from ..services import (
    DataQualityError,
    build_check_all_datas,
    dispatch_trade_analysis,
)
from ..services.ohlc import TIMEFRAME_TO_MS, resample_ohlcv
from ..services.analysis import SYSTEM_PROMPT
from ..version import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = PROJECT_ROOT / "public"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1/klines"
MINUTE_INTERVAL_MS = 60_000
REQUIRED_MINUTE_HOURS = 72
TARGET_TIMEFRAMES: tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Copilot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if PUBLIC_DIR.is_dir():
    app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")


def _extract_openai_error(response: httpx.Response) -> Any | None:
    """Extract a readable error payload from the OpenAI API."""

    if response is None:
        return None

    try:
        payload = response.json()
    except ValueError:
        text = response.text
        return text.strip()[:500] if text else None

    if isinstance(payload, Mapping):
        error_node = payload.get("error")
        if isinstance(error_node, Mapping):
            cleaned: dict[str, Any] = {}
            for key in ("message", "type", "code", "param"):
                value = error_node.get(key)
                if isinstance(value, str) and value.strip():
                    cleaned[key] = value.strip()[:500]
                elif value is not None:
                    cleaned[key] = value
            if cleaned:
                return cleaned
            return str(error_node)[:500]
        return payload

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return payload[:5]

    return payload


def _datetime_to_ms(moment: datetime | None) -> int | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        aligned = moment.replace(tzinfo=timezone.utc)
    else:
        aligned = moment.astimezone(timezone.utc)
    return int(aligned.timestamp() * 1000)


def _parse_binance_row(row: Any) -> Dict[str, float] | None:
    open_time: int | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume_value: float | None = None

    if isinstance(row, Mapping):
        time_candidate = None
        for key in ("openTime", "t", "time", "open_time"):
            candidate = row.get(key)
            if candidate is not None:
                time_candidate = candidate
                break
        if isinstance(time_candidate, (int, float)):
            open_time = int(time_candidate)

        def _numeric(key: str, fallback: str | None = None) -> float | None:
            value = row.get(key)
            if value is None and fallback is not None:
                value = row.get(fallback)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        open_price = _numeric("o", "open")
        high_price = _numeric("h", "high")
        low_price = _numeric("l", "low")
        close_price = _numeric("c", "close")
        volume_value = _numeric("v", "volume")
    elif isinstance(row, Sequence):
        try:
            open_time = int(row[0])
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            if len(row) > 5:
                volume_value = float(row[5])
        except (IndexError, TypeError, ValueError):
            return None

    if open_time is None or open_price is None or high_price is None or low_price is None or close_price is None:
        return None

    volume = float(volume_value) if volume_value is not None else 0.0
    return {
        "t": int(open_time),
        "o": float(open_price),
        "h": float(high_price),
        "l": float(low_price),
        "c": float(close_price),
        "v": volume,
    }


async def _download_recent_minutes(symbol: str, *, hours: int = REQUIRED_MINUTE_HOURS) -> List[Dict[str, float]]:
    if hours <= 0:
        raise ValueError("hours must be positive")

    required_bars = max(60, hours * 60)
    candles: Dict[int, Dict[str, float]] = {}
    limit = 1000
    end_time: int | None = None
    last_oldest: int | None = None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            while len(candles) < required_bars:
                params = {
                    "symbol": symbol.upper(),
                    "interval": "1m",
                    "limit": str(limit),
                }
                if end_time is not None and end_time > 0:
                    params["endTime"] = str(end_time)

                response = await client.get(BINANCE_FAPI_REST, params=params)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, Sequence) or not data:
                    break

                parsed: List[Dict[str, float]] = []
                for row in data:
                    candle = _parse_binance_row(row)
                    if candle is None:
                        continue
                    parsed.append(candle)
                if not parsed:
                    break

                for candle in parsed:
                    candles[candle["t"]] = candle

                oldest = min(candle["t"] for candle in parsed)
                if last_oldest is not None and oldest >= last_oldest:
                    break
                last_oldest = oldest
                end_time = oldest - MINUTE_INTERVAL_MS
                if end_time is not None and end_time <= 0:
                    break
    except httpx.HTTPError as exc:  # pragma: no cover - network failure guard
        logger.error(
            "Failed to fetch minute candles",
            exc_info=exc,
            extra={"symbol": symbol.upper(), "hours": hours},
        )
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить минутные свечи Binance",
        ) from exc

    ordered_times = sorted(candles)
    if not ordered_times:
        return []

    if len(ordered_times) > required_bars:
        ordered_times = ordered_times[-required_bars:]

    return [candles[ts] for ts in ordered_times]


def _generate_synthetic_minutes(hours: int = REQUIRED_MINUTE_HOURS) -> List[Dict[str, float]]:
    total = max(60, hours * 60)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - total * MINUTE_INTERVAL_MS
    price = 26_000.0
    candles: List[Dict[str, float]] = []
    for idx in range(total):
        ts = start_ms + idx * MINUTE_INTERVAL_MS
        drift = math.sin(idx / 180.0) * 120.0 + math.cos(idx / 300.0) * 85.0
        open_price = max(100.0, price + drift)
        close_variation = math.sin(idx / 45.0) * 90.0 + math.cos(idx / 60.0) * 45.0
        close_price = max(100.0, open_price + close_variation)
        high_price = max(open_price, close_price) + abs(math.sin(idx / 30.0)) * 75.0
        low_price = min(open_price, close_price) - abs(math.cos(idx / 42.0)) * 75.0
        volume = 120.0 + abs(math.sin(idx / 18.0)) * 80.0
        candles.append(
            {
                "t": ts,
                "o": round(open_price, 2),
                "h": round(high_price, 2),
                "l": round(max(1.0, low_price), 2),
                "c": round(close_price, 2),
                "v": round(volume, 2),
            }
        )
        price = close_price
    return candles


async def _build_runtime_snapshot(
    symbol: str,
    timeframe: str,
    *,
    selection_start_ms: int | None = None,
    selection_end_ms: int | None = None,
) -> Dict[str, Any]:
    symbol_norm = (symbol or "BTCUSDT").upper()
    timeframe_key = (timeframe or "1m").lower()

    fallback_reason: str | None = None
    try:
        minute_candles = await _download_recent_minutes(symbol_norm)
    except HTTPException as exc:
        logger.warning(
            "Falling back to synthetic minute candles",
            extra={"symbol": symbol_norm, "status_code": exc.status_code},
        )
        fallback_reason = f"binance_error_{exc.status_code}" if exc.status_code else "binance_error"
        minute_candles = _generate_synthetic_minutes(REQUIRED_MINUTE_HOURS)

    if len(minute_candles) < 3 * 24 * 60:
        logger.warning(
            "Minute dataset too short, generating synthetic candles",
            extra={"symbol": symbol_norm, "candles": len(minute_candles)},
        )
        fallback_reason = fallback_reason or "insufficient_minutes"
        minute_candles = _generate_synthetic_minutes(REQUIRED_MINUTE_HOURS)

    frames: Dict[str, Dict[str, Any]] = {}
    for tf in TARGET_TIMEFRAMES:
        if tf == "1m":
            candles = minute_candles
        else:
            interval_ms = TIMEFRAME_TO_MS.get(tf)
            if interval_ms is None:
                continue
            candles = resample_ohlcv(minute_candles, interval_ms)
        frames[tf] = {"tf": tf, "candles": candles}

    if timeframe_key not in frames:
        interval_ms = TIMEFRAME_TO_MS.get(timeframe_key)
        if interval_ms is not None:
            frames[timeframe_key] = {
                "tf": timeframe_key,
                "candles": resample_ohlcv(minute_candles, interval_ms),
            }

    selection_payload: Dict[str, int] | None = None
    if selection_start_ms is not None and selection_end_ms is not None:
        selection_payload = {
            "start": int(min(selection_start_ms, selection_end_ms)),
            "end": int(max(selection_start_ms, selection_end_ms)),
        }

    snapshot: Dict[str, Any] = {
        "id": f"live-{int(datetime.now(timezone.utc).timestamp()*1000)}",
        "symbol": symbol_norm,
        "tf": timeframe_key,
        "frames": frames,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "meta": {"source": {"kind": "binance_live", "symbol": symbol_norm}},
    }
    if selection_payload:
        snapshot["selection"] = selection_payload
    if fallback_reason:
        source_meta = snapshot["meta"].setdefault("source", {})
        if isinstance(source_meta, dict):
            source_meta["fallback"] = fallback_reason

    return snapshot


def _extract_last_price(payload: Mapping[str, Any]) -> float | None:
    latest = payload.get("latest_candle") if isinstance(payload, Mapping) else None
    if isinstance(latest, Mapping):
        for key in ("c", "close", "price"):
            value = latest.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue

    detailed = payload.get("datas_for_last_N_hours") if isinstance(payload, Mapping) else None
    if isinstance(detailed, Mapping):
        frames = detailed.get("frames")
        if isinstance(frames, Mapping):
            minute_frame = frames.get("1m")
            if isinstance(minute_frame, Mapping):
                candles = minute_frame.get("candles")
                if isinstance(candles, Sequence) and candles:
                    last = candles[-1]
                    if isinstance(last, Mapping):
                        value = last.get("c")
                        try:
                            if value is not None:
                                return float(value)
                        except (TypeError, ValueError):
                            pass
    return None

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @validator("content")
    def _validate_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("content cannot be empty")
        return value


class ChatSettings(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, max_length=120)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    api_base: str | None = Field(default=None, max_length=200)


class ChatRequest(BaseModel):
    system_prompt: str = Field(default=SYSTEM_PROMPT)
    messages: Sequence[ChatMessage]
    settings: ChatSettings = Field(default_factory=ChatSettings)
    api_key: str = Field(min_length=10)


class ChatResponse(BaseModel):
    reply: str
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


class TestEnvironmentRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", max_length=32)
    timeframe: str = Field(default="1m", max_length=8)
    start: datetime | None = None
    end: datetime | None = None


class CheckAllRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", max_length=32)
    timeframe: str = Field(default="1m", max_length=8)
    start: datetime | None = None
    end: datetime | None = None


class TradeAnalysisRequest(BaseModel):
    api_key: str = Field(min_length=10)
    model: str = Field(default=DEFAULT_MODEL, max_length=120)
    system_prompt: str | None = None
    api_base: str | None = Field(default=None, max_length=200)
    symbol: str = Field(default="BTCUSDT", max_length=32)
    timeframe: str = Field(default="1m", max_length=8)
    period: str = Field(default="last_4h", max_length=32)
    last_price: float | None = None
    check_all: Dict[str, Any] | None = None


def _render_test_environment_window(data: Mapping[str, Any]) -> str:
    pretty = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    return (
        "<!DOCTYPE html><html lang=\"ru\"><head><meta charset=\"UTF-8\" />"
        "<title>test environment</title><style>body{margin:0;background:#0f172a;color:#e2e8f0;font:14px/1.5 'JetBrains Mono',monospace;}"
        "pre{padding:24px;white-space:pre-wrap;word-break:break-word;}</style></head><body>"
        f"<pre>{pretty}</pre></body></html>"
    )


async def _call_chat_completion(payload: ChatRequest) -> ChatResponse:
    base_url = payload.settings.api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {payload.api_key}",
        "Content-Type": "application/json",
    }

    body: dict[str, Any] = {
        "model": payload.settings.model,
        "temperature": payload.settings.temperature,
        "messages": [
            {"role": "system", "content": payload.system_prompt},
            *[
                {"role": message.role, "content": message.content}
                for message in payload.messages
            ],
        ],
    }
    if payload.settings.top_p is not None:
        body["top_p"] = payload.settings.top_p

    logger.info(
        "ChatGPT request",
        extra={
            "endpoint": "chat.completions",
            "model": payload.settings.model,
            "message_count": len(body["messages"]),
            "temperature": payload.settings.temperature,
        },
    )

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, headers=headers, json=body)

    if not response.is_success:
        detail = _extract_openai_error(response)
        raise HTTPException(
            status_code=response.status_code,
            detail={"message": "OpenAI request failed", "openai_error": detail},
        )

    data = response.json()
    reply = ""
    try:
        reply = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError):
        reply = ""

    usage = data.get("usage") if isinstance(data, Mapping) else None
    raw = data if isinstance(data, Mapping) else None
    return ChatResponse(reply=reply or "", usage=usage, raw=raw)


async def _build_check_all_payload(
    symbol: str,
    timeframe: str,
    start: datetime | None,
    end: datetime | None,
) -> dict[str, Any]:
    selection_start_ms = _datetime_to_ms(start)
    selection_end_ms = _datetime_to_ms(end)

    snapshot = await _build_runtime_snapshot(
        symbol,
        timeframe,
        selection_start_ms=selection_start_ms,
        selection_end_ms=selection_end_ms,
    )

    error: MutableMapping[str, Any] | str | None = None
    try:
        check_all = build_check_all_datas(
            snapshot,
            now_utc=datetime.now(timezone.utc),
            selection_start_ms=selection_start_ms,
            selection_end_ms=selection_end_ms,
            hours=4,
        )
    except DataQualityError as exc:
        check_all = None
        error = exc.detail or str(exc)

    result: dict[str, Any] = {"snapshot": snapshot, "check_all": check_all}
    if error:
        result["error"] = error
    return result


@app.get("/api/chat/defaults")
async def chat_defaults() -> JSONResponse:
    return JSONResponse(
        {
            "system_prompt": SYSTEM_PROMPT,
            "settings": {
                "model": DEFAULT_MODEL,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": None,
            },
        }
    )


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest = Body(...)) -> JSONResponse:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    response = await _call_chat_completion(payload)
    return JSONResponse(response.dict())


@app.get("/api/check-all")
async def check_all_default(
    symbol: str = Query("BTCUSDT", max_length=32),
    timeframe: str = Query("1m", max_length=8),
) -> JSONResponse:
    data = await _build_check_all_payload(symbol, timeframe, None, None)
    return JSONResponse(data)


@app.post("/api/check-all")
async def check_all_custom(payload: CheckAllRequest = Body(...)) -> JSONResponse:
    data = await _build_check_all_payload(
        payload.symbol,
        payload.timeframe,
        payload.start,
        payload.end,
    )
    return JSONResponse(data)


@app.post("/api/test-environment")
async def create_test_environment(payload: TestEnvironmentRequest = Body(...)) -> JSONResponse:
    data = await _build_check_all_payload(
        payload.symbol,
        payload.timeframe,
        payload.start,
        payload.end,
    )
    html_payload = _render_test_environment_window(data["check_all"] or {})
    data["rendered_html"] = html_payload
    return JSONResponse(data)


@app.post("/api/trade-analysis")
async def trade_analysis(payload: TradeAnalysisRequest = Body(...)) -> JSONResponse:
    upload_dir = UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "ChatGPT request",
        extra={
            "endpoint": "trade-analysis",
            "model": payload.model,
            "symbol": payload.symbol,
            "timeframe": payload.timeframe,
        },
    )

    check_all_payload: Mapping[str, Any] | None
    if isinstance(payload.check_all, Mapping):
        check_all_payload = payload.check_all
    else:
        generated = await _build_check_all_payload(payload.symbol, payload.timeframe, None, None)
        check_all_payload = generated.get("check_all") if isinstance(generated, Mapping) else None

    if not isinstance(check_all_payload, Mapping):
        raise HTTPException(status_code=500, detail="Не удалось построить check_all_datas")

    last_price = payload.last_price
    if last_price is None:
        inferred_price = _extract_last_price(check_all_payload)
        if inferred_price is not None:
            last_price = inferred_price

    result = await dispatch_trade_analysis(
        check_all_payload,
        symbol=payload.symbol,
        period=payload.period,
        last_price=last_price,
        upload_dir=upload_dir,
        api_key=payload.api_key,
        model=payload.model,
        api_base=payload.api_base,
        system_prompt=payload.system_prompt,
    )

    body = {
        "status": result.status,
        "request_id": result.request_id,
        "trade_json": result.trade_json,
        "raw_text": result.raw_text,
        "latency_ms": result.latency_ms,
        "attachment_size": result.attachment_size,
        "attachment_sha256": result.attachment_sha256,
        "attachment_path": str(result.file_path),
    }
    return JSONResponse(body)


@app.get("/version")
async def version() -> JSONResponse:
    return JSONResponse({"version": APP_VERSION})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    try:
        html_content = TEMPLATES_DIR.joinpath("index.html").read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - deployment guard
        raise HTTPException(status_code=500, detail="Index template is missing") from exc
    return HTMLResponse(content=html_content)
