"""FastAPI application powering the trading copilot UI."""
from __future__ import annotations

import html
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from ..services import (
    DataQualityError,
    build_check_all_datas,
    build_placeholder_snapshot,
    dispatch_trade_analysis,
)
from ..services.analysis import SYSTEM_PROMPT
from ..version import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = PROJECT_ROOT / "public"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

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


class TradeAnalysisRequest(BaseModel):
    api_key: str = Field(min_length=10)
    model: str = Field(default=DEFAULT_MODEL, max_length=120)
    system_prompt: str | None = None
    api_base: str | None = Field(default=None, max_length=200)
    symbol: str = Field(default="BTCUSDT", max_length=32)
    timeframe: str = Field(default="1m", max_length=8)
    period: str = Field(default="last_4h", max_length=32)
    last_price: float | None = None


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


def _build_check_all_payload(symbol: str, timeframe: str, start: datetime | None, end: datetime | None) -> dict[str, Any]:
    snapshot = build_placeholder_snapshot(symbol=symbol, timeframe=timeframe)
    selection_start_ms = int(start.timestamp() * 1000) if start else None
    selection_end_ms = int(end.timestamp() * 1000) if end else None

    error: str | None = None
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
        error = str(exc)

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
    data = _build_check_all_payload(symbol, timeframe, None, None)
    return JSONResponse(data)


@app.post("/api/test-environment")
async def create_test_environment(payload: TestEnvironmentRequest = Body(...)) -> JSONResponse:
    data = _build_check_all_payload(payload.symbol, payload.timeframe, payload.start, payload.end)
    html_payload = _render_test_environment_window(data["check_all"] or {})
    data["rendered_html"] = html_payload
    return JSONResponse(data)


@app.post("/api/trade-analysis")
async def trade_analysis(payload: TradeAnalysisRequest = Body(...)) -> JSONResponse:
    snapshot = build_placeholder_snapshot(symbol=payload.symbol, timeframe=payload.timeframe)
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

    frames = snapshot.get("frames", {}) if isinstance(snapshot, Mapping) else {}
    frame_node = frames.get(payload.timeframe)
    candles: list[Mapping[str, Any]] = []
    if isinstance(frame_node, Mapping):
        raw_candles = frame_node.get("candles")
        if isinstance(raw_candles, list):
            candles = [item for item in raw_candles if isinstance(item, Mapping)]
    last_price = payload.last_price
    if last_price is None and candles:
        try:
            last_price = float(candles[-1].get("c"))
        except (ValueError, TypeError, AttributeError):
            last_price = None

    result = await dispatch_trade_analysis(
        snapshot,
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
