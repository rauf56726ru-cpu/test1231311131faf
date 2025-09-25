"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

import asyncio

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..services import (
    TIMEFRAME_WINDOWS,
    build_inspection_payload,
    fetch_bar_delta,
    fetch_ohlcv,
    fetch_session_vwap,
    fetch_tpo_profile,
    render_inspection_page,
)
from ..version import APP_VERSION

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


def _render_services_page() -> str:
    services = [
        {
            "title": "OHLCV History",
            "endpoint": "/ohlc",
            "description": "Возвращает свечи Binance с синхронизированными границами и метаданными закрытия.",
            "inputs": ["symbol (str)", "tf (str: 1m,3m,5m,15m,1h,4h,1d)"],
            "output": "{ symbol, tf, candles:[{t,o,h,l,c,v}], last_price, last_ts, next_close_ts, time_to_close_ms }",
        },
        {
            "title": "Bar Delta",
            "endpoint": "/delta",
            "description": "Считает buy/sell delta и CVD по барным границам на основе aggTrades.",
            "inputs": ["symbol (str)", "tf (str как в /ohlc)"],
            "output": "{ symbol, bar_delta:[{t,tf,delta,deltaMax,deltaMin,deltaPct,cvd}] }",
        },
        {
            "title": "Session VWAP",
            "endpoint": "/vwap",
            "description": "Агрегирует дневной и сессионный VWAP по окнам Meta за 5 дней.",
            "inputs": ["symbol (str)"],
            "output": "{ symbol, vwap:[{date,session,value}] }",
        },
        {
            "title": "TPO / Volume Profile",
            "endpoint": "/tpo",
            "description": "Строит TPO и volume profile по последним 2–5 сессиям выбранного окна.",
            "inputs": ["symbol (str)", "session (str: asia|london|ny)", "sessions (int 2-5)"],
            "output": "{ symbol, session, requested_sessions, sessions, tpo:[{date,session,VAL,VAH,POC}], profile:[{price,volume}] }",
        },
        {
            "title": "Diagnostics Bundle",
            "endpoint": "/diagnostics",
            "description": "Параллельно собирает ответы всех сервисов для проверки реальными данными.",
            "inputs": [
                "symbol (str)",
                "tf (str как в /ohlc)",
                "session (str: asia|london|ny)",
                "sessions (int 2-5)",
            ],
            "output": "{ symbol, tf, session, sessions, ohlc, delta, vwap, tpo }",
        },
    ]

    body = ["<!DOCTYPE html>", "<html lang=\"ru\">", "<head>", "<meta charset=\"utf-8\" />", "<title>Документация сервисов</title>", "<style>body{font-family:Inter,system-ui,-apple-system,sans-serif;padding:24px;max-width:960px;margin:0 auto;line-height:1.5;}section{margin-bottom:24px;border-bottom:1px solid #e0e0e0;padding-bottom:16px;}h1{margin-top:0;}code,pre{font-family:SFMono-Regular,ui-monospace,Menlo,Monaco,Consolas,monospace;}</style>", "</head>", "<body>", "<h1>Сервисы расчёта и данные Binance</h1>"]
    for service in services:
        body.append("<section>")
        body.append(f"<h2>{service['title']}</h2>")
        body.append(f"<p><strong>Endpoint:</strong> {service['endpoint']}</p>")
        body.append(f"<p>{service['description']}</p>")
        body.append("<p><strong>Вход:</strong> " + ", ".join(service["inputs"]) + "</p>")
        body.append(f"<p><strong>Выход:</strong> {service['output']}</p>")
        body.append("</section>")
    body.append("</body>")
    body.append("</html>")
    return "\n".join(body)


@app.get("/services", response_class=HTMLResponse)
async def services_page() -> HTMLResponse:
    return HTMLResponse(content=_render_services_page())


@app.get("/ohlc")
async def ohlc(
    symbol: str = Query(..., min_length=1, description="Trading pair, e.g. BTCUSDT"),
    tf: str = Query("1m", description="Timeframe: 1m,3m,5m,15m,1h,4h,1d"),
):
    timeframe = tf.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {tf}")
    try:
        payload = await fetch_ohlcv(symbol, timeframe)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors etc.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get("/delta")
async def delta(
    symbol: str = Query(..., min_length=1, description="Trading pair, e.g. BTCUSDT"),
    tf: str = Query("1m", description="Timeframe: 1m,3m,5m,15m,1h,4h,1d"),
):
    timeframe = tf.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {tf}")
    try:
        payload = await fetch_bar_delta(symbol, timeframe)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors etc.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get("/vwap")
async def vwap(symbol: str = Query(..., min_length=1, description="Trading pair")):
    try:
        payload = await fetch_session_vwap(symbol)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors etc.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get("/tpo")
async def tpo(
    symbol: str = Query(..., min_length=1, description="Trading pair, e.g. BTCUSDT"),
    session: str = Query("ny", description="Session name: asia, london, ny"),
    sessions: int = Query(5, ge=2, le=5, description="Number of sessions to aggregate"),
):
    try:
        payload = await fetch_tpo_profile(symbol, session=session, sessions=sessions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return payload


@app.get("/diagnostics")
async def diagnostics(
    symbol: str = Query(..., min_length=1, description="Trading pair, e.g. BTCUSDT"),
    tf: str = Query("1m", description="Timeframe: 1m,3m,5m,15m,1h,4h,1d"),
    session: str = Query("ny", description="Session name: asia, london, ny"),
    sessions: int = Query(5, ge=2, le=5, description="Number of sessions to aggregate"),
):
    timeframe = tf.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {tf}")
    try:
        ohlc_task = fetch_ohlcv(symbol, timeframe)
        delta_task = fetch_bar_delta(symbol, timeframe)
        vwap_task = fetch_session_vwap(symbol)
        tpo_task = fetch_tpo_profile(symbol, session=session, sessions=sessions)
        ohlc_payload, delta_payload, vwap_payload, tpo_payload = await asyncio.gather(
            ohlc_task,
            delta_task,
            vwap_task,
            tpo_task,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors etc.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "symbol": symbol.upper(),
        "tf": timeframe,
        "session": session.lower(),
        "sessions": sessions,
        "ohlc": ohlc_payload,
        "delta": delta_payload,
        "vwap": vwap_payload,
        "tpo": tpo_payload,
    }


@app.get("/inspection", response_class=HTMLResponse)
async def inspection(
    request: Request,
    symbol: str = Query("BTCUSDT", min_length=1, description="Trading pair, e.g. BTCUSDT"),
    tf: str = Query("1m", description="Timeframe: 1m,3m,5m,15m,1h,4h,1d"),
    session: str = Query("ny", description="Session name: asia, london, ny"),
    sessions: int = Query(5, ge=2, le=5, description="Number of sessions to aggregate"),
):
    timeframe = tf.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {tf}")

    try:
        payload = await build_inspection_payload(
            symbol,
            timeframe,
            session=session,
            sessions=sessions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors etc.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    accept_header = request.headers.get("accept", "").lower()
    if "application/json" in accept_header:
        return payload

    html = render_inspection_page(
        payload,
        symbol=symbol,
        timeframe=timeframe,
        session=session,
        sessions=sessions,
    )
    return HTMLResponse(content=html)


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


