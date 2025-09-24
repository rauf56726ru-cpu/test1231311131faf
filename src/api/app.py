"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..services import TIMEFRAME_WINDOWS, fetch_bar_delta, fetch_ohlcv
from ..version import APP_VERSION

app = FastAPI(title="Chart OHLC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict[str, str]:
    return {"version": APP_VERSION}
