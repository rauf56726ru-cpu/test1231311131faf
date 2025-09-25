"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..services import (
    build_inspection_payload,
    get_snapshot,
    register_snapshot,
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
    snapshot: str = Query(..., description="Snapshot identifier"),
) -> HTMLResponse:
    snapshot_data = get_snapshot(snapshot)
    if not snapshot_data:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    payload = build_inspection_payload(snapshot_data)

    accept_header = request.headers.get("accept", "").lower()
    if "application/json" in accept_header:
        return JSONResponse(payload)

    html = render_inspection_page(
        payload,
        snapshot_id=snapshot_data["id"],
        symbol=snapshot_data.get("symbol", "UNKNOWN"),
        timeframe=snapshot_data.get("tf", "1m"),
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


