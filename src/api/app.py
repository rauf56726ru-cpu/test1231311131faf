"""Minimal FastAPI app that exposes OHLCV history for the chart."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..services import (
    InsufficientDataError,
    build_inspection_payload,
    build_market_overview,
    build_placeholder_snapshot,
    DEFAULT_SYMBOL,
    get_snapshot,
    list_snapshots,
    register_snapshot,
    render_market_overview_html,
    render_inspection_page,
)
from ..version import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = PROJECT_ROOT / "public"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

app = FastAPI(title="Chart OHLC API")

template_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)
check_all_datas_template = template_env.get_template("check_all_datas.html")

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
        placeholder_payload = {
            "DATA": {
                "symbol": DEFAULT_SYMBOL,
                "frames": {},
                "selection": None,
                "delta_cvd": {},
                "vwap_tpo": {},
                "zones": {"status": "waiting", "detail": "Создайте первый снэпшот"},
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


def _parse_accept_header(header: str | None) -> Tuple[str | None, bool]:
    if not header:
        return "application/json", True

    parts = [part.strip() for part in header.split(",") if part.strip()]
    if not parts:
        return "application/json", True

    ranked: list[tuple[str, float, int]] = []
    for idx, part in enumerate(parts):
        mime = part
        q = 1.0
        if ";" in part:
            tokens = [token.strip() for token in part.split(";") if token.strip()]
            mime = tokens[0]
            for token in tokens[1:]:
                if token.startswith("q="):
                    try:
                        q = float(token[2:])
                    except ValueError:
                        q = 0.0
        ranked.append((mime.lower(), q, idx))

    ranked.sort(key=lambda item: (-item[1], item[2]))

    for mime, _, _ in ranked:
        if mime in {"application/json", "application/*", "*/*"}:
            return "application/json", True
        if mime in {"text/html", "text/*"}:
            return "text/html", True
    return None, False


@app.get("/inspection/check-all-datas")
async def inspection_check_all_datas(
    request: Request,
    hours: int = Query(4, ge=2, le=4),
) -> Response:
    accept_value, matched = _parse_accept_header(request.headers.get("accept"))
    if not matched or accept_value is None:
        raise HTTPException(status_code=406, detail="Not Acceptable")

    try:
        payload = build_market_overview(hours=hours)
    except InsufficientDataError:
        return Response(status_code=204)

    if accept_value == "application/json":
        if hasattr(payload, "model_dump"):
            content = payload.model_dump(mode="json", by_alias=True)
        else:
            content = payload.dict(by_alias=True)
        return JSONResponse(content)

    html = render_market_overview_html(payload, check_all_datas_template)
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


