"""Lightweight orderbook extraction helpers."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import httpx

from .http_client import RATE_LIMIT_STATUSES, request as http_request
from .tracing import TraceContext

BINANCE_FUTURES_BOOK = "https://fapi.binance.com/fapi/v1/depth"
_CACHE_TTL_SECONDS = 20.0
_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}
_CACHE_LOCK = asyncio.Lock()


class OrderbookUnavailable(RuntimeError):
    """Raised when a live orderbook snapshot cannot be retrieved."""


async def _request_orderbook(
    symbol: str, *, trace: TraceContext | None = None
) -> Dict[str, object]:
    params = {"symbol": symbol, "limit": "100"}
    scope = "orderbook.depth"
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        response = await http_request(
            "GET",
            BINANCE_FUTURES_BOOK,
            scope=scope,
            trace=trace,
            client=client,
            params=params,
            symbol=symbol,
            details="limit=100",
            max_retries=0,
            rate_limit_statuses=RATE_LIMIT_STATUSES,
        )

    status_code = response.status_code
    if status_code in RATE_LIMIT_STATUSES:
        raise OrderbookUnavailable(f"rate_limited:{status_code}")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise OrderbookUnavailable(str(exc)) from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise OrderbookUnavailable("unexpected_payload")
    return payload
async def fetch_orderbook(
    symbol: str,
    window_minutes: int,
    *,
    trace: TraceContext | None = None,
) -> Dict[str, object]:
    """Return high level orderbook metrics for the requested symbol."""

    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    cache_key = symbol_clean
    now = time.monotonic()
    cached = _CACHE.get(cache_key)
    if cached and cached[0] > now:
        if trace is not None:
            trace.info(
                "book.cache.hit",
                scope="orderbook.depth",
                symbol=symbol_clean,
                metrics={"ttl_ms": int((cached[0] - now) * 1000.0)},
            )
        return dict(cached[1])

    async with _CACHE_LOCK:
        now = time.monotonic()
        cached = _CACHE.get(cache_key)
        if cached and cached[0] > now:
            if trace is not None:
                trace.info(
                    "book.cache.hit",
                    scope="orderbook.depth",
                    symbol=symbol_clean,
                    metrics={"ttl_ms": int((cached[0] - now) * 1000.0)},
                )
            return dict(cached[1])
        if trace is not None:
            trace.info(
                "book.cache.miss",
                scope="orderbook.depth",
                symbol=symbol_clean,
            )
        try:
            depth = await _request_orderbook(symbol_clean, trace=trace)
        except OrderbookUnavailable as exc:
            if trace is not None:
                trace.warn(
                    "fallback.engaged",
                    scope="orderbook.depth",
                    symbol=symbol_clean,
                    details=str(exc),
                )
            raise
        expires = time.monotonic() + _CACHE_TTL_SECONDS
        _CACHE[cache_key] = (expires, depth)

    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    top_levels: List[Dict[str, float]] = []
    for side, levels in (("bid", bids), ("ask", asks)):
        for level in levels[:5]:
            try:
                price = float(level[0])
                size = float(level[1])
            except (IndexError, TypeError, ValueError):
                continue
            top_levels.append({"side": side, "p": price, "sz": size})

    total_bid = sum(level.get("sz", 0.0) for level in top_levels if level.get("side") == "bid")
    total_ask = sum(level.get("sz", 0.0) for level in top_levels if level.get("side") == "ask")
    imbalance = total_bid / total_ask if total_ask > 0 else 0.0

    average_size = (total_bid + total_ask) / max(len(top_levels), 1)
    spoof_threshold = average_size * 10
    spoofing_flags = [
        {"price": level["p"], "side": level["side"]}
        for level in top_levels
        if level["sz"] >= spoof_threshold and level["sz"] > 0
    ]

    return {
        "symbol": symbol_clean,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "window_minutes": window_minutes,
        "top_levels": top_levels,
        "imbalance": imbalance,
        "spoofing_flags": spoofing_flags,
    }
