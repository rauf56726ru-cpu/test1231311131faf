"""Lightweight orderbook extraction helpers."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import httpx

from .rate_limiter import get_global_rate_limiter
from .tracing import TraceContext

BINANCE_FUTURES_BOOK = "https://fapi.binance.com/fapi/v1/depth"
_CACHE_TTL_SECONDS = 20.0
_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}
_CACHE_LOCK = asyncio.Lock()
_RATE_LIMITER = get_global_rate_limiter()


async def _request_orderbook(
    symbol: str, *, trace: TraceContext | None = None
) -> Dict[str, object]:
    params = {"symbol": symbol, "limit": "100"}
    scope = "orderbook.depth"
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        async with _RATE_LIMITER.limit(scope=scope, trace=trace):
            response = await client.get(BINANCE_FUTURES_BOOK, params=params)
    await _RATE_LIMITER.note_used_weight(
        _parse_used_weight(response), scope=scope, trace=trace
    )
    if response.status_code in {418, 429}:
        await _RATE_LIMITER.apply_backoff(
            _parse_retry_after(response), scope=scope, trace=trace, reason="depth"
        )
        raise RuntimeError(f"Orderbook rate limited: {response.status_code}")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected book payload")
    return payload


def _parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_used_weight(response: httpx.Response) -> int | None:
    value = response.headers.get("X-MBX-USED-WEIGHT-1m")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


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
            trace.debug("book.cache.hit", symbol=symbol_clean)
        return dict(cached[1])

    async with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached and cached[0] > now:
            if trace is not None:
                trace.debug("book.cache.hit", symbol=symbol_clean, scope="orderbook")
            return dict(cached[1])
        if trace is not None:
            trace.debug("book.cache.miss", symbol=symbol_clean)
        depth = await _request_orderbook(symbol_clean, trace=trace)
        expires = now + _CACHE_TTL_SECONDS
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
