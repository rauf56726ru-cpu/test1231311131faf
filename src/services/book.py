"""Lightweight orderbook extraction helpers."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Tuple

import aiohttp

from .binance import (
    BinanceAPIException,
    BinanceRequestException,
    fetch_um_order_book,
)
from .http_client import RATE_LIMIT_STATUSES
from .tracing import TraceContext
from .vision_store import get_store
_CACHE_TTL_SECONDS = 20.0
_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}
_CACHE_LOCK = asyncio.Lock()


class OrderbookUnavailable(RuntimeError):
    """Raised when a live orderbook snapshot cannot be retrieved."""


async def _load_store_orderbook(symbol: str, window_minutes: int) -> Dict[str, object] | None:
    store = get_store()
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - max(1, window_minutes) * 60_000
    snapshots = await asyncio.to_thread(
        store.fetch_depth_snapshots,
        symbol,
        start_ms,
        end_ms,
        limit=1,
        descending=True,
    )
    if not snapshots:
        return None
    snapshot = snapshots[-1]
    bids = snapshot.get("bids") or []
    asks = snapshot.get("asks") or []
    top_levels: List[Dict[str, float]] = []
    for side, levels in (("bid", bids), ("ask", asks)):
        for level in levels[:5]:
            price: float | None = None
            size: float | None = None
            if isinstance(level, Mapping):
                price_value = level.get("price") or level.get("p")
                size_value = level.get("size") or level.get("qty") or level.get("sz")
                try:
                    price = float(price_value)
                    size = float(size_value)
                except (TypeError, ValueError):
                    price = None
                    size = None
            else:
                try:
                    price = float(level[0])
                    size = float(level[1]) if len(level) > 1 else None
                except (TypeError, ValueError, IndexError):
                    price = None
                    size = None
            if price is None or size is None:
                continue
            top_levels.append({"side": side, "p": price, "sz": size})

    if not top_levels:
        return None

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
        "symbol": symbol,
        "captured_at": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        "window_minutes": window_minutes,
        "top_levels": top_levels,
        "imbalance": imbalance,
        "spoofing_flags": spoofing_flags,
        "source": "vision_store",
    }


async def _request_orderbook(
    symbol: str, *, trace: TraceContext | None = None
) -> Dict[str, object]:
    scope = "orderbook.depth"
    try:
        payload = await fetch_um_order_book(symbol, limit=100)
    except BinanceAPIException as exc:
        status = getattr(exc, "status_code", None)
        if status in RATE_LIMIT_STATUSES:
            raise OrderbookUnavailable(f"rate_limited:{status}") from exc
        raise OrderbookUnavailable(str(exc)) from exc
    except (BinanceRequestException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise OrderbookUnavailable(str(exc)) from exc

    if trace is not None:
        trace.info(
            "book.fetch",
            scope=scope,
            symbol=symbol,
            details="limit=100",
        )

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

    store_result = await _load_store_orderbook(symbol_clean, window_minutes)
    if store_result:
        return store_result

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
