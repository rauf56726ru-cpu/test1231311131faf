"""Lightweight orderbook extraction helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import httpx

BINANCE_FUTURES_BOOK = "https://fapi.binance.com/fapi/v1/depth"


async def fetch_orderbook(symbol: str, window_minutes: int) -> Dict[str, object]:
    """Return high level orderbook metrics for the requested symbol."""

    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    params = {"symbol": symbol_clean, "limit": "100"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        response = await client.get(BINANCE_FUTURES_BOOK, params=params)
        response.raise_for_status()
        depth = response.json()
        if not isinstance(depth, dict):
            raise ValueError("Unexpected book payload")

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
    imbalance = 0.0
    if total_ask > 0:
        imbalance = total_bid / total_ask

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
