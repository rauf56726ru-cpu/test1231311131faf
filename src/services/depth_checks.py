"""Depth snapshot helpers for Binance Futures order book."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, MutableMapping, Sequence, Tuple

import httpx

from .binance import BINANCE_FAPI_BASE_URL
from .binance_ingest import _init_metrics, _request_with_backoff
from .tracing import TraceContext

DEPTH_ENDPOINT = f"{BINANCE_FAPI_BASE_URL}/depth"
DEFAULT_LIMIT = 500
TOP_N = 10


def _sum_top_levels(levels: Sequence[Sequence[Any]], depth: int) -> Tuple[float, float]:
    qty_total = 0.0
    notional_total = 0.0
    for price_str, qty_str, *_ in levels[:depth]:
        price = float(price_str)
        qty = float(qty_str)
        qty_total += qty
        notional_total += price * qty
    return qty_total, notional_total


async def fetch_depth_snapshot(
    *,
    symbol: str,
    client: httpx.AsyncClient | None,
    trace: TraceContext | None = None,
    timeout: float | httpx.Timeout | None = None,
    limit: int = DEFAULT_LIMIT,
    metrics: MutableMapping[str, int] | None = None,
) -> Dict[str, Any]:
    request_ts = int(time.time() * 1000)
    params = {"symbol": symbol.upper(), "limit": str(limit)}
    response = await _request_with_backoff(
        "GET",
        DEPTH_ENDPOINT,
        params=params,
        scope="ingest.depth",
        trace=trace,
        client=client,
        timeout=timeout,
        metrics=metrics,
    )
    if response.status_code != 200:
        raise RuntimeError(f"depth request failed with status {response.status_code}")
    payload = response.json()
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    best_bid = float(bids[0][0]) if bids else None
    best_ask = float(asks[0][0]) if asks else None
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
    top_bids_qty, top_bids_notional = _sum_top_levels(bids, TOP_N)
    top_asks_qty, top_asks_notional = _sum_top_levels(asks, TOP_N)
    imbalance = 0.0
    denom = top_bids_qty + top_asks_qty
    if denom > 1e-9:
        imbalance = (top_bids_qty - top_asks_qty) / denom
    snapshot_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    return {
        "ts": snapshot_ts,
        "ts_req": request_ts,
        "spread": spread,
        "imbalance": imbalance,
        "topN_liquidity": {
            "bids_qty": top_bids_qty,
            "asks_qty": top_asks_qty,
            "bids_notional": top_bids_notional,
            "asks_notional": top_asks_notional,
        },
    }


async def capture_depth_series(
    *,
    symbol: str,
    client: httpx.AsyncClient | None,
    trace: TraceContext | None = None,
    timeout: float | httpx.Timeout | None = None,
    snapshots: int = 3,
    labels: Sequence[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    metrics = _init_metrics()
    series: List[Dict[str, Any]] = []
    for _ in range(max(1, snapshots)):
        snapshot = await fetch_depth_snapshot(
            symbol=symbol,
            client=client,
            trace=trace,
            timeout=timeout,
            metrics=metrics,
        )
        series.append(snapshot)
    return series, metrics


__all__ = ["capture_depth_series"]
