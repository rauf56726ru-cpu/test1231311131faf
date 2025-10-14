"""Orderflow utilities with shared rate limiting and conservative fallbacks."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import httpx

from .http_client import RATE_LIMIT_STATUSES, TRANSIENT_STATUSES, request as http_request
from .tracing import TraceContext
from .vision_store import get_store

__all__ = [
    "fetch_footprint",
    "calculate_cvd",
    "compute_orderflow_aggregates",
]

BINANCE_FUTURES_AGG_TRADES = "https://fapi.binance.com/fapi/v1/aggTrades"
LOGGER = logging.getLogger(__name__)
_FOOTPRINT_LOCK = asyncio.Lock()
_MAX_WINDOW_HOURS = 4
_PER_BAR_MINUTES = 120
_PAGE_WINDOW_MS = 5 * 60_000
_MAX_BATCHES = 48
_AGG_INTERVALS_MINUTES: Mapping[str, int] = {"15m": 15, "1h": 60}


class OrderflowError(RuntimeError):
    """Raised when upstream orderflow data are invalid."""


@dataclass(slots=True)
class OrderflowSnapshot:
    """Container for compact orderflow data."""

    per_bar: List[Dict[str, Any]]
    aggregates: Dict[str, List[Dict[str, Any]]]

    def as_dict(self) -> Dict[str, Any]:
        return {"per_bar": self.per_bar, "aggregates": self.aggregates}


async def _load_trades_from_store(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    store = get_store()
    rows = await asyncio.to_thread(
        store.fetch_agg_trades,
        symbol,
        start_ms,
        end_ms,
    )
    if not rows:
        return []
    trades: List[Dict[str, Any]] = []
    for row in rows:
        trades.append(
            {
                "T": int(row["t"]),
                "p": float(row["p"]),
                "q": float(row["q"]),
                "m": bool(row.get("m")),
                "side": row.get("side"),
            }
        )
    return trades


def _isoformat(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _round_price(price: float, *, precision: int = 2) -> float:
    return round(price, precision)


def _empty_snapshot() -> OrderflowSnapshot:
    return OrderflowSnapshot(
        per_bar=[], aggregates={tf: [] for tf in _AGG_INTERVALS_MINUTES}
    )


def _compute_aggregates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    aggregates: Dict[str, List[Dict[str, Any]]] = {tf: [] for tf in _AGG_INTERVALS_MINUTES}
    if not rows:
        return aggregates

    sorted_rows = sorted(
        (
            row
            for row in rows
            if isinstance(row, Mapping)
            and (ts := row.get("ts")) is not None
            and isinstance(ts, (int, float))
        ),
        key=lambda row: int(row["ts"]),
    )

    for tf, minutes in _AGG_INTERVALS_MINUTES.items():
        interval_ms = minutes * 60_000
        buckets: List[Dict[str, Any]] = []
        bucket_start: int | None = None
        bucket_delta = 0.0
        bucket_volume = 0.0
        bucket_rows = 0
        running_cvd = 0.0

        def _flush(current_start: int) -> None:
            nonlocal bucket_delta, bucket_volume, bucket_rows, running_cvd
            running_cvd += bucket_delta
            buckets.append(
                {
                    "t": _isoformat(current_start),
                    "ts": current_start,
                    "delta_sum": bucket_delta,
                    "cvd_close": running_cvd,
                    "vol_sum": bucket_volume,
                    "bars": bucket_rows,
                }
            )
            bucket_delta = 0.0
            bucket_volume = 0.0
            bucket_rows = 0

        for row in sorted_rows:
            ts = int(row["ts"])
            bucket_id = (ts // interval_ms) * interval_ms
            if bucket_start is None:
                bucket_start = bucket_id
            if bucket_id != bucket_start:
                _flush(bucket_start)
                bucket_start = bucket_id
            delta = float(row.get("delta", 0.0))
            volume = float(
                row.get(
                    "volume",
                    float(row.get("ask", row.get("ask_vol", 0.0)))
                    + float(row.get("bid", row.get("bid_vol", 0.0))),
                )
            )
            bucket_delta += delta
            bucket_volume += volume
            bucket_rows += 1

        if bucket_start is not None and bucket_rows:
            _flush(bucket_start)

        aggregates[tf] = buckets

    return aggregates


async def _fetch_trades(
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    trace: TraceContext | None = None,
) -> tuple[List[Mapping[str, object]] | None, int | None]:
    params = {
        "symbol": symbol.upper(),
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": "1000",
    }
    scope = "orderflow.aggTrades"
    try:
        response = await http_request(
            "GET",
            BINANCE_FUTURES_AGG_TRADES,
            scope=scope,
            trace=trace,
            client=client,
            params=params,
            symbol=symbol,
            window=(start_ms, end_ms),
            details="limit=1000",
            max_retries=0,
            retry_statuses=TRANSIENT_STATUSES,
            rate_limit_statuses=RATE_LIMIT_STATUSES,
        )
    except httpx.RequestError:  # pragma: no cover - network failure
        return None, None

    if response.status_code == 200:
        payload = response.json()
        if not isinstance(payload, list):
            raise OrderflowError("Invalid trade payload structure")
        return payload, 200

    if response.status_code in RATE_LIMIT_STATUSES or response.status_code >= 500:
        return None, response.status_code

    response.raise_for_status()
    return None, response.status_code


def _build_minute_rows(trades: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    minute_buckets: MutableMapping[int, Dict[str, Any]] = {}
    for trade in trades:
        try:
            trade_time = int(trade["T"])
            price = float(trade["p"])
            quantity = float(trade["q"])
            buyer_is_maker = bool(trade["m"])
        except (KeyError, TypeError, ValueError):
            LOGGER.debug("Skipping malformed trade: %s", trade)
            continue
        if quantity <= 0:
            continue
        bucket = (trade_time // 60_000) * 60_000
        bucket_row = minute_buckets.get(bucket)
        if bucket_row is None:
            bucket_row = {
                "ts": bucket,
                "t": _isoformat(bucket),
                "price": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "volume": 0.0,
                "delta": 0.0,
                "imbalance": 0.0,
                "absorption": False,
                "absorption_high": False,
                "absorption_low": False,
                "imbalance_buy": False,
                "imbalance_sell": False,
                "large_trades_count": 0,
            }
            minute_buckets[bucket] = bucket_row
        prev_volume = bucket_row["volume"]
        bucket_row["volume"] += quantity
        if buyer_is_maker:
            bucket_row["bid"] += quantity
        else:
            bucket_row["ask"] += quantity
        if bucket_row["volume"] > 0:
            bucket_row["price"] = _round_price(
                (
                    bucket_row.get("price", 0.0) * prev_volume + price * quantity
                )
                / bucket_row["volume"]
            )

    minute_rows: List[Dict[str, Any]] = []
    for bucket, entry in sorted(minute_buckets.items(), key=lambda item: item[0]):
        bid = float(entry.get("bid", 0.0))
        ask = float(entry.get("ask", 0.0))
        delta = ask - bid
        entry["delta"] = delta
        entry["imbalance"] = ask / bid if bid > 0 else (ask if ask > 0 else 0.0)
        absorption = abs(delta) >= 1_000
        entry["absorption"] = absorption
        entry["absorption_high"] = absorption and delta < 0
        entry["absorption_low"] = absorption and delta > 0
        entry["imbalance_buy"] = ask > bid and bid > 0
        entry["imbalance_sell"] = bid > ask and ask > 0
        minute_rows.append(entry)

    return minute_rows


async def fetch_footprint(
    symbol: str, window_hours: int, *, trace: TraceContext | None = None
) -> Dict[str, Any]:
    """Build a footprint snapshot using Binance aggregated trades."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    lookback_hours = min(window_hours, _MAX_WINDOW_HOURS)
    start_time = end_time - timedelta(hours=lookback_hours)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    trace_ctx = trace.child(stage="footprint") if trace is not None else None

    batches = 0
    fallback_status: int | None = None
    last_window_end: int | None = None
    fallback_triggered = False
    trades: List[Mapping[str, Any]] = []

    store_rows = await _load_trades_from_store(symbol_clean, start_ms, end_ms)
    if store_rows:
        trades = store_rows
        if trace_ctx is not None:
            trace_ctx.info(
                "orderflow.data_source",
                scope="orderflow.aggTrades",
                symbol=symbol_clean,
                source="vision_store",
                rows=len(trades),
            )
    else:
        async with _FOOTPRINT_LOCK:
            cursor = start_ms
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                while cursor < end_ms and batches < _MAX_BATCHES:
                    page_end = min(end_ms, cursor + _PAGE_WINDOW_MS)
                    rows, status = await _fetch_trades(
                        client, symbol_clean, cursor, page_end, trace=trace_ctx
                    )
                    batches += 1
                    last_window_end = page_end
                    if status != 200:
                        if status in RATE_LIMIT_STATUSES or (
                            isinstance(status, int) and status >= 500
                        ) or status is None:
                            fallback_status = status or 0
                            fallback_triggered = True
                            break
                        raise OrderflowError(f"Unexpected aggTrades status: {status}")

                    if not rows:
                        cursor = page_end + 1
                        continue
                    trades.extend(rows)
                    last_trade_time = max(int(row.get("T", cursor)) for row in rows)
                    cursor = max(last_trade_time + 1, page_end + 1)

                if cursor < end_ms and not fallback_triggered:
                    fallback_triggered = True
                    if fallback_status is None:
                        fallback_status = 0

        if fallback_status is not None and trace_ctx is not None:
            trace_ctx.warn(
                "fallback.engaged",
                scope="orderflow.aggTrades",
                symbol=symbol_clean,
                window={"from": cursor, "to": last_window_end or cursor},
                details=f"status={fallback_status}",
            )

    if fallback_status is not None and trades and trace_ctx is not None:
        trace_ctx.warn(
            "fallback.engaged",
            scope="orderflow",
            symbol=symbol_clean,
            details=f"partial_status={fallback_status}",
            preserved=len(trades),
        )

    if not trades:
        if trace_ctx is not None:
            trace_ctx.warn(
                "fallback.engaged",
                scope="orderflow",
                symbol=symbol_clean,
                details="no_trades",
            )
        snapshot = _empty_snapshot().as_dict()
        if fallback_triggered:
            snapshot["meta"] = {
                "partial": True,
                "fallback": {
                    "status": fallback_status,
                    "preserved_trades": 0,
                    "batches": batches,
                },
            }
        return snapshot

    minute_rows = _build_minute_rows(trades)
    if not minute_rows:
        if trace_ctx is not None:
            trace_ctx.warn(
                "fallback.engaged",
                scope="orderflow",
                symbol=symbol_clean,
                details="minute_rows_empty",
            )
        return _empty_snapshot().as_dict()

    latest_ts = minute_rows[-1]["ts"]
    cutoff_ms = latest_ts - (_PER_BAR_MINUTES - 1) * 60_000
    per_bar_source = [row for row in minute_rows if row["ts"] >= cutoff_ms]
    running_cvd = 0.0
    per_bar: List[Dict[str, Any]] = []
    for row in per_bar_source:
        delta_value = float(row.get("delta", 0.0))
        running_cvd += delta_value
        entry = dict(row)
        entry["cvd"] = running_cvd
        per_bar.append(entry)

    aggregates = _compute_aggregates(minute_rows)

    if trace_ctx is not None:
        trace_ctx.info(
            "orderflow.per_bar_compact",
            scope="orderflow",
            symbol=symbol_clean,
            rows=len(per_bar),
            minutes=_PER_BAR_MINUTES,
        )
        for tf, series in aggregates.items():
            trace_ctx.info(
                "orderflow.delta_cvd_compact",
                scope=f"orderflow.{tf}",
                symbol=symbol_clean,
                rows=len(series),
                tf=tf,
            )

    snapshot = OrderflowSnapshot(per_bar=per_bar, aggregates=aggregates).as_dict()
    if fallback_triggered:
        snapshot["meta"] = {
            "partial": True,
            "fallback": {
                "status": fallback_status,
                "preserved_trades": len(trades),
                "batches": batches,
            },
        }
    return snapshot


def compute_orderflow_aggregates(
    per_bar_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Public helper to aggregate per-bar orderflow rows."""

    return _compute_aggregates(per_bar_rows)


async def calculate_cvd(
    symbol: str,
    window_hours: int,
    *,
    footprint_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    trace: TraceContext | None = None,
) -> Dict[str, Any]:
    """Calculate cumulative volume delta series from footprint data."""

    trace_ctx = trace.child(stage="cvd") if trace is not None else None

    snapshot: Mapping[str, Any]
    if footprint_rows is None:
        snapshot = await fetch_footprint(symbol, window_hours, trace=trace_ctx)
    elif isinstance(footprint_rows, Mapping) and "per_bar" in footprint_rows:
        snapshot = footprint_rows
    else:
        rows_list = list(footprint_rows or [])
        aggregates = _compute_aggregates(rows_list)
        snapshot = {"per_bar": rows_list, "aggregates": aggregates}

    per_bar_rows = [
        row
        for row in snapshot.get("per_bar", [])
        if isinstance(row, Mapping) and "t" in row
    ]

    cumulative_buy = 0.0
    cumulative_sell = 0.0
    series: List[Dict[str, Any]] = []
    for row in sorted(per_bar_rows, key=lambda item: int(item.get("ts", 0))):
        ask_volume = float(row.get("ask", row.get("ask_vol", 0.0)))
        bid_volume = float(row.get("bid", row.get("bid_vol", 0.0)))
        cumulative_buy += ask_volume
        cumulative_sell += bid_volume
        series.append(
            {
                "t": row.get("t"),
                "ts": row.get("ts"),
                "cvd_buy": cumulative_buy,
                "cvd_sell": cumulative_sell,
                "cvd_net": cumulative_buy - cumulative_sell,
                "delta": float(row.get("delta", 0.0)),
            }
        )

    aggregates = snapshot.get("aggregates")
    if not isinstance(aggregates, Mapping):
        aggregates = _compute_aggregates(per_bar_rows)

    if trace_ctx is not None:
        trace_ctx.info(
            "orderflow.delta_cvd_compact",
            scope="orderflow.cvd",
            symbol=symbol,
            rows=len(series),
            minutes=_PER_BAR_MINUTES,
        )

    return {"per_bar": series, "aggregates": dict(aggregates)}
