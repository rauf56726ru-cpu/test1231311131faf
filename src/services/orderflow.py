"""Orderflow helpers for footprint and cumulative volume delta calculations."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, MutableMapping, Tuple

import httpx

from .rate_limiter import get_global_rate_limiter
from .tracing import TraceContext

BINANCE_FUTURES_AGG_TRADES = "https://fapi.binance.com/fapi/v1/aggTrades"
LOGGER = logging.getLogger(__name__)
_FOOTPRINT_LOCK = asyncio.Lock()
_RATE_LIMITER = get_global_rate_limiter()
_MAX_WINDOW_HOURS = 4
_PER_BAR_MINUTES = 120
_RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}


class OrderflowError(RuntimeError):
    """Raised when upstream orderflow data are invalid."""


def _retry_after_seconds(response: httpx.Response) -> float | None:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def _used_weight(response: httpx.Response) -> int | None:
    header = response.headers.get("X-MBX-USED-WEIGHT-1m")
    if not header:
        return None
    try:
        return int(header)
    except ValueError:
        return None


async def _fetch_trades(
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    trace: TraceContext | None = None,
) -> List[Mapping[str, object]]:
    params = {
        "symbol": symbol.upper(),
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": "1000",
    }
    scope = "orderflow.aggTrades"
    for attempt in range(1, 3):
        if trace is not None:
            trace.debug(
                "fetch.batch_start",
                scope=scope,
                attempt=attempt,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        try:
            async with _RATE_LIMITER.limit(scope=scope, trace=trace):
                response = await client.get(BINANCE_FUTURES_AGG_TRADES, params=params)
        except httpx.RequestError as exc:  # pragma: no cover - network failure
            if attempt >= 2:
                raise OrderflowError(f"aggTrades request failed: {exc}") from exc
            await _RATE_LIMITER.apply_backoff(
                1.0,
                scope=scope,
                trace=trace,
                reason="request_error",
            )
            continue

        await _RATE_LIMITER.note_used_weight(
            _used_weight(response), scope=scope, trace=trace
        )

        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, list):
                raise OrderflowError("Invalid trade payload structure")
            if trace is not None:
                trace.info(
                    "fetch.batch_done",
                    scope=scope,
                    attempt=attempt,
                    rows=len(data),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            return data  # type: ignore[return-value]

        if response.status_code in _RETRYABLE_STATUS and attempt < 2:
            await _RATE_LIMITER.apply_backoff(
                _retry_after_seconds(response),
                scope=scope,
                trace=trace,
                reason=str(response.status_code),
            )
            if trace is not None:
                trace.warn(
                    "fetch.batch_retry",
                    scope=scope,
                    status=response.status_code,
                    attempt=attempt,
                )
            continue

        response.raise_for_status()

    raise OrderflowError("Unable to fetch aggTrades without hitting rate limits")


def _round_price(price: float, *, precision: int = 2) -> float:
    return round(price, precision)


async def fetch_footprint(
    symbol: str, window_hours: int, *, trace: TraceContext | None = None
) -> List[Dict[str, object]]:
    """Build a simple footprint profile from Binance aggregated trades."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_time = end_time - timedelta(hours=min(window_hours, _MAX_WINDOW_HOURS))
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    async with _FOOTPRINT_LOCK:
        footprint: MutableMapping[Tuple[int, float], Dict[str, float | int | str | bool]] = {}
        cursor = start_ms
        last_trade = start_ms
        trace_ctx = trace.child(stage="footprint") if trace is not None else None
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            while cursor < end_ms:
                rows = await _fetch_trades(
                    client, symbol_clean, cursor, end_ms, trace=trace_ctx
                )
                if not rows:
                    break
                for row in rows:
                    try:
                        trade_time = int(row["T"])  # trade time in ms
                        price = float(row["p"])
                        quantity = float(row["q"])
                        buyer_is_maker = bool(row["m"])
                    except (KeyError, TypeError, ValueError):
                        LOGGER.debug("Skipping malformed trade: %s", row)
                        continue
                    if quantity <= 0:
                        continue
                    bucket = (trade_time // 60_000) * 60_000
                    price_key = _round_price(price)
                    key = (bucket, price_key)
                    record = footprint.get(key)
                    if record is None:
                        record = {
                            "t": datetime.fromtimestamp(bucket / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                            "price": price_key,
                            "bid": 0.0,
                            "ask": 0.0,
                            "ms": bucket,
                        }
                        footprint[key] = record
                    if buyer_is_maker:
                        record["bid"] = float(record.get("bid", 0.0)) + quantity
                    else:
                        record["ask"] = float(record.get("ask", 0.0)) + quantity
                    last_trade = max(last_trade, trade_time)
                if len(rows) < 1000:
                    break
                cursor = last_trade + 1

    footprint_rows: List[Dict[str, object]] = []
    for (_, _), entry in sorted(footprint.items(), key=lambda item: (item[0][0], item[0][1])):
        bid = float(entry.get("bid", 0.0))
        ask = float(entry.get("ask", 0.0))
        delta = ask - bid
        imbalance = 0.0 if bid == 0 else ask / bid
        absorption = abs(delta) >= 1_000
        entry.update({"delta": delta, "imbalance": imbalance, "absorption": absorption})
        footprint_rows.append(entry)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_PER_BAR_MINUTES)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    compact_rows: List[Dict[str, object]] = []
    for row in footprint_rows:
        ts_ms = int(row.pop("ms", 0))
        if ts_ms < cutoff_ms:
            continue
        compact_rows.append(row)

    if trace_ctx is not None:
        trace_ctx.info(
            "orderflow.per_bar_compact",
            rows=len(compact_rows),
            minutes=_PER_BAR_MINUTES,
        )

    return compact_rows


async def calculate_cvd(
    symbol: str, window_hours: int, *, trace: TraceContext | None = None
) -> List[Dict[str, object]]:
    """Calculate cumulative volume delta series from footprint data."""

    trace_ctx = trace.child(stage="cvd") if trace is not None else None
    footprint_rows = await fetch_footprint(symbol, window_hours, trace=trace_ctx)

    per_minute: MutableMapping[str, Dict[str, float]] = defaultdict(lambda: {
        "cvd_buy": 0.0,
        "cvd_sell": 0.0,
        "cvd_net": 0.0,
    })

    for row in footprint_rows:
        ts = row.get("t")
        delta = float(row.get("delta", 0.0))
        bucket = per_minute[str(ts)]
        if delta >= 0:
            bucket["cvd_buy"] += delta
        else:
            bucket["cvd_sell"] += abs(delta)
        bucket["cvd_net"] = bucket["cvd_buy"] - bucket["cvd_sell"]

    cumulative_buy = 0.0
    cumulative_sell = 0.0
    cumulative_rows: List[Dict[str, object]] = []
    for ts, values in sorted(per_minute.items(), key=lambda item: item[0]):
        cumulative_buy += values["cvd_buy"]
        cumulative_sell += values["cvd_sell"]
        cumulative_rows.append(
            {
                "t": ts,
                "cvd_buy": cumulative_buy,
                "cvd_sell": cumulative_sell,
                "cvd_net": cumulative_buy - cumulative_sell,
            }
        )

    if trace_ctx is not None:
        trace_ctx.info(
            "orderflow.delta_cvd_compact",
            rows=len(cumulative_rows),
            minutes=_PER_BAR_MINUTES,
        )

    return cumulative_rows
