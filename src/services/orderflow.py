"""Orderflow helpers for footprint and cumulative volume delta calculations."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import httpx

BINANCE_FUTURES_AGG_TRADES = "https://fapi.binance.com/fapi/v1/aggTrades"
LOGGER = logging.getLogger(__name__)
_FOOTPRINT_LOCK = asyncio.Lock()


class OrderflowError(RuntimeError):
    """Raised when upstream orderflow data are invalid."""


async def _fetch_trades(symbol: str, start_ms: int, end_ms: int) -> List[Mapping[str, object]]:
    params = {
        "symbol": symbol.upper(),
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": "1000",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        response = await client.get(BINANCE_FUTURES_AGG_TRADES, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise OrderflowError("Invalid trade payload structure")
        return data  # type: ignore[return-value]


def _round_price(price: float, *, precision: int = 2) -> float:
    return round(price, precision)


async def fetch_footprint(symbol: str, window_hours: int) -> List[Dict[str, object]]:
    """Build a simple footprint profile from Binance aggregated trades."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_time = end_time - timedelta(hours=window_hours)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    async with _FOOTPRINT_LOCK:
        footprint: MutableMapping[Tuple[int, float], Dict[str, float | int | str | bool]] = {}
        cursor = start_ms
        last_trade = start_ms
        while cursor < end_ms:
            rows = await _fetch_trades(symbol_clean, cursor, end_ms)
            if not rows:
                break
            for row in rows:
                try:
                    trade_time = int(row["T"])  # trade time in ms
                    price = float(row["p"])
                    quantity = float(row["q"])
                    buyer_is_maker = bool(row["m"])
                except (KeyError, TypeError, ValueError) as exc:
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

    return footprint_rows


async def calculate_cvd(symbol: str, window_hours: int) -> List[Dict[str, object]]:
    """Calculate cumulative volume delta series from footprint data."""

    footprint_rows = await fetch_footprint(symbol, window_hours)

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

    return cumulative_rows
