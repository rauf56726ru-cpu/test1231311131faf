"""REST helpers for topping off recent Binance data."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List

import httpx

from src.common.config import AppConfig
from src.common.ts import ensure_epoch_ms
from src.storage.parquet import ParquetStorage

LOGGER = logging.getLogger(__name__)

REST_ENDPOINTS = {
    "um": "https://fapi.binance.com/fapi/v1/klines",
}

INTERVAL_MS = {
    "1m": 60_000,
}


@dataclass(slots=True)
class RestTopoffStats:
    """Summary of a REST top-off attempt."""

    added_rows: int
    requests: int
    steps: List[int] = field(default_factory=list)
    last_cursor: int = 0
    shortfall: bool = False


def normalise_klines_payload(payload: List[List]) -> List[Dict[str, object]]:
    """Convert raw Binance REST payload rows into storage-friendly records."""

    rows: List[Dict[str, object]] = []
    for entry in payload:
        try:
            open_time_raw = entry[0]
            open_price = float(entry[1])
            high_price = float(entry[2])
            low_price = float(entry[3])
            close_price = float(entry[4])
            volume = float(entry[5])
            trades = int(entry[8])
            taker_buy_base = float(entry[9])
            taker_buy_quote = float(entry[10])
        except (TypeError, ValueError, IndexError):
            LOGGER.warning("rest.topoff.invalid_row", extra={"raw": entry})
            continue
        try:
            ts_open = ensure_epoch_ms(open_time_raw)
        except ValueError:
            LOGGER.warning("rest.topoff.invalid_ts", extra={"value": open_time_raw})
            continue
        if ts_open == 0:
            LOGGER.warning("rest.topoff.zero_ts", extra={"value": open_time_raw})
            continue
        numeric_values = (
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            taker_buy_base,
            taker_buy_quote,
        )
        if any(math.isnan(value) or math.isinf(value) for value in numeric_values):
            LOGGER.warning("rest.topoff.nan_row", extra={"ts_open": ts_open})
            continue
        rows.append(
            {
                "ts_open": ts_open,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "taker_buy_vol": taker_buy_base,
                "taker_buy_quote": taker_buy_quote,
                "trades": trades,
            }
        )
    return rows


async def topoff_current_day(
    symbol: str,
    interval: str,
    market: str,
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    step_minutes: int = 1000,
    max_calls: int = 30,
    timeout_s: int = 10,
) -> RestTopoffStats:
    """Fetch missing candles for the current day from Binance REST API."""

    cfg = AppConfig.load()
    market_key = market.lower()
    base_url = REST_ENDPOINTS.get(market_key)
    if base_url is None:
        raise ValueError("Futures UM only")

    interval_key = interval.lower()
    interval_ms = INTERVAL_MS.get(interval_key)
    if interval_ms is None:
        raise ValueError(f"Unsupported interval '{interval}' for REST top-off")

    step_minutes = max(1, int(step_minutes))
    limit = min(step_minutes, 1000)
    step_ms = step_minutes * interval_ms

    storage = ParquetStorage(
        root=cfg.data_dir,
        market=market,
        index_path=cfg.duckdb_path,
    )

    existing = storage.load_window(symbol, interval, start_ts_ms, end_ts_ms)
    existing_ts = set(existing["ts_open"].tolist()) if not existing.empty else set()

    total_added = 0
    calls = 0
    cursor = start_ts_ms
    steps_used: set[int] = set()
    shortfall = False

    async with httpx.AsyncClient() as client:
        while cursor < end_ts_ms and calls < max_calls:
            window_end = min(end_ts_ms, cursor + step_ms)
            if window_end <= cursor:
                break
            params = {
                "symbol": symbol.upper(),
                "interval": interval_key,
                "limit": str(limit),
                "startTime": str(cursor),
                "endTime": str(window_end - 1),
            }
            try:
                response = await client.get(base_url, params=params, timeout=timeout_s)
            except httpx.RequestError as exc:  # pragma: no cover - network failure
                LOGGER.warning(
                    "rest.topoff.request_error",
                    extra={"symbol": symbol, "error": str(exc)},
                )
                shortfall = True
                break
            calls += 1
            if response.status_code != 200:
                LOGGER.warning(
                    "rest.topoff.bad_status",
                    extra={"symbol": symbol, "status": response.status_code},
                )
                shortfall = True
                break
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                shortfall = True
                cursor = min(cursor + step_ms, end_ts_ms)
                continue

            rows = normalise_klines_payload(payload)
            if not rows:
                shortfall = True
                cursor = min(cursor + step_ms, end_ts_ms)
                continue

            steps_used.add(step_minutes)
            new_rows = [row for row in rows if row["ts_open"] not in existing_ts]
            if new_rows:
                storage.write_rows(symbol, interval, new_rows)
                for row in new_rows:
                    existing_ts.add(row["ts_open"])
                total_added += len(new_rows)

            last_open = max(row["ts_open"] for row in rows)
            cursor_before = cursor
            cursor = min(last_open + interval_ms, end_ts_ms)

            window_span = max(0, window_end - cursor_before)
            expected = min(limit, window_span // interval_ms) if window_span >= interval_ms else len(rows)
            if expected == 0 and rows:
                expected = len(rows)

            if len(rows) < expected or cursor_before == cursor:
                shortfall = True
                break

    LOGGER.info(
        "rest.topoff.summary",
        extra={
            "symbol": symbol.upper(),
            "interval": interval_key,
            "market": market.lower(),
            "requests": calls,
            "added": total_added,
            "start_ts": start_ts_ms,
            "end_ts": end_ts_ms,
            "step_minutes": step_minutes,
            "shortfall": shortfall,
        },
    )
    return RestTopoffStats(
        added_rows=total_added,
        requests=calls,
        steps=sorted(steps_used),
        last_cursor=cursor,
        shortfall=shortfall,
    )


__all__ = ["RestTopoffStats", "normalise_klines_payload", "topoff_current_day"]
