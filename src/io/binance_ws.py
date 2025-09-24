"""Lightweight Binance websocket helpers.

The implementation is intentionally simplified so that it can operate in
paper/sandbox environments and be unit-tested without a real network
connection. Real deployments should extend this module with proper error
handling and authentication.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

import httpx
import websockets

import pandas as pd

from ..utils.logging import get_logger
from .binance_offline import OFFLINE_DATA, OfflineDataUnavailable

LOGGER = get_logger(__name__)

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"
BINANCE_REST_BASE = "https://fapi.binance.com"  # USDT-M futures REST endpoint
BINANCE_SOURCE = "binance_futures_usdtm"


@dataclass
class StreamMessage:
    stream: str
    payload: Dict[str, Any]


class BinanceWebSocketClient:
    """Minimal websocket client with subscribe helpers."""

    def __init__(self, stream_url: str = BINANCE_WS_BASE):
        self.stream_url = stream_url
        self._socket: Optional[websockets.WebSocketClientProtocol] = None

    async def __aenter__(self) -> "BinanceWebSocketClient":
        LOGGER.info("Connecting to Binance WS: %s", self.stream_url)
        self._socket = await websockets.connect(self.stream_url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._socket:
            await self._socket.close()
            LOGGER.info("WS connection closed")

    async def send(self, payload: Dict[str, Any]) -> None:
        if not self._socket:
            raise RuntimeError("Websocket not connected")
        await self._socket.send(json.dumps(payload))

    async def recv(self) -> StreamMessage:
        if not self._socket:
            raise RuntimeError("Websocket not connected")
        data = json.loads(await self._socket.recv())
        stream = data.get("stream", "")
        payload = data.get("data", data)
        return StreamMessage(stream=stream, payload=payload)

    async def listen(self) -> AsyncIterator[StreamMessage]:
        while True:
            msg = await self.recv()
            yield msg

    async def subscribe(self, streams: list[str]) -> None:
        payload = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        await self.send(payload)

    async def subscribe_kline(self, symbol: str, interval: str) -> None:
        await self.subscribe([f"{symbol.lower()}@kline_{interval}"])

    async def subscribe_trades(self, symbol: str) -> None:
        await self.subscribe([f"{symbol.lower()}@trade"])

    async def subscribe_depth(self, symbol: str) -> None:
        await self.subscribe([f"{symbol.lower()}@depth"])


def _to_milliseconds(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    try:
        return int(pd.Timestamp(value).timestamp() * 1000)
    except Exception:  # pragma: no cover - defensive conversion
        return None


async def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 500,
    start_time: Any | None = None,
    end_time: Any | None = None,
) -> list[list[Any]]:
    """Fetch historical klines via REST API with backoff safeguards."""

    url = f"{BINANCE_REST_BASE}/fapi/v1/klines"
    safe_limit = max(1, min(int(limit or 500), 1000))
    params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": safe_limit}
    start_ms = _to_milliseconds(start_time)
    end_ms = _to_milliseconds(end_time)
    if start_ms is not None:
        params["startTime"] = start_ms
    clamped_end: int | None = None
    if end_ms is not None:
        now_utc = datetime.now(timezone.utc)
        now_ms = int(now_utc.timestamp() * 1000)
        day_end = now_utc.replace(hour=23, minute=59, second=59, microsecond=999000)
        day_end_ms = int(day_end.timestamp() * 1000)
        clamped_end = min(int(end_ms), now_ms, day_end_ms)
        params["endTime"] = clamped_end

        params["endTime"] = min(int(end_ms), now_ms, day_end_ms)

    max_attempts = 3
    attempt = 0
    delay = 1.0
    last_error: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            async with httpx.AsyncClient(
                timeout=15.0, verify=False, headers={"User-Agent": "autotrading-bot"}
            ) as client:
                resp = await client.get(url, params=params)
            if resp.status_code in {418, 429}:
                last_error = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
                LOGGER.warning(
                    "Binance rate limit for %s %s (attempt %s/%s)",
                    symbol,
                    interval,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status in {418, 429} and attempt < max_attempts:
                LOGGER.warning(
                    "HTTP %s from Binance, backing off (attempt %s/%s)",
                    status,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if status == 451:
                LOGGER.warning(
                    "HTTP %s from Binance for %s %s; using offline fixture",
                    status,
                    symbol,
                    interval,
                )
                break
            if attempt >= max_attempts:
                LOGGER.warning(
                    "HTTP %s from Binance for %s %s after retries; falling back offline",
                    status,
                    symbol,
                    interval,
                )
                break
            LOGGER.warning(
                "HTTP %s from Binance for %s %s; falling back offline",
                status,
                symbol,
                interval,
            )
            break
        except (httpx.RequestError, httpx.HTTPError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                LOGGER.warning("Binance REST error after retries: %s", exc)
                break
            LOGGER.warning("Binance REST request failed (%s), retrying", exc)
            await asyncio.sleep(min(delay, 0.5))
            delay = min(delay * 2, 1.0)
            continue

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected Binance response format")
    return data


async def fetch_agg_trades(
    symbol: str,
    start_time: Any | None = None,
    end_time: Any | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Fetch aggregated trades for backfilling websocket gaps."""

    url = f"{BINANCE_REST_BASE}/fapi/v1/aggTrades"
    safe_limit = max(1, min(int(limit or 1000), 1000))
    params: Dict[str, Any] = {"symbol": symbol.upper(), "limit": safe_limit}
    start_ms = _to_milliseconds(start_time)
    end_ms = _to_milliseconds(end_time)
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    max_attempts = 3
    attempt = 0
    delay = 1.0
    last_error: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            async with httpx.AsyncClient(
                timeout=15.0, verify=False, headers={"User-Agent": "autotrading-bot"}
            ) as client:
                resp = await client.get(url, params=params)
            if resp.status_code in {418, 429}:
                last_error = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
                LOGGER.warning(
                    "Binance aggTrades rate limit for %s (attempt %s/%s)",
                    symbol,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status in {418, 429} and attempt < max_attempts:
                LOGGER.warning(
                    "HTTP %s from aggTrades, backing off (attempt %s/%s)",
                    status,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            LOGGER.warning(
                "HTTP %s from Binance aggTrades for %s; returning cached/empty", status, symbol
            )
            break
        except (httpx.RequestError, httpx.HTTPError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                LOGGER.warning("Binance aggTrades REST error after retries: %s", exc)
                break
            LOGGER.warning("Binance aggTrades request failed (%s), retrying", exc)
            await asyncio.sleep(min(delay, 0.5))
            delay = min(delay * 2, 1.0)
            continue

        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected aggTrades response format")
        normalized: list[dict[str, Any]] = []
        for entry in data:
            if isinstance(entry, dict):
                normalized.append(entry)
        return normalized

    if last_error is not None:
        LOGGER.debug("aggTrades fetch failed: %s", last_error)
    return []

    try:
        fallback = OFFLINE_DATA.klines(
            symbol,
            interval,
            limit=safe_limit,
            start_time=start_ms,
            end_time=clamped_end,
        )
    except OfflineDataUnavailable:
        fallback = []
    if fallback:
        return fallback
    if last_error is not None:
        raise RuntimeError(f"Binance REST error: {last_error}") from last_error
    LOGGER.error("Exceeded retry budget fetching klines for %s %s", symbol, interval)
    return []


async def fetch_klines_paginated(
    symbol: str,
    interval: str,
    *,
    end_time: Any | None = None,
    minimum: int = 0,
    date_from_ms: int | None = None,
    fetcher: Callable[..., Awaitable[list[list[Any]]]] | None = None,
) -> list[list[Any]]:
    """Fetch paginated klines using 1000-bar pages until coverage goals are met."""

    fetch = fetcher or fetch_klines
    now_utc = datetime.now(timezone.utc)
    now_ms = int(now_utc.timestamp() * 1000)
    day_end = now_utc.replace(hour=23, minute=59, second=59, microsecond=999000)
    day_end_ms = int(day_end.timestamp() * 1000)
    clamp_end = min(_to_milliseconds(end_time) or now_ms, now_ms, day_end_ms)

    pages: list[list[Any]] = []
    collected = 0
    safety = 0
    current_end = clamp_end

    while safety < 100 and current_end is not None:
        safety += 1
        batch = await fetch(
            symbol,
            interval,
            limit=1000,
            end_time=current_end,
        )
        if not batch:
            break
        pages.extend(batch)
        collected += len(batch)
        first_open = batch[0][0] if batch and isinstance(batch[0], (list, tuple)) else None
        if first_open is None:
            break
        reached_start = date_from_ms is not None and int(first_open) <= int(date_from_ms)
        reached_minimum = date_from_ms is None and minimum > 0 and collected >= minimum
        if reached_start or reached_minimum:
            break
        next_end = int(first_open) - 1
        if next_end <= 0:
            break
        current_end = next_end

    return pages


async def collect_stream(
    symbol: str,
    interval: str,
    handler: Callable[[StreamMessage], Awaitable[None]],
    streams: Optional[list[str]] = None,
) -> None:
    """Generic helper for consuming websocket streams."""
    if streams is None:
        streams = [f"{symbol.lower()}@kline_{interval}"]
    async with BinanceWebSocketClient() as client:
        await client.subscribe(streams)
        async for msg in client.listen():
            await handler(msg)
