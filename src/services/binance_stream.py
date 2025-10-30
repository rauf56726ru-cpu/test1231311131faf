"""Minimal Binance WebSocket streaming pipeline storing updates locally."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Mapping, Sequence, Tuple

import aiohttp

from .vision_store import get_store, VisionStore

LOGGER = logging.getLogger(__name__)

STREAM_URL = "wss://fstream.binance.com/stream"

AGG_STREAM_TEMPLATE = "{symbol}@aggTrade"
KLINE_STREAM_TEMPLATE = "{symbol}@kline_1m"


_STREAM_CACHE_LIMITS = {"aggTrades": 5000, "klines": 1000}
_STREAM_CACHE: Dict[str, Dict[str, Deque[Dict[str, object]]]] = {
    "aggTrades": defaultdict(lambda: deque(maxlen=_STREAM_CACHE_LIMITS["aggTrades"])),
    "klines": defaultdict(lambda: deque(maxlen=_STREAM_CACHE_LIMITS["klines"])),
}
_STREAM_CACHE_LOCK = asyncio.Lock()


async def _append_stream_cache(dataset: str, symbol: str, rows: Sequence[Mapping[str, object]]) -> None:
    if dataset not in _STREAM_CACHE or not rows:
        return
    async with _STREAM_CACHE_LOCK:
        cache = _STREAM_CACHE[dataset][symbol.upper()]
        for row in rows:
            cache.append(dict(row))


async def get_recent_stream_rows(
    dataset: str,
    symbol: str,
    since_ms: int | None = None,
) -> List[Dict[str, object]]:
    bucket = _STREAM_CACHE.get(dataset)
    if bucket is None:
        return []
    async with _STREAM_CACHE_LOCK:
        entries = list(bucket.get(symbol.upper(), ()))
    if since_ms is not None:
        entries = [row for row in entries if int(row.get("ts", 0)) >= since_ms]
    return [dict(row) for row in entries]


def _normalise_symbol(symbol: str) -> str:
    return symbol.strip().lower()


def _day_from_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


class BinanceStreamManager:
    """Maintain a shared websocket stream for aggTrades and 1m klines."""

    def __init__(self, symbols: Iterable[str], store: VisionStore | None = None) -> None:
        self._symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol})
        self._store = store or get_store()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._reconnect_backoff = 1.0
        self._buffers: Dict[str, Dict[Tuple[str, ...], List[Dict[str, object]]]] = {
            "aggTrades": {},
            "klines": {},
        }
        self._buffer_lock = asyncio.Lock()
        self._buffer_batch_limit = 200
        self._buffer_flush_interval = 2.0
        self._buffer_flush_task: asyncio.Task | None = None
        self._ingest_stats: dict[Tuple[str, str, str | None], dict[str, int]] = {}
        self._ingest_lock = asyncio.Lock()
        self._ingest_flush_task: asyncio.Task | None = None
        self._ingest_flush_interval = 5.0

    async def start(self) -> None:
        if not self._symbols:
            LOGGER.info("binance.stream: no symbols configured, skipping websocket start")
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="binance-stream")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._buffer_flush_task is not None:
            self._buffer_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._buffer_flush_task
            self._buffer_flush_task = None
        await self._flush_buffers()
        if self._ingest_flush_task is not None:
            self._ingest_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ingest_flush_task
            self._ingest_flush_task = None
        await self._flush_ingest_stats()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _run(self) -> None:
        streams = self._build_streams()
        if not streams:
            LOGGER.info("binance.stream: nothing to subscribe")
            return
        url = f"{STREAM_URL}?streams={'/'.join(streams)}"
        while not self._stop_event.is_set():
            try:
                if self._session is None:
                    self._session = aiohttp.ClientSession()
                LOGGER.info("binance.stream: connecting to %s", url)
                async with self._session.ws_connect(url, heartbeat=30, compress=0) as ws:
                    self._reconnect_backoff = 1.0
                    LOGGER.info("binance.stream: connected", extra={"streams": streams})
                    await self._listen(ws)
                await self._flush_buffers()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                LOGGER.warning(
                    "binance.stream: connection error, reconnecting",
                    exc_info=exc,
                )
                await self._flush_buffers()
                await asyncio.sleep(self._reconnect_backoff)
                self._reconnect_backoff = min(self._reconnect_backoff * 2, 60.0)
        await self._flush_buffers()
        await self._flush_ingest_stats()
        LOGGER.info("binance.stream: stopped")

    def _build_streams(self) -> List[str]:
        streams: List[str] = []
        for symbol in self._symbols:
            symbol_lc = _normalise_symbol(symbol)
            streams.append(AGG_STREAM_TEMPLATE.format(symbol=symbol_lc))
            streams.append(KLINE_STREAM_TEMPLATE.format(symbol=symbol_lc))
        return streams

    async def _listen(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    LOGGER.debug("binance.stream: failed to decode %s", msg.data[:100])
                    continue
                await self._handle_message(payload)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                LOGGER.warning("binance.stream: websocket error %s", ws.exception())
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break

    async def _handle_message(self, payload: Mapping[str, object]) -> None:
        stream = payload.get("stream")
        data = payload.get("data")
        if not isinstance(stream, str) or not isinstance(data, Mapping):
            return
        stream_lower = stream.lower()
        try:
            if stream_lower.endswith("@aggtrade"):
                await self._handle_agg_trade(stream_lower.split("@")[0], data)
            elif "@kline_" in stream_lower:
                await self._handle_kline(stream_lower.split("@")[0], stream_lower.split("@")[1], data)
        except Exception as exc:
            LOGGER.debug("binance.stream: handler error %s", exc, exc_info=exc)
            LOGGER.warning(
                "binance.stream: failed to process message",
                extra={"stream": stream, "error": str(exc)},
            )

    async def _handle_agg_trade(self, symbol_lc: str, data: Mapping[str, object]) -> None:
        try:
            trade_id = int(data["a"])  # aggregate trade ID
            price = float(data["p"])
            qty = float(data["q"])
            ts = int(data["T"])
            buyer_maker = bool(data["m"])
        except (KeyError, TypeError, ValueError):
            return
        side = "sell" if buyer_maker else "buy"
        day = _day_from_ts(ts)
        row = {
            "agg_id": trade_id,
            "ts": ts,
            "price": price,
            "qty": qty,
            "side": side,
            "buyer_maker": buyer_maker,
        }
        symbol = symbol_lc.upper()
        LOGGER.debug(
            "binance.stream: agg_trade",
            extra={"symbol": symbol, "ts": ts, "price": price, "qty": qty, "side": side},
        )
        await self._enqueue_buffer("aggTrades", (symbol, day), row)

    async def _handle_kline(self, symbol_lc: str, stream_suffix: str, data: Mapping[str, object]) -> None:
        kline = data.get("k")
        if not isinstance(kline, Mapping):
            return
        try:
            open_time = int(kline["t"])
            close_time = int(kline["T"])
            open_price = float(kline["o"])
            high_price = float(kline["h"])
            low_price = float(kline["l"])
            close_price = float(kline["c"])
            volume = float(kline["v"])
            quote_volume = float(kline.get("q", 0.0))
            trades = int(kline.get("n", 0))
            interval = kline.get("i", "1m")
        except (KeyError, TypeError, ValueError):
            return
        row = {
            "ts": close_time,
            "open_time": open_time,
            "close_time": close_time,
            "o": open_price,
            "h": high_price,
            "l": low_price,
            "c": close_price,
            "v": volume,
            "quote_volume": quote_volume,
            "trades": trades,
        }
        symbol = symbol_lc.upper()
        day = _day_from_ts(open_time)
        LOGGER.debug(
            "binance.stream: kline",
            extra={
                "symbol": symbol,
                "interval": interval,
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "close": close_price,
                "volume": volume,
            },
        )
        await self._enqueue_buffer("klines", (symbol, interval, day), row)

    async def _record_ingest(
        self,
        dataset: str,
        *,
        symbol: str,
        interval: str | None,
        count: int,
        latest_ts: int,
    ) -> None:
        if count <= 0:
            return
        key: Tuple[str, str, str | None] = (dataset, symbol, interval)
        async with self._ingest_lock:
            stats = self._ingest_stats.get(key)
            if stats is None:
                stats = {"count": 0, "latest_ts": 0}
                self._ingest_stats[key] = stats
            stats["count"] += count
            if latest_ts > stats["latest_ts"]:
                stats["latest_ts"] = latest_ts
            if self._ingest_flush_task is None or self._ingest_flush_task.done():
                self._ingest_flush_task = asyncio.create_task(self._flush_ingest_stats_delayed())

    async def _flush_ingest_stats_delayed(self) -> None:
        try:
            await asyncio.sleep(self._ingest_flush_interval)
            await self._flush_ingest_stats()
        finally:
            self._ingest_flush_task = None

    async def _flush_ingest_stats(self) -> None:
        async with self._ingest_lock:
            if not self._ingest_stats:
                return
            snapshot = self._ingest_stats
            self._ingest_stats = {}
        for (dataset, symbol, interval), stats in snapshot.items():
            extra = {
                "dataset": dataset,
                "symbol": symbol,
                "count": stats.get("count", 0),
                "latest_ts": stats.get("latest_ts"),
            }
            if interval:
                extra["interval"] = interval
            LOGGER.info("binance.stream: stored batch", extra=extra)

    async def _enqueue_buffer(
        self,
        dataset: str,
        key: Tuple[str, ...],
        row: Dict[str, object],
    ) -> None:
        flush_payload: Tuple[str, Tuple[str, ...], List[Dict[str, object]]] | None = None
        async with self._buffer_lock:
            bucket_map = self._buffers.setdefault(dataset, {})
            bucket = bucket_map.setdefault(key, [])
            bucket.append(row)
            if len(bucket) >= self._buffer_batch_limit:
                flush_payload = (dataset, key, bucket_map.pop(key))
            if (self._buffer_flush_task is None or self._buffer_flush_task.done()) and any(
                buckets for buckets in self._buffers.values()
            ):
                self._buffer_flush_task = asyncio.create_task(self._flush_buffers_delayed())
        if flush_payload is not None:
            await self._flush_buffer(*flush_payload)

    async def _flush_buffers_delayed(self) -> None:
        try:
            await asyncio.sleep(self._buffer_flush_interval)
            await self._flush_buffers()
        finally:
            self._buffer_flush_task = None

    async def _flush_buffers(self) -> None:
        flush_jobs: List[Tuple[str, Tuple[str, ...], List[Dict[str, object]]]] = []
        async with self._buffer_lock:
            for dataset, buckets in self._buffers.items():
                keys = list(buckets.keys())
                for key in keys:
                    rows = buckets.pop(key, [])
                    if rows:
                        flush_jobs.append((dataset, key, rows))
        for dataset, key, rows in flush_jobs:
            try:
                await self._flush_buffer(dataset, key, rows)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "binance.stream: failed to flush buffer",
                    extra={"dataset": dataset, "key": key},
                    exc_info=exc,
                )

    async def _flush_buffer(
        self,
        dataset: str,
        key: Tuple[str, ...],
        rows: List[Dict[str, object]],
    ) -> None:
        if not rows:
            return
        if dataset == "aggTrades":
            symbol, day = key
            stats = await asyncio.to_thread(self._store.insert_agg_trades, symbol, day, rows)
            if getattr(stats, "inserted", 0):
                latest_ts = max(int(row.get("ts", 0)) for row in rows)
                await _append_stream_cache("aggTrades", symbol, rows)
                await self._record_ingest(
                    "aggTrades",
                    symbol=symbol,
                    interval=None,
                    count=stats.inserted,
                    latest_ts=latest_ts,
                )
        elif dataset == "klines":
            symbol, interval, day = key
            stats = await asyncio.to_thread(self._store.upsert_klines, symbol, interval, day, rows)
            if getattr(stats, "inserted", 0):
                latest_ts = max(
                    int(row.get("ts") or row.get("close_time") or 0)
                    for row in rows
                )
                await _append_stream_cache("klines", symbol, rows)
                await self._record_ingest(
                    "klines",
                    symbol=symbol,
                    interval=interval,
                    count=stats.inserted,
                    latest_ts=latest_ts,
                )
