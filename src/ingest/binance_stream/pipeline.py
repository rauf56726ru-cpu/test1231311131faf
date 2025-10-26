"""Realtime Binance Futures ingestion with micro-batching and ring buffer retention."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Awaitable, Callable, Deque, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.common.config import AppConfig
from src.services.binance import (
    BinanceAPIException,
    BinanceRequestException,
    fetch_um_klines,
)
from src.storage.parquet import ParquetStorage, StorageWriteStats

LOGGER = logging.getLogger(__name__)

MINUTE_MS = 60_000
WINDOW_HOURS_DEFAULT = 72
WS_DEFAULT_URL = "wss://fstream.binance.com/stream"
RING_LOG_INTERVAL = 300  # seconds


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_symbol(symbol: str) -> str:
    value = (symbol or "").strip().upper()
    if not value:
        raise ValueError("symbol cannot be empty")
    return value


@dataclass(slots=True)
class StreamEvent:
    """Container for raw websocket messages."""

    event_type: str
    symbol: str
    event_time_ms: int
    received_time_ms: int
    payload: Mapping[str, object]
    stream: str | None = None


@dataclass(slots=True)
class BinanceStreamConfig:
    """Runtime configuration for the stream ingestion pipeline."""

    symbols: Sequence[str]
    interval: str = "1m"
    flush_interval_seconds: float = 1.0
    window_hours: int = WINDOW_HOURS_DEFAULT
    websocket_url: str = WS_DEFAULT_URL
    include_agg_trades: bool = False
    queue_size: int = 10_000
    rest_timeout: float = 10.0
    rest_max_retries: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 32.0

    def __post_init__(self) -> None:
        if self.interval != "1m":
            raise ValueError("Only 1m interval is supported for the stream pipeline")
        if not self.symbols:
            raise ValueError("At least one symbol must be configured")
        self.symbols = tuple(_coerce_symbol(symbol) for symbol in self.symbols)
        if self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if self.window_hours <= 0:
            raise ValueError("window_hours must be positive")
        if self.queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if self.reconnect_base_delay <= 0:
            raise ValueError("reconnect_base_delay must be positive")
        if self.reconnect_max_delay < self.reconnect_base_delay:
            raise ValueError("reconnect_max_delay must be >= reconnect_base_delay")


class SymbolTimeframeRingBuffer:
    """Thread-safe sliding window storage for recent candles."""

    __slots__ = ("_window_ms", "_entries", "_lock", "_last_log_ms")

    def __init__(self, window_hours: int) -> None:
        self._window_ms = int(window_hours * 60 * MINUTE_MS)
        self._entries: Dict[Tuple[str, str], Deque[Mapping[str, object]]] = {}
        self._lock = RLock()
        self._last_log_ms: Dict[Tuple[str, str], int] = {}

    def append(self, symbol: str, interval: str, candle: Mapping[str, object]) -> int:
        key = (symbol, interval)
        ts_open = int(candle["ts_open"])
        cutoff = ts_open - self._window_ms
        with self._lock:
            bucket = self._entries.get(key)
            if bucket is None:
                bucket = deque()
                self._entries[key] = bucket
            bucket.append(candle)
            while bucket and int(bucket[0]["ts_open"]) < cutoff:
                bucket.popleft()
            size = len(bucket)

            now_ms = _now_ms()
            last_logged = self._last_log_ms.get(key, 0)
            if now_ms - last_logged >= RING_LOG_INTERVAL * 1000:
                self._last_log_ms[key] = now_ms
                LOGGER.info(
                    "stream.ring.size",
                    extra={"symbol": symbol, "interval": interval, "entries": size},
                )
            return size

    def latest_ts(self, symbol: str, interval: str) -> int | None:
        key = (symbol, interval)
        with self._lock:
            bucket = self._entries.get(key)
            if not bucket:
                return None
            return int(bucket[-1]["ts_open"])

    def snapshot(self, symbol: str, interval: str) -> List[Mapping[str, object]]:
        key = (symbol, interval)
        with self._lock:
            bucket = self._entries.get(key)
            if not bucket:
                return []
            return list(bucket)


class MicroBatcher:
    """Aggregate incoming events into short micro-batches."""

    __slots__ = ("_queue", "_flush_interval", "_handler", "_pending", "_stop", "_task")

    def __init__(
        self,
        queue: asyncio.Queue[StreamEvent],
        *,
        flush_interval: float,
        handler: Callable[[Sequence[StreamEvent]], Awaitable[None]],
    ) -> None:
        self._queue = queue
        self._flush_interval = flush_interval
        self._handler = handler
        self._pending: List[StreamEvent] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="binance-micro-batcher")
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_flush = loop.time() + self._flush_interval
        while not self._stop.is_set():
            timeout = max(0.0, next_flush - loop.time())
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._flush()
                next_flush = loop.time() + self._flush_interval
                continue
            if self._stop.is_set():
                break
            self._pending.append(event)
            self._queue.task_done()
            now = loop.time()
            if now >= next_flush:
                await self._flush()
                next_flush = now + self._flush_interval

        if self._pending:
            await self._flush()
        self._task = None

    async def _flush(self) -> None:
        if not self._pending:
            return
        batch = list(self._pending)
        self._pending.clear()
        await self._handler(batch)


class FuturesRestClient:
    """Thin wrapper around the shared Binance REST helpers for futures klines."""

    __slots__ = ("_timeout", "_max_retries")

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_retries: int = 5,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Mapping[str, object]]:
        symbol_clean = _coerce_symbol(symbol)
        if interval != "1m":
            raise ValueError("Only 1m klines supported for fetch_klines")

        result: List[Mapping[str, object]] = []
        cursor = start_ms
        while cursor <= end_ms:
            slice_end = min(end_ms, cursor + (1000 * MINUTE_MS) - 1)
            rows = await self._fetch_slice(
                symbol_clean,
                interval,
                start_time=cursor,
                end_time=slice_end,
            )
            if not rows:
                break

            normalised = [_normalise_rest_row(row) for row in rows if row]
            result.extend(normalised)
            last_ts = normalised[-1]["ts_open"] if normalised else slice_end
            cursor = int(last_ts) + MINUTE_MS

            if len(rows) < 1000:
                break

        return result

    async def _fetch_slice(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: int,
        end_time: int,
    ) -> Sequence[Sequence[object]]:
        attempt = 0
        while True:
            try:
                return await fetch_um_klines(
                    symbol,
                    interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000,
                )
            except BinanceAPIException as exc:  # pragma: no cover - network variability
                attempt += 1
                retryable = getattr(exc, "status_code", None) in {418, 429, 500, 502, 503, 504}
                if not retryable or attempt > self._max_retries:
                    LOGGER.error(
                        "stream.rest.error",
                        extra={
                            "symbol": symbol,
                            "interval": interval,
                            "attempt": attempt,
                            "status": getattr(exc, "status_code", None),
                            "message": str(exc),
                        },
                    )
                    raise
                delay = min(self._timeout, (2 ** (attempt - 1)) * 0.5)
                LOGGER.warning(
                    "stream.rest.retry",
                    extra={
                        "symbol": symbol,
                        "interval": interval,
                        "attempt": attempt,
                        "delay": delay,
                        "status": getattr(exc, "status_code", None),
                    },
                )
                await asyncio.sleep(delay)
            except BinanceRequestException as exc:  # pragma: no cover - network variability
                attempt += 1
                if attempt > self._max_retries:
                    LOGGER.error(
                        "stream.rest.request_error",
                        extra={
                            "symbol": symbol,
                            "interval": interval,
                            "attempt": attempt,
                            "message": str(exc),
                        },
                    )
                    raise
                delay = min(self._timeout, (2 ** (attempt - 1)) * 0.5)
                LOGGER.warning(
                    "stream.rest.retry",
                    extra={
                        "symbol": symbol,
                        "interval": interval,
                        "attempt": attempt,
                        "delay": delay,
                        "status": getattr(exc, "status_code", None),
                    },
                )
                await asyncio.sleep(delay)


def _normalise_rest_row(row: Sequence[object]) -> Dict[str, object]:
    try:
        return {
            "ts_open": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "taker_buy_vol": float(row[9]),
            "taker_buy_quote": float(row[10]),
            "trades": int(row[8]),
        }
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"unable to normalise kline row: {row!r}") from exc


class BinanceWebsocketConsumer:
    """Threaded websocket wrapper that reconnects with exponential backoff."""

    __slots__ = (
        "_config",
        "_loop",
        "_queue",
        "_task",
        "_client",
        "_running",
        "_current_delay",
        "_backoff_task",
    )

    def __init__(
        self,
        config: BinanceStreamConfig,
        *,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[StreamEvent],
    ) -> None:
        self._config = config
        self._loop = loop
        self._queue = queue
        self._task: asyncio.Future[None] | None = None
        self._client = None
        self._running = False
        self._current_delay = config.reconnect_base_delay
        self._backoff_task: asyncio.TimerHandle | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        LOGGER.info(
            "stream.ws.start",
            extra={
                "url": self._config.websocket_url,
                "symbols": list(self._config.symbols),
                "streams": self._describe_streams(),
            },
        )
        self._schedule_connect(delay=0.0)

    def stop(self) -> None:
        self._running = False
        if self._backoff_task is not None:
            self._backoff_task.cancel()
            self._backoff_task = None
        if self._client is not None:
            try:
                self._client.stop()
            finally:
                self._client = None
        LOGGER.info("stream.ws.stop")

    def _schedule_connect(self, *, delay: float) -> None:
        if not self._running:
            return
        if self._backoff_task is not None:
            self._backoff_task.cancel()
        self._backoff_task = self._loop.call_later(delay, self._connect_once)

    def _connect_once(self) -> None:
        if not self._running:
            return
        try:
            from binance.websocket.websocket_client import BinanceWebsocketClient
        except ImportError as exc:  # pragma: no cover - dependency
            LOGGER.error("stream.ws.import_error", extra={"error": str(exc)})
            return

        try:
            self._client = BinanceWebsocketClient(
                stream_url=self._config.websocket_url,
                on_message=self._handle_message,
                on_close=self._handle_close,
                on_error=self._handle_error,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("stream.ws.connect_failed", extra={"error": str(exc)})
            self._schedule_reconnect()
            return

        self._current_delay = self._config.reconnect_base_delay
        self._subscribe()

    def _subscribe(self) -> None:
        if self._client is None:
            return
        streams = self._build_streams()
        try:
            self._client.subscribe(streams)
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("stream.ws.subscribe_failed", extra={"error": str(exc)})
            self._schedule_reconnect()
            return
        LOGGER.info("stream.ws.subscribed", extra={"streams": streams})

    def _build_streams(self) -> List[str]:
        entries: list[str] = []
        suffixes = ["kline_1m"]
        if self._config.include_agg_trades:
            suffixes.append("aggTrade")
        for symbol in self._config.symbols:
            for suffix in suffixes:
                entries.append(f"{symbol.lower()}@{suffix}")
        return entries

    def _describe_streams(self) -> Sequence[str]:
        return self._build_streams()

    def _handle_message(self, _manager, message: str) -> None:
        received_ms = _now_ms()
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            LOGGER.debug("stream.ws.invalid_json", extra={"message": message})
            return

        data = payload.get("data", payload)
        if not isinstance(data, Mapping):
            return
        event_type = str(data.get("e") or "")
        symbol = str(data.get("s") or data.get("ps") or "").upper()
        event_time_ms = int(data.get("E") or received_ms)
        if not event_type or not symbol:
            return

        stream_name = payload.get("stream")
        event = StreamEvent(
            event_type=event_type,
            symbol=symbol,
            event_time_ms=event_time_ms,
            received_time_ms=received_ms,
            payload=data,
            stream=stream_name,
        )
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            LOGGER.warning(
                "stream.ws.queue_full",
                extra={"symbol": symbol, "event_type": event_type},
            )

    def _handle_close(self, *_args) -> None:  # pragma: no cover - network dependent
        LOGGER.warning("stream.ws.closed")
        self._schedule_reconnect()

    def _handle_error(self, _manager, error: Exception) -> None:  # pragma: no cover - network dependent
        LOGGER.warning("stream.ws.error", extra={"error": str(error)})
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if not self._running:
            return
        delay = self._current_delay
        self._current_delay = min(self._current_delay * 2, self._config.reconnect_max_delay)
        LOGGER.warning("stream.ws.reconnect", extra={"delay": delay})
        self._schedule_connect(delay=delay)


@dataclass(slots=True)
class _SymbolState:
    last_candle_ts: int | None = None
    last_gap_ts: int | None = None


class BinanceFuturesStreamIngestor:
    """High-level pipeline orchestrating websocket consumption, batching, and storage."""

    __slots__ = (
        "_config",
        "_queue",
        "_loop",
        "_consumer",
        "_batcher",
        "_ring",
        "_storage",
        "_rest",
        "_states",
        "_shutdown_event",
        "_batch_task",
    )

    def __init__(self, config: BinanceStreamConfig) -> None:
        self._config = config
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=config.queue_size)
        self._consumer = BinanceWebsocketConsumer(config, loop=self._loop, queue=self._queue)
        self._batcher = MicroBatcher(self._queue, flush_interval=config.flush_interval_seconds, handler=self._handle_batch)
        self._ring = SymbolTimeframeRingBuffer(window_hours=config.window_hours)
        app_config = AppConfig.load()
        self._storage = ParquetStorage(
            root=app_config.data_dir,
            market=app_config.market,
            index_path=app_config.duckdb_path,
        )
        self._rest = FuturesRestClient(timeout=config.rest_timeout, max_retries=config.rest_max_retries)
        self._states: Dict[str, _SymbolState] = {_coerce_symbol(sym): _SymbolState() for sym in config.symbols}
        self._shutdown_event = asyncio.Event()
        self._batch_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        LOGGER.info(
            "stream.pipeline.start",
            extra={
                "symbols": list(self._config.symbols),
                "flush_interval": self._config.flush_interval_seconds,
                "window_hours": self._config.window_hours,
            },
        )
        await self._perform_initial_backfill()
        self._consumer.start()
        self._batch_task = self._batcher.start()

    async def run_forever(self) -> None:
        await self.start()
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        self._consumer.stop()
        await self._batcher.stop()

    async def _perform_initial_backfill(self) -> None:
        end_ms = _now_ms()
        start_ms = end_ms - (self._config.window_hours * 60 * MINUTE_MS)
        LOGGER.info("stream.backfill.start", extra={"start_ms": start_ms, "end_ms": end_ms})
        for symbol in self._config.symbols:
            try:
                candles = await self._rest.fetch_klines(symbol, self._config.interval, start_ms, end_ms)
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.error("stream.backfill.error", extra={"symbol": symbol, "error": str(exc)})
                continue
            if not candles:
                LOGGER.warning("stream.backfill.empty", extra={"symbol": symbol})
                continue
            for candle in candles:
                self._ring.append(symbol, self._config.interval, candle)
            await self._write_to_storage(symbol, candles)
            last_ts = int(candles[-1]["ts_open"])
            self._states[symbol].last_candle_ts = last_ts
            LOGGER.info(
                "stream.backfill.complete",
                extra={"symbol": symbol, "rows": len(candles), "last_ts": last_ts},
            )

    async def _handle_batch(self, batch: Sequence[StreamEvent]) -> None:
        if not batch:
            return

        closed_klines: Dict[Tuple[str, int], Tuple[StreamEvent, Mapping[str, object]]] = {}
        ws_lags: List[int] = []
        for event in batch:
            if event.event_type != "kline":
                continue
            data = event.payload.get("k")
            if not isinstance(data, Mapping):
                continue
            if not data.get("x"):
                continue  # ignore open candles
            try:
                ts_open = int(data["t"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (event.symbol, ts_open)
            closed_klines[key] = (event, data)
            lag = max(0, event.received_time_ms - event.event_time_ms)
            ws_lags.append(lag)

        if not closed_klines:
            return

        per_symbol: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
        processed = 0

        for (symbol, _), (event, data) in sorted(closed_klines.items(), key=lambda item: item[0][1]):
            candle = _normalise_stream_kline(data)
            if not candle:
                continue
            if not await self._handle_candle(symbol, candle):
                continue
            per_symbol[symbol].append(candle)
            processed += 1

        for symbol, candles in per_symbol.items():
            await self._write_to_storage(symbol, candles)

        LOGGER.info(
            "stream.batch.flush",
            extra={
                "batch_events": len(batch),
                "closed_candles": processed,
                "symbols": list(per_symbol.keys()),
                "avg_ws_lag_ms": _safe_avg(ws_lags),
                "max_ws_lag_ms": max(ws_lags) if ws_lags else None,
            },
        )

    async def _handle_candle(self, symbol: str, candle: Mapping[str, object]) -> bool:
        state = self._states.setdefault(symbol, _SymbolState())
        ts_open = int(candle["ts_open"])
        last_ts = state.last_candle_ts
        if last_ts is not None and ts_open <= last_ts:
            return False

        if last_ts is not None and ts_open > last_ts + MINUTE_MS:
            missing_start = last_ts + MINUTE_MS
            missing_end = ts_open - MINUTE_MS
            LOGGER.warning(
                "stream.gap.detected",
                extra={"symbol": symbol, "from_ts": missing_start, "to_ts": missing_end},
            )
            try:
                gap_candles = await self._rest.fetch_klines(symbol, self._config.interval, missing_start, missing_end)
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.error(
                    "stream.gap.fetch_failed",
                    extra={"symbol": symbol, "from_ts": missing_start, "to_ts": missing_end, "error": str(exc)},
                )
            else:
                if gap_candles:
                    LOGGER.info(
                        "stream.gap.filled",
                        extra={"symbol": symbol, "rows": len(gap_candles), "from_ts": missing_start, "to_ts": missing_end},
                    )
                    for gap in gap_candles:
                        self._ring.append(symbol, self._config.interval, gap)
                    await self._write_to_storage(symbol, gap_candles)
                    state.last_candle_ts = int(gap_candles[-1]["ts_open"])
                else:
                    LOGGER.warning(
                        "stream.gap.empty",
                        extra={"symbol": symbol, "from_ts": missing_start, "to_ts": missing_end},
                    )

        ring_size = self._ring.append(symbol, self._config.interval, candle)
        state.last_candle_ts = ts_open
        LOGGER.debug(
            "stream.candle.processed",
            extra={
                "symbol": symbol,
                "ts_open": ts_open,
                "ring_size": ring_size,
            },
        )
        return True

    async def _write_to_storage(self, symbol: str, candles: Iterable[Mapping[str, object]]) -> None:
        if not candles:
            return

        def _write() -> Sequence[StorageWriteStats]:
            return self._storage.write_rows(symbol, self._config.interval, candles)

        stats = await asyncio.to_thread(_write)
        for stat in stats:
            LOGGER.info(
                "stream.parquet.write",
                extra={
                    "symbol": stat.symbol,
                    "interval": stat.interval,
                    "rows": stat.rows,
                    "path": stat.path,
                    "start_ts": stat.start_ts,
                    "end_ts": stat.end_ts,
                },
            )


def _normalise_stream_kline(data: Mapping[str, object]) -> Dict[str, object] | None:
    try:
        return {
            "ts_open": int(data["t"]),
            "open": float(data["o"]),
            "high": float(data["h"]),
            "low": float(data["l"]),
            "close": float(data["c"]),
            "volume": float(data["v"]),
            "taker_buy_vol": float(data.get("V") or 0.0),
            "taker_buy_quote": float(data.get("Q") or 0.0),
            "trades": int(data.get("n") or 0),
        }
    except (KeyError, TypeError, ValueError):
        LOGGER.debug("stream.kline.normalise_failed", extra={"data": data})
        return None


def _safe_avg(values: Sequence[int]) -> int | None:
    if not values:
        return None
    return int(sum(values) / len(values))


__all__ = [
    "BinanceStreamConfig",
    "BinanceFuturesStreamIngestor",
    "StreamEvent",
    "MicroBatcher",
    "SymbolTimeframeRingBuffer",
]
