"""Realtime UM futures ingest with WS micro-batching and minute ring buffer."""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Sequence, Tuple

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .binance import (
    fetch_um_index_price_klines,
    fetch_um_klines,
    fetch_um_mark_price_klines,
    fetch_um_premium_index_klines,
)
from .timeutils import ensure_ms_epoch

LOGGER = logging.getLogger(__name__)

UM_STREAM_HOST = "wss://fstream.binance.com"
STREAM_TEMPLATE = "{symbol}@{stream}"
KLINE_STREAM = "kline_1m"
AGGTRADE_STREAM = "aggTrade"
BOOKTICKER_STREAM = "bookTicker"
MINUTE_MS = 60_000


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _align_minute(ts: int) -> int:
    return (ts // MINUTE_MS) * MINUTE_MS


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class UMIngestConfig:
    """Runtime configuration for UM minute ingest."""

    symbols: Tuple[str, ...]
    microbatch_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_base_seconds: float = 1.0
    ws_heartbeat: float = 30.0
    dedup_ttl_seconds: float = 180.0
    queue_maxsize: int = 10_000
    ring_days: int = 1
    include_previous_day: bool = True
    flush_interval_seconds: float = 60.0
    parquet_root: Path = Path("var/um_ingest")
    parquet_file: str = "minute.parquet"
    rest_tail_minutes: int = 1440
    rest_batch_minutes: int = 500
    coverage_threshold: float = 0.99
    log_samples: int = 500

    def __post_init__(self) -> None:
        clean_symbols: List[str] = []
        for symbol in self.symbols:
            cleaned = (symbol or "").strip().upper()
            if not cleaned:
                continue
            clean_symbols.append(cleaned)
        if not clean_symbols:
            raise ValueError("At least one symbol required")
        object.__setattr__(self, "symbols", tuple(dict.fromkeys(clean_symbols)))
        if self.microbatch_seconds <= 0.0:
            raise ValueError("microbatch_seconds must be positive")
        if self.dedup_ttl_seconds <= 0.0:
            raise ValueError("dedup_ttl_seconds must be positive")
        if self.queue_maxsize < 1000:
            raise ValueError("queue_maxsize is too small")
        if self.ring_days < 1:
            raise ValueError("ring_days must be >= 1")
        if self.flush_interval_seconds < 5.0:
            raise ValueError("flush_interval_seconds must be >= 5 seconds")
        object.__setattr__(self, "parquet_root", Path(self.parquet_root))


@dataclass(slots=True)
class MinuteRecord:
    """Minute-level composite derived from WS streams."""

    symbol: str
    ts_min: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    buyer_volume: float
    seller_volume: float
    delta: float
    cvd: float
    spread_bp_p50: float
    spread_bp_p95: float
    l1_imbalance_p50: float
    source_flags: Mapping[str, bool] = field(default_factory=dict)
    mark_price: float | None = None
    index_price: float | None = None
    premium_index: float | None = None
    basis_bp: float | None = None
    funding_rate: float | None = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts_min": self.ts_min,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trades": self.trades,
            "buyer_volume": self.buyer_volume,
            "seller_volume": self.seller_volume,
            "delta": self.delta,
            "cvd": self.cvd,
            "spread_bp_p50": self.spread_bp_p50,
            "spread_bp_p95": self.spread_bp_p95,
            "l1_imbalance_p50": self.l1_imbalance_p50,
            "has_kline": bool(self.source_flags.get(KLINE_STREAM)),
            "has_aggtrade": bool(self.source_flags.get(AGGTRADE_STREAM)),
            "has_bookticker": bool(self.source_flags.get(BOOKTICKER_STREAM)),
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "premium_index": self.premium_index,
            "basis_bp": self.basis_bp,
            "funding_rate": self.funding_rate,
        }


class DedupCache:
    """Keep track of processed event identifiers with TTL eviction."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._entries: "OrderedDict[str, float]" = OrderedDict()

    def add(self, key: str) -> None:
        now = time.monotonic()
        self._entries[key] = now
        self._entries.move_to_end(key)
        self._trim(now)

    def seen(self, key: str) -> bool:
        now = time.monotonic()
        ts = self._entries.get(key)
        if ts is None:
            return False
        if now - ts > self._ttl:
            self._entries.pop(key, None)
            return False
        return True

    def _trim(self, now: float) -> None:
        threshold = now - self._ttl
        while self._entries:
            _, ts = self._entries.items().__iter__().__next__()
            if ts >= threshold:
                break
            self._entries.popitem(last=False)


class MinuteAccumulator:
    """Aggregate WS payloads into minute-level composite."""

    __slots__ = (
        "symbol",
        "minute_ts",
        "kline",
        "agg_buy",
        "agg_sell",
        "agg_trades",
        "book_spreads",
        "book_imbalances",
        "flags",
        "first_event_ms",
        "last_event_ms",
    )

    def __init__(self, symbol: str, minute_ts: int) -> None:
        self.symbol = symbol
        self.minute_ts = minute_ts
        self.kline: Dict[str, Any] | None = None
        self.agg_buy = 0.0
        self.agg_sell = 0.0
        self.agg_trades = 0
        self.book_spreads: List[float] = []
        self.book_imbalances: List[float] = []
        self.flags: Dict[str, bool] = defaultdict(bool)
        self.first_event_ms: int | None = None
        self.last_event_ms: int | None = None

    def ingest_kline(self, payload: Mapping[str, Any]) -> None:
        kline = payload.get("k")
        if not isinstance(kline, Mapping):
            return
        self.flags[KLINE_STREAM] = True
        self.kline = {
            "open": _as_float(kline.get("o")),
            "high": _as_float(kline.get("h")),
            "low": _as_float(kline.get("l")),
            "close": _as_float(kline.get("c")),
            "volume": _as_float(kline.get("v")),
            "trades": int(kline.get("n", 0)),
            "closed": bool(kline.get("x")),
        }
        self._touch(payload)

    def ingest_aggtrade(self, payload: Mapping[str, Any]) -> None:
        qty = _as_float(payload.get("q"))
        is_buyer_taker = not bool(payload.get("m"))
        if is_buyer_taker:
            self.agg_buy += qty
        else:
            self.agg_sell += qty
        self.agg_trades += 1
        self.flags[AGGTRADE_STREAM] = True
        self._touch(payload)

    def ingest_bookticker(self, payload: Mapping[str, Any]) -> None:
        bid = _as_float(payload.get("b"))
        ask = _as_float(payload.get("a"))
        bid_qty = _as_float(payload.get("B"))
        ask_qty = _as_float(payload.get("A"))
        if bid <= 0.0 or ask <= 0.0 or ask <= bid:
            return
        spread_bp = ((ask - bid) / ((ask + bid) / 2.0)) * 10_000
        imbalance_den = bid_qty + ask_qty
        imbalance = 0.0
        if imbalance_den > 0.0:
            imbalance = (bid_qty - ask_qty) / imbalance_den
        self.book_spreads.append(spread_bp)
        self.book_imbalances.append(imbalance)
        self.flags[BOOKTICKER_STREAM] = True
        self._touch(payload)

    def finalise(self, cvd_base: float) -> MinuteRecord | None:
        if self.kline is None:
            return None
        buyer_volume = self.agg_buy
        seller_volume = self.agg_sell
        delta = buyer_volume - seller_volume
        cvd = cvd_base + delta
        spreads = self.book_spreads or [0.0]
        imbalances = self.book_imbalances or [0.0]
        median_spread = statistics.median(spreads)
        p95_spread = statistics.quantiles(spreads, n=100, method="inclusive")[94] if len(spreads) > 1 else median_spread
        median_imbalance = statistics.median(imbalances)
        return MinuteRecord(
            symbol=self.symbol,
            ts_min=self.minute_ts,
            open=self.kline["open"],
            high=self.kline["high"],
            low=self.kline["low"],
            close=self.kline["close"],
            volume=self.kline["volume"],
            trades=self.kline["trades"],
            buyer_volume=buyer_volume,
            seller_volume=seller_volume,
            delta=delta,
            cvd=cvd,
            spread_bp_p50=median_spread,
            spread_bp_p95=p95_spread,
            l1_imbalance_p50=median_imbalance,
            source_flags=dict(self.flags),
        )

    def _touch(self, payload: Mapping[str, Any]) -> None:
        now = int(payload.get("E") or 0)
        if now <= 0:
            return
        if self.first_event_ms is None or now < self.first_event_ms:
            self.first_event_ms = now
        if self.last_event_ms is None or now > self.last_event_ms:
            self.last_event_ms = now


class MinuteRingBuffer:
    """Keep minute composites in RAM with deduplication."""

    def __init__(self, max_days: int, include_previous: bool) -> None:
        self._max_days = max_days
        self._include_previous = include_previous
        self._store: Dict[str, "OrderedDict[int, MinuteRecord]"] = defaultdict(OrderedDict)
        self._cvd_cache: Dict[str, float] = defaultdict(float)

    def cvd_base(self, symbol: str) -> float:
        return self._cvd_cache[symbol]

    def upsert(self, record: MinuteRecord) -> None:
        index = self._store[record.symbol]
        index[record.ts_min] = record
        index.move_to_end(record.ts_min)
        self._cvd_cache[record.symbol] = record.cvd
        self._prune(record.symbol)

    def _prune(self, symbol: str) -> None:
        index = self._store[symbol]
        if not index:
            return
        cutoff_days = self._max_days + (1 if self._include_previous else 0)
        latest = next(reversed(index))
        latest_dt = datetime.fromtimestamp(latest / 1000, UTC)
        floor = (latest_dt - timedelta(days=cutoff_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        floor_ms = int(floor.timestamp() * 1000)
        keys_to_remove = [key for key in index if key < floor_ms]
        for key in keys_to_remove:
            index.pop(key, None)

    def minutes(self, symbol: str) -> List[MinuteRecord]:
        return list(self._store.get(symbol, {}).values())

    def all_records(self) -> Iterable[MinuteRecord]:
        for index in self._store.values():
            for record in index.values():
                yield record

    def flush_to_parquet(self, config: UMIngestConfig) -> Dict[str, int]:
        config.parquet_root.mkdir(parents=True, exist_ok=True)
        write_counts: Dict[str, int] = {}
        by_symbol: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for record in self.all_records():
            dt = datetime.fromtimestamp(record.ts_min / 1000, UTC)
            partition = dt.strftime("date=%Y-%m-%d")
            by_symbol[record.symbol][partition].append(record.to_row())

        for symbol, partitions in by_symbol.items():
            sym_root = config.parquet_root / symbol.lower()
            for partition, rows in partitions.items():
                if not rows:
                    continue
                partition_dir = sym_root / partition
                partition_dir.mkdir(parents=True, exist_ok=True)
                path = partition_dir / config.parquet_file
                frame = pd.DataFrame(rows)
                frame.sort_values("ts_min", inplace=True)
                frame.drop_duplicates(["symbol", "ts_min"], keep="last", inplace=True)
                if path.exists():
                    existing = pq.read_table(path)
                    existing_df = existing.to_pandas()
                    frame = pd.concat([existing_df, frame], ignore_index=True)
                    frame.sort_values("ts_min", inplace=True)
                    frame.drop_duplicates(["symbol", "ts_min"], keep="last", inplace=True)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                pq.write_table(table, path)
                write_counts[path.as_posix()] = len(frame)
        return write_counts


class MetricsTracker:
    """Collect lag, batch sizes, and coverage stats."""

    def __init__(self, config: UMIngestConfig) -> None:
        self._lag_samples: Deque[int] = deque(maxlen=config.log_samples)
        self._batch_sizes: Deque[int] = deque(maxlen=config.log_samples)
        self._stream_presence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._stream_expected: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._latest_minute: Dict[str, int] = defaultdict(int)
        self._largest_gap: Dict[str, int] = defaultdict(int)
        self._coverage_threshold = config.coverage_threshold
        self._minute_count: Dict[str, int] = defaultdict(int)
        self._first_minute: Dict[str, int] = {}

    def add_lag(self, value: int) -> None:
        if value >= 0:
            self._lag_samples.append(value)

    def add_batch_size(self, size: int) -> None:
        if size > 0:
            self._batch_sizes.append(size)

    def update_stream_presence(self, record: MinuteRecord) -> None:
        symbol = record.symbol
        minute = record.ts_min
        prev_minute = self._latest_minute.get(symbol)
        if prev_minute:
            gap = max(0, (minute - prev_minute) // MINUTE_MS - 1)
            if gap > self._largest_gap[symbol]:
                self._largest_gap[symbol] = gap
        self._latest_minute[symbol] = minute
        if symbol not in self._first_minute:
            self._first_minute[symbol] = minute
        self._minute_count[symbol] += 1

        for stream, flag in record.source_flags.items():
            self._stream_expected[symbol][stream] += 1
            if flag:
                self._stream_presence[symbol][stream] += 1

    def snapshot(self) -> Dict[str, Any]:
        lag = list(self._lag_samples)
        batches = list(self._batch_sizes)
        lag_avg = sum(lag) / len(lag) if lag else 0.0
        lag_p95 = statistics.quantiles(lag, n=20, method="inclusive")[18] if len(lag) >= 20 else lag_avg
        batch_avg = sum(batches) / len(batches) if batches else 0.0
        symbol_metrics: Dict[str, Any] = {}
        all_symbols = set(self._minute_count.keys()) | set(self._stream_expected.keys())
        for symbol in all_symbols:
            first_minute = self._first_minute.get(symbol)
            latest_minute = self._latest_minute.get(symbol)
            minutes_found = self._minute_count.get(symbol, 0)
            expected_minutes = minutes_found
            if first_minute is not None and latest_minute and latest_minute >= first_minute:
                expected_minutes = int(((latest_minute - first_minute) // MINUTE_MS) + 1)
            coverage_pct = round(minutes_found / max(expected_minutes, 1), 4)
            streams_payload: Dict[str, Any] = {}
            expected_streams = self._stream_expected.get(symbol, {})
            present_streams = self._stream_presence.get(symbol, {})
            for stream, expected in expected_streams.items():
                found = present_streams.get(stream, 0)
                streams_payload[stream] = {
                    "found": int(found),
                    "expected": int(expected),
                    "coverage_pct": round(found / expected, 4) if expected else None,
                }
            symbol_metrics[symbol] = {
                "minutes": {
                    "found": int(minutes_found),
                    "expected": int(expected_minutes),
                    "coverage_pct": coverage_pct,
                },
                "largest_gap_min": int(self._largest_gap.get(symbol, 0)),
                "streams": streams_payload,
            }

        return {
            "lag": {
                "avg_ms": round(lag_avg, 2),
                "p95_ms": round(lag_p95, 2),
                "samples": len(lag),
            },
            "batch": {
                "avg_size": round(batch_avg, 2),
                "samples": len(batches),
            },
            "symbols": symbol_metrics,
        }

    def log_if_needed(self) -> None:
        snapshot = self.snapshot()
        warn = False
        for symbol, metrics in snapshot.get("symbols", {}).items():
            streams = metrics.get("streams", {})
            for stream, stats in streams.items():
                pct = stats.get("coverage_pct") or 0.0
                if pct < self._coverage_threshold:
                    warn = True
                    LOGGER.warning(
                        "Coverage below threshold",
                        extra={
                            "symbol": symbol,
                            "stream": stream,
                            "pct": round(pct * 100.0, 2),
                            "threshold": self._coverage_threshold * 100.0,
                        },
                    )
        levels = LOGGER.warning if warn else LOGGER.info
        levels("UM ingest metrics", extra={"metrics": snapshot})


class UMIngestService:
    """End-to-end ingestion service combining WS and REST filling."""

    def __init__(self, config: UMIngestConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=config.queue_maxsize)
        self._dedup = DedupCache(config.dedup_ttl_seconds)
        self._ring = MinuteRingBuffer(config.ring_days, config.include_previous_day)
        self._metrics = MetricsTracker(config)
        self._session: aiohttp.ClientSession | None = None
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[Any]] = []
        self._flush_task: asyncio.Task[Any] | None = None
        self._metrics_task: asyncio.Task[Any] | None = None
        self._duplicates_dropped = 0
        self._ws_disconnects = 0

    async def start(self) -> None:
        if self._session is not None:
            raise RuntimeError("Service already started")
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        await self._prime_rest_tail()
        self._tasks.append(asyncio.create_task(self._consume_loop(), name="um-ingest-consumer"))
        self._tasks.append(asyncio.create_task(self._ws_loop(), name="um-ingest-ws"))
        self._flush_task = asyncio.create_task(self._flush_loop(), name="um-ingest-flush")
        self._metrics_task = asyncio.create_task(self._metrics_loop(), name="um-ingest-metrics")

    async def stop(self) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        if self._flush_task:
            self._flush_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._flush_task:
            await asyncio.gather(self._flush_task, return_exceptions=True)
        if self._metrics_task:
            await asyncio.gather(self._metrics_task, return_exceptions=True)
        if self._session:
            await self._session.close()
            self._session = None

    async def _ws_loop(self) -> None:
        assert self._session is not None
        delay = self._config.reconnect_base_seconds
        streams = self._build_streams()
        url = f"{UM_STREAM_HOST}/stream?streams={'/'.join(streams)}"
        while not self._stop_event.is_set():
            try:
                await self._ws_consume(url)
                if not self._stop_event.is_set():
                    self._ws_disconnects += 1
                delay = self._config.reconnect_base_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("WS loop error", exc_info=exc)
                if not self._stop_event.is_set():
                    self._ws_disconnects += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._config.reconnect_max_seconds)

    async def _ws_consume(self, url: str) -> None:
        assert self._session is not None
        LOGGER.info("Connecting WS stream", extra={"url": url})
        async with self._session.ws_connect(url, heartbeat=self._config.ws_heartbeat) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        self._queue.put_nowait(payload)
                    except asyncio.QueueFull:
                        LOGGER.warning("Ingest queue is full; dropping payload")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WS error: {msg}")

    async def _consume_loop(self) -> None:
        window = self._config.microbatch_seconds
        while not self._stop_event.is_set():
            batch: List[Dict[str, Any]] = []
            start = time.monotonic()
            while True:
                if self._stop_event.is_set():
                    break
                remaining = start + window - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=min(window, remaining))
                except asyncio.TimeoutError:
                    break
                batch.append(item)
                if len(batch) >= 5000:
                    break
            if batch:
                self._metrics.add_batch_size(len(batch))
                self._process_batch(batch)

    def _process_batch(self, batch: Sequence[Mapping[str, Any]]) -> None:
        by_symbol: Dict[Tuple[str, int], MinuteAccumulator] = {}
        for payload in batch:
            stream = payload.get("stream")
            data = payload.get("data")
            if not isinstance(stream, str) or not isinstance(data, Mapping):
                continue
            symbol = data.get("s")
            if not isinstance(symbol, str):
                continue
            symbol = symbol.upper()
            event_ms = ensure_ms_epoch(data.get("E"))
            if event_ms is None:
                LOGGER.warning("um.ingest.invalid_event_ts", extra={"stream": stream, "symbol": symbol})
                continue
            minute_ts = _align_minute(event_ms)
            if minute_ts <= 0:
                continue
            dedup_id = None
            if stream.endswith(KLINE_STREAM):
                kline = data.get("k")
                if isinstance(kline, Mapping):
                    start_ts = ensure_ms_epoch(kline.get("t"))
                    if start_ts is None:
                        LOGGER.warning("um.ingest.invalid_kline_ts", extra={"symbol": symbol})
                        continue
                    dedup_id = f"{start_ts}:{int(bool(kline.get('x')))}"
            elif stream.endswith(AGGTRADE_STREAM):
                dedup_id = data.get("a")
            elif stream.endswith(BOOKTICKER_STREAM):
                dedup_id = data.get("u")
            if dedup_id is None:
                dedup_id = event_ms
            key = f"{stream}:{symbol}:{dedup_id}"
            if self._dedup.seen(key):
                self._duplicates_dropped += 1
                continue
            self._dedup.add(key)
            lag = _utc_ms() - event_ms
            self._metrics.add_lag(lag)
            accumulator = by_symbol.setdefault(
                (symbol, minute_ts),
                MinuteAccumulator(symbol, minute_ts),
            )
            if stream.endswith(KLINE_STREAM):
                accumulator.ingest_kline(data)
            elif stream.endswith(AGGTRADE_STREAM):
                accumulator.ingest_aggtrade(data)
            elif stream.endswith(BOOKTICKER_STREAM):
                accumulator.ingest_bookticker(data)
        for accumulator in by_symbol.values():
            record = accumulator.finalise(self._ring.cvd_base(accumulator.symbol))
            if record is None:
                continue
            self._ring.upsert(record)
            self._metrics.update_stream_presence(record)

    async def _flush_loop(self) -> None:
        interval = self._config.flush_interval_seconds
        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            try:
                written = self._ring.flush_to_parquet(self._config)
                if written:
                    LOGGER.info("Flushed UM minutes to parquet", extra={"files": written})
            except Exception as exc:
                LOGGER.exception("Failed to flush UM minutes", exc_info=exc)

    async def _metrics_loop(self) -> None:
        interval = 60.0
        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            snapshot = self._metrics.snapshot()
            degradations: List[Dict[str, Any]] = []
            for symbol, metrics in snapshot.get("symbols", {}).items():
                minutes = metrics.get("minutes", {})
                coverage_pct = minutes.get("coverage_pct") or 0.0
                if coverage_pct < self._config.coverage_threshold:
                    degradations.append({
                        "symbol": symbol,
                        "reason": "low_coverage",
                        "value": coverage_pct,
                    })
                if metrics.get("largest_gap_min", 0) > 1:
                    degradations.append({
                        "symbol": symbol,
                        "reason": "large_gap",
                        "value": metrics.get("largest_gap_min", 0),
                    })
                streams = metrics.get("streams", {})
                for stream_name, stream_stats in streams.items():
                    pct = stream_stats.get("coverage_pct") or 0.0
                    if pct < self._config.coverage_threshold:
                        degradations.append({
                            "symbol": symbol,
                            "reason": "stream_low_coverage",
                            "stream": stream_name,
                            "value": pct,
                        })

            lag_p95 = snapshot.get("lag", {}).get("p95_ms", 0.0)
            if lag_p95 and lag_p95 > 1500.0:
                degradations.append({
                    "reason": "lag_high",
                    "value": lag_p95,
                })

            metrics_payload = {
                "type": "um_metrics",
                "timestamp_ms": _utc_ms(),
                "snapshot": snapshot,
                "duplicates_dropped": self._duplicates_dropped,
                "ws_disconnects": self._ws_disconnects,
                "degradations": degradations,
            }
            try:
                LOGGER.info(json.dumps(metrics_payload, ensure_ascii=False))
            except Exception:
                LOGGER.info("um.metrics", extra=metrics_payload)

    def _build_streams(self) -> List[str]:
        streams: List[str] = []
        for symbol in self._config.symbols:
            lower = symbol.lower()
            streams.extend(
                [
                    STREAM_TEMPLATE.format(symbol=lower, stream=KLINE_STREAM),
                    STREAM_TEMPLATE.format(symbol=lower, stream=AGGTRADE_STREAM),
                    STREAM_TEMPLATE.format(symbol=lower, stream=BOOKTICKER_STREAM),
                ]
            )
        return streams

    async def _prime_rest_tail(self) -> None:
        lookback = self._config.rest_tail_minutes
        now = _utc_ms()
        start_ms = now - lookback * MINUTE_MS
        end_ms = now
        tasks: List[asyncio.Task[None]] = []
        for symbol in self._config.symbols:
            tasks.append(asyncio.create_task(self._fetch_rest_tail(symbol, start_ms, end_ms)))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_rest_tail(self, symbol: str, start_ms: int, end_ms: int) -> None:
        LOGGER.info(
            "Backfilling tail via REST",
            extra={"symbol": symbol, "start_ms": start_ms, "end_ms": end_ms},
        )
        klines = await fetch_um_klines(symbol, "1m", start_time=start_ms, end_time=end_ms)
        mark = await fetch_um_mark_price_klines(symbol, "1m", start_time=start_ms, end_time=end_ms)
        index = await fetch_um_index_price_klines(symbol, "1m", start_time=start_ms, end_time=end_ms)
        premium = await fetch_um_premium_index_klines(symbol, "1m", start_time=start_ms, end_time=end_ms)

        mark_map = {int(row[0]): row for row in mark if len(row) >= 2}
        index_map = {int(row[0]): row for row in index if len(row) >= 2}
        premium_map = {int(row[0]): row for row in premium if len(row) >= 2}

        cvd_base = self._ring.cvd_base(symbol)
        for row in klines:
            if len(row) < 11:
                continue
            ts_raw = row[0]
            ts = ensure_ms_epoch(ts_raw)
            if ts is None:
                LOGGER.warning("um.ingest.rest_tail.invalid_ts", extra={"symbol": symbol, "value": ts_raw})
                continue
            minute_ts = _align_minute(ts)
            buyer_volume = _as_float(row[9])
            seller_volume = max(_as_float(row[5]) - buyer_volume, 0.0)
            delta = buyer_volume - seller_volume
            cvd_base += delta
            mark_row = mark_map.get(minute_ts)
            index_row = index_map.get(minute_ts)
            premium_row = premium_map.get(minute_ts)
            mark_close = _as_float(mark_row[4]) if mark_row and len(mark_row) > 4 else None
            index_close = _as_float(index_row[4]) if index_row and len(index_row) > 4 else None
            premium_close = _as_float(premium_row[4]) if premium_row and len(premium_row) > 4 else None
            basis_bp = None
            if mark_close is not None and index_close is not None and index_close > 0.0:
                basis_bp = ((mark_close - index_close) / index_close) * 10_000
            record = MinuteRecord(
                symbol=symbol,
                ts_min=minute_ts,
                open=_as_float(row[1]),
                high=_as_float(row[2]),
                low=_as_float(row[3]),
                close=_as_float(row[4]),
                volume=_as_float(row[5]),
                trades=int(row[8]),
                buyer_volume=buyer_volume,
                seller_volume=seller_volume,
                delta=delta,
                cvd=cvd_base,
                spread_bp_p50=0.0,
                spread_bp_p95=0.0,
                l1_imbalance_p50=0.0,
                source_flags={KLINE_STREAM: True, AGGTRADE_STREAM: False, BOOKTICKER_STREAM: False},
                mark_price=mark_close,
                index_price=index_close,
                premium_index=premium_close,
                basis_bp=basis_bp,
                funding_rate=premium_close,
            )
            if mark_row:
                record.source_flags["mark_price"] = True
            if index_row:
                record.source_flags["index_price"] = True
            if premium_row:
                record.source_flags["premium_index"] = True
            self._ring.upsert(record)

    async def snapshot_minutes(self, symbol: str) -> List[Dict[str, Any]]:
        return [record.to_row() for record in self._ring.minutes(symbol)]


__all__ = [
    "UMIngestConfig",
    "UMIngestService",
    "MinuteRecord",
]
