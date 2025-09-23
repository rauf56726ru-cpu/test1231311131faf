"""Market data helpers with REST history and simulation fallbacks."""
from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - ccxt is optional for tests
    ccxt = None  # type: ignore

from ..io.binance_ws import BINANCE_SOURCE, fetch_klines, fetch_klines_paginated
from ..utils.logging import get_logger
from ..utils.timeframes import compute_refresh_seconds, interval_to_seconds
from .candles_store import CandleStore

LOGGER = get_logger(__name__)

HISTORY_LOOKBACK_DAYS = 30
PRELOAD_CANDLES_PER_INTERVAL = 10_000

# Binance futures REST API does not expose a native 10m interval, so we
# aggregate 1m candles locally when callers request 10m history.
AGGREGATED_INTERVALS: Dict[str, Tuple[str, int]] = {
    "3m": ("1m", 3),
    "5m": ("1m", 5),
    "10m": ("1m", 10),
    "15m": ("1m", 15),
    "30m": ("1m", 30),
    "1h": ("1m", 60),
    "2h": ("1m", 120),
    "4h": ("1m", 240),
}

Candle = Dict[str, float]
Key = Tuple[str, str]


@dataclass
class SimulationState:
    last_ts_ms: int
    last_close: float
    interval_ms: int
    rng: random.Random


class MarketDataProvider:
    """Fetch OHLC data with fallbacks and lightweight caching."""

    def __init__(
        self,
        *,
        store: CandleStore | None = None,
        fetch_fraction: float = 0.25,
        fetch_min_seconds: float = 1.0,
        fetch_max_seconds: float = 15.0,
    ) -> None:
        self._history: Dict[Key, List[Dict[str, float]]] = {}
        self._locks: Dict[Key, asyncio.Lock] = {}
        self._sim_state: Dict[Key, SimulationState] = {}
        self._remote_retry_at: Dict[Key, float] = {}
        self._last_fetch_at: Dict[Key, float] = {}
        self.source = BINANCE_SOURCE
        self._fetch_fraction = max(0.0, float(fetch_fraction))
        self._fetch_min_seconds = max(0.0, float(fetch_min_seconds))
        self._fetch_max_seconds = float(fetch_max_seconds)
        self._preload_limit = PRELOAD_CANDLES_PER_INTERVAL
        self._preloaded: Set[Key] = set()
        self._store = store
        self._store_loaded: Set[Key] = set()
        self._lock_owners: Dict[Key, asyncio.Task] = {}

    @asynccontextmanager
    async def _acquire_key(self, key: Key):
        lock = self._locks.setdefault(key, asyncio.Lock())
        current = asyncio.current_task()
        if current is None:
            async with lock:
                yield
            return
        owner = self._lock_owners.get(key)
        if owner is current:
            yield
            return
        await lock.acquire()
        self._lock_owners[key] = current
        try:
            yield
        finally:
            self._lock_owners.pop(key, None)
            lock.release()

    async def _get_base_history(self, symbol: str, interval: str) -> List[Dict[str, float]]:
        base_key = self._key(symbol, interval)
        async with self._acquire_key(base_key):
            await self._hydrate_from_store(base_key, symbol, interval)
            return [dict(bar) for bar in self._history.get(base_key, [])]

    async def _rebuild_aggregated_history(
        self,
        symbol: str,
        interval: str,
        *,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        base_interval, ratio = self._resolve_base_interval(interval)
        if ratio <= 1:
            dest_key = self._key(symbol, interval)
            async with self._acquire_key(dest_key):
                candles = [dict(bar) for bar in self._history.get(dest_key, [])]
            if limit is None:
                return candles
            return candles[-max(1, int(limit)) :] if candles else []
        base_candles = await self._get_base_history(symbol, base_interval)
        if not base_candles:
            stored: List[Dict[str, float]] = []
        else:
            aggregated = self._aggregate(base_candles, interval, limit=0)
            cap = self._history_cap(interval)
            if len(aggregated) > cap:
                aggregated = aggregated[-cap:]
            stored = [dict(bar) for bar in aggregated]
        dest_key = self._key(symbol, interval)
        async with self._acquire_key(dest_key):
            self._history[dest_key] = list(stored)
        if not stored:
            return []
        filtered = stored
        if range_start is not None or range_end is not None:
            start = range_start if range_start is not None else -10**15
            end = range_end if range_end is not None else 10**15
            filtered = [
                bar
                for bar in stored
                if start <= int(bar.get("ts_ms_utc", 0)) <= end
            ]
        if limit is not None and limit > 0:
            filtered = filtered[-max(1, int(limit)) :]
        return [dict(bar) for bar in filtered]

    def _interval_ms(self, interval: str) -> int:
        seconds = interval_to_seconds(interval) or 60.0
        return int(max(1.0, seconds) * 1000)

    def _resolve_base_interval(self, interval: str) -> Tuple[str, int]:
        base, ratio = AGGREGATED_INTERVALS.get(interval, (interval, 1))
        return base, ratio

    def _chunk_end(self, start_ms: int, interval: str, chunk_size: int = 1000) -> int:
        interval_ms = self._interval_ms(interval)
        safe_start = max(0, int(start_ms))
        safe_size = max(1, int(chunk_size))
        return safe_start + interval_ms * safe_size - 1

    def _estimate_required(self, interval: str, date_from_ms: Optional[int]) -> int:
        if date_from_ms is None:
            return 0
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        span = max(0, now_ms - int(date_from_ms))
        interval_ms = self._interval_ms(interval)
        if interval_ms <= 0:
            return 0
        return int(span // interval_ms) + 5

    def _history_cap(self, interval: str) -> int:
        seconds = interval_to_seconds(interval) or 60.0
        approx = int(HISTORY_LOOKBACK_DAYS * 86_400 / max(1.0, seconds)) + 10
        return max(self._preload_limit, approx)

    def _merge_history(
        self, key: Key, interval: str, new_bars: Iterable[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        existing = {int(bar["ts_ms_utc"]): dict(bar) for bar in self._history.get(key, [])}
        for bar in new_bars:
            try:
                ts = int(bar["ts_ms_utc"])
            except (KeyError, TypeError, ValueError):
                continue
            existing[ts] = dict(bar)
        merged = [existing[idx] for idx in sorted(existing)]
        cap = self._history_cap(key[1])
        if len(merged) > cap:
            merged = merged[-cap:]
        self._history[key] = merged
        return merged

    def _filter_future(self, bars: Iterable[Dict[str, float]], interval: str) -> List[Dict[str, float]]:
        interval_ms = self._interval_ms(interval)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        cutoff = now_ms + interval_ms
        filtered: List[Dict[str, float]] = []
        for bar in bars:
            try:
                ts = int(bar["ts_ms_utc"])
            except (KeyError, TypeError, ValueError):
                continue
            if ts <= cutoff:
                filtered.append(dict(bar))
        return filtered

    async def _hydrate_from_store(self, key: Key, symbol: str, interval: str) -> List[Dict[str, float]]:
        if self._store is None or key in self._store_loaded:
            return self._history.get(key, [])
        stored = await self._store.read(symbol, interval)
        sanitized = self._filter_future(stored, interval) if stored else []
        if sanitized:
            cap = self._history_cap(interval)
            trimmed = [dict(bar) for bar in sanitized[-cap:]]
            self._history[key] = trimmed
            self._preloaded.add(key)
            self._last_fetch_at[key] = time.time()
        else:
            self._history.setdefault(key, [])
        self._store_loaded.add(key)
        return self._history.get(key, [])

    async def _persist_history(self, symbol: str, interval: str) -> None:
        if self._store is None:
            return
        base_interval, ratio = self._resolve_base_interval(interval)
        key = self._key(symbol, interval)
        snapshot = [dict(bar) for bar in self._history.get(key, [])]
        await self._store.write(symbol, interval, snapshot)

    async def _clear_store(self, symbol: str, interval: str) -> None:
        if self._store is None:
            return
        key = self._key(symbol, interval)
        await self._store.clear(symbol, interval)
        self._store_loaded.discard(key)

    async def _fetch_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        chunk_size: int = 1000,
    ) -> List[Dict[str, float]]:
        if start_ms is None or end_ms is None:
            return []
        start = max(0, int(start_ms))
        end = max(start, int(end_ms))
        fetch_interval, ratio = self._resolve_base_interval(interval)
        fetch_interval_ms = self._interval_ms(fetch_interval)
        interval_ms = self._interval_ms(interval)
        safe_chunk = max(1, min(int(chunk_size), 1000))
        target_limit = safe_chunk * ratio
        fetch_limit = max(1, min(int(target_limit), 1000))
        if ratio > 1:
            aligned_start = max(0, start - (start % interval_ms if interval_ms else 0))
            fetch_start = max(0, aligned_start - fetch_interval_ms * ratio)
            fetch_end = end + interval_ms
        else:
            fetch_start = start
            fetch_end = end
        fetcher_fn = globals().get("fetch_klines", fetch_klines)
        rows: List[List[float]] = []
        current_start = fetch_start
        current_end = min(self._chunk_end(current_start, fetch_interval, fetch_limit), fetch_end)
        safety = 0
        while current_start <= fetch_end and safety < 1000:
            safety += 1
            raw = await fetcher_fn(
                symbol,
                fetch_interval,
                limit=fetch_limit,
                start_time=current_start,
                end_time=current_end,
            )
            if not raw:
                break
            rows.extend(raw)
            last_row = raw[-1] if raw else None
            last_open = int(last_row[0]) if last_row and isinstance(last_row[0], (int, float)) else None
            if last_open is None:
                break
            next_start = last_open + fetch_interval_ms
            if next_start <= current_start:
                break
            if next_start > fetch_end or len(raw) < fetch_limit:
                if next_start > fetch_end:
                    break
                current_start = next_start
            else:
                current_start = next_start
            current_end = min(self._chunk_end(current_start, fetch_interval, fetch_limit), fetch_end)
        normalized = self._normalize_binance(rows, fetch_interval)
        if ratio > 1 and normalized:
            normalized = self._aggregate(normalized, interval, limit=len(normalized))
        normalized = self._filter_future(normalized, interval)
        result: List[Dict[str, float]] = []
        for bar in normalized:
            try:
                ts = int(bar["ts_ms_utc"])
            except (KeyError, TypeError, ValueError):
                continue
            if start <= ts <= end:
                result.append(dict(bar))
        return result

    async def _preload_interval(
        self,
        symbol: str,
        interval: str,
        *,
        candles: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        target = max(self._preload_limit, int(candles)) if candles else self._preload_limit
        base_interval, ratio = self._resolve_base_interval(interval)
        if ratio > 1:
            base_key = self._key(symbol, base_interval)
            base_target = max(target * ratio, self._preload_limit)
            if base_key not in self._preloaded:
                await self._preload_interval(symbol, base_interval, candles=base_target)
            aggregated = await self._rebuild_aggregated_history(
                symbol,
                interval,
                range_start=None,
                range_end=None,
                limit=target,
            )
            if aggregated:
                agg_key = self._key(symbol, interval)
                self._preloaded.add(agg_key)
                await self._persist_history(symbol, interval)
            return aggregated
        key = self._key(symbol, interval)
        interval_ms = self._interval_ms(interval)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        async with self._acquire_key(key):
            await self._hydrate_from_store(key, symbol, interval)
            existing = list(self._history.get(key, []))
            if existing:
                if len(existing) >= target:
                    return existing[-target:]
                latest_ts = int(existing[-1].get("ts_ms_utc", 0))
                recent_enough = bool(latest_ts) and now_ms - latest_ts <= interval_ms * 3
                if recent_enough:
                    return existing[-min(len(existing), target):]
        start_ms = max(0, now_ms - interval_ms * target)
        fetched = await self._fetch_range(symbol, interval, start_ms, now_ms)
        if fetched:
            async with self._acquire_key(key):
                merged = self._merge_history(key, interval, fetched[-target:])
                self._preloaded.add(key)
                await self._persist_history(symbol, interval)
                return merged
        async with self._acquire_key(key):
            return self._history.get(key, []) or []

    async def preload_cache(
        self,
        symbol: str,
        intervals: Iterable[str],
        *,
        candles: Optional[int] = None,
    ) -> Dict[str, int]:
        results: Dict[str, int] = {}
        for frame in intervals:
            merged = await self._preload_interval(symbol, frame, candles=candles)
            results[frame] = len(merged)
        return results

    async def history(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        *,
        date_from_ms: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, float]]:
        key = self._key(symbol, interval)
        safe_limit = max(1, int(limit))
        interval_ms = self._interval_ms(interval)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        now_sec = time.time()
        desired = max(safe_limit, self._preload_limit)
        range_start = int(date_from_ms) if date_from_ms is not None else max(0, now_ms - interval_ms * desired)
        range_end = now_ms
        base_interval, ratio = self._resolve_base_interval(interval)
        if ratio > 1:
            base_target = max(self._preload_limit, desired * ratio)
            if force_refresh:
                await self.history(
                    symbol,
                    base_interval,
                    limit=base_target,
                    date_from_ms=range_start,
                    force_refresh=True,
                )
            else:
                base_history = await self._get_base_history(symbol, base_interval)
                if not base_history:
                    await self._preload_interval(symbol, base_interval, candles=base_target)
            candles = await self._rebuild_aggregated_history(
                symbol,
                interval,
                range_start=range_start,
                range_end=range_end,
                limit=None,
            )
            if not candles:
                return []
            if len(candles) > safe_limit:
                candles = candles[-safe_limit:]
            self._preloaded.add(key)
            await self._persist_history(symbol, interval)
            return [dict(bar) for bar in candles]
        need_fetch = False
        async with self._acquire_key(key):
            await self._hydrate_from_store(key, symbol, interval)
            if force_refresh:
                self._history[key] = []
                self._preloaded.discard(key)
                await self._clear_store(symbol, interval)
                candles: List[Dict[str, float]] = []
                need_fetch = True
            else:
                candles = list(self._history.get(key, []))
            if candles and not need_fetch:
                try:
                    earliest = int(candles[0]["ts_ms_utc"])
                    latest = int(candles[-1]["ts_ms_utc"])
                except (KeyError, TypeError, ValueError, IndexError):
                    earliest = 0
                    latest = 0
                covers_start = earliest <= range_start
                margin = max(interval_ms * 5, 120_000)
                covers_end = latest >= range_end - margin
                if date_from_ms is not None:
                    covers_start = earliest <= range_start
                enough_span = covers_start and covers_end
                if enough_span and len(candles) >= safe_limit:
                    window = [bar for bar in candles if range_start <= int(bar.get("ts_ms_utc", 0)) <= range_end]
                    if not window:
                        window = candles[-safe_limit:]
                    return [dict(bar) for bar in window[-safe_limit:]]
                fresh_enough = (now_sec - self._last_fetch_at.get(key, 0.0)) < max(
                    self._fetch_min_seconds, margin / 1000.0
                )
                if fresh_enough and len(candles) >= safe_limit:
                    window = [bar for bar in candles if range_start <= int(bar.get("ts_ms_utc", 0)) <= range_end]
                    if not window:
                        window = candles[-safe_limit:]
                    return [dict(bar) for bar in window[-safe_limit:]]
                need_fetch = len(candles) < safe_limit or not covers_end
            if not candles:
                need_fetch = True
        fetched: List[Dict[str, float]] = []
        if need_fetch:
            fetched = await self._fetch_range(symbol, interval, range_start, range_end)
        if fetched:
            async with self._acquire_key(key):
                candles = self._merge_history(key, interval, fetched)
                self._last_fetch_at[key] = now_sec
                self._preloaded.add(key)
                await self._persist_history(symbol, interval)
                window = [bar for bar in candles if range_start <= int(bar.get("ts_ms_utc", 0)) <= range_end]
                if not window:
                    window = candles[-safe_limit:]
                return [dict(bar) for bar in window[-safe_limit:]]
        async with self._acquire_key(key):
            candles = self._history.get(key, []) or []
            if not candles:
                return []
            window = [bar for bar in candles if range_start <= int(bar.get("ts_ms_utc", 0)) <= range_end]
            if not window:
                window = candles[-safe_limit:]
            return [dict(bar) for bar in window[-safe_limit:]]

    async def fetch_gap(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Dict[str, float]]:
        key = self._key(symbol, interval)
        start = int(min(start_ms, end_ms))
        end = int(max(start_ms, end_ms))
        base_interval, ratio = self._resolve_base_interval(interval)
        if ratio > 1:
            base_ms = self._interval_ms(base_interval)
            padded_end = end + base_ms * max(0, ratio - 1)
            await self.fetch_gap(symbol, base_interval, start, padded_end)
            candles = await self._rebuild_aggregated_history(
                symbol,
                interval,
                range_start=start,
                range_end=end,
                limit=None,
            )
            agg_key = self._key(symbol, interval)
            if candles:
                self._preloaded.add(agg_key)
                await self._persist_history(symbol, interval)
            return [dict(bar) for bar in candles]

        interval_ms = self._interval_ms(interval)
        segments: List[Tuple[int, int]] = []
        async with self._acquire_key(key):
            await self._hydrate_from_store(key, symbol, interval)
            candles = list(self._history.get(key, []))
            if not candles:
                segments.append((start, end))
            else:
                try:
                    earliest = int(candles[0]["ts_ms_utc"])
                    latest = int(candles[-1]["ts_ms_utc"])
                except (KeyError, TypeError, ValueError, IndexError):
                    earliest = start
                    latest = start
                if start < earliest:
                    gap_end = min(end, earliest - interval_ms)
                    if gap_end >= start:
                        segments.append((start, gap_end))
                if end > latest:
                    gap_start = max(start, latest + interval_ms)
                    if gap_start <= end:
                        segments.append((gap_start, end))
        fetched_any = False
        for seg_start, seg_end in segments:
            if seg_start > seg_end:
                continue
            batch = await self._fetch_range(symbol, interval, seg_start, seg_end)
            if batch:
                async with self._acquire_key(key):
                    self._merge_history(key, interval, batch)
                    self._preloaded.add(key)
                    fetched_any = True
        async with self._acquire_key(key):
            if fetched_any:
                await self._persist_history(symbol, interval)
            return [
                dict(bar)
                for bar in self._history.get(key, [])
                if start <= int(bar.get("ts_ms_utc", 0)) <= end
            ]

    async def next_candle(
        self,
        symbol: str,
        interval: str,
        last_known: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        base_interval, ratio = self._resolve_base_interval(interval)
        if ratio > 1:
            await self.next_candle(symbol, base_interval)
            candles = await self._rebuild_aggregated_history(symbol, interval, limit=None)
            if not candles:
                return None
            latest = candles[-1]
            if last_known:
                try:
                    known_ts = int(last_known.get("ts_ms_utc", 0))
                except (TypeError, ValueError, AttributeError):
                    known_ts = 0
                if known_ts and int(latest.get("ts_ms_utc", 0)) <= known_ts:
                    return None
            return dict(latest)
        key = self._key(symbol, interval)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            history = self._history.get(key)
            if not history:
                history = await self._bootstrap_history(key, symbol, interval, 500)
            history = history or []
            last_time = int(history[-1]["ts_ms_utc"]) if history else 0
            if last_known:
                try:
                    known_ts = int(last_known.get("ts_ms_utc", last_time))
                except (TypeError, ValueError, AttributeError):
                    known_ts = last_time
                last_time = max(last_time, known_ts)
            now = time.time()
            retry_at = self._remote_retry_at.get(key, 0.0)
            min_fetch_delay = compute_refresh_seconds(
                interval,
                fraction=self._fetch_fraction,
                min_seconds=self._fetch_min_seconds,
                max_seconds=self._fetch_max_seconds,
            )
            last_fetch = self._last_fetch_at.get(key, 0.0)
            should_fetch = now >= retry_at and (now - last_fetch >= min_fetch_delay)
            new_bar: Optional[Dict[str, float]] = None
            fetched: List[Dict[str, float]] = []
            if should_fetch:
                try:
                    fetch_interval, ratio = self._resolve_base_interval(interval)
                    start_ms = last_time if last_time else None
                    if ratio > 1 and start_ms is not None:
                        start_ms = max(0, start_ms - self._interval_ms(fetch_interval) * ratio)
                    raw = await fetch_klines(
                        symbol,
                        fetch_interval,
                        limit=1000,
                        start_time=start_ms,
                    )
                    base_rows = self._normalize_binance(raw, fetch_interval)
                    if ratio > 1 and base_rows:
                        agg_rows = self._aggregate(base_rows, interval, limit=len(base_rows))
                        fetched = [
                            bar
                            for bar in agg_rows
                            if int(bar.get("ts_ms_utc", 0)) > last_time
                        ]
                    else:
                        fetched = [
                            bar
                            for bar in base_rows
                            if int(bar.get("ts_ms_utc", 0)) > last_time
                        ]
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("REST fetch failed for %s %s: %s", symbol, interval, exc)
                    self._remote_retry_at[key] = now + max(min_fetch_delay, 30.0)
                else:
                    self._last_fetch_at[key] = now
                    if fetched:
                        history = self._merge_history(key, interval, fetched)
                        await self._persist_history(symbol, interval)
                        latest = history[-1] if history else None
                        if latest and int(latest.get("ts_ms_utc", 0)) > last_time:
                            new_bar = latest
                    else:
                        history = self._history.get(key, history) or history
            if new_bar is None:
                return None
            if not history or int(history[-1].get("ts_ms_utc", 0)) != int(new_bar.get("ts_ms_utc", 0)):
                history = self._merge_history(key, interval, [new_bar])
                await self._persist_history(symbol, interval)
            return dict(new_bar)

    async def _bootstrap_history(
        self,
        key: Key,
        symbol: str,
        interval: str,
        limit: int,
        *,
        date_from_ms: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        preload_target = max(limit, self._preload_limit)
        async with self._acquire_key(key):
            await self._hydrate_from_store(key, symbol, interval)
            cached = list(self._history.get(key, []))
            if cached:
                return cached
        try:
            fetched = await self._preload_interval(symbol, interval, candles=preload_target)
        except Exception as exc:  # pragma: no cover - runtime fallback
            LOGGER.warning("Failed to bootstrap %s %s via REST: %s", symbol, interval, exc)
            fetched = []
        candles = self._filter_future(list(fetched), interval) if fetched else []
        async with self._acquire_key(key):
            if not candles:
                self._history[key] = []
                self._sim_state.pop(key, None)
                await self._clear_store(symbol, interval)
                return []
            merged = self._merge_history(key, interval, candles)
            if not merged:
                self._history[key] = []
                self._sim_state.pop(key, None)
                await self._clear_store(symbol, interval)
                return []
            await self._persist_history(symbol, interval)
            last = merged[-1]
            interval_ms = self._interval_ms(interval)
            seed = hash((symbol.upper(), interval)) & 0xFFFF
            self._sim_state[key] = SimulationState(
                last_ts_ms=int(last["ts_ms_utc"]),
                last_close=float(last["close"]),
                interval_ms=interval_ms,
                rng=random.Random(seed),
            )
            return merged

    async def _fetch_latest(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        *,
        date_from_ms: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        base_interval, ratio = self._resolve_base_interval(interval)
        minimum = max(limit * ratio, self._estimate_required(base_interval, date_from_ms))
        fetcher_fn = globals().get("fetch_klines", fetch_klines)
        raw_pages = await fetch_klines_paginated(
            symbol,
            base_interval,
            end_time=None,
            minimum=minimum,
            date_from_ms=date_from_ms,
            fetcher=fetcher_fn,
        )
        normalized = self._normalize_binance(raw_pages, base_interval)
        if normalized:
            if ratio > 1:
                agg_limit = max(len(normalized), self._estimate_required(interval, date_from_ms))
                aggregated = self._aggregate(normalized, interval, agg_limit)
                if date_from_ms is not None:
                    aggregated = [bar for bar in aggregated if int(bar["ts_ms_utc"]) >= int(date_from_ms)]
                if aggregated:
                    return aggregated[-limit:] if limit else aggregated
            return normalized[-limit:] if limit else normalized
        base_limit = min(1000, max(limit * ratio, ratio))
        raw = await fetch_klines(symbol, base_interval, limit=base_limit)
        normalized = self._normalize_binance(raw, base_interval)
        if normalized:
            if ratio > 1:
                aggregated = self._aggregate(normalized, interval, limit=len(normalized))
                if aggregated:
                    return aggregated[-limit:] if limit else aggregated
            return normalized[-limit:] if limit else normalized
        ccxt_bars = await self._fetch_ccxt(symbol, interval, limit)
        if ccxt_bars:
            return ccxt_bars
        raise RuntimeError("No market data available")

    async def _fetch_ccxt(self, symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
        if ccxt is None:  # pragma: no cover - optional dependency
            return []
        return await asyncio.to_thread(self._fetch_ccxt_sync, symbol, interval, limit)

    def _fetch_ccxt_sync(self, symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
        exchanges = ["binance", "bybit", "okx"]
        ccxt_symbol = self._to_ccxt_symbol(symbol)
        for name in exchanges:
            try:
                exchange = getattr(ccxt, name)({"enableRateLimit": True})  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                continue
            try:
                ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
            except Exception:
                ohlcv = []
            finally:
                try:
                    exchange.close()
                except Exception:  # pragma: no cover - cleanup best effort
                    pass
            if not ohlcv:
                continue
            return [
                {
                    "ts_ms_utc": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5] if len(row) > 5 else 0.0),
                    "close_time_ms_utc": int(row[0]),
                    "timestamp": datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc),
                }
                for row in ohlcv
            ]
        return []

    def _normalize_binance(self, data: Iterable[Iterable], interval: str) -> List[Dict[str, float]]:
        candles: Dict[int, Dict[str, float]] = {}
        for entry in data:
            try:
                open_time_ms = int(entry[0])
                open_price = float(entry[1])
                high_price = float(entry[2])
                low_price = float(entry[3])
                close_price = float(entry[4])
            except (IndexError, TypeError, ValueError):
                continue
            close_time_ms = open_time_ms
            if len(entry) > 6:
                try:
                    close_time_ms = int(entry[6])
                except (TypeError, ValueError):
                    close_time_ms = open_time_ms
            candle = {
                "ts_ms_utc": open_time_ms,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": float(entry[5]) if len(entry) > 5 else 0.0,
                "close_time_ms_utc": close_time_ms,
                "timestamp": datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc),
            }
            candles[int(open_time_ms)] = candle
        ordered = [candles[idx] for idx in sorted(candles)]
        return self._filter_future(ordered, interval)

    def _aggregate(
        self, candles: List[Dict[str, float]], interval: str, limit: int
    ) -> List[Dict[str, float]]:
        _, ratio = self._resolve_base_interval(interval)
        if ratio <= 1:
            return candles[-limit:] if limit else list(candles)
        if not candles:
            return []
        frame = pd.DataFrame(candles)
        if "ts_ms_utc" not in frame.columns:
            return []
        frame = frame.sort_values("ts_ms_utc").copy()
        frame["ts"] = pd.to_datetime(frame["ts_ms_utc"], unit="ms", utc=True)
        frame.set_index("ts", inplace=True)
        interval_ms = self._interval_ms(interval)
        freq = pd.to_timedelta(interval_ms, unit="ms")
        resampled = frame.resample(freq, origin="epoch", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "close_time_ms_utc": "last",
            }
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        if limit:
            resampled = resampled.tail(limit)
        aggregated: List[Dict[str, float]] = []
        for ts, row in resampled.iterrows():
            ts_obj = ts.to_pydatetime()
            open_ts_ms = int(ts_obj.timestamp() * 1000)
            close_ts_ms = int(row.get("close_time_ms_utc") or (open_ts_ms + interval_ms - 1))
            aggregated.append(
                {
                    "ts_ms_utc": open_ts_ms,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                    "close_time_ms_utc": close_ts_ms,
                    "timestamp": ts_obj,
                }
            )
        return self._filter_future(aggregated, interval)

    def _generate_history(self, symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
        interval_seconds = interval_to_seconds(interval) or 60.0
        interval_ms = int(max(1.0, interval_seconds) * 1000)
        now = int(time.time() * 1000)
        start = now - interval_ms * max(limit - 1, 1)
        rng = random.Random(hash((symbol.upper(), interval)) & 0xFFFF)
        price = 100.0 + (rng.random() - 0.5) * 10
        candles: List[Dict[str, float]] = []
        for i in range(limit):
            ts = start + interval_ms * i
            candle = self._simulate_bar(ts, price, rng, interval_ms=interval_ms)
            candles.append(candle)
            price = candle["close"]
        return candles

    def _simulate_next(self, key: Key, symbol: str, interval: str) -> Dict[str, float]:
        state = self._sim_state.get(key)
        if state is None:
            interval_seconds = interval_to_seconds(interval) or 60.0
            seed = hash((symbol.upper(), interval)) & 0xFFFF
            state = SimulationState(
                last_ts_ms=int(time.time() * 1000),
                last_close=100.0,
                interval_ms=int(max(1.0, interval_seconds) * 1000),
                rng=random.Random(seed),
            )
            self._sim_state[key] = state
        next_time = int(state.last_ts_ms + max(1, state.interval_ms))
        candle = self._simulate_bar(next_time, state.last_close, state.rng, interval_ms=state.interval_ms)
        state.last_ts_ms = int(candle["ts_ms_utc"])
        state.last_close = float(candle["close"])
        return candle

    def _simulate_bar(
        self, ts: int, base_price: float, rng: random.Random, interval_ms: Optional[int] = None
    ) -> Dict[str, float]:
        drift = (rng.random() - 0.5) * max(base_price * 0.01, 0.5)
        close = max(1e-6, base_price + drift)
        spread = abs(drift) + rng.random() * max(base_price * 0.005, 0.1)
        high = max(base_price, close) + spread * 0.5
        low = min(base_price, close) - spread * 0.5
        volume = abs(drift) * 500 + rng.random() * 50
        close_ts = int(ts + max(1, int(interval_ms or 0)) - 1) if interval_ms else int(ts)
        return {
            "ts_ms_utc": int(ts),
            "open": float(base_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
            "close_time_ms_utc": close_ts,
            "timestamp": datetime.fromtimestamp(int(ts) / 1000.0, tz=timezone.utc),
        }

    def _key(self, symbol: str, interval: str) -> Key:
        return symbol.upper(), interval

    def _to_ccxt_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol
        symbol = symbol.upper()
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        if symbol.endswith("USD"):
            return f"{symbol[:-3]}/USD"
        if symbol.endswith("USDC"):
            return f"{symbol[:-4]}/USDC"
        if len(symbol) > 3:
            return f"{symbol[:-3]}/{symbol[-3:]}"
        return symbol
