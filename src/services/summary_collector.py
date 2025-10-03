"""Helpers for filling recent OHLCV gaps for inspection summary builds."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

from .candles_repository import CandleRepository, UpsertStats, get_repository
from .ohlc_sanitizer import sanitize_candles
from .ohlc import TIMEFRAME_TO_MS
from .timeutils import ensure_ms_epoch
from .check_all_datas import _normalise_binance_row
from .tracing import TraceContext, log_event, new_rid

LOGGER = logging.getLogger(__name__)
UTC = timezone.utc

BINANCE_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
MAX_PAGE_LIMIT = 1000
DEFAULT_TOKEN_RATE = 75.0  # tokens per second
DEFAULT_TOKEN_BURST = 150
MAX_CONCURRENCY = 4
GAP_CONCURRENCY = 2
RATE_DELAY_MIN = 0.020
RATE_DELAY_MAX = 0.040
BACKOFF_BASE_MS = 0.2
BACKOFF_MAX_MS = 1.6


class GapCollectionError(RuntimeError):
    """Raised when an unexpected failure occurs while filling gaps."""


@dataclass(slots=True)
class IntervalSummary:
    gaps_total: int
    gaps_filled: int
    candles_written: int
    dropped_candles: int
    requests: int
    remaining_gaps: List[Dict[str, int]]
    fetch_ms: float
    db_write_ms: float


@dataclass(slots=True)
class CollectionSummary:
    symbol: str
    start_ms: int
    end_ms: int
    intervals: Dict[str, IntervalSummary]
    requests: int
    candles_written: int
    dropped_candles: int
    fetch_ms: float
    db_write_ms: float
    compute_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "range": {"start_ms": self.start_ms, "end_ms": self.end_ms},
            "requests": self.requests,
            "candles_written": self.candles_written,
            "dropped_candles": self.dropped_candles,
            "fetch_ms": self.fetch_ms,
            "db_write_ms": self.db_write_ms,
            "compute_ms": self.compute_ms,
            "intervals": {
                tf: {
                    "gaps_total": summary.gaps_total,
                    "gaps_filled": summary.gaps_filled,
                    "candles_written": summary.candles_written,
                    "dropped_candles": summary.dropped_candles,
                    "requests": summary.requests,
                    "remaining_gaps": summary.remaining_gaps,
                    "fetch_ms": summary.fetch_ms,
                    "db_write_ms": summary.db_write_ms,
                }
                for tf, summary in self.intervals.items()
            },
        }


class _TokenBucket:
    __slots__ = ("_rate", "_capacity", "_tokens", "_updated")

    def __init__(self, rate: float, capacity: int) -> None:
        self._rate = max(0.1, float(rate))
        self._capacity = max(1, int(capacity))
        self._tokens = float(self._capacity)
        self._updated = time.monotonic()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = max(0.0, now - self._updated)
            self._updated = now
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate,
            )
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            deficit = 1.0 - self._tokens
            delay = max(deficit / self._rate, RATE_DELAY_MIN)
            await asyncio.sleep(min(delay, RATE_DELAY_MAX))


async def _request_klines(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    bucket: _TokenBucket,
    trace: TraceContext | None = None,
    rid: str | None = None,
    window: Mapping[str, int] | None = None,
) -> List[Sequence[Any]]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": str(limit),
    }
    backoff = BACKOFF_BASE_MS
    for attempt in range(6):
        await bucket.acquire()
        try:
            response = await client.get(BINANCE_ENDPOINT, params=params)
        except httpx.RequestError as exc:  # pragma: no cover - network failure
            if trace:
                log_event(
                    level="WARN",
                    event="fetch.retry",
                    cid=trace.cid,
                    rid=rid,
                    user_action=trace.user_action,
                    symbol=symbol,
                    tf=interval,
                    window=window,
                    details=f"request_error:{exc.__class__.__name__}",
                    metrics={"attempt": attempt + 1},
                )
            if attempt >= 5:
                raise GapCollectionError(f"Request failure: {exc}") from exc
            await asyncio.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, BACKOFF_MAX_MS)
            continue

        if response.status_code in {418, 429}:
            if trace:
                log_event(
                    level="WARN",
                    event="fetch.rate_limited",
                    cid=trace.cid,
                    rid=rid,
                    user_action=trace.user_action,
                    symbol=symbol,
                    tf=interval,
                    window=window,
                    details=f"http_{response.status_code}",
                    metrics={"backoff_ms": backoff * 1000.0, "attempt": attempt + 1},
                )
            await asyncio.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, BACKOFF_MAX_MS)
            continue

        if response.status_code >= 500:
            if trace:
                log_event(
                    level="WARN",
                    event="fetch.retry",
                    cid=trace.cid,
                    rid=rid,
                    user_action=trace.user_action,
                    symbol=symbol,
                    tf=interval,
                    window=window,
                    details=f"http_{response.status_code}",
                    metrics={"attempt": attempt + 1, "backoff_ms": backoff * 1000.0},
                )
            if attempt >= 5:
                raise GapCollectionError(f"Binance error {response.status_code}")
            await asyncio.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, BACKOFF_MAX_MS)
            continue

        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        await asyncio.sleep(random.uniform(RATE_DELAY_MIN, RATE_DELAY_MAX))
        return payload  # type: ignore[return-value]
    return []


def _align_to_interval(timestamp_ms: int, interval_ms: int) -> int:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (timestamp_ms // interval_ms) * interval_ms


def _compute_gaps(
    first_expected: int,
    last_expected: int,
    interval_ms: int,
    existing: Sequence[int],
) -> List[Dict[str, int]]:
    if first_expected > last_expected:
        return []
    ordered = sorted({ts for ts in existing if first_expected <= ts <= last_expected})
    gaps: List[Dict[str, int]] = []
    cursor = first_expected
    for open_ms in ordered:
        if open_ms < cursor:
            continue
        if open_ms > cursor:
            gap_start = cursor
            gap_end = open_ms - interval_ms
            if gap_end >= gap_start:
                count = int((gap_end - gap_start) // interval_ms + 1)
                gaps.append({"from": gap_start, "to": gap_end, "count": count})
        cursor = open_ms + interval_ms
    if cursor <= last_expected:
        gap_start = cursor
        gap_end = last_expected
        count = int((gap_end - gap_start) // interval_ms + 1)
        gaps.append({"from": gap_start, "to": gap_end, "count": count})
    return gaps


async def _upsert_sanitised(
    repository: CandleRepository,
    *,
    symbol: str,
    interval: str,
    candles: Sequence[Mapping[str, Any]],
    stage: str,
) -> tuple[UpsertStats, List[Dict[str, Any]]]:
    if not candles:
        return UpsertStats(written=0, dropped_ts=0, dropped_ohlc=0), []
    sanitised = sanitize_candles(candles, stage=stage)
    stats = await asyncio.to_thread(
        repository.upsert_candles,
        symbol,
        interval,
        sanitised.candles,
        stage=stage,
    )
    return stats, sanitised.candles


async def _fetch_existing(
    repository: CandleRepository,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[int]:
    return await asyncio.to_thread(
        repository.fetch_open_times,
        symbol,
        interval,
        start_ms,
        end_ms,
    )


async def _fill_gap(
    repository: CandleRepository,
    *,
    symbol: str,
    interval: str,
    gap: Mapping[str, int],
    interval_ms: int,
    client: httpx.AsyncClient,
    bucket: _TokenBucket,
    trace: TraceContext | None = None,
) -> IntervalSummary:
    fetch_ms = 0.0
    db_write_ms = 0.0
    gap_start = int(gap["from"])
    gap_end = int(gap["to"])
    progress = await asyncio.to_thread(
        repository.load_gap_progress,
        symbol,
        interval,
        gap_start,
    )
    if progress is not None:
        resume_from = max(gap_start, progress + interval_ms)
    else:
        resume_from = gap_start

    page_span = interval_ms * MAX_PAGE_LIMIT
    cursor = resume_from
    written = 0
    dropped_ts = 0
    dropped_ohlc = 0
    requests = 0

    while cursor <= gap_end:
        page_end = min(gap_end, cursor + page_span - interval_ms)
        # Match the ChartGapViewer strategy by sizing the Binance page limit to the
        # actual gap width.  This keeps requests tight to the missing window while
        # still respecting the hard 1000 candle ceiling enforced by the REST API.
        approx_bars = max(1, ((page_end - cursor) // interval_ms) + 1)
        page_limit = min(MAX_PAGE_LIMIT, max(approx_bars, 50))
        batch_window = {"from_ms": cursor, "to_ms": page_end}
        batch_rid = trace.new_rid() if trace else new_rid()
        if trace:
            log_event(
                level="INFO",
                event="fetch.batch_start",
                cid=trace.cid,
                rid=batch_rid,
                user_action=trace.user_action,
                symbol=symbol,
                tf=interval,
                window=batch_window,
                metrics={"expected_bars": approx_bars, "limit": page_limit},
            )
        request_start = time.perf_counter()
        request_kwargs = {
            "symbol": symbol,
            "interval": interval,
            "start_ms": cursor,
            "end_ms": page_end + interval_ms,
            "limit": page_limit,
            "bucket": bucket,
        }
        if trace:
            request_kwargs.update({"trace": trace, "rid": batch_rid, "window": batch_window})
        raw_rows = await _request_klines(client, **request_kwargs)
        elapsed_ms = (time.perf_counter() - request_start) * 1000.0
        fetch_ms += elapsed_ms
        requests += 1
        if trace:
            log_event(
                level="INFO",
                event="fetch.batch_done",
                cid=trace.cid,
                rid=batch_rid,
                user_action=trace.user_action,
                symbol=symbol,
                tf=interval,
                window=batch_window,
                metrics={"ms": elapsed_ms, "bars": len(raw_rows) if raw_rows else 0},
            )
        if not raw_rows:
            cursor += page_span
            continue

        normalised: List[Dict[str, Any]] = []
        for row in raw_rows:
            candle = _normalise_binance_row(row)
            if candle is None:
                continue
            ts_raw = ensure_ms_epoch(candle.get("t"))
            if ts_raw is None:
                continue
            aligned = _align_to_interval(ts_raw, interval_ms)
            if aligned < gap_start or aligned > gap_end:
                continue
            candle["t"] = aligned
            normalised.append(candle)

        if not normalised:
            cursor += page_span
            continue

        dedup: Dict[int, Dict[str, Any]] = {}
        for candle in normalised:
            dedup[int(candle["t"])] = candle
        normalised = [dedup[key] for key in sorted(dedup.keys())]
        write_start = time.perf_counter()
        stats, sanitised = await _upsert_sanitised(
            repository,
            symbol=symbol,
            interval=interval,
            candles=normalised,
            stage=f"summary_fetch.{interval}",
        )
        db_elapsed = (time.perf_counter() - write_start) * 1000.0
        db_write_ms += db_elapsed
        written += stats.written
        dropped_ts += stats.dropped_ts
        dropped_ohlc += stats.dropped_ohlc
        if trace:
            log_event(
                level="INFO",
                event="db.bulk_upsert",
                cid=trace.cid,
                rid=new_rid(),
                user_action=trace.user_action,
                symbol=symbol,
                tf=interval,
                window=batch_window,
                metrics={
                    "rows": stats.written,
                    "dropped": stats.dropped_ts + stats.dropped_ohlc,
                    "ms": db_elapsed,
                },
            )

        if not sanitised:
            cursor += page_span
            continue

        last_open = sanitised[-1]["t"]
        await asyncio.to_thread(
            repository.update_gap_progress,
            symbol,
            interval,
            gap_start,
            last_open,
        )
        cursor = last_open + interval_ms

    await asyncio.to_thread(repository.clear_gap_progress, symbol, interval, gap_start)

    return IntervalSummary(
        gaps_total=1,
        gaps_filled=1 if written > 0 else 0,
        candles_written=written,
        dropped_candles=dropped_ts + dropped_ohlc,
        requests=requests,
        remaining_gaps=[],
        fetch_ms=fetch_ms,
        db_write_ms=db_write_ms,
    )


async def collect_recent_summary(
    symbol: str,
    *,
    days: int = 3,
    end_ms: Optional[int] = None,
    intervals: Optional[Sequence[str]] = None,
    repository: Optional[CandleRepository] = None,
    start_ms: Optional[int] = None,
    trace: TraceContext | None = None,
) -> CollectionSummary:
    """Ensure recent OHLC coverage exists for the requested symbol."""

    repo = repository or get_repository()
    now_ms = end_ms if end_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    span_days = max(1, int(days))
    if start_ms is not None:
        try:
            start_candidate = int(start_ms)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            start_candidate = now_ms - span_days * 86_400_000
        start_ms = max(0, start_candidate)
    else:
        start_ms = max(0, now_ms - span_days * 86_400_000)
    if start_ms > now_ms:
        start_ms = now_ms

    target_intervals = list(intervals or TIMEFRAME_TO_MS.keys())
    summaries: Dict[str, IntervalSummary] = {}
    total_requests = 0
    total_written = 0
    total_dropped = 0
    total_fetch_ms = 0.0
    total_db_write_ms = 0.0
    compute_start = time.perf_counter()

    bucket = _TokenBucket(DEFAULT_TOKEN_RATE, DEFAULT_TOKEN_BURST)
    async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0)) as client:
        interval_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def _collect_for_interval(interval: str) -> Tuple[str, IntervalSummary]:
            async with interval_semaphore:
                interval_ms = TIMEFRAME_TO_MS.get(interval)
                if not interval_ms:
                    empty_summary = IntervalSummary(
                        gaps_total=0,
                        gaps_filled=0,
                        candles_written=0,
                        dropped_candles=0,
                        requests=0,
                        remaining_gaps=[],
                        fetch_ms=0.0,
                        db_write_ms=0.0,
                    )
                    return interval, empty_summary

                first_expected = _align_to_interval(start_ms, interval_ms)
                last_expected = _align_to_interval(now_ms, interval_ms)
                if end_ms is None:
                    tail_open = now_ms - last_expected < interval_ms
                    last_closed = last_expected - interval_ms if tail_open else last_expected
                else:
                    last_closed = last_expected
                if last_closed < first_expected:
                    empty_summary = IntervalSummary(
                        gaps_total=0,
                        gaps_filled=0,
                        candles_written=0,
                        dropped_candles=0,
                        requests=0,
                        remaining_gaps=[],
                        fetch_ms=0.0,
                        db_write_ms=0.0,
                    )
                    return interval, empty_summary

                fetch_existing_start = time.perf_counter()
                existing = await _fetch_existing(
                    repo,
                    symbol=symbol,
                    interval=interval,
                    start_ms=first_expected,
                    end_ms=last_closed,
                )
                fetch_existing_ms = (time.perf_counter() - fetch_existing_start) * 1000.0
                if trace:
                    log_event(
                        level="INFO",
                        event="data.availability_checked",
                        cid=trace.cid,
                        rid=trace.new_rid(),
                        user_action=trace.user_action,
                        symbol=symbol,
                        tf=interval,
                        window={"from_ms": first_expected, "to_ms": last_closed},
                        metrics={"bars": len(existing), "ms": fetch_existing_ms},
                    )
                gaps = _compute_gaps(first_expected, last_closed, interval_ms, existing)
                if trace and gaps:
                    total_minutes = sum(
                        int(gap["count"]) * (interval_ms / 60_000) for gap in gaps
                    )
                    log_event(
                        level="INFO",
                        event="gaps.detected",
                        cid=trace.cid,
                        rid=trace.new_rid(),
                        user_action=trace.user_action,
                        symbol=symbol,
                        tf=interval,
                        window={"from_ms": first_expected, "to_ms": last_closed},
                        metrics={"gaps": len(gaps), "minutes": total_minutes},
                    )
                if not gaps:
                    summary = IntervalSummary(
                        gaps_total=0,
                        gaps_filled=0,
                        candles_written=0,
                        dropped_candles=0,
                        requests=0,
                        remaining_gaps=[],
                        fetch_ms=fetch_existing_ms,
                        db_write_ms=0.0,
                    )
                    return interval, summary

                merged: List[Dict[str, int]] = []
                current: Dict[str, int] | None = None
                min_merge = max(15 * 60_000, interval_ms)
                for gap in gaps:
                    gap_from = int(gap["from"])
                    gap_to = int(gap["to"])
                    if current is None:
                        current = {"from": gap_from, "to": gap_to}
                        continue
                    if gap_from - current["to"] <= min_merge:
                        current["to"] = max(current["to"], gap_to)
                    else:
                        merged.append(current)
                        current = {"from": gap_from, "to": gap_to}
                if current is not None:
                    merged.append(current)
                if trace and merged:
                    merged_minutes = sum(
                        max(0, (item["to"] - item["from"]) // 60_000 + 1)
                        * (interval_ms / 60_000)
                        for item in merged
                    )
                    log_event(
                        level="DEBUG",
                        event="gaps.merged",
                        cid=trace.cid,
                        rid=trace.new_rid(),
                        user_action=trace.user_action,
                        symbol=symbol,
                        tf=interval,
                        window={"from_ms": first_expected, "to_ms": last_closed},
                        metrics={"windows": len(merged), "minutes": merged_minutes},
                    )

                gap_semaphore = asyncio.Semaphore(max(1, GAP_CONCURRENCY))

                async def _fill_single_gap(merged_gap: Mapping[str, int]) -> IntervalSummary:
                    async with gap_semaphore:
                        return await _fill_gap(
                            repo,
                            symbol=symbol,
                            interval=interval,
                            gap=merged_gap,
                            interval_ms=interval_ms,
                            client=client,
                            bucket=bucket,
                            trace=trace,
                        )

                gap_tasks = [_fill_single_gap(gap) for gap in merged]
                gap_summaries = await asyncio.gather(*gap_tasks) if gap_tasks else []
                refreshed_start = time.perf_counter()
                refreshed = await _fetch_existing(
                    repo,
                    symbol=symbol,
                    interval=interval,
                    start_ms=first_expected,
                    end_ms=last_closed,
                )
                refreshed_ms = (time.perf_counter() - refreshed_start) * 1000.0
                remaining = _compute_gaps(first_expected, last_closed, interval_ms, refreshed)
                summary = IntervalSummary(
                    gaps_total=len(gaps),
                    gaps_filled=sum(1 for item in gap_summaries if item.candles_written > 0),
                    candles_written=sum(item.candles_written for item in gap_summaries),
                    dropped_candles=sum(item.dropped_candles for item in gap_summaries),
                    requests=sum(item.requests for item in gap_summaries),
                    remaining_gaps=remaining,
                    fetch_ms=fetch_existing_ms + sum(item.fetch_ms for item in gap_summaries) + refreshed_ms,
                    db_write_ms=sum(item.db_write_ms for item in gap_summaries),
                )
                return interval, summary

        interval_tasks = [_collect_for_interval(interval) for interval in target_intervals]
        for interval_task in asyncio.as_completed(interval_tasks):
            interval, summary = await interval_task
            summaries[interval] = summary
            total_requests += summary.requests
            total_written += summary.candles_written
            total_dropped += summary.dropped_candles
            total_fetch_ms += summary.fetch_ms
            total_db_write_ms += summary.db_write_ms

    compute_ms = (time.perf_counter() - compute_start) * 1000.0
    collection = CollectionSummary(
        symbol=symbol.upper(),
        start_ms=start_ms,
        end_ms=now_ms,
        intervals=summaries,
        requests=total_requests,
        candles_written=total_written,
        dropped_candles=total_dropped,
        fetch_ms=total_fetch_ms,
        db_write_ms=total_db_write_ms,
        compute_ms=compute_ms,
    )

    bars_by_tf = {tf: summary.candles_written for tf, summary in summaries.items()}
    LOGGER.info(
        "summary.collect",
        extra={
            "symbol": symbol.upper(),
            "requests": total_requests,
            "candles_written": total_written,
            "dropped_candles": total_dropped,
            "bars_written_by_tf": bars_by_tf,
            "fetch_ms": round(collection.fetch_ms, 3),
            "db_write_ms": round(collection.db_write_ms, 3),
        },
    )

    return collection


__all__ = ["collect_recent_summary", "CollectionSummary"]
