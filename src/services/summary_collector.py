"""Helpers for filling recent OHLCV gaps for inspection summary builds."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from .candles_repository import CandleRepository, UpsertStats, get_repository
from .ohlc_sanitizer import sanitize_candles
from .ohlc import TIMEFRAME_TO_MS
from .timeutils import ensure_ms_epoch
from .check_all_datas import _normalise_binance_row

LOGGER = logging.getLogger(__name__)
UTC = timezone.utc

BINANCE_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
MAX_PAGE_LIMIT = 1000
DEFAULT_TOKEN_RATE = 30.0  # tokens per second
DEFAULT_TOKEN_BURST = 30
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


@dataclass(slots=True)
class CollectionSummary:
    symbol: str
    start_ms: int
    end_ms: int
    intervals: Dict[str, IntervalSummary]
    requests: int
    candles_written: int
    dropped_candles: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "range": {"start_ms": self.start_ms, "end_ms": self.end_ms},
            "requests": self.requests,
            "candles_written": self.candles_written,
            "dropped_candles": self.dropped_candles,
            "intervals": {
                tf: {
                    "gaps_total": summary.gaps_total,
                    "gaps_filled": summary.gaps_filled,
                    "candles_written": summary.candles_written,
                    "dropped_candles": summary.dropped_candles,
                    "requests": summary.requests,
                    "remaining_gaps": summary.remaining_gaps,
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
            if attempt >= 5:
                raise GapCollectionError(f"Request failure: {exc}") from exc
            await asyncio.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, BACKOFF_MAX_MS)
            continue

        if response.status_code in {418, 429}:
            await asyncio.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, BACKOFF_MAX_MS)
            continue

        if response.status_code >= 500:
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
) -> IntervalSummary:
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
        raw_rows = await _request_klines(
            client,
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=page_end + interval_ms,
            limit=MAX_PAGE_LIMIT,
            bucket=bucket,
        )
        requests += 1
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
        stats, sanitised = await _upsert_sanitised(
            repository,
            symbol=symbol,
            interval=interval,
            candles=normalised,
            stage=f"summary_fetch.{interval}",
        )
        written += stats.written
        dropped_ts += stats.dropped_ts
        dropped_ohlc += stats.dropped_ohlc

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
    )


async def collect_recent_summary(
    symbol: str,
    *,
    days: int = 3,
    end_ms: Optional[int] = None,
    intervals: Optional[Sequence[str]] = None,
    repository: Optional[CandleRepository] = None,
    start_ms: Optional[int] = None,
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

    bucket = _TokenBucket(DEFAULT_TOKEN_RATE, DEFAULT_TOKEN_BURST)
    async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0)) as client:
        for interval in target_intervals:
            interval_ms = TIMEFRAME_TO_MS.get(interval)
            if not interval_ms:
                continue
            first_expected = _align_to_interval(start_ms, interval_ms)
            last_expected = _align_to_interval(now_ms, interval_ms)
            existing = await _fetch_existing(
                repo,
                symbol=symbol,
                interval=interval,
                start_ms=first_expected,
                end_ms=last_expected,
            )
            gaps = _compute_gaps(first_expected, last_expected, interval_ms, existing)
            if not gaps:
                summaries[interval] = IntervalSummary(
                    gaps_total=0,
                    gaps_filled=0,
                    candles_written=0,
                    dropped_candles=0,
                    requests=0,
                    remaining_gaps=[],
                )
                continue

            gap_summaries: List[IntervalSummary] = []
            for gap in gaps:
                summary = await _fill_gap(
                    repo,
                    symbol=symbol,
                    interval=interval,
                    gap=gap,
                    interval_ms=interval_ms,
                    client=client,
                    bucket=bucket,
                )
                gap_summaries.append(summary)
                total_requests += summary.requests
                total_written += summary.candles_written
                total_dropped += summary.dropped_candles

            refreshed = await _fetch_existing(
                repo,
                symbol=symbol,
                interval=interval,
                start_ms=first_expected,
                end_ms=last_expected,
            )
            remaining = _compute_gaps(first_expected, last_expected, interval_ms, refreshed)
            summaries[interval] = IntervalSummary(
                gaps_total=len(gaps),
                gaps_filled=sum(1 for item in gap_summaries if item.candles_written > 0),
                candles_written=sum(item.candles_written for item in gap_summaries),
                dropped_candles=sum(item.dropped_candles for item in gap_summaries),
                requests=sum(item.requests for item in gap_summaries),
                remaining_gaps=remaining,
            )

    return CollectionSummary(
        symbol=symbol.upper(),
        start_ms=start_ms,
        end_ms=now_ms,
        intervals=summaries,
        requests=total_requests,
        candles_written=total_written,
        dropped_candles=total_dropped,
    )


__all__ = ["collect_recent_summary", "CollectionSummary"]
