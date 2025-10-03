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

from . import tracing
from .candles_repository import CandleRepository, UpsertStats, get_repository
from .ohlc_sanitizer import sanitize_candles
from .ohlc import TIMEFRAME_TO_MS
from .timeutils import ensure_ms_epoch
from .check_all_datas import _normalise_binance_row
from .progress import ProgressReporter, emit_progress

LOGGER = logging.getLogger(__name__)
TRACE_LOGGER = tracing.LOGGER.getChild("summary_collector")
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
    TRACE_LOGGER.debug(
        "summary_collector:upsert_sanitised",
        extra={
            "symbol": symbol,
            "interval": interval,
            "stage": stage,
            "incoming": len(candles),
            "sanitised": len(sanitised.candles),
            "dropped_ts": getattr(sanitised, "invalid_ts", 0),
            "dropped_ohlc": getattr(sanitised, "invalid_ohlc", 0),
        },
    )
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
    progress: Optional[ProgressReporter] = None,
) -> IntervalSummary:
    gap_start = int(gap["from"])
    gap_end = int(gap["to"])
    resume_marker = await asyncio.to_thread(
        repository.load_gap_progress,
        symbol,
        interval,
        gap_start,
    )
    if resume_marker is not None:
        resume_from = max(gap_start, resume_marker + interval_ms)
    else:
        resume_from = gap_start

    page_span = interval_ms * MAX_PAGE_LIMIT
    cursor = resume_from
    written = 0
    dropped_ts = 0
    dropped_ohlc = 0
    requests = 0

    await emit_progress(
        progress,
        "summary_collector:gap_begin",
        symbol=symbol,
        interval=interval,
        gap_start=gap_start,
        gap_end=gap_end,
        resume_from=resume_from,
    )

    while cursor <= gap_end:
        page_end = min(gap_end, cursor + page_span - interval_ms)
        # Match the ChartGapViewer strategy by sizing the Binance page limit to the
        # actual gap width.  This keeps requests tight to the missing window while
        # still respecting the hard 1000 candle ceiling enforced by the REST API.
        approx_bars = max(1, ((page_end - cursor) // interval_ms) + 1)
        page_limit = min(MAX_PAGE_LIMIT, max(approx_bars, 50))
        raw_rows = await _request_klines(
            client,
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=page_end + interval_ms,
            limit=page_limit,
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

        TRACE_LOGGER.debug(
            "summary_collector:normalised_batch",
            extra={
                "symbol": symbol,
                "interval": interval,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "requested_limit": page_limit,
                "raw_rows": len(raw_rows),
                "normalised_rows": len(normalised),
            },
        )
        await emit_progress(
            progress,
            "summary_collector:normalised_batch",
            symbol=symbol,
            interval=interval,
            gap_start=gap_start,
            gap_end=gap_end,
            requested_limit=page_limit,
            raw_rows=len(raw_rows),
            normalised_rows=len(normalised),
        )

        if not normalised:
            cursor += page_span
            continue

        dedup: Dict[int, Dict[str, Any]] = {}
        for candle in normalised:
            dedup[int(candle["t"])] = candle
        normalised = [dedup[key] for key in sorted(dedup.keys())]
        TRACE_LOGGER.debug(
            "summary_collector:deduplicated_batch",
            extra={
                "symbol": symbol,
                "interval": interval,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "deduped_rows": len(normalised),
            },
        )
        await emit_progress(
            progress,
            "summary_collector:deduplicated_batch",
            symbol=symbol,
            interval=interval,
            gap_start=gap_start,
            gap_end=gap_end,
            deduped_rows=len(normalised),
        )
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

        TRACE_LOGGER.debug(
            "summary_collector:batch_persisted",
            extra={
                "symbol": symbol,
                "interval": interval,
                "written": stats.written,
                "dropped_ts": stats.dropped_ts,
                "dropped_ohlc": stats.dropped_ohlc,
            },
        )
        await emit_progress(
            progress,
            "summary_collector:batch_persisted",
            symbol=symbol,
            interval=interval,
            written=stats.written,
            dropped_ts=stats.dropped_ts,
            dropped_ohlc=stats.dropped_ohlc,
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

        TRACE_LOGGER.debug(
            "summary_collector:gap_progress_updated",
            extra={
                "symbol": symbol,
                "interval": interval,
                "gap_start": gap_start,
                "last_open": last_open,
            },
        )
        await emit_progress(
            progress,
            "summary_collector:gap_progress_updated",
            symbol=symbol,
            interval=interval,
            gap_start=gap_start,
            last_open=last_open,
        )

    await asyncio.to_thread(repository.clear_gap_progress, symbol, interval, gap_start)

    TRACE_LOGGER.debug(
        "summary_collector:gap_completed",
        extra={
            "symbol": symbol,
            "interval": interval,
            "gap_start": gap_start,
            "gap_end": gap_end,
            "written": written,
            "dropped_ts": dropped_ts,
            "dropped_ohlc": dropped_ohlc,
            "requests": requests,
        },
    )
    await emit_progress(
        progress,
        "summary_collector:gap_completed",
        symbol=symbol,
        interval=interval,
        gap_start=gap_start,
        gap_end=gap_end,
        written=written,
        dropped_ts=dropped_ts,
        dropped_ohlc=dropped_ohlc,
        requests=requests,
    )

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
    progress: Optional[ProgressReporter] = None,
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

    TRACE_LOGGER.debug(
        "summary_collector:start",
        extra={
            "symbol": symbol,
            "start_ms": start_ms,
            "end_ms": now_ms,
            "days": span_days,
        },
    )
    await emit_progress(
        progress,
        "summary_collector:start",
        symbol=symbol,
        start_ms=start_ms,
        end_ms=now_ms,
        days=span_days,
    )

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
            if end_ms is None:
                # Live invocations should not expect the still-forming candle at the
                # window tail.  Skipping that bar prevents an endless loop where
                # the collector keeps re-requesting the most recent minute even
                # though the exchange has not closed it yet.
                tail_open = now_ms - last_expected < interval_ms
                last_closed = last_expected - interval_ms if tail_open else last_expected
            else:
                last_closed = last_expected
            if last_closed < first_expected:
                summaries[interval] = IntervalSummary(
                    gaps_total=0,
                    gaps_filled=0,
                    candles_written=0,
                    dropped_candles=0,
                    requests=0,
                    remaining_gaps=[],
                )
                TRACE_LOGGER.debug(
                    "summary_collector:interval_skipped",
                    extra={
                        "symbol": symbol,
                        "interval": interval,
                        "first_expected": first_expected,
                        "last_closed": last_closed,
                    },
                )
                await emit_progress(
                    progress,
                    "summary_collector:interval_skipped",
                    symbol=symbol,
                    interval=interval,
                    first_expected=first_expected,
                    last_closed=last_closed,
                )
                continue
            existing = await _fetch_existing(
                repo,
                symbol=symbol,
                interval=interval,
                start_ms=first_expected,
                end_ms=last_closed,
            )
            TRACE_LOGGER.debug(
                "summary_collector:fetched_existing",
                extra={
                    "symbol": symbol,
                    "interval": interval,
                    "first_expected": first_expected,
                    "last_closed": last_closed,
                    "existing": len(existing),
                },
            )
            await emit_progress(
                progress,
                "summary_collector:fetched_existing",
                symbol=symbol,
                interval=interval,
                first_expected=first_expected,
                last_closed=last_closed,
                existing=len(existing),
            )
            gaps = _compute_gaps(first_expected, last_closed, interval_ms, existing)
            TRACE_LOGGER.debug(
                "summary_collector:computed_gaps",
                extra={
                    "symbol": symbol,
                    "interval": interval,
                    "gap_count": len(gaps),
                    "first_expected": first_expected,
                    "last_closed": last_closed,
                },
            )
            await emit_progress(
                progress,
                "summary_collector:computed_gaps",
                symbol=symbol,
                interval=interval,
                gap_count=len(gaps),
                first_expected=first_expected,
                last_closed=last_closed,
            )
            if not gaps:
                summaries[interval] = IntervalSummary(
                    gaps_total=0,
                    gaps_filled=0,
                    candles_written=0,
                    dropped_candles=0,
                    requests=0,
                    remaining_gaps=[],
                )
                await emit_progress(
                    progress,
                    "summary_collector:interval_complete",
                    symbol=symbol,
                    interval=interval,
                    first_expected=first_expected,
                    last_closed=last_closed,
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
                    progress=progress,
                )
                gap_summaries.append(summary)
                total_requests += summary.requests
                total_written += summary.candles_written
                total_dropped += summary.dropped_candles
                TRACE_LOGGER.debug(
                    "summary_collector:interval_progress",
                    extra={
                        "symbol": symbol,
                        "interval": interval,
                        "gap_from": gap.get("from"),
                        "gap_to": gap.get("to"),
                        "written_total": total_written,
                        "requests_total": total_requests,
                    },
                )
                await emit_progress(
                    progress,
                    "summary_collector:interval_progress",
                    symbol=symbol,
                    interval=interval,
                    gap_from=gap.get("from"),
                    gap_to=gap.get("to"),
                    written_total=total_written,
                    requests_total=total_requests,
                )

            refreshed = await _fetch_existing(
                repo,
                symbol=symbol,
                interval=interval,
                start_ms=first_expected,
                end_ms=last_closed,
            )
            remaining = _compute_gaps(first_expected, last_closed, interval_ms, refreshed)
            summaries[interval] = IntervalSummary(
                gaps_total=len(gaps),
                gaps_filled=sum(1 for item in gap_summaries if item.candles_written > 0),
                candles_written=sum(item.candles_written for item in gap_summaries),
                dropped_candles=sum(item.dropped_candles for item in gap_summaries),
                requests=sum(item.requests for item in gap_summaries),
                remaining_gaps=remaining,
            )
            TRACE_LOGGER.debug(
                "summary_collector:interval_finished",
                extra={
                    "symbol": symbol,
                    "interval": interval,
                    "gaps_total": len(gaps),
                    "candles_written": summaries[interval].candles_written,
                    "dropped_candles": summaries[interval].dropped_candles,
                    "requests": summaries[interval].requests,
                    "remaining_gaps": len(remaining),
                },
            )
            await emit_progress(
                progress,
                "summary_collector:interval_finished",
                symbol=symbol,
                interval=interval,
                gaps_total=len(gaps),
                candles_written=summaries[interval].candles_written,
                dropped_candles=summaries[interval].dropped_candles,
                requests=summaries[interval].requests,
                remaining_gaps=len(remaining),
            )

    summary = CollectionSummary(
        symbol=symbol.upper(),
        start_ms=start_ms,
        end_ms=now_ms,
        intervals=summaries,
        requests=total_requests,
        candles_written=total_written,
        dropped_candles=total_dropped,
    )

    TRACE_LOGGER.debug(
        "summary_collector:finished",
        extra={
            "symbol": summary.symbol,
            "requests": summary.requests,
            "candles_written": summary.candles_written,
            "dropped_candles": summary.dropped_candles,
        },
    )
    await emit_progress(
        progress,
        "summary_collector:finished",
        symbol=summary.symbol,
        requests=summary.requests,
        candles_written=summary.candles_written,
        dropped_candles=summary.dropped_candles,
    )

    return summary


__all__ = ["collect_recent_summary", "CollectionSummary"]
