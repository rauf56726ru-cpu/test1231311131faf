"""Helpers for filling recent OHLCV gaps for inspection summary builds."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from . import tracing
from .tracing import TraceContext
from .candles_repository import CandleRepository, UpsertStats, get_repository
from .ohlc_sanitizer import sanitize_candles
from .ohlc import TIMEFRAME_TO_MS, resample_ohlcv
from .timeutils import ensure_ms_epoch
from .check_all_datas import _normalise_binance_row
from .progress import ProgressReporter, emit_progress
from .http_client import RATE_LIMIT_STATUSES, TRANSIENT_STATUSES, request as http_request

LOGGER = logging.getLogger(__name__)
TRACE_LOGGER = tracing.LOGGER.getChild("summary_collector")
UTC = timezone.utc

BINANCE_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
MAX_PAGE_LIMIT = 1000
MAX_CONCURRENCY = 4
MERGE_GAP_JOIN_MS = 15 * 60_000
BULK_UPSERT_CHUNK = 750


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


async def _request_klines(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    trace: TraceContext | None = None,
) -> List[Sequence[Any]]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": str(limit),
    }
    scope = f"summary.{interval}"
    try:
        response = await http_request(
            "GET",
            BINANCE_ENDPOINT,
            scope=scope,
            trace=trace,
            client=client,
            params=params,
            symbol=symbol,
            window=(start_ms, end_ms),
            details=f"interval={interval},limit={limit}",
            max_retries=1,
            retry_statuses=TRANSIENT_STATUSES,
            rate_limit_statuses=RATE_LIMIT_STATUSES,
        )
    except httpx.RequestError as exc:  # pragma: no cover - network failure
        raise GapCollectionError(f"Request failure: {exc}") from exc

    if response.status_code == 200:
        payload = response.json()
        return payload if isinstance(payload, list) else []

    if response.status_code in RATE_LIMIT_STATUSES or response.status_code >= 500:
        return []

    response.raise_for_status()
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


def _gap_count(start_ms: int, end_ms: int, interval_ms: int) -> int:
    if end_ms < start_ms:
        return 0
    return int((end_ms - start_ms) // interval_ms + 1)


def _merge_adjacent_gaps(
    gaps: Sequence[Dict[str, int]],
    *,
    interval_ms: int,
) -> List[Dict[str, int]]:
    if not gaps:
        return []
    merged: List[Dict[str, int]] = []
    current = dict(gaps[0])
    for gap in gaps[1:]:
        next_start = int(gap.get("from", 0))
        current_end = int(current.get("to", next_start))
        separation = max(0, next_start - (current_end + interval_ms))
        if separation <= MERGE_GAP_JOIN_MS:
            current["to"] = max(current_end, int(gap.get("to", current_end)))
            current["count"] = _gap_count(int(current.get("from", next_start)), int(current["to"]), interval_ms)
        else:
            current.setdefault("count", _gap_count(int(current.get("from", next_start)), current_end, interval_ms))
            merged.append(current)
            current = dict(gap)
    current_start = int(current.get("from", 0))
    current_end = int(current.get("to", current_start))
    current["count"] = _gap_count(current_start, current_end, interval_ms)
    merged.append(current)
    return merged


async def _upsert_sanitised(
    repository: CandleRepository,
    *,
    symbol: str,
    interval: str,
    candles: Sequence[Mapping[str, Any]],
    stage: str,
    trace: TraceContext | None = None,
    chunk_size: int = BULK_UPSERT_CHUNK,
) -> tuple[UpsertStats, List[Dict[str, Any]]]:
    if not candles:
        return UpsertStats(written=0, dropped_ts=0, dropped_ohlc=0), []

    sanitised = sanitize_candles(candles, stage=stage)
    invalid_ts = getattr(sanitised, "invalid_ts", 0)
    invalid_ohlc = getattr(sanitised, "invalid_ohlc", 0)
    payload = sanitised.candles

    TRACE_LOGGER.debug(
        "summary_collector:upsert_sanitised",
        extra={
            "symbol": symbol,
            "interval": interval,
            "stage": stage,
            "incoming": len(candles),
            "sanitised": len(payload),
            "dropped_ts": invalid_ts,
            "dropped_ohlc": invalid_ohlc,
        },
    )

    if not payload:
        return UpsertStats(written=0, dropped_ts=invalid_ts, dropped_ohlc=invalid_ohlc), []

    total_written = 0
    total_ts = invalid_ts
    total_ohlc = invalid_ohlc
    stored: List[Dict[str, Any]] = []
    chunk_limit = max(1, int(chunk_size))

    for index in range(0, len(payload), chunk_limit):
        chunk = payload[index : index + chunk_limit]
        stats = await asyncio.to_thread(
            repository.upsert_candles,
            symbol,
            interval,
            chunk,
            stage=stage,
        )
        total_written += stats.written
        total_ts += stats.dropped_ts
        total_ohlc += stats.dropped_ohlc
        stored.extend(chunk)
        if trace is not None:
            trace.info(
                "db.bulk_upsert",
                symbol=symbol,
                interval=interval,
                stage=stage,
                chunk=len(chunk),
                written=stats.written,
                dropped_ts=stats.dropped_ts,
                dropped_ohlc=stats.dropped_ohlc,
            )

    return UpsertStats(written=total_written, dropped_ts=total_ts, dropped_ohlc=total_ohlc), stored


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


async def _fetch_candle_range(
    repository: CandleRepository,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        repository.fetch_candles,
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
    progress: Optional[ProgressReporter] = None,
    trace: TraceContext | None = None,
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

    if trace is not None:
        trace.info(
            "fetch.gap_begin",
            symbol=symbol,
            interval=interval,
            gap_from=gap_start,
            gap_to=gap_end,
            resume_from=resume_from,
            expected=int(gap.get("count", 0)),
        )

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
        if trace is not None:
            trace.debug(
                "fetch.batch_start",
                symbol=symbol,
                interval=interval,
                gap_from=gap_start,
                gap_to=gap_end,
                cursor=cursor,
                page_end=page_end,
                limit=page_limit,
                request_index=requests + 1,
            )
        request_kwargs = {
            "symbol": symbol,
            "interval": interval,
            "start_ms": cursor,
            "end_ms": page_end + interval_ms,
            "limit": page_limit,
        }
        if trace is not None:
            request_kwargs["trace"] = trace
        raw_rows = await _request_klines(
            client,
            **request_kwargs,
        )
        requests += 1
        if trace is not None:
            trace.info(
                "fetch.batch_done",
                symbol=symbol,
                interval=interval,
                gap_from=gap_start,
                gap_to=gap_end,
                cursor=cursor,
                page_end=page_end,
                rows=len(raw_rows),
            )
        if not raw_rows:
            if trace is not None:
                trace.debug(
                    "fetch.batch_empty",
                    symbol=symbol,
                    interval=interval,
                    gap_from=gap_start,
                    gap_to=gap_end,
                    cursor=cursor,
                    page_end=page_end,
                )
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
        if trace is not None:
            trace.debug(
                "fetch.batch_normalised",
                symbol=symbol,
                interval=interval,
                gap_from=gap_start,
                gap_to=gap_end,
                deduped_rows=len(normalised),
            )
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
            trace=trace,
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
        if trace is not None:
            trace.debug(
                "fetch.progress_updated",
                symbol=symbol,
                interval=interval,
                gap_from=gap_start,
                gap_to=gap_end,
                last_open=last_open,
            )

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
    if trace is not None:
        trace.info(
            "fetch.gap_complete",
            symbol=symbol,
            interval=interval,
            gap_from=gap_start,
            gap_to=gap_end,
            written=written,
            dropped_ts=dropped_ts,
            dropped_ohlc=dropped_ohlc,
            requests=requests,
        )

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

    trace_ctx = trace.child(component="summary_collector", symbol=symbol.upper()) if trace else None
    if trace_ctx is not None:
        trace_ctx.info(
            "collector.start",
            start_ms=start_ms,
            end_ms=now_ms,
            days=span_days,
        )

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

    requested_intervals = list(intervals or TIMEFRAME_TO_MS.keys())
    seen_intervals: set[str] = set()
    target_intervals: List[str] = []
    for interval in requested_intervals:
        if interval in seen_intervals:
            continue
        seen_intervals.add(interval)
        target_intervals.append(interval)
    if "1m" in seen_intervals:
        target_intervals = ["1m"] + [tf for tf in target_intervals if tf != "1m"]
    elif intervals is None:
        target_intervals.insert(0, "1m")

    aligned_starts: List[int] = []
    for interval in target_intervals:
        interval_ms = TIMEFRAME_TO_MS.get(interval)
        if interval_ms is None:
            continue
        aligned_starts.append(_align_to_interval(start_ms, interval_ms))
    base_start_ms = min(aligned_starts) if aligned_starts else start_ms

    summaries: Dict[str, IntervalSummary] = {}
    total_requests = 0
    total_written = 0
    total_dropped = 0

    minute_first_expected: Optional[int] = None
    minute_last_closed: Optional[int] = None
    minute_series: Optional[List[Dict[str, Any]]] = None

    pending: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=10.0)) as client:
        for interval in target_intervals:
            interval_ms = TIMEFRAME_TO_MS.get(interval)
            if not interval_ms:
                continue

            first_expected = _align_to_interval(base_start_ms, interval_ms)
            last_expected = _align_to_interval(now_ms, interval_ms)
            if end_ms is None:
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
                if trace_ctx is not None:
                    trace_ctx.debug(
                        "interval.skipped",
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
            if trace_ctx is not None:
                trace_ctx.debug(
                    "availability.checked",
                    interval=interval,
                    first_expected=first_expected,
                    last_closed=last_closed,
                    existing=len(existing),
                )

            gaps = _compute_gaps(first_expected, last_closed, interval_ms, existing)
            merged_gaps = _merge_adjacent_gaps(gaps, interval_ms=interval_ms)

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
            if trace_ctx is not None:
                trace_ctx.info(
                    "gaps.detected",
                    interval=interval,
                    total=len(gaps),
                    merged=len(merged_gaps),
                    first_expected=first_expected,
                    last_closed=last_closed,
                )
                trace_ctx.info(
                    "gaps.merged",
                    interval=interval,
                    total=len(gaps),
                    merged=len(merged_gaps),
                )

            if interval == "1m":
                minute_first_expected = first_expected
                minute_last_closed = last_closed

                if not merged_gaps:
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
                    if trace_ctx is not None:
                        trace_ctx.info(
                            "interval.complete",
                            interval=interval,
                            written=0,
                            requests=0,
                        )
                    minute_series = await _fetch_candle_range(
                        repo,
                        symbol=symbol,
                        interval="1m",
                        start_ms=first_expected,
                        end_ms=last_closed,
                    )
                    continue

                gap_summaries: List[IntervalSummary] = []
                for gap in merged_gaps:
                    summary = await _fill_gap(
                        repo,
                        symbol=symbol,
                        interval=interval,
                        gap=gap,
                        interval_ms=interval_ms,
                        client=client,
                        progress=progress,
                        trace=trace_ctx,
                    )
                    gap_summaries.append(summary)
                    total_requests += summary.requests
                    total_written += summary.candles_written
                    total_dropped += summary.dropped_candles

                refreshed_minutes = await _fetch_existing(
                    repo,
                    symbol=symbol,
                    interval=interval,
                    start_ms=first_expected,
                    end_ms=last_closed,
                )
                remaining_minutes = _compute_gaps(
                    first_expected, last_closed, interval_ms, refreshed_minutes
                )
                summaries[interval] = IntervalSummary(
                    gaps_total=len(merged_gaps),
                    gaps_filled=sum(1 for item in gap_summaries if item.candles_written > 0),
                    candles_written=sum(item.candles_written for item in gap_summaries),
                    dropped_candles=sum(item.dropped_candles for item in gap_summaries),
                    requests=sum(item.requests for item in gap_summaries),
                    remaining_gaps=remaining_minutes,
                )
                TRACE_LOGGER.debug(
                    "summary_collector:interval_finished",
                    extra={
                        "symbol": symbol,
                        "interval": interval,
                        "gaps_total": len(merged_gaps),
                        "candles_written": summaries[interval].candles_written,
                        "dropped_candles": summaries[interval].dropped_candles,
                        "requests": summaries[interval].requests,
                        "remaining_gaps": len(remaining_minutes),
                    },
                )
                await emit_progress(
                    progress,
                    "summary_collector:interval_finished",
                    symbol=symbol,
                    interval=interval,
                    gaps_total=len(merged_gaps),
                    candles_written=summaries[interval].candles_written,
                    dropped_candles=summaries[interval].dropped_candles,
                    requests=summaries[interval].requests,
                    remaining_gaps=len(remaining_minutes),
                )
                minute_series = await _fetch_candle_range(
                    repo,
                    symbol=symbol,
                    interval="1m",
                    start_ms=first_expected,
                    end_ms=last_closed,
                )
                if trace_ctx is not None:
                    trace_ctx.info(
                        "interval.complete",
                        interval=interval,
                        written=summaries[interval].candles_written,
                        requests=summaries[interval].requests,
                        remaining=len(remaining_minutes),
                    )
                continue

            if minute_first_expected is None or minute_last_closed is None or minute_series is None:
                summaries[interval] = IntervalSummary(
                    gaps_total=len(merged_gaps),
                    gaps_filled=0,
                    candles_written=0,
                    dropped_candles=0,
                    requests=0,
                    remaining_gaps=merged_gaps,
                )
                if trace_ctx is not None:
                    trace_ctx.warn(
                        "interval.skipped",
                        interval=interval,
                        reason="missing_minute_cache",
                    )
                continue

            pending.append(
                {
                    "interval": interval,
                    "interval_ms": interval_ms,
                    "first_expected": first_expected,
                    "last_closed": last_closed,
                    "existing": existing,
                    "gaps": merged_gaps,
                }
            )

    async def _process_higher_interval(params: Dict[str, Any]) -> tuple[str, IntervalSummary, int, int]:
        interval = params["interval"]
        interval_ms = params["interval_ms"]
        first_expected = params["first_expected"]
        last_closed = params["last_closed"]
        existing = params["existing"]
        gaps = params["gaps"]

        filtered = [
            candle
            for candle in (minute_series or [])
            if first_expected <= int(candle.get("t", 0)) <= last_closed
        ]
        aggregated_raw = resample_ohlcv(filtered, interval_ms)
        aggregated: List[Dict[str, Any]] = []
        for item in aggregated_raw:
            if not isinstance(item, Mapping):
                continue
            ts = _align_to_interval(int(item.get("t", 0)), interval_ms)
            if ts < first_expected or ts > last_closed:
                continue
            aggregated.append(
                {
                    "t": ts,
                    "o": float(item.get("o", 0.0)),
                    "h": float(item.get("h", 0.0)),
                    "l": float(item.get("l", 0.0)),
                    "c": float(item.get("c", 0.0)),
                    "v": float(item.get("v", 0.0)),
                }
            )

        aggregated.sort(key=lambda candle: candle["t"])
        stats, _ = await _upsert_sanitised(
            repo,
            symbol=symbol,
            interval=interval,
            candles=aggregated,
            stage=f"summary_resample.{interval}",
            trace=trace_ctx,
        )

        refreshed = await _fetch_existing(
            repo,
            symbol=symbol,
            interval=interval,
            start_ms=first_expected,
            end_ms=last_closed,
        )
        remaining = _compute_gaps(first_expected, last_closed, interval_ms, refreshed)
        gaps_filled = max(0, len(gaps) - len(remaining))
        existing_set = {int(ts) for ts in existing}
        refreshed_set = {int(ts) for ts in refreshed}
        new_candles = max(0, len(refreshed_set - existing_set))

        summary = IntervalSummary(
            gaps_total=len(gaps),
            gaps_filled=gaps_filled,
            candles_written=new_candles,
            dropped_candles=stats.dropped_ts + stats.dropped_ohlc,
            requests=0,
            remaining_gaps=remaining,
        )
        if trace_ctx is not None:
            trace_ctx.info(
                "interval.complete",
                interval=interval,
                written=new_candles,
                dropped=stats.dropped_ts + stats.dropped_ohlc,
                remaining=len(remaining),
            )
        return interval, summary, new_candles, stats.dropped_ts + stats.dropped_ohlc

    if pending:
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def _runner(params: Dict[str, Any]) -> tuple[str, IntervalSummary, int, int]:
            async with semaphore:
                return await _process_higher_interval(params)

        results = await asyncio.gather(*(_runner(params) for params in pending))
        for interval, summary, written, dropped in results:
            summaries[interval] = summary
            total_written += written
            total_dropped += dropped

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
    if trace_ctx is not None:
        trace_ctx.info(
            "collector.finished",
            requests=summary.requests,
            written=summary.candles_written,
            dropped=summary.dropped_candles,
        )

    return summary

__all__ = ["collect_recent_summary", "CollectionSummary"]
















