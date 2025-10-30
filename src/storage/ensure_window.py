"""Ensure data windows by fetching Binance UM futures candles online."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.common.config import AppConfig
from src.common.ts import ensure_epoch_ms
from src.ingest.rest_topoff import normalise_klines_payload
import asyncio

import aiohttp

from src.services.binance import (
    BinanceAPIException,
    BinanceRateLimitBudgetExceeded,
    BinanceRequestException,
    fetch_um_klines,
)

LOGGER = logging.getLogger(__name__)

INTERVAL_MS = {
    "1m": 60_000,
}

PRIMARY_LIMIT = 1_000
FALLBACK_LIMIT = 200
TARGET_COVERAGE = 0.99
MAX_PRIMARY_PAGES = 12  # 12 * 1000 covers 12_000 minutes > 72h
MAX_GAP_ATTEMPTS = 6


@dataclass(slots=True)
class EnsureResult:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    minutes_expected: int
    minutes_found: int
    coverage: float
    coverage_pct: float
    source_sequence: list[str] = field(default_factory=list)
    read_counts: dict[str, int] = field(default_factory=dict)
    archives_used: list[str] = field(default_factory=list)
    added_rows: int = 0
    frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


class InsufficientCoverageError(RuntimeError):
    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        found: int,
        expected: int,
        source_sequence: Sequence[str],
        coverage_pct: float | None = None,
    ) -> None:
        message = f"insufficient coverage for {symbol} {interval}: found {found} of {expected} minutes"
        super().__init__(message)
        self.symbol = symbol
        self.interval = interval
        self.found = found
        self.expected = expected
        self.source_sequence = list(source_sequence)
        self.coverage_pct = coverage_pct if coverage_pct is not None else (
            100.0 if expected == 0 else (found / expected) * 100.0
        )


def _expected_minutes(start_ms: int, end_ms: int, interval_ms: int) -> Tuple[int, List[int]]:
    expected = max(0, (end_ms - start_ms) // interval_ms)
    stamps = [start_ms + i * interval_ms for i in range(expected)]
    return expected, stamps


def _ingest_payload(
    storage: Dict[int, Dict[str, object]],
    payload: List[List],
    start_ms: int,
    end_ms: int,
) -> int:
    inserted = 0
    for row in normalise_klines_payload(payload):
        ts_open = row["ts_open"]
        if ts_open < start_ms or ts_open >= end_ms:
            continue
        if ts_open in storage:
            continue
        storage[ts_open] = row
        inserted += 1
    return inserted


async def _page_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
) -> List[List]:
    if start_ms >= end_ms:
        return []
    # ``fetch_um_klines`` expects endTime inclusive.
    try:
        payload = await fetch_um_klines(
            symbol,
            interval,
            start_time=start_ms,
            end_time=end_ms - 1,
            limit=limit,
        )
    except BinanceRateLimitBudgetExceeded as exc:
        LOGGER.warning(
            "ensure_window.rate_limit_pending",
            extra={
                "symbol": symbol,
                "interval": interval,
                "retry_after": round(exc.retry_after, 3),
                "start": start_ms,
                "end": end_ms,
            },
        )
        await asyncio.sleep(max(exc.retry_after, 0.0) + 0.25)
        return []
    except BinanceAPIException as exc:
        status = getattr(exc, "status_code", None)
        if status in {418, 429}:
            LOGGER.warning(
                "ensure_window.rate_limited",
                extra={"symbol": symbol, "interval": interval, "status": status, "start": start_ms, "end": end_ms},
            )
            return []
        raise
    except (BinanceRequestException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        LOGGER.debug(
            "ensure_window.page_failed",
            exc_info=exc,
            extra={"symbol": symbol, "interval": interval, "start": start_ms, "end": end_ms},
        )
        return []
    return payload


def _coverage(current: int, expected: int) -> float:
    if expected <= 0:
        return 1.0
    return current / expected


async def _collect_minutes(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
) -> Tuple[Dict[int, Dict[str, object]], int, float]:
    records: Dict[int, Dict[str, object]] = {}
    expected_count, expected_ts = _expected_minutes(start_ms, end_ms, interval_ms)
    requests = 0
    coverage = 0.0
    cursor = start_ms
    limit = PRIMARY_LIMIT
    pages_used = 0

    while cursor < end_ms and pages_used < MAX_PRIMARY_PAGES:
        batch_end = min(end_ms, cursor + interval_ms * limit)
        payload = await _page_binance_klines(
            symbol,
            interval,
            cursor,
            batch_end,
            limit=limit,
        )
        requests += 1
        pages_used += 1

        if not payload:
            cursor = batch_end
            continue

        inserted = _ingest_payload(records, payload, start_ms, end_ms)
        if inserted == 0 and limit > FALLBACK_LIMIT:
            # Retry with tighter window if no new data arrived.
            limit = FALLBACK_LIMIT
            continue
        limit = PRIMARY_LIMIT

        last_ts = None
        for entry in reversed(payload):
            try:
                ts_candidate = ensure_epoch_ms(entry[0])
            except ValueError:
                continue
            if start_ms <= ts_candidate < end_ms:
                last_ts = ts_candidate
                break
        if last_ts is None:
            cursor = batch_end
        else:
            cursor = max(last_ts + interval_ms - interval_ms, cursor + interval_ms)

        coverage = _coverage(len(records), expected_count)
        if coverage >= TARGET_COVERAGE:
            break

    if coverage >= TARGET_COVERAGE or expected_count == 0:
        return records, requests, coverage

    missing_ts = [ts for ts in expected_ts if ts not in records]
    attempts = 0
    while missing_ts and coverage < TARGET_COVERAGE and attempts < MAX_GAP_ATTEMPTS:
        attempts += 1
        gap_start = missing_ts[0]
        gap_end = min(end_ms, gap_start + interval_ms * FALLBACK_LIMIT)
        payload = await _page_binance_klines(
            symbol,
            interval,
            max(start_ms, gap_start - interval_ms),
            gap_end,
            limit=FALLBACK_LIMIT,
        )
        requests += 1
        if not payload:
            missing_ts = [ts for ts in missing_ts if ts > gap_start]
            continue
        _ingest_payload(records, payload, start_ms, end_ms)
        coverage = _coverage(len(records), expected_count)
        missing_ts = [ts for ts in expected_ts if ts not in records]

    return records, requests, coverage


def _build_frame(records: Dict[int, Dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["ts_open", "open", "high", "low", "close", "volume"])
    ordered = [records[key] for key in sorted(records.keys())]
    frame = pd.DataFrame(ordered)
    frame.sort_values("ts_open", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


async def ensure_window_real(
    symbol: str,
    start_ts: int,
    end_ts: int,
    interval: str = "1m",
) -> EnsureResult:
    """Fetch a continuous window of UM futures candles using Binance REST."""

    cfg = AppConfig.load()
    if cfg.market.lower() != "um":
        raise ValueError("Futures UM only")

    interval_key = (interval or "").strip().lower()
    interval_ms = INTERVAL_MS.get(interval_key)
    if interval_ms is None:
        raise ValueError(f"Unsupported interval '{interval}' for ensure_window_real")

    start_ms = ensure_epoch_ms(start_ts)
    end_ms_raw = ensure_epoch_ms(end_ts)
    if end_ms_raw <= start_ms:
        raise ValueError("end_ts must be greater than start_ts")

    start_ms = (start_ms // interval_ms) * interval_ms
    end_closed_ms = (end_ms_raw // interval_ms) * interval_ms
    if end_closed_ms <= start_ms:
        return EnsureResult(
            symbol=symbol.upper(),
            interval=interval_key,
            start_ms=start_ms,
            end_ms=end_closed_ms,
            minutes_expected=0,
            minutes_found=0,
            coverage=1.0,
            coverage_pct=100.0,
        )

    expected_minutes, _ = _expected_minutes(start_ms, end_closed_ms, interval_ms)

    records, requests, coverage = await _collect_minutes(
        symbol.upper(),
        interval_key,
        start_ms,
        end_closed_ms,
        interval_ms,
    )

    minutes_found = len(records)
    coverage_value = coverage if expected_minutes else 1.0
    coverage_pct = round(coverage_value * 100.0, 4)

    source_sequence = ["binance_rest"]
    read_counts = {"binance_rest": minutes_found}
    frame = _build_frame(records)

    ensure_meta = SimpleNamespace(
        coverage=coverage_value,
        source_sequence=list(source_sequence),
    )
    frame.attrs["ensure"] = ensure_meta

    LOGGER.info(
        "ensure_window.rest_summary",
        extra={
            "symbol": symbol.upper(),
            "interval": interval_key,
            "minutes_expected": expected_minutes,
            "minutes_found": minutes_found,
            "requests": requests,
            "coverage_pct": coverage_pct,
        },
    )

    if ensure_meta.coverage < TARGET_COVERAGE:
        raise InsufficientCoverageError(
            symbol=symbol.upper(),
            interval=interval_key,
            found=minutes_found,
            expected=expected_minutes,
            source_sequence=source_sequence,
            coverage_pct=coverage_pct,
        )

    result = EnsureResult(
        symbol=symbol.upper(),
        interval=interval_key,
        start_ms=start_ms,
        end_ms=end_closed_ms,
        minutes_expected=expected_minutes,
        minutes_found=minutes_found,
        coverage=ensure_meta.coverage,
        coverage_pct=coverage_pct,
        source_sequence=source_sequence,
        read_counts=read_counts,
        archives_used=[],
        added_rows=minutes_found,
        frame=frame,
    )
    return result


__all__ = ["EnsureResult", "InsufficientCoverageError", "ensure_window_real"]
