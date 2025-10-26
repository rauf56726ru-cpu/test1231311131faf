"""Snapshot builder for the inspection check-all endpoint."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import copy
import json
import logging
import math
import numbers
import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime, date
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    Set,
    TYPE_CHECKING,
)

import aiohttp
import pandas as pd

import src.services.inspection as inspection
from .binance import (
    BinanceAPIException,
    BinanceRequestException,
    fetch_um_klines,
)
from .candles_repository import get_repository
from .inspection import build_htf_section
from .liquidity import (
    build_liquidity_snapshot,
    normalise_symbol_for_tick,
    resolve_liquidity_tick_size,
)
from .orderflow_pipeline import OrderflowConfig, build_orderflow_block as _build_orderflow_block
from .pipeline_presets import SUMMARY_72H_PRESET, resolve_pipeline_preset
from .presets import resolve_profile_config
from .profile import build_compact_vwap_profiles, build_profile_package
from .ohlc_sanitizer import SanitizedCandles, sanitize_candles
from .smc import SMCConfig, detect_smc_blocks
from .zones import Config as ZonesConfig, detect_zones, compute_atr
from .settings import get_settings
from .pipeline_utils import (
    align_to_interval as _align_to_interval,
    build_expected_times as _build_expected_times,
    coerce_float as _coerce_float,
    coerce_iso_timestamp as _coerce_iso_timestamp,
    isoformat_utc as _isoformat_utc,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from .timeutils import safe_datetime_from_ms
from .progress import ProgressReporter, emit_progress
from .vision_ingest import ingest_binance_vision, DATASET_KLINES
from .vision_store import get_store
from .session_analysis import ZoneDetectionConfig, build_session_snapshot, build_72h_context
from src.common.config import AppConfig
from src.storage.ensure_window import ensure_window_real, InsufficientCoverageError


def _expected_minutes(interval_ms: int, start_ts: int, end_ts: int) -> int:
    if end_ts < start_ts:
        return 0
    return int((end_ts - start_ts) // interval_ms + 1)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .summary_collector import CollectionSummary
UTC = timezone.utc

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None  # type: ignore[assignment]

VWAP_SESSION_TZ = ZoneInfo("Europe/Berlin") if ZoneInfo else timezone.utc
LOGGER = logging.getLogger(__name__)
MS_IN_HOUR = 3_600_000
MS_IN_DAY = 86_400_000
VALID_HOUR_WINDOWS = {1, 2, 3, 4}
VALUE_AREA_PCT = 0.70
REPOSITORY_LOOKBACK_MS = 5 * MS_IN_DAY
LIQUIDITY_STRICT_MAX_HOURS = 12
ORDERFLOW_STRICT_MAX_HOURS = 12
SMC_TIMEOUT_SECONDS = 6.0
PIPELINE_PRESET_DEFAULT = SUMMARY_72H_PRESET
DEFAULT_ALL_OHLCV_TFS: Tuple[str, ...] = tuple(
    dict.fromkeys(("1m",) + PIPELINE_PRESET_DEFAULT.rollup_timeframes)
)
DEFAULT_TIMEFRAME_SUMMARY_ORDER: Tuple[str, ...] = tuple(
    tf for tf in ("1m", "5m", "15m", "1h", "4h", "1d") if tf in DEFAULT_ALL_OHLCV_TFS
)
ZONE_FOCUS_WINDOW_HOURS = PIPELINE_PRESET_DEFAULT.zones.focus_window_hours
ZONE_ALLOWED_STATUSES: Set[str] = {"open", "fresh", "tapped", "mitigated", "invalidated"}
ZONE_FOCUS_ALLOWED_TFS: Tuple[str, ...] = ("15m", "1h")

_EQUAL_LIQUIDITY_TIMEFRAMES: Tuple[str, ...] = ("15m", "1h", "4h")
_EQUAL_LIQUIDITY_REL_TOLERANCE = {
    "15m": 0.0005,
    "1h": 0.0003,
    "4h": 0.0002,
}
_EQUAL_LIQUIDITY_MIN_SEPARATION = {
    "15m": 5,
    "1h": 6,
    "4h": 6,
}
_EQUAL_LIQUIDITY_PIVOT_RADIUS = {
    "15m": 2,
    "1h": 3,
    "4h": 4,
}

try:
    from .ohlc import (
        TIMEFRAME_TO_MS,
        aggregate_1m_to_1h,
        build_multi_timeframe_ohlcv,
        fetch_ohlcv,
        fetch_ohlcv_sync,
        resample_ohlcv,
    )
    from .ohlcv import build_multi_tf_ohlcv, MinuteDataUnavailable
except ImportError:  # pragma: no cover - circular import guard
    TIMEFRAME_TO_MS = {"1m": MS_IN_HOUR // 60}

    def resample_ohlcv(*args, **kwargs):  # type: ignore[override]
        raise ImportError("resample_ohlcv is unavailable")

    def aggregate_1m_to_1h(*args, **kwargs):  # type: ignore[override]
        raise ImportError("aggregate_1m_to_1h is unavailable")

    def build_multi_timeframe_ohlcv(*args, **kwargs):  # type: ignore[override]
        raise ImportError("build_multi_timeframe_ohlcv is unavailable")

    async def fetch_ohlcv(*args, **kwargs):  # type: ignore[override]
        raise ImportError("fetch_ohlcv is unavailable")

    def fetch_ohlcv_sync(*args, **kwargs):  # type: ignore[override]
        raise ImportError("fetch_ohlcv_sync is unavailable")

MINUTE_INTERVAL_MS = TIMEFRAME_TO_MS.get("1m", MS_IN_HOUR // 60)

VWAP_TPO_SESSIONS: Tuple[Tuple[str, dtime, dtime], ...] = tuple(
    (session.name, session.open, session.close) for session in PIPELINE_PRESET_DEFAULT.sessions
)

_RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}
_MAX_RETRIES = 5

_BUILD_TIMEOUT_SECONDS = 30.0
_ASYNC_BUILD_TIMEOUT_SECONDS = 180.0
PIPELINE_HEARTBEAT_SECONDS = 3.0


class _PipelineProgress:
    """Emit structured progress updates with automatic heartbeats."""

    __slots__ = (
        "_reporter",
        "_trace",
        "_stage_start",
        "_heartbeat_tasks",
        "_heartbeat_interval",
        "_current_stage",
        "_label_map",
        "_stage_history",
    )

    def __init__(
        self,
        reporter: ProgressReporter | None,
        trace_ctx: "TraceContext | None",
        *,
        heartbeat_seconds: float = PIPELINE_HEARTBEAT_SECONDS,
        label_map: Mapping[str, str] | None = None,
    ) -> None:
        self._reporter = reporter
        self._trace = trace_ctx.child(stage="progress") if trace_ctx is not None else None
        self._stage_start: Dict[str, float] = {}
        self._heartbeat_tasks: Dict[str, asyncio.Task[None]] = {}
        self._heartbeat_interval = max(0.5, float(heartbeat_seconds))
        self._current_stage: str | None = None
        self._label_map = dict(label_map or {})
        self._stage_history: Dict[str, Dict[str, Any]] = {}

    async def start(self, stage: str, **payload: Any) -> None:
        self._cancel(stage)
        started_at = time.perf_counter()
        self._stage_start[stage] = started_at
        self._current_stage = stage
        self._stage_history[stage] = {"status": "running"}
        await self._emit(stage, "start", payload, duration_ms=None)
        if self._reporter is not None:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._heartbeat_loop(stage))
            self._heartbeat_tasks[stage] = task

    async def complete(self, stage: str, **payload: Any) -> None:
        duration_ms = self._compute_duration(stage)
        self._cancel(stage)
        self._stage_history[stage] = self._summarise_stage("complete", duration_ms, payload)
        await self._emit(stage, "complete", payload, duration_ms=duration_ms)
        if self._current_stage == stage:
            self._current_stage = None

    async def fail(self, stage: str, **payload: Any) -> None:
        duration_ms = self._compute_duration(stage)
        self._cancel(stage)
        self._stage_history[stage] = self._summarise_stage("error", duration_ms, payload)
        await self._emit(stage, "error", payload, duration_ms=duration_ms, level="warn")
        if self._current_stage == stage:
            self._current_stage = None

    async def shutdown(self) -> None:
        for stage in list(self._heartbeat_tasks.keys()):
            self._cancel(stage)

    async def _emit(
        self,
        stage: str,
        status: str,
        payload: Mapping[str, Any],
        *,
        duration_ms: float | None,
        level: str = "info",
    ) -> None:
        label = self._label_map.get(stage, stage)
        message: Dict[str, Any] = {
            "stage": stage,
            "label": label,
            "status": status,
        }
        if duration_ms is not None:
            message["duration_ms"] = round(duration_ms, 3)
        message.update({key: value for key, value in payload.items() if value is not None})
        if self._trace is not None:
            log_method = getattr(self._trace, level, self._trace.info)
            log_method("pipeline.stage", **message)
        if self._reporter is not None:
            await emit_progress(
                self._reporter,
                "inspection.pipeline.stage",
                **message,
            )

    def _compute_duration(self, stage: str) -> float | None:
        started_at = self._stage_start.get(stage)
        if started_at is None:
            return None
        return (time.perf_counter() - started_at) * 1000.0

    def _cancel(self, stage: str) -> None:
        task = self._heartbeat_tasks.pop(stage, None)
        if task is not None:
            task.cancel()
        self._stage_start.pop(stage, None)

    async def _heartbeat_loop(self, stage: str) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                duration_ms = self._compute_duration(stage)
                label = self._label_map.get(stage, stage)
                heartbeat_payload = {
                    "stage": stage,
                    "label": label,
                    "status": "heartbeat",
                    "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
                }
                if self._trace is not None:
                    self._trace.warn(
                        "pipeline.heartbeat",
                        stage=stage,
                        duration_ms=heartbeat_payload["duration_ms"],
                    )
                if self._reporter is not None:
                    await emit_progress(
                        self._reporter,
                        "inspection.pipeline.heartbeat",
                        **{k: v for k, v in heartbeat_payload.items() if v is not None},
                    )
        except asyncio.CancelledError:  # pragma: no cover - task cancellation
            pass

    def _summarise_stage(
        self,
        status: str,
        duration_ms: float | None,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"status": status}
        if duration_ms is not None:
            summary["duration_ms"] = round(duration_ms, 3)
        if payload:
            summary["details"] = self._summarise_payload(payload)
        return summary

    def _summarise_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            elif isinstance(value, Mapping):
                nested = {
                    str(nested_key): nested_value
                    for nested_key, nested_value in value.items()
                    if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
                }
                if nested:
                    result[key] = nested
        return result

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self._stage_history)



def _format_progress_payload(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(payload)


def _combine_progress_reporters(
    *reporters: ProgressReporter | None,
) -> ProgressReporter | None:
    callbacks: List[ProgressReporter] = []
    for reporter in reporters:
        if reporter is None:
            continue
        if any(callback is reporter for callback in callbacks):
            continue
        callbacks.append(reporter)

    if not callbacks:
        return None

    async def _combined(event: str, message: Dict[str, Any]) -> None:
        for callback in callbacks:
            try:
                result = callback(event, dict(message))
            except Exception:
                LOGGER.exception(
                    "Progress reporter failed",
                    extra={"event": event},
                )
                continue
            if result is None:
                continue
            if inspect.isawaitable(result):
                try:
                    await result
                except Exception:
                    LOGGER.exception(
                        "Progress reporter await failed",
                        extra={"event": event},
                    )

    return _combined


def _wrap_progress_reporter(
    symbol: str | None,
    reporter: ProgressReporter | None,
) -> ProgressReporter | None:
    if reporter is not None and getattr(reporter, "_console_wrapped", False):
        return reporter

    symbol_label = symbol or "UNKNOWN"

    async def _report(event: str, message: Dict[str, Any]) -> None:
        LOGGER.info(
            "pipeline.progress | symbol=%s | event=%s | payload=%s",
            symbol_label,
            event,
            _format_progress_payload(message),
        )
        if reporter is None:
            return
        try:
            result = reporter(event, dict(message))
        except Exception:
            LOGGER.exception(
                "Downstream progress reporter failed",
                extra={"symbol": symbol_label, "event": event},
            )
            return
        if result is None:
            return
        if inspect.isawaitable(result):
            try:
                await result
            except Exception:
                LOGGER.exception(
                    "Downstream progress reporter await failed",
                    extra={"symbol": symbol_label, "event": event},
                )

    setattr(_report, "_console_wrapped", True)
    return _report

_EXPECTED_OHLCV_TFS: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")
_EXPECTED_ORDERFLOW_TFS: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h")
_EXPECTED_ORDERFLOW_METRICS: Tuple[str, ...] = ("footprint", "delta", "cvd")
_EXPECTED_ZONE_KEYS: Tuple[str, ...] = (
    "fvg",
    "fvl",
    "ob",
    "mb",
    "bb",
    "rb",
    "pb",
    "sr",
    "profile_levels",
)

_PER_BAR_MIN_LENGTH = 240
_PER_BAR_HARD_CAP = 7200
_fetch_binance_klines = None  # overrideable for tests/backfills
_fetch_binance_agg_trades = None  # overrideable for tests/backfills

ORDERFLOW_REQUIRED_HOURS = PIPELINE_PRESET_DEFAULT.orderflow.window_hours
ORDERFLOW_WINDOW_MS = ORDERFLOW_REQUIRED_HOURS * MS_IN_HOUR
_AGG_TRADES_PAGE_MS = ORDERFLOW_WINDOW_MS  # default page span; overridden by preset



def _per_bar_target_length(window_minutes: int | None) -> int:
    if window_minutes is None or window_minutes <= 0:
        return _PER_BAR_MIN_LENGTH
    return max(_PER_BAR_MIN_LENGTH, min(int(window_minutes), _PER_BAR_HARD_CAP))


def _build_orderflow_proxy_from_minutes(
    minute_series: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    target_timeframes: Sequence[str],
    aggregated_timeframes: Sequence[str],
    target_length: int,
    window_minutes: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    filtered_minutes: List[Dict[str, Any]] = []
    for candle in minute_series:
        ts = _safe_int(candle.get("t"))
        if ts is None or ts < start_ms or ts > end_ms:
            continue
        filtered_minutes.append(
            {
                "t": ts,
                "o": _coerce_float(candle.get("o")),
                "h": _coerce_float(candle.get("h")),
                "l": _coerce_float(candle.get("l")),
                "c": _coerce_float(candle.get("c")),
                "v": _coerce_float(candle.get("v")),
            }
        )

    if not filtered_minutes:
        blank_block = {tf: {"per_bar": [], "summary": {}} for tf in target_timeframes}
        diag: Dict[str, Any] = {
            "delta_source": "proxy",
            "series_lengths": {tf: 0 for tf in target_timeframes},
            "trimmed": {},
            "coverage": {
                "expected_minutes": 0,
                "with_trades": 0,
                "missing_minutes": 0,
                "gaps": [],
                "reconstructed_minutes": 0,
            },
            "target_length": target_length,
            "window_minutes": window_minutes,
            "source": "proxy",
        }
        return blank_block, diag

    filtered_minutes.sort(key=lambda candle: candle["t"])

    per_bar_1m: List[Dict[str, Any]] = []
    running_cvd = 0.0
    for candle in filtered_minutes:
        ts = candle["t"]
        volume = float(candle.get("v") or 0.0)
        delta = float(candle.get("c") or 0.0) - float(candle.get("o") or 0.0)
        running_cvd += delta
        per_bar_1m.append(
            {
                "ts": ts,
                "t": _isoformat_utc(ts),
                "delta": delta,
                "cvd": running_cvd,
                "bid_vol": 0.0,
                "ask_vol": volume,
                "volume": volume,
                "has_trades": False,
                "reconstructed": False,
                "large_trades_count": 0,
                "absorption_high": False,
                "absorption_low": False,
                "imbalance_buy": False,
                "imbalance_sell": False,
            }
        )

    def _aggregate_series(series: Sequence[Mapping[str, Any]], interval_ms: int) -> List[Dict[str, Any]]:
        if not series:
            return []
        buckets: Dict[int, List[Mapping[str, Any]]] = {}
        for entry in series:
            ts = _safe_int(entry.get("ts"))
            if ts is None:
                continue
            bucket = (ts // interval_ms) * interval_ms
            buckets.setdefault(bucket, []).append(entry)
        aggregated: List[Dict[str, Any]] = []
        for bucket_ts in sorted(buckets):
            group = sorted(buckets[bucket_ts], key=lambda item: int(item.get("ts", bucket_ts)))
            delta_sum = sum(float(item.get("delta", 0.0)) for item in group)
            cvd_close = float(group[-1].get("cvd", 0.0))
            vol_sum = sum(float(item.get("volume", 0.0)) for item in group)
            aggregated.append(
                {
                    "ts": bucket_ts,
                    "t": _isoformat_utc(bucket_ts),
                    "delta_sum": delta_sum,
                    "cvd_close": cvd_close,
                    "vol_sum": vol_sum,
                    "bars": len(group),
                }
            )
        return aggregated

    orderflow_block: Dict[str, Dict[str, Any]] = {
        "1m": {
            "per_bar": per_bar_1m,
            "summary": {
                "delta_sum": float(per_bar_1m[-1]["cvd"]) if per_bar_1m else 0.0,
                "count": len(per_bar_1m),
            },
        }
    }

    minute_interval_ms = MINUTE_INTERVAL_MS
    for tf in aggregated_timeframes:
        interval_ms = TIMEFRAME_TO_MS.get(tf)
        if interval_ms is None or interval_ms <= minute_interval_ms:
            continue
        aggregated_series = _aggregate_series(per_bar_1m, interval_ms)
        orderflow_block.setdefault(tf, {})["per_bar"] = aggregated_series
        orderflow_block[tf]["summary"] = {
            "delta_sum": sum(item.get("delta_sum", 0.0) for item in aggregated_series),
            "cvd_close": aggregated_series[-1].get("cvd_close", 0.0) if aggregated_series else 0.0,
            "count": len(aggregated_series),
        }

    for tf in target_timeframes:
        orderflow_block.setdefault(tf, orderflow_block.get(tf, {"per_bar": [], "summary": {}}))

    series_lengths = {tf: len((orderflow_block.get(tf) or {}).get("per_bar", [])) for tf in orderflow_block}
    series_lengths.setdefault("1m_raw", len(per_bar_1m))
    diag = {
        "delta_source": "proxy",
        "series_lengths": series_lengths,
        "trimmed": {},
        "coverage": {
            "expected_minutes": len(per_bar_1m),
            "with_trades": 0,
            "missing_minutes": 0,
            "gaps": [],
            "reconstructed_minutes": 0,
        },
        "target_length": target_length,
        "window_minutes": window_minutes,
        "source": "proxy",
    }
    return orderflow_block, diag

_COLD_BACKFILL_MIN_REQUIRED: Dict[str, int] = {
    tf: 1 for tf in _EXPECTED_OHLCV_TFS
}


@dataclass(slots=True)
class _SnapshotContext:
    """Prepared context for building inspection snapshots."""

    symbol: str
    now: datetime
    frames: Dict[str, List[MutableMapping[str, Any]]]
    raw_meta: Mapping[str, Any] | None
    stream_price: float | None
    stream_ts: int | None
    has_now_override: bool


def _iter_stream_candidates(
    snapshot: Mapping[str, Any], raw_meta: Mapping[str, Any] | None
) -> Iterable[Mapping[str, Any]]:
    """Yield candidate live-tick payloads from a snapshot and its metadata."""

    for key in ("stream", "live", "live_price", "live_tick"):
        candidate = snapshot.get(key)
        if isinstance(candidate, Mapping):
            yield candidate

    if isinstance(raw_meta, Mapping):
        for key in ("stream", "live", "live_price", "ticker"):
            candidate = raw_meta.get(key)
            if isinstance(candidate, Mapping):
                yield candidate


def _resolve_stream_from_context(
    snapshot: Mapping[str, Any], raw_meta: Mapping[str, Any] | None
) -> Tuple[float | None, int | None]:
    """Extract the first valid stream price/timestamp pair from the snapshot."""

    for candidate in _iter_stream_candidates(snapshot, raw_meta):
        price, ts = _normalise_stream_point(candidate)
        if price is not None and ts is not None:
            return price, ts
    return None, None


def _prepare_snapshot_context(
    snapshot: Mapping[str, Any], now_utc: datetime | None
) -> _SnapshotContext:
    """Assemble the deterministic build context for an inspection snapshot."""

    frames = _normalise_frames(snapshot) or {}
    raw_meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), Mapping) else None
    symbol = str(snapshot.get("symbol") or snapshot.get("pair") or "UNKNOWN").upper()

    if now_utc is None:
        now_dt = datetime.now(UTC)
    else:
        if now_utc.tzinfo is None:
            now_dt = now_utc.replace(tzinfo=UTC)
        else:
            now_dt = now_utc.astimezone(UTC)

    stream_price, stream_ts = _resolve_stream_from_context(snapshot, raw_meta)

    return _SnapshotContext(
        symbol=symbol,
        now=now_dt,
        frames=frames,
        raw_meta=raw_meta,
        stream_price=stream_price,
        stream_ts=stream_ts,
        has_now_override=now_utc is not None,
    )


async def _maybe_ingest_vision_data(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    preset,
    allow_network: bool,
    trace: "TraceContext | None",
) -> Dict[str, Any] | None:
    if not allow_network or start_ms >= end_ms:
        return None

    store = get_store()

    start_day = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end_day = datetime.fromtimestamp(max(start_ms, end_ms - MINUTE_INTERVAL_MS) / 1000, tz=timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    last_closed_day = today - timedelta(days=1)
    required_days: set[str] = set()
    cursor = start_day
    while cursor <= end_day and cursor <= last_closed_day:
        required_days.add(cursor.isoformat())
        cursor += timedelta(days=1)

    def _fetch_metrics() -> List[Dict[str, Any]]:
        return store.fetch_ingestion_metrics("klines", symbol=symbol, interval="1m", limit=100)

    coverage_ok = False
    if not required_days:
        coverage_ok = True
    else:
        try:
            metrics = await asyncio.to_thread(_fetch_metrics)
            present_days = {entry.get("day") for entry in metrics if int(entry.get("count", 0)) > 0}
            coverage_ok = required_days.issubset(present_days)
            if not coverage_ok:
                for day in required_days:
                    if day in present_days:
                        continue
                    day_obj = date.fromisoformat(day)
                    day_start = int(datetime.combine(day_obj, dtime.min, tzinfo=timezone.utc).timestamp() * 1000)
                    day_end = day_start + MS_IN_DAY
                    sample = await asyncio.to_thread(
                        store.fetch_klines,
                        symbol,
                        "1m",
                        day_start,
                        day_end,
                        1,
                    )
                    if sample:
                        continue
                    break
                else:
                    coverage_ok = True
        except Exception:
            coverage_ok = False

    if coverage_ok:
        return {"status": "cached", "symbol": symbol, "window": {"start": start_ms, "end": end_ms}}

    ingest_trace = trace.child(stage="vision_ingest") if trace is not None else None
    summary = await ingest_binance_vision(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        datasets=(DATASET_KLINES,),
        klines_intervals=preset.rollup_timeframes,
        trace=ingest_trace,
        source="pipeline.check_all",
    )
    return summary


def _insufficient_from_context(
    context: _SnapshotContext, *, now_override: datetime | None = None
) -> Dict[str, Any]:
    """Materialise an insufficient-data payload for the provided context."""

    now_dt = now_override or context.now
    return _build_insufficient_payload(
        symbol=context.symbol,
        now=now_dt,
        stream_price=context.stream_price,
        stream_ts=context.stream_ts,
    )

class DataQualityError(RuntimeError):
    """Raised when the inspected snapshot fails deterministic data checks."""

    def __init__(self, detail: Mapping[str, Any]):
        super().__init__("Market data continuity validation failed")
        self.detail = dict(detail)


class BinanceDownloadError(RuntimeError):
    """Raised when Binance minute candles could not be fetched fully."""

    def __init__(self, downloaded: int, message: str):
        super().__init__(message)
        self.downloaded = int(downloaded)


class _TimeBudgetExceeded(RuntimeError):
    """Raised when the snapshot build exceeds the allocated time budget."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"Time budget exceeded while {stage}")


class _TimeBudget:
    """Helper for enforcing a soft timeout while building the snapshot."""

    __slots__ = ("deadline",)

    def __init__(self, seconds: float | None):
        if seconds is None or seconds <= 0:
            self.deadline = None
        else:
            self.deadline = time.monotonic() + seconds

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()

    def raise_if_exceeded(self, stage: str) -> None:
        if self.deadline is None:
            return
        if time.monotonic() >= self.deadline:
            raise _TimeBudgetExceeded(stage)


async def build_check_all_datas_async(
    snapshot: Mapping[str, Any],
    *,
    timeout: float | None = _ASYNC_BUILD_TIMEOUT_SECONDS,
    network_backfill: bool = True,
    trace: TraceContext | None = None,
    progress: ProgressReporter | None = None,
    **kwargs: Any,
) -> Dict[str, Any] | None:
    """Execute ``build_check_all_datas`` with a timeout that respects cancellation."""

    context = _prepare_snapshot_context(snapshot, kwargs.get("now_utc"))
    build_kwargs = dict(kwargs)
    build_kwargs.setdefault("network_backfill", network_backfill)
    downstream_progress = _combine_progress_reporters(build_kwargs.get("progress"), progress)
    build_kwargs["progress"] = _wrap_progress_reporter(context.symbol, downstream_progress)
    if trace is not None:
        build_kwargs.setdefault("trace", trace)

    async def _invoke() -> Dict[str, Any] | None:
        return await build_check_all_datas(snapshot, **build_kwargs)

    LOGGER.info(
        "pipeline.run.start | symbol=%s | window_hours=%s | hours=%s | strict_window=%s | network_backfill=%s | timeout=%s",
        context.symbol,
        build_kwargs.get("window_hours"),
        build_kwargs.get("hours"),
        build_kwargs.get("strict_window"),
        build_kwargs.get("network_backfill"),
        timeout,
    )
    result: Dict[str, Any] | None
    try:
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(_invoke(), timeout)
        else:
            result = await _invoke()
    except asyncio.TimeoutError:
        LOGGER.warning(
            "Check-all build timed out",
            extra={
                "symbol": context.symbol,
                "timeout": timeout,
                "has_now_override": context.has_now_override,
                "hours": build_kwargs.get("hours"),
                "window_hours": build_kwargs.get("window_hours"),
                "strict_window": build_kwargs.get("strict_window"),
                "network_backfill": build_kwargs.get("network_backfill"),
            },
        )
        result = _insufficient_from_context(context)
        LOGGER.info(
            "pipeline.run.complete | symbol=%s | status=%s | reason=timeout",
            context.symbol,
            result.get("status") if isinstance(result, Mapping) else None,
        )
        return result
    LOGGER.info(
        "pipeline.run.complete | symbol=%s | status=%s",
        context.symbol,
        result.get("status") if isinstance(result, Mapping) else None,
    )
    return result


def _iso_to_ms(value: Any) -> int | None:
    """Parse an ISO-8601 string into a UTC millisecond timestamp."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return int(parsed.timestamp() * 1000)


def _round_float_value(value: float, ndigits: int = 3) -> float:
    """Round a floating-point value to a stable number of decimal places."""

    if not math.isfinite(value):
        return value

    digits = max(0, int(ndigits))
    rounded = round(value, digits)

    if digits > 2:
        magnitude = abs(rounded)
        integer_digits = 1
        if magnitude >= 1:
            integer_digits = len(str(int(magnitude)))
        if integer_digits > 6:
            rounded = round(value, 2)

    return rounded


def round_floats(obj: Any, ndigits: int = 3) -> Any:
    """Recursively round floats within mappings and sequences."""

    if isinstance(obj, Mapping):
        return {key: round_floats(val, ndigits) for key, val in obj.items()}
    if isinstance(obj, list):
        return [round_floats(item, ndigits) for item in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats(item, ndigits) for item in obj)
    if isinstance(obj, set):
        return {round_floats(item, ndigits) for item in obj}
    if isinstance(obj, float):
        return _round_float_value(obj, ndigits)
    return obj


def _filter_profile_entries(profile: Sequence[Any]) -> List[Any]:
    """Filter profile rows to drop entries with zero volume while keeping order."""

    filtered: List[Any] = []
    for entry in profile:
        if isinstance(entry, Mapping):
            volume = entry.get("volume")
            try:
                volume_value = float(volume)
            except (TypeError, ValueError):
                filtered.append(entry)
                continue

            if math.isfinite(volume_value) and volume_value == 0.0:
                continue

        filtered.append(entry)

    return filtered


def _deduplicate_sorted(
    candles: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return candles sorted by timestamp with the last occurrence kept."""

    seen: Dict[int, Dict[str, Any]] = {}
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        seen[ts] = {
            "t": ts,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        }

    ordered_times = sorted(seen)
    return [seen[ts] for ts in ordered_times]


def _normalise_external_candles(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Normalise external candle payloads into the internal shape."""

    if payload is None:
        return []

    if isinstance(payload, Mapping):
        raw_candles = payload.get("candles")
        if not isinstance(raw_candles, Sequence):
            return []
        source: Sequence[Mapping[str, Any]] = raw_candles  # type: ignore[assignment]
    elif isinstance(payload, Sequence):
        source = [item for item in payload if isinstance(item, Mapping)]  # type: ignore[list-item]
    else:
        return []

    normalised: List[Dict[str, Any]] = []
    for item in source:
        ts = _safe_int(item.get("t"))
        if ts is None:
            continue
        normalised.append(
            {
                "t": ts,
                "o": _coerce_float(item.get("o")),
                "h": _coerce_float(item.get("h")),
                "l": _coerce_float(item.get("l")),
                "c": _coerce_float(item.get("c")),
                "v": _coerce_float(item.get("v")),
            }
        )

    normalised.sort(key=lambda candle: candle["t"])
    return normalised


def _merge_candle_collections(
    existing: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two candle collections preferring values from the existing set."""

    combined: List[Mapping[str, Any]] = []
    if incoming:
        combined.extend(incoming)
    if existing:
        combined.extend(existing)
    return _deduplicate_sorted(combined)


async def _load_repository_candles(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    repository = get_repository()
    if repository.__class__.__name__ == "InMemoryRepository":
        candles = repository.fetch_candles(symbol, interval, start_ms, end_ms)
    else:
        candles = await asyncio.to_thread(
            repository.fetch_candles,
            symbol,
            interval,
            start_ms,
            end_ms,
        )
    return [dict(item) for item in candles]


def _resolve_window_end_ms(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now_ms: int,
    selection_end_ms: int | None,
    stream_ts: int | None,
) -> int:
    """Resolve a best-effort window end timestamp based on available hints."""

    candidates: List[int] = [now_ms]
    if selection_end_ms is not None:
        candidates.append(selection_end_ms)
    if stream_ts is not None:
        candidates.append(stream_ts)

    for candles in frames.values():
        if not candles:
            continue
        last_ts = _safe_int(candles[-1].get("t"))
        if last_ts is not None:
            candidates.append(last_ts)

    return max(candidates) if candidates else now_ms


async def _backfill_timeframe_with_rest(
    frames: MutableMapping[str, List[MutableMapping[str, Any]]],
    *,
    symbol: str,
    timeframe: str,
    window_end_ms: int,
    window_hours: int,
    fetcher: Callable[..., Awaitable[Mapping[str, Any]]] | Callable[..., Mapping[str, Any]] | None = None,
    minimum_required: int | None = None,
    allow_network: bool = True,
) -> bool:
    """Ensure the requested timeframe has at least the required candles."""

    interval_ms = TIMEFRAME_TO_MS.get(timeframe)
    if interval_ms is None:
        return False

    if minimum_required is None:
        minimum_required = _COLD_BACKFILL_MIN_REQUIRED.get(timeframe, 1)
    minimum_required = max(1, int(minimum_required))

    existing = frames.get(timeframe, [])
    window_start_ms = max(0, window_end_ms - max(1, window_hours) * MS_IN_HOUR)

    available = 0
    for candle in existing:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        if window_start_ms <= ts <= window_end_ms:
            available += 1
            if available >= minimum_required:
                return False

    if not allow_network:
        return False

    fetch_callable = fetcher or fetch_ohlcv
    try:
        if asyncio.iscoroutinefunction(fetch_callable):
            fetched_payload = await fetch_callable(symbol, timeframe, hours=max(1, window_hours))  # type: ignore[arg-type]
        else:
            result = fetch_callable(symbol, timeframe, hours=max(1, window_hours))
            if asyncio.iscoroutine(result):
                fetched_payload = await result  # type: ignore[assignment]
            else:
                fetched_payload = result
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "Cold backfill request failed",
            exc_info=exc,
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "window_hours": window_hours,
            },
        )
        return False

    fetched_candles = _normalise_external_candles(fetched_payload)
    if not fetched_candles:
        return False

    merged = _merge_candle_collections(existing, fetched_candles)
    frames[timeframe] = [dict(candle) for candle in merged]
    return True


def _build_insufficient_payload(
    *,
    symbol: str,
    now: datetime,
    stream_price: float | None,
    stream_ts: int | None,
) -> Dict[str, Any]:
    """Return a deterministic payload when cold backfill could not seed data."""

    frames_stub: Dict[str, Sequence[Mapping[str, Any]]] = {
        tf: [] for tf in _EXPECTED_OHLCV_TFS
    }
    (
        last_price,
        last_iso,
        last_tf,
        age_sec,
        _,
        price_source,
        _diagnostics,
    ) = _resolve_last_price(
        frames_stub,
        now=now,
        stream_price=stream_price,
        stream_ts=stream_ts,
        tick_size=None,
    )

    meta_block: Dict[str, Any] = {
        "symbol": symbol,
        "tz": "Europe/Berlin",
        "last_price": last_price,
        "last_ts_utc": last_iso,
        "last_tf": last_tf,
        "last_price_source": price_source,
        "insufficient_reason": "stale_or_unseeded_buffers",
        "stale": True,
    }
    if age_sec is not None:
        meta_block["snapshot_age_sec"] = age_sec

    timeframe_order: Tuple[str, ...] = DEFAULT_TIMEFRAME_SUMMARY_ORDER
    session_names: Tuple[str, ...] = ("asia", "london", "ny")
    empty_timeframes_data: Dict[str, Dict[str, Any]] = {
        tf: {
            "zones": {"eqh": 0, "eql": 0, "fvg": 0, "ob": 0},
            "sweeps": 0,
            "atr": None,
            "vwap_sessions": {session: False for session in session_names},
        }
        for tf in timeframe_order
    }
    availability_timeframes: Dict[str, Dict[str, Any]] = {
        tf: {
            "zones": {"eqh": False, "eql": False, "fvg": False, "ob": False},
            "sweeps": 0,
            "atr": False,
            "vwap_sessions": {session: False for session in session_names},
        }
        for tf in timeframe_order
    }

    data_payload = {
        "symbol": symbol,
        "ohlcv": {tf: {"candles": []} for tf in _EXPECTED_OHLCV_TFS},
        "orderflow": {tf: {"per_bar": []} for tf in _EXPECTED_ORDERFLOW_TFS},
        "vwap_tpo": {
            "daily": {},
            "sessions": {session: {} for session in ("asia", "london", "ny")},
        },
        "tpo": {"composite_day": {}},
        "prev_day": {},
        "zones": {key: [] for key in _EXPECTED_ZONE_KEYS},
        "liquidity": {"eqh": [], "eql": []},
        "risk_prefs": {"rr_min": 2.5, "risk_per_trade_pct": 1.0},
        "context": {"globalBias": "neutral", "narrative": "", "openOppositeZones": False},
        "timeframes": empty_timeframes_data,
    }

    availability_payload = {"timeframes": availability_timeframes}

    missing_fields: Set[str] = set()
    missing_fields.update(f"ohlcv.{tf}" for tf in _EXPECTED_OHLCV_TFS)
    for session in ("asia", "london", "ny"):
        missing_fields.add(f"vwap_tpo.sessions.{session}")
        for metric in ("poc", "vah", "val", "ib_high", "ib_low"):
            missing_fields.add(f"vwap_tpo.sessions.{session}.{metric}")
    missing_fields.update(f"zones.{key}" for key in _EXPECTED_ZONE_KEYS)
    missing_fields.update(f"orderflow.{tf}" for tf in _EXPECTED_ORDERFLOW_TFS)
    missing_fields.update(f"orderflow.{metric}" for metric in _EXPECTED_ORDERFLOW_METRICS)

    meta_block["diagnostics"] = {
        "liquidity": {},
        "zones": {},
        "liquidity_levels": {},
        "orderflow": {},
        "vwap_tpo": {},
        "movement": {},
        "coverage": {},
        "api": {"requests": 0, "retries": 0, "rate_limit_hits": 0, "backoffs": 0},
        "sessions": {session: {"present": False} for session in session_names},
    }

    payload = {
        "status": "insufficient_data",
        "meta": meta_block,
        "data": data_payload,
        "availability": availability_payload,
        "missing_fields": sorted(missing_fields),
        "notes": [],
        "timing": {
            "fetch_ms": 0.0,
            "db_ms": 0.0,
            "compute_ms": 0.0,
            "serialize_ms": 0.0,
            "size_bytes": 0,
        },
    }
    return round_floats(payload)


def _build_minute_missing_payload(
    context: _SnapshotContext,
    *,
    now: datetime,
    missing: MinuteDataUnavailable,
) -> Dict[str, Any]:
    """Return a deterministic payload when minute coverage is incomplete."""

    payload = _build_insufficient_payload(
        symbol=context.symbol,
        now=now,
        stream_price=context.stream_price,
        stream_ts=context.stream_ts,
    )
    payload["status"] = "minute_missing"

    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta["insufficient_reason"] = "minute_missing"
        meta["minute_missing"] = int(missing.missing_count)
        meta["minute_window"] = {"start_ms": missing.start_ms, "end_ms": missing.end_ms}
        meta["minute_coverage_pct"] = round(missing.coverage_pct, 3)

    note = (
        f"Missing {missing.missing_count} minute candles between "
        f"{missing.start_ms} and {missing.end_ms}"
    )
    notes = payload.get("notes")
    if isinstance(notes, list):
        if note not in notes:
            notes.append(note)
    else:
        payload["notes"] = [note]

    return payload


def _finalise_payload(
    payload: Dict[str, Any],
    *,
    status: str,
    pipeline_start: float,
    fetch_ms: float,
    db_ms: float,
    trace_ctx: TraceContext | None,
) -> Dict[str, Any]:
    """Attach timing metadata, measure size, and emit the terminal trace event."""

    total_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000.0
    compute_ms = max(0.0, total_elapsed_ms - fetch_ms - db_ms)
    serialize_start = time.perf_counter()
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
    size_bytes = len(encoded.encode("utf-8"))

    timing_block = payload.setdefault("timing", {})
    timing_block.update(
        {
            "fetch_ms": round(fetch_ms, 2),
            "db_ms": round(db_ms, 2),
            "compute_ms": round(compute_ms, 2),
            "serialize_ms": round(serialize_ms, 2),
            "size_bytes": size_bytes,
        }
    )

    if trace_ctx is not None:
        trace_ctx.info(
            "output.publish",
            scope="output",
            status=status,
            size_bytes=size_bytes,
        )
        trace_ctx.info(
            "pipeline.done",
            scope="pipeline",
            status=status,
            fetch_ms=timing_block["fetch_ms"],
            db_ms=timing_block["db_ms"],
            compute_ms=timing_block["compute_ms"],
            serialize_ms=timing_block["serialize_ms"],
            size_bytes=size_bytes,
        )

    return round_floats(payload)


def build_inspection_error_payload(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    missing_fields: Sequence[str] | None = None,
    reason: str = "invalid_timestamps",
) -> Dict[str, Any]:
    """Return a deterministic insufficient-data payload for unexpected errors."""

    context = _prepare_snapshot_context(snapshot, now_utc)
    payload = _build_insufficient_payload(
        symbol=context.symbol,
        now=context.now,
        stream_price=context.stream_price,
        stream_ts=context.stream_ts,
    )
    if missing_fields:
        existing = set(payload.get("missing_fields", []))
        payload["missing_fields"] = sorted(existing.union(set(missing_fields)))
    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta["insufficient_reason"] = reason
        meta["sanitized"] = True
    return payload


def _resolve_last_price(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now: datetime,
    stream_price: float | None = None,
    stream_ts: int | None = None,
    tick_size: float | None = None,
) -> Tuple[
    float | None,
    str | None,
    str | None,
    int | None,
    str | None,
    str,
    Dict[str, Any],
]:
    """Return last price metadata using the most granular available timeframe.

    The function prefers a live stream tick when present, falling back to the
    most granular OHLCV close otherwise. It also emits diagnostics when the
    stream price diverges from OHLCV beyond one tick while being delayed.
    """

    priority = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")
    candle_tf: str | None = None
    candle_price: float | None = None
    candle_ts: int | None = None

    for tf in priority:
        candles = frames.get(tf)
        if not candles:
            continue
        deduped = _deduplicate_sorted(candles)
        if not deduped:
            continue
        last_candle = deduped[-1]
        ts = _safe_int(last_candle.get("t"))
        price = _safe_float(last_candle.get("c"))
        if ts is None or price is None:
            continue
        candle_tf = tf
        candle_price = price
        candle_ts = ts
        break

    diagnostics: Dict[str, Any] = {}
    price_source = "ohlcv"

    last_price: float | None = candle_price
    last_ts: int | None = candle_ts
    last_tf: str | None = candle_tf

    if stream_price is not None and stream_ts is not None:
        price_source = "stream"
        last_price = stream_price
        last_ts = stream_ts
        last_tf = "stream"
        diagnostics["stream_ts"] = stream_ts
        diagnostics["stream_price"] = stream_price

    last_iso: str | None = _isoformat_utc(last_ts) if last_ts is not None else None

    insufficient_reason: str | None = None
    snapshot_age_sec: int | None = None

    if last_ts is not None:
        now_ms = int(now.timestamp() * 1000)
        snapshot_age_sec = max(0, (now_ms - last_ts) // 1000)
        if snapshot_age_sec > 5 * 60:
            insufficient_reason = f"stale_snapshot_{snapshot_age_sec}"
    else:
        insufficient_reason = "missing_live_last_price"

    if candle_price is None or candle_ts is None:
        diagnostics["ohlcv_missing"] = True
    else:
        diagnostics["ohlcv_price"] = candle_price
        diagnostics["ohlcv_ts"] = candle_ts
        diagnostics["ohlcv_tf"] = candle_tf

    if price_source == "stream" and candle_price is not None and candle_ts is not None:
        tick = float(tick_size) if tick_size else None
        if tick is not None and tick > 0:
            diff = abs(stream_price - candle_price)
            interval_ms = TIMEFRAME_TO_MS.get(candle_tf or "1m", 60_000)
            candle_close_ms = candle_ts + interval_ms
            lag_ms = abs(stream_ts - candle_close_ms)
            diagnostics["stream_vs_candle_diff"] = diff
            diagnostics["stream_vs_candle_lag_ms"] = lag_ms
            if diff > tick and lag_ms > 2000:
                diagnostics["mismatch"] = True
                LOGGER.warning(
                    "Stream vs OHLCV mismatch detected",
                    extra={
                        "stream_price": stream_price,
                        "ohlcv_price": candle_price,
                        "tick_size": tick,
                        "lag_ms": lag_ms,
                        "diff": diff,
                        "ohlcv_tf": candle_tf,
                    },
                )

    LOGGER.info(
        "Resolved last price for inspection snapshot",
        extra={
            "source": price_source,
            "tf": last_tf,
            "ts": last_ts,
            "age_sec": snapshot_age_sec,
            "insufficient_reason": insufficient_reason,
        },
    )

    return (
        last_price,
        last_iso,
        last_tf,
        snapshot_age_sec,
        insufficient_reason,
        price_source,
        diagnostics,
    )


def _summarise_missing_times(
    expected: Sequence[int],
    available: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, int]]:
    gaps: List[Dict[str, int]] = []
    current_start: int | None = None
    current_count = 0

    for ts in expected:
        if ts not in available:
            if current_start is None:
                current_start = ts
                current_count = 1
            else:
                current_count += 1
        elif current_start is not None:
            gaps.append({"from": current_start, "to": ts - MINUTE_INTERVAL_MS, "count": current_count})
            current_start = None
            current_count = 0

    if current_start is not None:
        last_missing_ts = expected[-1]
        gaps.append({"from": current_start, "to": last_missing_ts, "count": current_count})

    return gaps


def _normalise_binance_row(row: Sequence[object]) -> Dict[str, Any] | None:
    try:
        open_time = int(row[0])
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        volume = float(row[5])
    except (IndexError, TypeError, ValueError):
        return None
    return {
        "t": open_time,
        "o": open_price,
        "h": high_price,
        "l": low_price,
        "c": close_price,
        "v": volume,
    }


async def _request_binance_minutes_async(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
    budget: _TimeBudget | None = None,
) -> List[Sequence[object]]:
    delay = 0.5
    for attempt in range(_MAX_RETRIES):
        if budget is not None:
            budget.raise_if_exceeded("request_binance_minutes")
        try:
            data = await fetch_um_klines(
                symbol,
                "1m",
                start_time=start_ms,
                end_time=end_ms,
                limit=limit,
            )
            return list(data)
        except BinanceAPIException as exc:
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                if budget is not None:
                    budget.raise_if_exceeded("request_binance_minutes_backoff")
                    remaining = budget.remaining()
                    if remaining is not None and remaining <= 0:
                        raise
                    await asyncio.sleep(min(delay, max(0.0, remaining)))
                else:
                    await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
                continue
            raise
        except (BinanceRequestException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt < _MAX_RETRIES - 1:
                if budget is not None:
                    budget.raise_if_exceeded("request_binance_minutes_retry")
                    remaining = budget.remaining()
                    if remaining is not None and remaining <= 0:
                        raise
                    await asyncio.sleep(min(delay, max(0.0, remaining)))
                else:
                    await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
                continue
            raise BinanceDownloadError(0, str(exc)) from exc
    return []


async def _download_missing_minutes_async(
    symbol: str,
    start_ms: int,
    end_ms: int,
    gaps: Sequence[Mapping[str, int]],
    *,
    budget: _TimeBudget | None = None,
) -> List[Dict[str, Any]]:
    if not gaps:
        return []

    fetched: List[Dict[str, Any]] = []
    downloaded = 0

    try:
        for gap in gaps:
            if budget is not None:
                budget.raise_if_exceeded("download_missing_minutes_gap")
            gap_start = int(gap["from"])
            gap_end = int(gap["to"])
            cursor = gap_start
            while cursor <= gap_end:
                if budget is not None:
                    budget.raise_if_exceeded("download_missing_minutes_cursor")
                chunk_end = min(
                    gap_end,
                    cursor + (1000 - 1) * MINUTE_INTERVAL_MS,
                )
                request_end = chunk_end + MINUTE_INTERVAL_MS
                if callable(_fetch_binance_klines):
                    fetch_result = _fetch_binance_klines(
                        symbol,
                        "1m",
                        cursor,
                        request_end,
                        1000,
                    )
                    if inspect.isawaitable(fetch_result):  # type: ignore[arg-type]
                        raw_rows = await fetch_result  # type: ignore[assignment]
                    else:
                        raw_rows = fetch_result
                else:
                    raw_rows = await _request_binance_minutes_async(
                        symbol,
                        cursor,
                        request_end,
                        limit=1000,
                        budget=budget,
                    )
                if not raw_rows:
                    break

                last_open = None
                for row in raw_rows:
                    candle = _normalise_binance_row(row)
                    if candle is None:
                        continue
                    ts = candle["t"]
                    if ts < start_ms or ts > end_ms:
                        continue
                    fetched.append(candle)
                    downloaded += 1
                    last_open = ts

                if last_open is None:
                    break
                cursor = last_open + MINUTE_INTERVAL_MS
                if cursor > gap_end:
                    break
    except (BinanceAPIException, BinanceRequestException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise BinanceDownloadError(downloaded, str(exc)) from exc
    except BinanceDownloadError:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise BinanceDownloadError(downloaded, str(exc)) from exc

    return fetched


async def _call_download_missing_minutes_async(
    symbol: str,
    start_ms: int,
    end_ms: int,
    gaps: Sequence[Mapping[str, int]],
    *,
    budget: _TimeBudget | None,
    allow_network: bool = True,
) -> List[Dict[str, Any]]:
    """Invoke `_download_missing_minutes` while tolerating legacy stubs without budget."""

    if not allow_network:
        return []

    if budget is None:
        return await _download_missing_minutes_async(symbol, start_ms, end_ms, gaps)

    try:
        return await _download_missing_minutes_async(
            symbol,
            start_ms,
            end_ms,
            gaps,
            budget=budget,
        )
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" not in message or "budget" not in message:
            raise
        return await _download_missing_minutes_async(symbol, start_ms, end_ms, gaps)


def _aggregate_from_minutes(
    minute_index: Mapping[int, Mapping[str, Any]],
    open_time: int,
    interval_ms: int,
) -> Dict[str, Any] | None:
    end_exclusive = open_time + interval_ms
    cursor = open_time
    bucket: List[Mapping[str, Any]] = []

    while cursor < end_exclusive:
        candle = minute_index.get(cursor)
        if candle is None:
            fallback = minute_index.get(cursor - MINUTE_INTERVAL_MS)
            if fallback is None and bucket:
                fallback = bucket[-1]
            if fallback is not None:
                candle = {
                    "t": cursor,
                    "o": fallback["c"],
                    "h": fallback["c"],
                    "l": fallback["c"],
                    "c": fallback["c"],
                    "v": 0.0,
                }
        if candle is not None:
            bucket.append(candle)
        cursor += MINUTE_INTERVAL_MS

    if not bucket:
        return None

    high = max(item["h"] for item in bucket)
    low = min(item["l"] for item in bucket)
    return {
        "t": open_time,
        "o": bucket[0]["o"],
        "h": high,
        "l": low,
        "c": bucket[-1]["c"],
        "v": sum(item["v"] for item in bucket),
    }


def _normalise_stream_point(candidate: Mapping[str, Any] | None) -> Tuple[float | None, int | None]:
    if not isinstance(candidate, Mapping):
        return None, None

    price: float | None = None
    for key in ("price", "last_price", "p", "value", "close"):
        price = _safe_float(candidate.get(key))
        if price is not None:
            break

    ts_raw: int | None = None
    for key in ("ts", "timestamp", "time", "t", "ts_ms", "event_time"):
        value = candidate.get(key)
        if value is None:
            continue
        ts_raw = _safe_int(value)
        if ts_raw is not None:
            if ts_raw < 10_000_000_000:
                ts_raw *= 1000
            break

    if price is None or ts_raw is None:
        return None, None

    return price, ts_raw


def _pivot_radius_for_equal_levels(tf: str) -> int:
    settings = get_settings().eql_settings
    mapping = getattr(settings, "pivot_radius_by_tf", {}) or {}
    value = mapping.get(tf)
    if value is not None:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and numeric > 0:
            return numeric
    fallback = _EQUAL_LIQUIDITY_PIVOT_RADIUS.get(tf)
    if fallback is not None:
        return fallback
    return max(1, int(getattr(settings, "min_separation_bars", 4) // 2) or 1)


def _minimum_separation_for_equal_levels(tf: str) -> int:
    settings = get_settings().eql_settings
    mapping = getattr(settings, "min_separation_bars_by_tf", {}) or {}
    value = mapping.get(tf)
    if value is not None:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and numeric > 0:
            return numeric
    fallback = _EQUAL_LIQUIDITY_MIN_SEPARATION.get(tf)
    if fallback is not None:
        return fallback
    try:
        global_default = int(settings.min_separation_bars)
    except (TypeError, ValueError):
        global_default = 5
    return max(1, global_default)


def _bps_to_ratio(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric <= 0:
        return max(0.0, default)
    if numeric > 1:
        return numeric / 10_000.0
    return numeric


def _relative_tolerance_for_equal_levels(tf: str) -> float:
    settings = get_settings().eql_settings
    mapping = getattr(settings, "tolerance_bps_by_tf", {}) or {}
    value = mapping.get(tf)
    if value is not None:
        ratio = _bps_to_ratio(value, default=0.0)
        if ratio > 0:
            return ratio
    fallback = _EQUAL_LIQUIDITY_REL_TOLERANCE.get(tf)
    if fallback is not None:
        return fallback
    base_ratio = _bps_to_ratio(getattr(settings, "tolerance_bps", 0.0), default=0.0003)
    return base_ratio or 0.0003


def _apply_zone_threshold_overrides(cfg: ZonesConfig) -> ZonesConfig:
    thresholds = get_settings().zone_thresholds
    sr_tolerance = _bps_to_ratio(
        getattr(thresholds, "tolerance_bps", 0.0),
        default=cfg.sr_merge_pct,
    )
    if sr_tolerance > 0:
        cfg.sr_merge_pct = sr_tolerance
    max_fill = getattr(thresholds, "max_fill_percent", None)
    if max_fill is not None:
        try:
            fill_value = float(max_fill)
        except (TypeError, ValueError):
            fill_value = None
        if fill_value is not None:
            fill_ratio = fill_value / 100.0 if fill_value > 1 else fill_value
            cfg.mitigation_fill_ratio = max(0.0, min(1.0, fill_ratio))
    min_displacement = getattr(thresholds, "min_displacement", None)
    try:
        displacement_value = float(min_displacement) if min_displacement is not None else None
    except (TypeError, ValueError):
        displacement_value = None
    if displacement_value is not None and displacement_value > 0:
        cfg.ob_distance_atr = displacement_value
    min_sep_bars = getattr(thresholds, "min_separation_bars", None)
    try:
        min_sep_numeric = int(min_sep_bars) if min_sep_bars is not None else None
    except (TypeError, ValueError):
        min_sep_numeric = None
    if min_sep_numeric is not None and min_sep_numeric > 0:
        cfg.base_min_bars = max(cfg.base_min_bars, min_sep_numeric)
    return cfg


def _detect_equal_levels_for_timeframe(
    candles: Sequence[Mapping[str, Any]],
    *,
    tf: str,
    kind: str,
) -> List[Dict[str, Any]]:
    radius = max(1, _pivot_radius_for_equal_levels(tf))
    minimum_separation = max(1, _minimum_separation_for_equal_levels(tf))
    tolerance_ratio = max(0.0, _relative_tolerance_for_equal_levels(tf))
    length = len(candles)
    if length < 2 * radius + 1:
        return []

    pivots: List[Dict[str, Any]] = []
    price_key = "h" if kind == "high" else "l"

    for idx in range(radius, length - radius):
        candle = candles[idx]
        pivot_price = _safe_float(candle.get(price_key))
        pivot_ts = _safe_int(candle.get("t"))
        if pivot_price is None or pivot_ts is None:
            continue
        is_pivot = True
        for offset in range(1, radius + 1):
            left = candles[idx - offset]
            right = candles[idx + offset]
            left_price = _safe_float(left.get(price_key))
            right_price = _safe_float(right.get(price_key))
            if left_price is not None:
                if kind == "high" and pivot_price < left_price:
                    is_pivot = False
                    break
                if kind == "low" and pivot_price > left_price:
                    is_pivot = False
                    break
            if right_price is not None:
                if kind == "high" and pivot_price < right_price:
                    is_pivot = False
                    break
                if kind == "low" and pivot_price > right_price:
                    is_pivot = False
                    break
        if not is_pivot:
            continue
        pivots.append({"idx": idx, "price": pivot_price, "ts": pivot_ts})

    if len(pivots) < 2:
        return []

    equal_levels: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for j in range(1, len(pivots)):
        pivot_j = pivots[j]
        best_candidate = None
        best_diff = None
        for i in range(j):
            pivot_i = pivots[i]
            if (pivot_i["idx"], pivot_j["idx"]) in seen_pairs:
                continue
            if pivot_j["idx"] - pivot_i["idx"] < minimum_separation:
                continue
            average_price = (pivot_i["price"] + pivot_j["price"]) / 2.0
            if average_price <= 0:
                continue
            price_diff = abs(pivot_j["price"] - pivot_i["price"])
            tolerance = tolerance_ratio * average_price
            if price_diff <= tolerance:
                if best_diff is None or price_diff < best_diff:
                    best_candidate = pivot_i
                    best_diff = price_diff
        if best_candidate is None:
            continue
        seen_pairs.add((best_candidate["idx"], pivot_j["idx"]))
        second_touch_ts = pivot_j["ts"]
        equal_levels.append(
            {
                "price": (best_candidate["price"] + pivot_j["price"]) / 2.0,
                "ts": second_touch_ts,
            }
        )

    equal_levels.sort(key=lambda item: item["ts"])
    for entry in equal_levels:
        entry["ts"] = _isoformat_utc(entry["ts"])
    return equal_levels


def build_equal_liquidity_levels(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_levels: Mapping[str, int] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Detect simplified EQH/EQL pools from timeframe candles."""

    eqh_levels: List[Dict[str, Any]] = []
    eql_levels: List[Dict[str, Any]] = []

    for tf in _EQUAL_LIQUIDITY_TIMEFRAMES:
        candles = frames.get(tf)
        if not isinstance(candles, Sequence):
            continue
        eqh_levels.extend(
            _detect_equal_levels_for_timeframe(candles, tf=tf, kind="high")
        )
        eql_levels.extend(
            _detect_equal_levels_for_timeframe(candles, tf=tf, kind="low")
        )

    if isinstance(max_levels, Mapping):
        limit_eqh = max(0, int(max_levels.get("eqh", len(eqh_levels)) or len(eqh_levels)))
        limit_eql = max(0, int(max_levels.get("eql", len(eql_levels)) or len(eql_levels)))
        if limit_eqh and len(eqh_levels) > limit_eqh:
            eqh_levels = eqh_levels[-limit_eqh:]
        if limit_eql and len(eql_levels) > limit_eql:
            eql_levels = eql_levels[-limit_eql:]
        if limit_eqh == 0:
            eqh_levels = []
        if limit_eql == 0:
            eql_levels = []

    return {"eqh": eqh_levels, "eql": eql_levels}


def _coerce_candle(entry: Mapping[str, Any]) -> MutableMapping[str, Any] | None:
    """Normalise a raw candle mapping into numeric OHLCV fields."""

    raw_ts = (
        entry.get("t")
        or entry.get("time")
        or entry.get("openTime")
        or entry.get("open_time")
    )
    timestamp_ms = _safe_int(raw_ts)
    if timestamp_ms is None:
        return None

    candle: MutableMapping[str, Any] = {
        "t": timestamp_ms,
        "o": _coerce_float(entry.get("o", entry.get("open"))),
        "h": _coerce_float(entry.get("h", entry.get("high"))),
        "l": _coerce_float(entry.get("l", entry.get("low"))),
        "c": _coerce_float(entry.get("c", entry.get("close"))),
        "v": _coerce_float(entry.get("v", entry.get("volume"))),
    }

    return candle


def _extract_raw_candles(snapshot: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    frames = snapshot.get("frames")
    primary_tf = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()

    if isinstance(frames, Mapping):
        target = frames.get(primary_tf)
        if target is None and frames:
            target = next(iter(frames.values()))
        if isinstance(target, Mapping):
            candles = target.get("candles", [])
        else:
            candles = target
    else:
        candles = snapshot.get("candles", [])

    if candles is None:
        return []

    try:
        return list(candles)  # type: ignore[arg-type]
    except TypeError:
        return []


def _normalise_candles(snapshot: Mapping[str, Any]) -> List[MutableMapping[str, Any]]:
    candles: List[MutableMapping[str, Any]] = []
    for entry in _extract_raw_candles(snapshot):
        if not isinstance(entry, Mapping):
            continue
        candle = _coerce_candle(entry)
        if candle is None:
            continue
        candles.append(candle)

    candles.sort(key=lambda item: item["t"])
    return candles


def _normalise_frames(snapshot: Mapping[str, Any]) -> Dict[str, List[MutableMapping[str, Any]]]:
    frames: Dict[str, List[MutableMapping[str, Any]]] = {}
    raw_frames = snapshot.get("frames")

    if isinstance(raw_frames, Mapping):
        for key, frame in raw_frames.items():
            candles: List[MutableMapping[str, Any]] = []
            raw_candles = []
            if isinstance(frame, Mapping):
                raw_candles = frame.get("candles", [])
            else:
                raw_candles = frame
            if raw_candles is None:
                raw_candles = []
            try:
                iterator = list(raw_candles)  # type: ignore[arg-type]
            except TypeError:
                iterator = []
            for entry in iterator:
                if not isinstance(entry, Mapping):
                    continue
                candle = _coerce_candle(entry)
                if candle is None:
                    continue
                candles.append(candle)
            candles.sort(key=lambda item: item["t"])
            frames[str(key).lower()] = candles

    if not frames:
        default_tf = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()
        frames[default_tf] = _normalise_candles(snapshot)

    return frames


def _timeframe_interval_ms(tf_key: str) -> int | None:
    return TIMEFRAME_TO_MS.get(tf_key)


def _infer_min_interval_ms(candles: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the smallest positive timestamp delta observed in a series."""

    prev_ts: int | None = None
    min_delta: int | None = None
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        if prev_ts is not None:
            delta = ts - prev_ts
            if delta > 0 and (min_delta is None or delta < min_delta):
                min_delta = delta
        prev_ts = ts
    return min_delta


def _series_needs_resample(
    candles: Sequence[Mapping[str, Any]],
    tf_key: str,
) -> bool:
    """Detect whether a timeframe series still contains minute-resolution bars."""

    interval_ms = _timeframe_interval_ms(tf_key)
    if interval_ms is None or interval_ms <= MINUTE_INTERVAL_MS:
        return False
    inferred = _infer_min_interval_ms(candles)
    if inferred is None:
        return False
    return inferred < interval_ms


def _resample_minutes_to_tf(
    minute_candles: Sequence[Mapping[str, Any]],
    tf_key: str,
) -> List[Dict[str, Any]]:
    """Aggregate 1m candles into the requested timeframe."""

    interval_ms = _timeframe_interval_ms(tf_key)
    if interval_ms is None or interval_ms <= MINUTE_INTERVAL_MS:
        return []
    aggregated = resample_ohlcv(minute_candles, interval_ms)
    aggregated.sort(key=lambda candle: _safe_int(candle.get("t")) or 0)
    return [
        {
            "t": _safe_int(candle.get("t")) or 0,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        }
        for candle in aggregated
        if _safe_int(candle.get("t")) is not None
    ]


def _ensure_minute_frame(
    frames: MutableMapping[str, List[MutableMapping[str, Any]]],
    *,
    primary_key: str,
    primary_candles: Sequence[Mapping[str, Any]],
) -> None:
    existing = frames.get("1m")
    if isinstance(existing, list) and existing:
        return

    if primary_key == "1m":
        frames["1m"] = [dict(candle) for candle in primary_candles]
        return

    ordered_frames = sorted(
        frames.items(),
        key=lambda item: _timeframe_interval_ms(item[0]) or float("inf"),
    )

    for tf_key, candles in ordered_frames:
        interval_ms = _timeframe_interval_ms(tf_key)
        if interval_ms is None or interval_ms < MINUTE_INTERVAL_MS:
            continue
        ratio = interval_ms // MINUTE_INTERVAL_MS
        if ratio <= 0:
            continue
        expanded: List[MutableMapping[str, Any]] = []
        for candle in candles:
            ts = _safe_int(candle.get("t"))
            if ts is None:
                continue
            open_price = _coerce_float(candle.get("o"))
            high_price = _coerce_float(candle.get("h"))
            low_price = _coerce_float(candle.get("l"))
            close_price = _coerce_float(candle.get("c"))
            volume = _coerce_float(candle.get("v"))
            portion = volume / ratio if ratio else volume
            for idx in range(ratio):
                expanded.append(
                    {
                        "t": ts + idx * MINUTE_INTERVAL_MS,
                        "o": open_price,
                        "h": high_price,
                        "l": low_price,
                        "c": close_price,
                        "v": portion,
                    }
                )
        if expanded:
            frames["1m"] = expanded
            return

    frames["1m"] = [
        {
            "t": _safe_int(candle.get("t")) or 0,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        }
        for candle in primary_candles
    ]


def _latest_candle_before(
    candles: Sequence[Mapping[str, Any]],
    *,
    end_ms: int | None,
) -> Mapping[str, Any] | None:
    if not candles:
        return None
    if end_ms is None:
        return candles[-1]
    for entry in reversed(candles):
        if not isinstance(entry, Mapping):
            continue
        ts = _safe_int(entry.get("t"))
        if ts is None:
            continue
        if ts <= end_ms:
            return entry
    return candles[-1]


def _primary_frame_key(snapshot: Mapping[str, Any], frames: Mapping[str, Sequence[Mapping[str, Any]]]) -> str | None:
    preferred = str(snapshot.get("tf") or snapshot.get("timeframe") or "").lower()
    if preferred and preferred in frames:
        return preferred
    if frames:
        return next(iter(frames))
    return None


def _filter_candles(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> List[Dict[str, Any]]:
    if start_ms is None and end_ms is None:
        return [dict(candle) for candle in candles]

    result: List[Dict[str, Any]] = []
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts > end_ms:
            continue
        result.append({
            "t": ts,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        })
    return result


def _summarise(candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not candles:
        return {"count": 0, "open": None, "close": None, "high": None, "low": None, "volume": 0.0}

    highs = [float(item.get("h", 0.0)) for item in candles]
    lows = [float(item.get("l", 0.0)) for item in candles]
    volumes = [float(item.get("v", 0.0)) for item in candles]

    return {
        "count": len(candles),
        "open": float(candles[0].get("o", 0.0)),
        "close": float(candles[-1].get("c", 0.0)),
        "high": max(highs) if highs else None,
        "low": min(lows) if lows else None,
        "volume": float(sum(volumes)),
    }


def _build_delta_series(candles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    cumulative = 0.0
    for candle in candles:
        open_price = float(candle.get("o", 0.0))
        close_price = float(candle.get("c", 0.0))
        volume = float(candle.get("v", 0.0))
        net = (close_price - open_price) * volume
        cumulative += net
        delta_pct = ((close_price - open_price) / open_price * 100.0) if open_price else 0.0
        series.append(
            {
                "t": candle.get("t"),
                "delta": net,
                "deltaPct": delta_pct,
                "cvd": cumulative,
            }
        )
    return series


def _summarise_delta_series(series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not series:
        return {"count": 0, "net_delta": 0.0, "cvd_change": 0.0, "delta_pct_total": 0.0}
    net = sum(float(item.get("delta", 0.0)) for item in series)
    cvd_change = float(series[-1].get("cvd", 0.0)) - float(series[0].get("cvd", 0.0))
    delta_pct_total = sum(float(item.get("deltaPct", 0.0)) for item in series)
    return {
        "count": len(series),
        "net_delta": net,
        "cvd_change": cvd_change,
        "delta_pct_total": delta_pct_total,
    }


def _resolve_orderflow_config(meta: Mapping[str, Any] | None) -> OrderflowConfig:
    if not isinstance(meta, Mapping):
        return OrderflowConfig()

    source = None
    for key in ("orderflow", "order_flow", "orderFlow"):
        candidate = meta.get(key)
        if isinstance(candidate, Mapping):
            source = candidate
            break

    if source is None:
        return OrderflowConfig()

    config = OrderflowConfig()

    def _float(name: str, default: float) -> float:
        value = source.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int) -> int:
        value = source.get(name)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    for key in ("imbalance_ratio", "imbalanceThreshold", "imbalance_threshold"):
        if key in source:
            config.imbalance_ratio = max(0.0, _float(key, config.imbalance_ratio))
            break

    for key in ("absorption_ratio", "absorptionThreshold", "absorption_threshold"):
        if key in source:
            config.absorption_ratio = max(0.0, _float(key, config.absorption_ratio))
            break

    for key in ("atr_period", "atrPeriod"):
        if key in source:
            config.atr_period = max(1, _int(key, config.atr_period))
            break

    for key in ("atr_band_k", "atrBandK", "atr_band_multiplier"):
        if key in source:
            config.atr_band_k = max(0.0, _float(key, config.atr_band_k))
            break

    for key in ("large_trade_min_qty", "large_trade_qty", "large_trade_trigger"):
        if key in source:
            config.large_trade_min_qty = max(0.0, _float(key, config.large_trade_min_qty))
            break

    for key in ("large_trade_lookback_minutes", "largeTradeLookbackMinutes"):
        if key in source:
            config.large_trade_lookback_minutes = max(1, _int(key, config.large_trade_lookback_minutes))
            break

    for key in ("large_trade_percentile", "largeTradePercentile"):
        if key in source:
            percentile = _float(key, config.large_trade_percentile)
            if 0.0 < percentile < 1.0:
                config.large_trade_percentile = percentile
            break

    if "epsilon" in source:
        config.epsilon = max(1e-12, _float("epsilon", config.epsilon))

    return config



def _build_agg_trade_minutes(
    trades: Sequence[Mapping[str, Any]] | None,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> Dict[str, Any]:
    """Aggregate Binance agg trade rows into per-minute metrics."""

    minute_ms = MINUTE_INTERVAL_MS
    if not trades:
        return {"status": "insufficient_data", "minutes": [], "range": None, "totals": None}

    minute_stats: Dict[int, Dict[str, float]] = {}
    seen_ids: Set[int] = set()
    total_buy = 0.0
    total_sell = 0.0
    min_minute: int | None = None
    max_minute: int | None = None

    for index, trade in enumerate(trades):
        if not isinstance(trade, Mapping):
            continue

        ts_value = trade.get("T")
        if ts_value is None:
            ts_value = trade.get("t")
        ts = _safe_int(ts_value)
        if ts is None:
            continue

        agg_id_value = trade.get("a")
        agg_id: int | None
        try:
            agg_id = int(agg_id_value) if agg_id_value is not None else None
        except (TypeError, ValueError):
            agg_id = None
        if agg_id is not None:
            if agg_id in seen_ids:
                continue
            seen_ids.add(agg_id)

        price_value = trade.get("p", trade.get("price"))
        qty_value = trade.get("q", trade.get("qty"))
        if price_value is None:
            raise ValueError(f"missing price at index {index}")
        if qty_value is None:
            raise ValueError(f"missing quantity at index {index}")

        try:
            price = float(price_value)
        except (TypeError, ValueError):
            raise ValueError(f"invalid price at index {index}") from None
        try:
            quantity = float(qty_value)
        except (TypeError, ValueError):
            raise ValueError(f"invalid quantity at index {index}") from None

        if not math.isfinite(price):
            raise ValueError(f"invalid price at index {index}")
        if not math.isfinite(quantity):
            raise ValueError(f"invalid quantity at index {index}")
        if quantity <= 0.0:
            continue

        maker_flag = trade.get("m")
        is_sell: bool
        if isinstance(maker_flag, bool):
            is_sell = maker_flag
        elif maker_flag in (0, 1):
            is_sell = bool(maker_flag)
        elif isinstance(maker_flag, str):
            lowered = maker_flag.strip().lower()
            if lowered in {"true", "1", "t", "yes", "y"}:
                is_sell = True
            elif lowered in {"false", "0", "f", "no", "n"}:
                is_sell = False
            else:
                side_candidate = str(trade.get("side") or "").strip().lower()
                is_sell = side_candidate == "sell"
        else:
            side_candidate = str(trade.get("side") or "").strip().lower()
            if side_candidate == "sell":
                is_sell = True
            elif side_candidate == "buy":
                is_sell = False
            else:
                is_sell = False

        minute_ts = _align_to_interval(ts, minute_ms)
        stats = minute_stats.get(minute_ts)
        if stats is None:
            stats = {"buy_vol": 0.0, "sell_vol": 0.0, "notional": 0.0, "qty": 0.0}
            minute_stats[minute_ts] = stats

        if is_sell:
            stats["sell_vol"] += quantity
            total_sell += quantity
        else:
            stats["buy_vol"] += quantity
            total_buy += quantity
        stats["qty"] += quantity
        stats["notional"] += price * quantity

        if min_minute is None or minute_ts < min_minute:
            min_minute = minute_ts
        if max_minute is None or minute_ts > max_minute:
            max_minute = minute_ts

    if not minute_stats or min_minute is None or max_minute is None:
        return {"status": "insufficient_data", "minutes": [], "range": None, "totals": None}

    if start_ms is not None:
        aligned_start = _align_to_interval(start_ms, minute_ms)
    else:
        aligned_start = min_minute

    if end_ms is not None and end_ms > 0:
        adjusted_end = end_ms - 1
        if adjusted_end < 0:
            adjusted_end = 0
        aligned_end = _align_to_interval(adjusted_end, minute_ms)
    else:
        aligned_end = max_minute

    range_start = min(aligned_start, min_minute)
    range_end = max(aligned_end, max_minute)

    minutes: List[Dict[str, Any]] = []
    running_cvd = 0.0
    cursor = range_start
    while cursor <= range_end:
        stats = minute_stats.get(cursor)
        if stats is not None:
            buy_vol = stats["buy_vol"]
            sell_vol = stats["sell_vol"]
            delta = buy_vol - sell_vol
            running_cvd += delta
            total_vol = buy_vol + sell_vol
            qty_sum = stats["qty"]
            vwap = stats["notional"] / qty_sum if qty_sum > 0.0 else None
            minutes.append(
                {
                    "ts_min": cursor,
                    "vol": total_vol,
                    "buy_vol": buy_vol,
                    "sell_vol": sell_vol,
                    "delta": delta,
                    "cvd": running_cvd,
                    "vwap_min": vwap,
                }
            )
        else:
            minutes.append(
                {
                    "ts_min": cursor,
                    "vol": 0.0,
                    "buy_vol": 0.0,
                    "sell_vol": 0.0,
                    "delta": 0.0,
                    "cvd": running_cvd,
                    "vwap_min": None,
                }
            )
        cursor += minute_ms

    status = "ok" if minutes else "insufficient_data"
    totals = {
        "vol": total_buy + total_sell,
        "buy_vol": total_buy,
        "sell_vol": total_sell,
        "delta": total_buy - total_sell,
        "cvd_close": minutes[-1]["cvd"] if minutes else 0.0,
    } if minutes else None

    return {
        "status": status,
        "minutes": minutes,
        "range": {
            "start_ms": range_start,
            "end_ms": range_end,
            "count": len(minutes),
        },
        "totals": totals,
    }




def _compute_vwap(candles: Sequence[Mapping[str, Any]]) -> float:
    total_pv = 0.0
    total_volume = 0.0
    for candle in candles:
        high = float(candle.get("h", 0.0))
        low = float(candle.get("l", 0.0))
        close = float(candle.get("c", 0.0))
        volume = float(candle.get("v", 0.0))
        typical_price = (high + low + close) / 3.0
        total_pv += typical_price * volume
        total_volume += volume
    if total_volume <= 0:
        return 0.0
    return total_pv / total_volume


def _compute_vwap_stats(
    candles: Sequence[Mapping[str, Any]]
) -> Tuple[float, float] | None:
    total_pv = 0.0
    total_p2v = 0.0
    total_volume = 0.0
    valid = 0
    for candle in candles:
        volume = float(candle.get("v", 0.0))
        if volume <= 0.0:
            continue
        high = float(candle.get("h", 0.0))
        low = float(candle.get("l", 0.0))
        close = float(candle.get("c", 0.0))
        typical_price = (high + low + close) / 3.0
        if not math.isfinite(typical_price):
            continue
        total_pv += typical_price * volume
        total_p2v += typical_price * typical_price * volume
        total_volume += volume
        valid += 1
    if total_volume <= 0.0:
        return None
    value = total_pv / total_volume
    if valid < 2:
        sigma = 0.0
    else:
        variance = max(total_p2v / total_volume - value * value, 0.0)
        sigma = math.sqrt(variance)
    return value, sigma


def _build_sigma_levels(center: float, sigma: float) -> List[Dict[str, float]]:
    return [
        {"k": k, "price_minus": center - sigma * k, "price_plus": center + sigma * k}
        for k in (1, 2)
    ]


def _build_vwap_sigma_block(
    candles: Sequence[Mapping[str, Any]], *, basis: str
) -> Dict[str, Any]:
    stats = _compute_vwap_stats(candles)
    if stats is None:
        center = _compute_vwap(candles)
        sigma = 0.0
    else:
        center, sigma = stats
    return {"basis": basis, "sigma": _build_sigma_levels(center, sigma)}


def _typical_price(candle: Mapping[str, Any]) -> float:
    high = float(candle.get("h", 0.0))
    low = float(candle.get("l", 0.0))
    close = float(candle.get("c", 0.0))
    return (high + low + close) / 3.0


def _determine_bin_size(prices: Sequence[float], tick_size: float | None) -> float | None:
    finite_prices = [price for price in prices if math.isfinite(price)]
    if not finite_prices:
        return float(tick_size) if tick_size and tick_size > 0 else None

    average_price = sum(finite_prices) / len(finite_prices)
    adaptive_step = abs(average_price) * 1e-4
    if adaptive_step <= 0:
        adaptive_step = max(abs(finite_prices[0]) * 1e-4, 1e-6)

    tick = float(tick_size) if tick_size and tick_size > 0 else None
    step = adaptive_step if adaptive_step > 0 else None
    if tick is not None:
        if step is None:
            return tick
        return max(tick, step)
    return step


def _build_volume_profile_stats(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    tick_size: float | None,
    value_area_pct: float = VALUE_AREA_PCT,
) -> Dict[str, Any]:
    window_start_iso = _isoformat_utc(start_ms)
    window_end_iso = _isoformat_utc(end_ms)

    if end_ms < start_ms:
        return {
            "vwap": 0.0,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    scoped = [
        candle
        for candle in candles
        if isinstance(candle, Mapping)
        and (ts := _safe_int(candle.get("t"))) is not None
        and start_ms <= ts <= end_ms
    ]

    if not scoped:
        return {
            "vwap": 0.0,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    session_high: float | None = None
    session_low: float | None = None

    def _attach_extrema(payload: Dict[str, Any]) -> Dict[str, Any]:
        if session_high is not None and session_low is not None:
            payload["session_high"] = session_high
            payload["session_low"] = session_low
        return payload

    vwap_value = _compute_vwap(scoped)
    prices: List[float] = []
    volumes: List[float] = []
    for candle in scoped:
        high_value = _safe_float(candle.get("h") or candle.get("high"))
        low_value = _safe_float(candle.get("l") or candle.get("low"))
        if high_value is not None:
            session_high = (
                high_value if session_high is None else max(session_high, high_value)
            )
        if low_value is not None:
            session_low = low_value if session_low is None else min(session_low, low_value)

        volume = float(candle.get("v", 0.0))
        if volume <= 0:
            continue
        price = _typical_price(candle)
        if not math.isfinite(price):
            continue
        prices.append(price)
        volumes.append(volume)

    if not prices or not volumes:
        return _attach_extrema(
            {
                "vwap": vwap_value,
                "poc": None,
                "vah": None,
                "val": None,
                "window": {"start": window_start_iso, "end": window_end_iso},
            }
        )

    bin_size = _determine_bin_size(prices, tick_size)
    if not bin_size or bin_size <= 0:
        return _attach_extrema(
            {
                "vwap": vwap_value,
                "poc": None,
                "vah": None,
                "val": None,
                "window": {"start": window_start_iso, "end": window_end_iso},
            }
        )

    min_price = min(prices)
    max_price = max(prices)
    start_bin = math.floor(min_price / bin_size) * bin_size
    bins_count = max(1, int(math.floor((max_price - start_bin) / bin_size)) + 1)

    histogram = [0.0 for _ in range(bins_count)]
    for price, volume in zip(prices, volumes):
        index = int(math.floor((price - start_bin) / bin_size + 1e-9))
        if index < 0:
            index = 0
        elif index >= bins_count:
            index = bins_count - 1
        histogram[index] += volume

    total_volume = sum(histogram)
    if total_volume <= 0:
        return _attach_extrema(
            {
                "vwap": vwap_value,
                "poc": None,
                "vah": None,
                "val": None,
                "window": {"start": window_start_iso, "end": window_end_iso},
            }
        )

    poc_index = max(range(len(histogram)), key=lambda idx: histogram[idx])
    poc_price = start_bin + poc_index * bin_size

    threshold = total_volume * max(0.0, min(1.0, value_area_pct))
    coverage = histogram[poc_index]
    left = right = poc_index

    while coverage < threshold and (left > 0 or right < len(histogram) - 1):
        next_left = histogram[left - 1] if left > 0 else -1.0
        next_right = histogram[right + 1] if right < len(histogram) - 1 else -1.0

        if next_left < 0 and next_right < 0:
            break

        if next_right > next_left:
            right += 1
            coverage += max(0.0, next_right)
        elif next_left > next_right:
            left -= 1
            coverage += max(0.0, next_left)
        else:
            if next_left >= 0 and left > 0:
                left -= 1
                coverage += max(0.0, next_left)
            if coverage < threshold and next_right >= 0 and right < len(histogram) - 1:
                right += 1
                coverage += max(0.0, next_right)

    val_price = start_bin + left * bin_size
    vah_price = start_bin + right * bin_size

    return _attach_extrema(
        {
            "vwap": vwap_value,
            "poc": round(poc_price, 12),
            "vah": round(vah_price, 12),
            "val": round(val_price, 12),
            "window": {"start": window_start_iso, "end": window_end_iso},
        }
    )


def _build_prev_day_block(
    candles: Sequence[Mapping[str, Any]],
    *,
    daily_start_ms: int,
    tick_size: float | None,
) -> Dict[str, float | None]:
    """Compute previous-day reference levels from minute candles."""

    prev_end_ms = daily_start_ms - MINUTE_INTERVAL_MS
    prev_start_ms = daily_start_ms - MS_IN_DAY
    if prev_end_ms < prev_start_ms:
        prev_end_ms = prev_start_ms

    scoped = _filter_candles(candles, start_ms=prev_start_ms, end_ms=prev_end_ms)
    summary = _summarise(scoped)
    profile = _build_volume_profile_stats(
        candles,
        start_ms=prev_start_ms,
        end_ms=prev_end_ms,
        tick_size=tick_size,
        value_area_pct=VALUE_AREA_PCT,
    )

    def _float_or_none(value: Any) -> float | None:
        return _safe_float(value)

    close_value: float | None = None
    if scoped:
        close_value = _safe_float(scoped[-1].get("c"))

    return {
        "pdh": _float_or_none(summary.get("high")),
        "pdl": _float_or_none(summary.get("low")),
        "close": close_value,
        "poc": _float_or_none((profile or {}).get("poc")),
        "vah": _float_or_none((profile or {}).get("vah")),
        "val": _float_or_none((profile or {}).get("val")),
    }


def _start_of_day_ms(timestamp_ms: int) -> int:
    dt = safe_datetime_from_ms(timestamp_ms, UTC)
    if dt is None:
        return 0
    start_dt = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    return int(start_dt.timestamp() * 1000)


def _session_window(
    anchor_ms: int,
    start_time: dtime,
    end_time: dtime,
    *,
    session_tz: timezone = VWAP_SESSION_TZ,
) -> tuple[int, int, int]:
    anchor_aligned = _align_to_interval(anchor_ms, MINUTE_INTERVAL_MS)
    anchor_dt_utc = safe_datetime_from_ms(anchor_aligned, UTC)
    if anchor_dt_utc is None:
        anchor_dt_utc = datetime(1970, 1, 1, tzinfo=UTC)

    tz = session_tz or UTC
    anchor_local = anchor_dt_utc.astimezone(tz)
    session_start_local = datetime.combine(anchor_local.date(), start_time, tzinfo=tz)
    session_end_local = datetime.combine(anchor_local.date(), end_time, tzinfo=tz)

    if end_time <= start_time:
        session_end_local += timedelta(days=1)

    session_start_utc = session_start_local.astimezone(UTC)
    session_end_utc = session_end_local.astimezone(UTC)

    start_ms = int(session_start_utc.timestamp() * 1000)
    end_boundary_ms = int(session_end_utc.timestamp() * 1000)
    raw_end_ms = end_boundary_ms - MINUTE_INTERVAL_MS
    if raw_end_ms < start_ms:
        raw_end_ms = start_ms

    end_ms = min(raw_end_ms, anchor_aligned)
    close_ms = end_boundary_ms

    if end_ms < start_ms:
        end_ms = start_ms
    return start_ms, end_ms, close_ms


def _session_window_for_day(
    session_day: date,
    start_time: dtime,
    end_time: dtime,
    *,
    session_tz: timezone = VWAP_SESSION_TZ,
) -> tuple[int, int, int]:
    tz = session_tz or UTC
    session_start_local = datetime.combine(session_day, start_time, tzinfo=tz)
    session_end_local = datetime.combine(session_day, end_time, tzinfo=tz)
    if end_time <= start_time:
        session_end_local += timedelta(days=1)

    session_start_utc = session_start_local.astimezone(UTC)
    session_end_utc = session_end_local.astimezone(UTC)
    start_ms = int(session_start_utc.timestamp() * 1000)
    close_ms = int(session_end_utc.timestamp() * 1000)
    end_ms = max(start_ms, close_ms - MINUTE_INTERVAL_MS)
    return start_ms, end_ms, close_ms


def _compute_initial_balance_extrema(
    candles: Sequence[Mapping[str, Any]],
    *,
    session_start_ms: int,
    minutes: int = 60,
) -> tuple[float | None, float | None]:
    """Determine the high/low for the initial balance slice of a session."""

    if minutes <= 0:
        return None, None

    cutoff_ms = session_start_ms + minutes * MINUTE_INTERVAL_MS
    ib_high: float | None = None
    ib_low: float | None = None

    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None or ts < session_start_ms or ts >= cutoff_ms:
            continue
        high_val = _safe_float(candle.get("h"))
        low_val = _safe_float(candle.get("l"))
        if high_val is not None:
            ib_high = high_val if ib_high is None else max(ib_high, high_val)
        if low_val is not None:
            ib_low = low_val if ib_low is None else min(ib_low, low_val)

    return ib_high, ib_low


def _compute_session_completeness(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    close_ms: int,
    interval_ms: int,
    timeframe: str,
) -> Dict[str, Any]:
    """Estimate how much of the session window is covered by the supplied candles."""

    interval = interval_ms if interval_ms and interval_ms > 0 else MINUTE_INTERVAL_MS
    expected_span_ms = max(0, close_ms - start_ms)
    if interval <= 0:
        interval = MINUTE_INTERVAL_MS
    expected_bars = expected_span_ms // interval if expected_span_ms > 0 else 0

    seen: Set[int] = set()
    earliest: int | None = None
    latest: int | None = None
    actual_end = min(end_ms, close_ms - (interval if interval > 0 else MINUTE_INTERVAL_MS))
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        if ts < start_ms or ts > actual_end:
            continue
        if ts in seen:
            continue
        seen.add(ts)
        if earliest is None or ts < earliest:
            earliest = ts
        if latest is None or ts > latest:
            latest = ts

    observed_bars = len(seen)
    missing_bars = max(expected_bars - observed_bars, 0)
    if expected_bars <= 0:
        coverage_ratio = 1.0 if observed_bars == 0 else 0.0
    else:
        coverage_ratio = min(1.0, observed_bars / expected_bars)

    if expected_bars <= 0:
        status = "na"
    elif observed_bars == 0:
        status = "empty"
    elif coverage_ratio >= 0.999:
        status = "complete"
    else:
        status = "partial"

    return {
        "tf": timeframe,
        "interval_ms": interval,
        "bars_expected": expected_bars,
        "bars_observed": observed_bars,
        "missing_bars": missing_bars,
        "coverage_ratio": coverage_ratio,
        "status": status,
        "observed_window": {
            "start_ms": earliest,
            "end_ms": latest,
        },
    }


def _normalise_zone_status(entry: Mapping[str, Any], zone_key: str) -> str:
    """Map raw detector status to the canonical shortlist vocabulary."""

    raw_status = str(entry.get("status") or "").strip().lower()
    if zone_key == "fvg":
        mitigation_value = str(entry.get("mitigation") or "").strip().lower()
        if mitigation_value == "mitigated":
            return "mitigated"
        if raw_status in {"fulfilled", "fill"}:
            return "tapped"
        if raw_status == "inverted":
            return "tapped" if entry.get("inverted") else "open"
        if not raw_status:
            return "open"
        if raw_status not in ZONE_ALLOWED_STATUSES:
            return "open"
        return raw_status
    if zone_key == "ob":
        fill_ratio = _safe_float(entry.get("fill_ratio"))
        if fill_ratio is not None and fill_ratio >= 0.6 and raw_status != "invalidated":
            return "mitigated"
        if not raw_status:
            return "fresh"
        if raw_status not in ZONE_ALLOWED_STATUSES:
            if raw_status in {"fulfilled", "fill"}:
                return "mitigated"
            return "fresh"
        return raw_status
    return raw_status


def _latest_atr_value(
    candles: Sequence[Mapping[str, Any]],
    *,
    period: int = 14,
) -> float | None:
    """Return the latest finite ATR value for the supplied candle slice."""

    if not candles or period <= 0:
        return None
    if len(candles) <= period:
        return None
    atr_series = compute_atr(candles, period=period)
    for value in reversed(atr_series):
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
    return None


def _session_atr_value(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    period: int = 14,
) -> float | None:
    """Compute ATR for a session window using the configured period."""

    if not candles:
        return None
    scoped = _filter_candles(candles, start_ms=start_ms, end_ms=end_ms)
    if not scoped:
        return None
    return _latest_atr_value(scoped, period=period)


def _zone_focus_timestamp(entry: Mapping[str, Any]) -> int | None:
    """Extract the most representative timestamp from a zone payload."""

    for key in ("created_utc", "origin_utc", "formed_at_utc", "last_touched_utc"):
        iso_value = entry.get(key)
        ts_iso = _iso_to_ms(iso_value)
        if ts_iso is not None:
            return ts_iso
    for key in ("created_at", "origin_ms", "ts", "t"):
        ts_numeric = _safe_int(entry.get(key))
        if ts_numeric is not None:
            return ts_numeric
    return None


def _zone_reference_price(entry: Mapping[str, Any], zone_key: str) -> float | None:
    """Return the mid-price used for distance calculations."""

    if zone_key == "fvg":
        mid_value = _safe_float(entry.get("mid"))
        if mid_value is not None:
            return mid_value
        top_value = _safe_float(entry.get("top"))
        bot_value = _safe_float(entry.get("bot"))
        if top_value is not None and bot_value is not None:
            return (top_value + bot_value) / 2.0
        return None
    mean_value = _safe_float(entry.get("mean"))
    if mean_value is not None:
        return mean_value
    open_value = _safe_float(entry.get("open"))
    close_value = _safe_float(entry.get("close"))
    if open_value is not None and close_value is not None:
        return (open_value + close_value) / 2.0
    return None


def _build_zone_focus(
    zones: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    *,
    last_price: float | None,
    reference_ms: int,
    window_hours: int = ZONE_FOCUS_WINDOW_HOURS,
    top_limits: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Assemble a 72h shortlist of the most relevant FVG and OB zones."""

    window_hours = max(1, int(window_hours))
    window_ms = window_hours * MS_IN_HOUR
    cutoff_ms = max(0, reference_ms - window_ms)
    status_priority = {"fresh": 0, "open": 0, "tapped": 1, "mitigated": 2, "invalidated": 3}

    focus_payload: Dict[str, Any] = {
        "window_hours": window_hours,
        "cutoff_utc": _isoformat_utc(cutoff_ms),
        "last_price": last_price,
        "fvg": [],
        "ob": [],
        "meta": {
            "counts": {"fvg": 0, "ob": 0},
            "reasons": {"fvg": [], "ob": []},
            "distance_reference_available": last_price is not None,
        },
    }

    if last_price is None:
        focus_payload["meta"].setdefault("notes", []).append("last_price_unavailable")

    def _process(zone_key: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        reason_flags: set[str] = set()
        if not isinstance(zones, Mapping):
            reason_flags.add("no_candidates")
            return [], sorted(reason_flags)
        series = zones.get(zone_key)
        if not isinstance(series, Sequence) or not series:
            reason_flags.add("no_candidates")
            return [], sorted(reason_flags)

        collected: List[Tuple[Tuple[float, float, float], Dict[str, Any]]] = []
        for entry in series:
            if not isinstance(entry, Mapping):
                continue
            tf_value_raw = entry.get("tf")
            tf_value = str(tf_value_raw).lower() if isinstance(tf_value_raw, str) else str(tf_value_raw or "").lower()
            if tf_value not in ZONE_FOCUS_ALLOWED_TFS:
                reason_flags.add(f"tf_excluded:{tf_value or 'unknown'}")
                continue

            formed_ms = _zone_focus_timestamp(entry)
            if formed_ms is None:
                reason_flags.add("missing_timestamp")
                continue
            if formed_ms < cutoff_ms:
                reason_flags.add("stale")
                continue

            status_value = _normalise_zone_status(entry, zone_key)
            if status_value and status_value not in ZONE_ALLOWED_STATUSES:
                reason_flags.add(f"status_excluded:{status_value}")
                continue

            reference_price = _zone_reference_price(entry, zone_key)
            if reference_price is None:
                reason_flags.add("missing_price")
                continue

            distance: float | None = None
            distance_pct: float | None = None
            if last_price is not None:
                distance = abs(reference_price - last_price)
                if last_price:
                    distance_pct = distance / last_price

            last_touch_iso = None
            for key in ("last_touched_utc", "last_touch_utc", "last_touched_iso", "last_touch_iso"):
                if last_touch_iso:
                    break
                candidate = entry.get(key)
                coerced = _coerce_iso_timestamp(candidate)
                if coerced:
                    last_touch_iso = coerced
            if last_touch_iso is None:
                last_touch_iso = _coerce_iso_timestamp(entry.get("last_touched"))
            if last_touch_iso is None:
                last_touch_iso = _coerce_iso_timestamp(entry.get("last_touch_ms"))

            age_minutes_float = max(0.0, (reference_ms - formed_ms) / 60000.0)
            age_minutes = int(age_minutes_float)
            age_hours = round(age_minutes_float / 60.0, 3)

            payload: Dict[str, Any] = {
                "tf": entry.get("tf"),
                "status": status_value,
                "formed_utc": entry.get("created_utc")
                or entry.get("origin_utc")
                or _isoformat_utc(formed_ms),
                "last_touch_utc": last_touch_iso,
                "age_minutes": age_minutes,
                "age_hours": age_hours,
                "distance": distance,
                "distance_pct": distance_pct,
            }
            if zone_key == "fvg":
                payload.update(
                    {
                        "direction": entry.get("direction"),
                        "top": _safe_float(entry.get("top")),
                        "bot": _safe_float(entry.get("bot")),
                        "mid": reference_price,
                    }
                )
            else:
                payload.update(
                    {
                        "type": entry.get("type"),
                        "open": _safe_float(entry.get("open")),
                        "close": _safe_float(entry.get("close")),
                        "mean": reference_price,
                    }
                )
            sweep_links = entry.get("sweep_links")
            if isinstance(sweep_links, Sequence) and sweep_links:
                payload["sweeps"] = [dict(link) for link in sweep_links if isinstance(link, Mapping)]

            status_rank = status_priority.get(status_value, float(len(status_priority)))
            distance_rank = distance if distance is not None else float("inf")
            collected.append(((status_rank, distance_rank, -formed_ms), payload))

        if not collected:
            reason_flags.add("no_recent")
            return [], sorted(reason_flags)

        collected.sort(key=lambda item: item[0])
        shortlisted = [payload for _, payload in collected]
        return shortlisted, sorted(reason_flags)

    fvg_entries, fvg_reasons = _process("fvg")
    ob_entries, ob_reasons = _process("ob")

    if isinstance(top_limits, Mapping):
        fvg_limit = int(top_limits.get("fvg", len(fvg_entries)) or len(fvg_entries))
        ob_limit = int(top_limits.get("ob", len(ob_entries)) or len(ob_entries))
        fvg_limit = max(0, fvg_limit)
        ob_limit = max(0, ob_limit)
        focus_payload["fvg"] = fvg_entries[:fvg_limit] if fvg_limit else []
        focus_payload["ob"] = ob_entries[:ob_limit] if ob_limit else []
    else:
        focus_payload["fvg"] = fvg_entries
        focus_payload["ob"] = ob_entries

    fvg_count = len(focus_payload["fvg"])
    ob_count = len(focus_payload["ob"])
    focus_payload["meta"]["counts"]["fvg"] = fvg_count
    focus_payload["meta"]["counts"]["ob"] = ob_count
    focus_payload["meta"]["reasons"]["fvg"] = fvg_reasons
    focus_payload["meta"]["reasons"]["ob"] = ob_reasons
    focus_payload["meta"]["has_recent"] = bool(fvg_count or ob_count)

    return focus_payload


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    return False


def _validate_showcase_a_items(items: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"item[{idx}]:not_mapping")
            continue
        tf_value = item.get("tf")
        if not isinstance(tf_value, str) or not tf_value.strip():
            errors.append(f"item[{idx}].tf")
        type_value = item.get("type")
        if not isinstance(type_value, str) or not type_value.strip():
            errors.append(f"item[{idx}].type")
        price_value = item.get("price")
        if price_value is not None and not _is_number(price_value):
            errors.append(f"item[{idx}].price")
        status_value = item.get("status")
        if not isinstance(status_value, str) or not status_value.strip():
            errors.append(f"item[{idx}].status")
        formed_value = item.get("formed_at")
        if formed_value is not None and (not isinstance(formed_value, str) or not formed_value.strip()):
            errors.append(f"item[{idx}].formed_at")
        last_touch = item.get("last_touch")
        if last_touch is not None and (not isinstance(last_touch, str) or not last_touch.strip()):
            errors.append(f"item[{idx}].last_touch")
        distance_value = item.get("distance_from_last")
        if distance_value is not None and not _is_number(distance_value):
            errors.append(f"item[{idx}].distance_from_last")
        age_value = item.get("age_hours")
        if age_value is not None and not _is_number(age_value):
            errors.append(f"item[{idx}].age_hours")
    return errors


def _build_showcase_a(
    zones_focus: Mapping[str, Any] | None,
    *,
    window_hours: int,
    last_price: float | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    allowed_statuses = {"open", "fresh", "tapped"}
    items: List[Dict[str, Any]] = []
    missing_modules: Set[str] = set()
    reasons: List[str] = []

    if not isinstance(zones_focus, Mapping):
        missing_modules.add("zones.recent")
        reasons.append("zones_focus_unavailable")
        zones_focus = {}

    counts_raw = zones_focus.get("meta", {}).get("counts") if isinstance(zones_focus.get("meta"), Mapping) else {}
    counts = {
        "fvg": _safe_int((counts_raw or {}).get("fvg")) or 0,
        "ob": _safe_int((counts_raw or {}).get("ob")) or 0,
    }

    distance_reference_available = False
    meta_block = zones_focus.get("meta")
    if isinstance(meta_block, Mapping):
        distance_reference_available = bool(meta_block.get("distance_reference_available"))
        reasons_block = meta_block.get("reasons")
        if isinstance(reasons_block, Mapping):
            for reason_list in reasons_block.values():
                if isinstance(reason_list, Sequence):
                    reasons.extend(str(reason) for reason in reason_list if reason)

    if not distance_reference_available or last_price is None:
        missing_modules.add("meta.last_price")

    for zone_key in ("fvg", "ob"):
        series = zones_focus.get(zone_key)
        if not isinstance(series, Sequence):
            continue
        for entry in series:
            if not isinstance(entry, Mapping):
                continue
            status_value = str(entry.get("status") or "").lower()
            if status_value not in allowed_statuses:
                continue
            raw_tf = entry.get("tf")
            tf_value = ""
            if isinstance(raw_tf, str):
                tf_value = raw_tf.strip().lower()
            elif raw_tf is not None:
                tf_value = str(raw_tf).strip().lower()
            if not tf_value:
                tf_value = "unknown"
                reasons.append(f"{zone_key}:missing_tf")

            if zone_key == "fvg":
                direction = entry.get("direction")
                if isinstance(direction, str) and direction.strip():
                    type_value = f"fvg:{direction.strip().lower()}"
                else:
                    type_value = "fvg"
                price_value = _safe_float(entry.get("mid"))
                if price_value is None:
                    price_value = _safe_float(entry.get("top"))
                if price_value is None:
                    price_value = _safe_float(entry.get("bot"))
            else:
                subtype = entry.get("type")
                if isinstance(subtype, str) and subtype.strip():
                    type_value = f"ob:{subtype.strip().lower()}"
                else:
                    type_value = "ob"
                price_value = _safe_float(entry.get("mean"))
                if price_value is None:
                    price_value = _safe_float(entry.get("open"))
                if price_value is None:
                    price_value = _safe_float(entry.get("close"))
            if price_value is None:
                reasons.append(f"{zone_key}:missing_price")

            formed_at = entry.get("formed_utc")
            if not isinstance(formed_at, str) or not formed_at.strip():
                formed_at = None
                reasons.append(f"{zone_key}:missing_formed_at")

            last_touch = entry.get("last_touch_utc")
            if not isinstance(last_touch, str) or not last_touch.strip():
                last_touch = None

            distance_value = _safe_float(entry.get("distance"))
            age_hours_value = _safe_float(entry.get("age_hours"))

            item = {
                "tf": tf_value,
                "type": type_value,
                "price": price_value,
                "status": status_value,
                "formed_at": formed_at,
                "last_touch": last_touch,
                "distance_from_last": distance_value,
                "age_hours": age_hours_value,
            }
            items.append(item)

    validation_errors = _validate_showcase_a_items(items)

    reason_text = None
    if not items:
        reason_candidates = sorted(set(reason for reason in reasons if reason))
        reason_text = "; ".join(reason_candidates) if reason_candidates else "no_open_zones"

    completeness = {
        "window_hours": window_hours,
        "counts": counts,
        "open_entries": len(items),
        "has_last_price": distance_reference_available and last_price is not None,
        "missing_modules": sorted(missing_modules),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
    }

    showcase = {
        "items": items,
        "completeness": completeness,
        "sources": {
            "zones": "zones.recent",
            "last_price": "meta.last_price",
        },
    }
    if reason_text:
        showcase["reason"] = reason_text

    diagnostics = {
        "missing_modules": sorted(missing_modules),
        "reason_flags": sorted(set(reason for reason in reasons if reason)),
        "validation_errors": validation_errors,
    }

    return showcase, diagnostics


def _validate_showcase_b_items(items: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"item[{idx}]:not_mapping")
            continue
        session_value = item.get("session")
        if not isinstance(session_value, str) or not session_value.strip():
            errors.append(f"item[{idx}].session")
        vwap_value = item.get("vwap")
        if vwap_value is not None and not _is_number(vwap_value):
            errors.append(f"item[{idx}].vwap")
        for key in ("IBH", "IBL", "POC", "VAH", "VAL", "delta_15m", "cvd_agg"):
            field_value = item.get(key)
            if field_value is not None and not _is_number(field_value):
                errors.append(f"item[{idx}].{key}")
        for key in ("sd1", "sd2"):
            sd_block = item.get(key)
            if sd_block is None:
                continue
            if not isinstance(sd_block, Mapping):
                errors.append(f"item[{idx}].{key}")
                continue
            for sub_key in ("minus", "plus"):
                sub_value = sd_block.get(sub_key)
                if sub_value is not None and not _is_number(sub_value):
                    errors.append(f"item[{idx}].{key}.{sub_key}")
        fresh_block = item.get("fresh_intraday")
        if fresh_block is not None:
            if not isinstance(fresh_block, Mapping):
                errors.append(f"item[{idx}].fresh_intraday")
            else:
                for sub_key in ("fvg", "ob"):
                    sub_value = fresh_block.get(sub_key)
                    if sub_value is not None and not isinstance(sub_value, int):
                        errors.append(f"item[{idx}].fresh_intraday.{sub_key}")
        sweeps_block = item.get("sweeps")
        if sweeps_block is not None:
            if not isinstance(sweeps_block, list):
                errors.append(f"item[{idx}].sweeps")
            else:
                for sweep_idx, sweep in enumerate(sweeps_block):
                    if not isinstance(sweep, Mapping):
                        errors.append(f"item[{idx}].sweeps[{sweep_idx}]")
    return errors


def _build_showcase_b(
    sessions: Mapping[str, Any] | None,
    *,
    orderflow: Mapping[str, Any],
    zones_focus: Mapping[str, Any] | None,
    liquidity: Mapping[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    session_names = ("asia", "london", "ny")
    items: List[Dict[str, Any]] = []
    missing_modules: Set[str] = set()
    reasons: List[str] = []

    if not isinstance(sessions, Mapping):
        sessions = {}
        missing_modules.add("vwap_tpo.sessions")
        reasons.append("sessions_unavailable")

    sweeps_series: List[Dict[str, Any]] = []
    if isinstance(liquidity, Mapping):
        sweeps_raw = liquidity.get("sweeps")
        if isinstance(sweeps_raw, Sequence):
            sweeps_series = [dict(entry) for entry in sweeps_raw if isinstance(entry, Mapping)]
    if not sweeps_series:
        missing_modules.add("liquidity.sweeps")
        reasons.append("sweeps_missing")

    sweeps_excerpt = sweeps_series[:5]

    delta_15m_value: float | None = None
    of_15m = orderflow.get("15m") if isinstance(orderflow, Mapping) else None
    if isinstance(of_15m, Mapping):
        summary = of_15m.get("summary")
        if isinstance(summary, Mapping):
            delta_15m_value = _safe_float(summary.get("delta_sum"))
    if delta_15m_value is None:
        missing_modules.add("orderflow.15m.delta_sum")
        reasons.append("delta_15m_missing")

    cvd_agg_value: float | None = None
    of_1m = orderflow.get("1m") if isinstance(orderflow, Mapping) else None
    if isinstance(of_1m, Mapping):
        per_bar = of_1m.get("per_bar")
        if isinstance(per_bar, Sequence):
            for entry in reversed(per_bar):
                if not isinstance(entry, Mapping):
                    continue
                cvd_candidate = _safe_float(entry.get("cvd"))
                if cvd_candidate is not None:
                    cvd_agg_value = cvd_candidate
                    break
    if cvd_agg_value is None:
        missing_modules.add("orderflow.1m.cvd")
        reasons.append("cvd_missing")

    fresh_counts = {"fvg": 0, "ob": 0}
    if isinstance(zones_focus, Mapping):
        for zone_key in ("fvg", "ob"):
            series = zones_focus.get(zone_key)
            if isinstance(series, Sequence):
                for entry in series:
                    if not isinstance(entry, Mapping):
                        continue
                    if str(entry.get("status") or "").lower() == "fresh":
                        fresh_counts[zone_key] += 1
    else:
        missing_modules.add("zones.recent")
        reasons.append("zones_focus_unavailable")

    sessions_presence: Dict[str, bool] = {}

    def _clone_sd(block: Mapping[str, Any] | None) -> Dict[str, float | None]:
        result = {"minus": None, "plus": None}
        if not isinstance(block, Mapping):
            return result
        result["minus"] = _safe_float(block.get("minus"))
        result["plus"] = _safe_float(block.get("plus"))
        return result

    for session_name in session_names:
        session_payload = sessions.get(session_name)
        if not isinstance(session_payload, Mapping) or not session_payload:
            sessions_presence[session_name] = False
            missing_modules.add(f"vwap_tpo.sessions.{session_name}")
            reasons.append(f"session_missing:{session_name}")
            continue

        sessions_presence[session_name] = True
        item = {
            "session": session_name,
            "vwap": _safe_float(session_payload.get("vwap")),
            "sd1": _clone_sd(session_payload.get("sd1")),  # type: ignore[arg-type]
            "sd2": _clone_sd(session_payload.get("sd2")),  # type: ignore[arg-type]
            "IBH": _safe_float(session_payload.get("ib_high")),
            "IBL": _safe_float(session_payload.get("ib_low")),
            "POC": _safe_float(session_payload.get("poc")),
            "VAH": _safe_float(session_payload.get("vah")),
            "VAL": _safe_float(session_payload.get("val")),
            "fresh_intraday": dict(fvg=fresh_counts["fvg"], ob=fresh_counts["ob"]),
            "delta_15m": delta_15m_value,
            "cvd_agg": cvd_agg_value,
            "sweeps": sweeps_excerpt,
        }
        items.append(item)

    validation_errors = _validate_showcase_b_items(items)

    reason_text = None
    if not items:
        reason_candidates = sorted(set(reason for reason in reasons if reason))
        reason_text = "; ".join(reason_candidates) if reason_candidates else "sessions_missing"

    completeness = {
        "sessions": sessions_presence,
        "missing_modules": sorted(missing_modules),
        "has_delta_15m": delta_15m_value is not None,
        "has_cvd": cvd_agg_value is not None,
        "has_sweeps": bool(sweeps_series),
        "fresh_intraday": fresh_counts,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
    }

    showcase = {
        "items": items,
        "completeness": completeness,
        "sources": {
            "sessions": "vwap_tpo.sessions",
            "orderflow_15m": "orderflow.15m.summary",
            "orderflow_1m": "orderflow.1m.per_bar",
            "zones": "zones.recent",
            "liquidity": "liquidity.sweeps",
        },
    }
    if reason_text:
        showcase["reason"] = reason_text

    diagnostics = {
        "missing_modules": sorted(missing_modules),
        "reason_flags": sorted(set(reason for reason in reasons if reason)),
        "validation_errors": validation_errors,
        "fresh_counts": fresh_counts,
        "delta_15m": delta_15m_value,
        "cvd_agg": cvd_agg_value,
        "sweeps_sample": len(sweeps_excerpt),
    }

    return showcase, diagnostics


def _build_final_showcases(
    zones_focus: Mapping[str, Any] | None,
    *,
    window_hours: int,
    last_price: float | None,
    sessions: Mapping[str, Any] | None,
    orderflow: Mapping[str, Any],
    liquidity: Mapping[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    showcase_a, diag_a = _build_showcase_a(
        zones_focus,
        window_hours=window_hours,
        last_price=last_price,
    )
    showcase_b, diag_b = _build_showcase_b(
        sessions,
        orderflow=orderflow,
        zones_focus=zones_focus,
        liquidity=liquidity,
    )

    return {"A": showcase_a, "B": showcase_b}, {"A": diag_a, "B": diag_b}


def _quantise_price(value: float | None, tick_size: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    if tick_size is None or tick_size <= 0:
        return float(value)
    return round(value / tick_size) * tick_size


def _dedupe_sweeps(
    sweeps: Sequence[Mapping[str, Any]] | None,
    *,
    tick_size: float | None,
) -> List[Dict[str, Any]]:
    if not sweeps:
        return []
    deduped: Dict[tuple[Any, Any, str], Dict[str, Any]] = {}
    for entry in sweeps:
        if not isinstance(entry, Mapping):
            continue
        ts = _safe_int(entry.get("t"))
        level_price = _safe_float(entry.get("level_price"))
        sweep_type = str(entry.get("type") or "")
        if ts is None or level_price is None or not sweep_type:
            continue
        quant_price = _quantise_price(level_price, tick_size)
        key = (ts, quant_price, sweep_type)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = dict(entry)
            continue
        current_tf = str(existing.get("tf") or "")
        new_tf = str(entry.get("tf") or "")
        # Prefer finer timeframe (minutes before hours)
        if current_tf.endswith("h") and new_tf.endswith("m"):
            deduped[key] = dict(entry)
    return list(deduped.values())


def _link_sweeps_to_zones(
    sweeps: Sequence[Mapping[str, Any]] | None,
    zones: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    *,
    tick_size: float | None,
    tolerance_ms: int = 60_000,
) -> List[Dict[str, Any]]:
    if not sweeps:
        return []
    zone_candidates: List[Dict[str, Any]] = []
    if isinstance(zones, Mapping):
        for zone_key in ("fvg", "ob"):
            zone_series = zones.get(zone_key)
            if not isinstance(zone_series, Sequence):
                continue
            for zone_entry in zone_series:
                if not isinstance(zone_entry, MutableMapping):
                    continue
                formed_ms = _zone_focus_timestamp(zone_entry)
                if formed_ms is None:
                    continue
                reference_price = _zone_reference_price(zone_entry, zone_key)
                if reference_price is None:
                    continue
                quant_price = _quantise_price(reference_price, tick_size)
                zone_candidates.append(
                    {
                        "zone": zone_entry,
                        "zone_type": zone_key,
                        "tf": zone_entry.get("tf"),
                        "status": zone_entry.get("status"),
                        "price": quant_price,
                        "ts": formed_ms,
                    }
                )
    linked: List[Dict[str, Any]] = []
    for entry in sweeps:
        if not isinstance(entry, Mapping):
            continue
        sweep_ts = _safe_int(entry.get("t"))
        retest_ts = _safe_int(entry.get("retest_t"))
        if sweep_ts is None and retest_ts is None:
            continue
        level_price = _quantise_price(_safe_float(entry.get("level_price")), tick_size)
        new_entry = dict(entry)
        matches: List[Dict[str, Any]] = []
        for candidate in zone_candidates:
            zone_ts = candidate["ts"]
            if sweep_ts is not None and abs(zone_ts - sweep_ts) <= tolerance_ms:
                pass
            elif retest_ts is not None and abs(zone_ts - retest_ts) <= tolerance_ms:
                pass
            else:
                continue
            zone_price = candidate["price"]
            if level_price is not None and zone_price is not None:
                max_delta = tick_size or 0.0
                if max_delta <= 0:
                    max_delta = max(abs(level_price), abs(zone_price)) * 1e-6 + 1e-6
                dynamic_delta = max(abs(zone_price) * 0.001, 1e-6)
                max_delta = max(max_delta * 3.0, dynamic_delta)
                if abs(level_price - zone_price) > max_delta:
                    continue
            zone_payload = candidate["zone"]
            zone_payload.setdefault("sweep_links", []).append(
                {
                    "type": entry.get("type"),
                    "tf": entry.get("tf"),
                    "timestamp": _isoformat_utc(sweep_ts) if sweep_ts is not None else None,
                }
            )
            matches.append(
                {
                    "zone_type": candidate["zone_type"],
                    "tf": candidate["tf"],
                    "status": zone_payload.get("status"),
                    "price": zone_price,
                    "formed_utc": zone_payload.get("created_utc") or zone_payload.get("origin_utc"),
                }
            )
        if matches:
            new_entry["zone_links"] = matches
        linked.append(new_entry)
    return linked


def _extract_range(candidate: Mapping[str, Any]) -> tuple[int, int] | None:
    for key in ("t", "time", "timestamp", "ts"):
        ts = _safe_int(candidate.get(key))
        if ts is not None:
            return ts, ts
    start = candidate.get("start") or candidate.get("from") or candidate.get("begin")
    end = candidate.get("end") or candidate.get("to") or candidate.get("finish")
    start_ts = _safe_int(start)
    end_ts = _safe_int(end)
    if start_ts is None and end_ts is None:
        return None
    if start_ts is None:
        start_ts = end_ts or 0
    if end_ts is None:
        end_ts = start_ts
    return start_ts, end_ts


def _range_intersects(range_tuple: tuple[int, int], start_ms: int | None, end_ms: int | None) -> bool:
    if start_ms is None and end_ms is None:
        return True
    start, end = range_tuple
    if end_ms is not None and start > end_ms:
        return False
    if start_ms is not None and end < start_ms:
        return False
    return True


def _filter_indicator_block(value: Any, start_ms: int | None, end_ms: int | None) -> Any:
    if isinstance(value, Mapping):
        range_tuple = _extract_range(value)
        if range_tuple and not _range_intersects(range_tuple, start_ms, end_ms):
            return None
        result: Dict[str, Any] = {}
        for key, inner in value.items():
            filtered = _filter_indicator_block(inner, start_ms, end_ms)
            if filtered is None and isinstance(inner, (Mapping, list, tuple, set)):
                continue
            result[key] = filtered if filtered is not None else inner
        return result

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        filtered_items: List[Any] = []
        has_range = False
        for item in value:
            if isinstance(item, Mapping):
                item_range = _extract_range(item)
                if item_range:
                    has_range = True
                    if not _range_intersects(item_range, start_ms, end_ms):
                        continue
            filtered = _filter_indicator_block(item, start_ms, end_ms)
            if filtered is None and isinstance(item, (Mapping, list, tuple, set)):
                continue
            filtered_items.append(filtered if filtered is not None else item)
        return filtered_items if has_range else filtered_items

    return value


def _build_profile_level_map(
    profile_tpo: Sequence[Mapping[str, Any]] | None,
) -> Dict[str, Dict[str, float]]:
    levels: Dict[str, Dict[str, float]] = {}
    if not profile_tpo:
        return levels
    for entry in profile_tpo:
        if not isinstance(entry, Mapping):
            continue
        session_raw = entry.get("session") or "daily"
        session = str(session_raw).lower()
        session_levels = levels.setdefault(session, {})
        for key, target in (("POC", "poc"), ("VAH", "vah"), ("VAL", "val")):
            value = entry.get(key)
            if value is None:
                continue
            try:
                session_levels[target] = float(value)
            except (TypeError, ValueError):
                continue
    return levels


def _collect_nested_events(source: Any, events: List[Mapping[str, Any]]) -> None:
    if isinstance(source, Mapping):
        for key in ("events", "flags", "signals"):
            value = source.get(key)
            if isinstance(value, Sequence):
                for entry in value:
                    if isinstance(entry, Mapping):
                        events.append(entry)
        for key in ("structure", "data", "payload"):
            nested = source.get(key)
            if nested is not None:
                _collect_nested_events(nested, events)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            _collect_nested_events(item, events)


def _extract_structure_events(snapshot: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    events: List[Mapping[str, Any]] = []
    for key in ("smt", "zones", "structure", "indicators"):
        candidate = snapshot.get(key)
        if candidate is not None:
            _collect_nested_events(candidate, events)
    return events


def _collect_ob_candidates(source: Any, accumulator: List[Mapping[str, Any]]) -> None:
    if isinstance(source, Mapping):
        ob_payload = source.get("ob")
        if isinstance(ob_payload, Sequence):
            for entry in ob_payload:
                if isinstance(entry, Mapping):
                    accumulator.append(entry)
        for key in ("zones", "data", "payload"):
            nested = source.get(key)
            if nested is not None:
                _collect_ob_candidates(nested, accumulator)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            _collect_ob_candidates(item, accumulator)


def _resolve_smc_config(meta: Mapping[str, Any] | None) -> SMCConfig:
    if not isinstance(meta, Mapping):
        return SMCConfig()
    config_source = meta.get("smc") or meta.get("SMC")
    if not isinstance(config_source, Mapping):
        return SMCConfig()
    kwargs: Dict[str, Any] = {}
    if "min_block_size" in config_source:
        try:
            kwargs["min_block_size"] = float(config_source["min_block_size"])
        except (TypeError, ValueError):
            pass
    if "displacement_factor" in config_source:
        try:
            kwargs["displacement_factor"] = float(config_source["displacement_factor"])
        except (TypeError, ValueError):
            pass
    if "displacement_lookback" in config_source:
        try:
            kwargs["displacement_lookback"] = int(config_source["displacement_lookback"])
        except (TypeError, ValueError):
            pass
    if "ttl_bars" in config_source:
        try:
            kwargs["ttl_bars"] = int(config_source["ttl_bars"])
        except (TypeError, ValueError):
            pass
    return SMCConfig(**kwargs)


def _inject_smc_blocks(target: MutableMapping[str, Any] | None, blocks: Sequence[Mapping[str, Any]]) -> None:
    if not blocks or not isinstance(target, MutableMapping):
        return
    zones = target.get("zones")
    if isinstance(zones, MutableMapping):
        existing = zones.get("ob")
        if isinstance(existing, list):
            existing.extend(dict(block) for block in blocks)
        else:
            zones["ob"] = [dict(block) for block in blocks]
    elif isinstance(zones, list):
        zones.append({"ob": [dict(block) for block in blocks]})
    else:
        target["zones"] = {"ob": [dict(block) for block in blocks]}


def _select_indicator_timeframes(data: Any, targets: Sequence[str]) -> Any:
    if not isinstance(data, Mapping):
        return data
    lowered = {str(key).lower(): key for key in data.keys()}
    result: Dict[str, Any] = {}
    for target in targets:
        key = lowered.get(target)
        if key is None:
            continue
        result[key] = data[key]
    return result


def _filter_agg_trades(
    payload: Any,
    *,
    start_ms: int | None,
    end_ms: int | None,
    include_trades: bool,
) -> Dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None

    trades = payload.get("agg")
    filtered_trades: List[Dict[str, Any]] = []
    if isinstance(trades, Sequence):
        for entry in trades:
            if not isinstance(entry, Mapping):
                continue
            ts = _safe_int(entry.get("t"))
            if ts is None:
                continue
            if start_ms is not None and ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            filtered_trades.append({
                "t": ts,
                "p": _coerce_float(entry.get("p")),
                "q": _coerce_float(entry.get("q")),
                "side": entry.get("side"),
            })

    summary = {
        "count": len(filtered_trades),
        "buy": sum(1 for trade in filtered_trades if str(trade.get("side")).lower() == "buy"),
        "sell": sum(1 for trade in filtered_trades if str(trade.get("side")).lower() == "sell"),
        "volume": sum(float(trade.get("q", 0.0)) for trade in filtered_trades),
    }

    minutes_payload: Dict[str, Any]
    if filtered_trades:
        try:
            minutes_payload = _build_agg_trade_minutes(
                filtered_trades,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        except ValueError as exc:
            minutes_payload = {
                "status": "error",
                "error": str(exc),
                "minutes": [],
                "range": None,
                "totals": None,
            }
    else:
        minutes_payload = {
            "status": "insufficient_data",
            "minutes": [],
            "range": None,
            "totals": None,
        }

    result: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "summary": summary,
        "minutes": minutes_payload,
    }
    if include_trades:
        result["trades"] = filtered_trades
    return result


def _build_daily_vwap(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> Dict[str, Any] | None:
    daily_keys = [key for key in ("1d", "1h", "4h") if key in frames]
    if not daily_keys:
        return None

    source_key = daily_keys[0]
    filtered = _filter_candles(frames[source_key], start_ms=start_ms, end_ms=end_ms)
    if not filtered:
        return None
    stats = _compute_vwap_stats(filtered)
    if stats is None:
        vwap_value = _compute_vwap(filtered)
        sigma_payload = {"basis": "daily", "sigma": _build_sigma_levels(vwap_value, 0.0)}
    else:
        vwap_value, sigma_value = stats
        sigma_payload = {"basis": "daily", "sigma": _build_sigma_levels(vwap_value, sigma_value)}
    return {
        "timeframe": source_key,
        "value": vwap_value,
        "summary": _summarise(filtered),
        "vwap_sigma": sigma_payload,
    }


async def build_check_all_datas(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    selection_start_ms: int | None = None,
    selection_end_ms: int | None = None,
    hours: int | None = None,
    window_hours: int | None = None,
    window_start_override_ms: int | None = None,
    strict_window: bool = False,
    network_backfill: bool = True,
    trace: TraceContext | None = None,
    progress: ProgressReporter | None = None,
) -> Dict[str, Any] | None:
    """Create an enriched payload for the snapshot health endpoint."""

    trace_ctx = trace
    pipeline_start = time.perf_counter()
    fetch_ms = 0.0
    db_ms = 0.0

    status = "ok"

    notes: List[str] = []
    invalid_ts_total = 0
    invalid_ohlc_total = 0
    invalid_candle_stages: Dict[str, Dict[str, int]] = {}
    sanitized_applied = False
    sessions_empty_flag = False
    minute_coverage_diag: Dict[str, Any] | None = None
    stage_labels = {
        "gaps": "gaps",
        "rollups": "rollups",
        "delta": "delta",
        "vwap_tpo": "vwap/tpo",
        "zones": "zones",
        "vitrines": "vitrines",
    }
    progress_tracker = _PipelineProgress(
        progress,
        trace_ctx,
        label_map=stage_labels,
    )

    async def _finish(result: Dict[str, Any] | None) -> Dict[str, Any] | None:
        stage_snapshot = progress_tracker.snapshot()
        await progress_tracker.shutdown()
        if stage_snapshot and isinstance(result, MutableMapping):
            meta_block = result.get("meta")
            if isinstance(meta_block, MutableMapping):
                meta_block.setdefault("stage_timing", stage_snapshot)
        return result

    expected_count = 0
    minute_missing_before = 0
    enforce_minute_coverage = False
    time_gaps: List[Dict[str, Any]] = []

    def _build_gap_metrics(
        missing_after: int,
        downloaded: int,
        relaxed: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        coverage_before_ratio = 1.0
        coverage_after_ratio = 1.0
        if expected_count > 0:
            coverage_before_ratio = max(0.0, (expected_count - minute_missing_before) / expected_count)
            coverage_after_ratio = max(0.0, (expected_count - missing_after) / expected_count)
        bar_counts = {
            "expected": expected_count,
            "missing_before": minute_missing_before,
            "missing_after": missing_after,
            "downloaded": downloaded,
        }
        agg_counts = {
            "gap_count": len(time_gaps),
            "relaxed": bool(relaxed),
        }
        completeness = {
            "coverage_before": round(coverage_before_ratio, 6),
            "coverage_after": round(coverage_after_ratio, 6),
            "enforced": enforce_minute_coverage,
            "relaxed": relaxed,
        }
        return bar_counts, agg_counts, completeness

    def _record_invalid(stage: str, *, invalid_ts: int = 0, invalid_ohlc: int = 0) -> None:
        nonlocal invalid_ts_total, invalid_ohlc_total
        if invalid_ts == 0 and invalid_ohlc == 0:
            return
        entry = invalid_candle_stages.setdefault(stage, {"invalid_ts": 0, "invalid_ohlc": 0})
        if invalid_ts:
            entry["invalid_ts"] += int(invalid_ts)
            invalid_ts_total += int(invalid_ts)
        if invalid_ohlc:
            entry["invalid_ohlc"] += int(invalid_ohlc)
            invalid_ohlc_total += int(invalid_ohlc)

    def _apply_sanitizer(stage: str, candles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        nonlocal sanitized_applied
        sanitized_applied = True
        result = sanitize_candles(candles, stage=stage)
        _record_invalid(stage, invalid_ts=result.invalid_ts, invalid_ohlc=result.invalid_ohlc)
        return result.candles

    def _register_invalid_candle(_ts: int, stage: str) -> None:
        _record_invalid(stage, invalid_ts=1)

    budget = _TimeBudget(_BUILD_TIMEOUT_SECONDS)

    context = _prepare_snapshot_context(snapshot, now_utc)
    frames = context.frames
    raw_meta = context.raw_meta

    pipeline_meta = raw_meta.get("pipeline") if isinstance(raw_meta, Mapping) else None
    preset_name = PIPELINE_PRESET_DEFAULT.name
    overrides: Mapping[str, Any] | None = None
    if isinstance(pipeline_meta, Mapping):
        preset_candidate = pipeline_meta.get("preset")
        if isinstance(preset_candidate, str) and preset_candidate.strip():
            preset_name = preset_candidate.strip()
        override_candidate = pipeline_meta.get("overrides")
        if isinstance(override_candidate, Mapping):
            overrides = override_candidate
    try:
        pipeline_preset = resolve_pipeline_preset(preset_name, overrides)
    except ValueError as exc:
        LOGGER.error("Unknown pipeline preset %s, falling back to %s", preset_name, PIPELINE_PRESET_DEFAULT.name)
        pipeline_preset = PIPELINE_PRESET_DEFAULT

    rollup_timeframes = pipeline_preset.rollup_timeframes
    orderflow_timeframes = pipeline_preset.orderflow.timeframes
    orderflow_required_hours = pipeline_preset.orderflow.window_hours
    orderflow_window_ms = max(MINUTE_INTERVAL_MS, orderflow_required_hours * MS_IN_HOUR)
    orderflow_window_minutes = max(orderflow_required_hours * 60, 1)
    agg_trades_page_ms = max(MINUTE_INTERVAL_MS, pipeline_preset.orderflow.page_span_minutes * 60_000)
    zone_settings = pipeline_preset.zones
    zone_focus_window_hours = zone_settings.focus_window_hours
    zone_top_n = {key: int(value) for key, value in zone_settings.top_n.items()}
    zone_min_bars_default = {key: int(value) for key, value in zone_settings.min_bars_per_tf.items()}
    zone_min_bars_strict = (
        {key: int(value) for key, value in zone_settings.min_bars_strict.items()}
        if zone_settings.min_bars_strict
        else None
    )
    zone_warmup_bars = {key: int(value) for key, value in zone_settings.warmup_bars_per_tf.items()}
    zone_pivot_overrides = {key: int(value) for key, value in zone_settings.pivot_overrides.items()}
    zone_resample = bool(zone_settings.resample_when_sparse)
    atr_adaptation = dict(zone_settings.atr_adaptation)
    resample_when_sparse = bool(pipeline_preset.resample_when_sparse)
    session_windows = tuple((session.name, session.open, session.close) for session in pipeline_preset.sessions)
    modules_enabled = dict(pipeline_preset.modules)
    combined_ohlcv_tfs = tuple(dict.fromkeys(("1m",) + rollup_timeframes))
    timeframe_summary_order = tuple(tf for tf in combined_ohlcv_tfs if tf != "3m")
    aggregated_orderflow_timeframes = tuple(tf for tf in orderflow_timeframes if tf != "1m")
    orderflow_enabled = bool(modules_enabled.get("orderflow", True))
    orderflow_allow_network = bool(pipeline_preset.orderflow.allow_network)

    smc_detection_cfg = ZoneDetectionConfig()
    total_top_n = sum(zone_top_n.values())
    if total_top_n > 0:
        smc_detection_cfg.top_n_zones = int(total_top_n)
    smc_session_payload: Dict[str, Any] | None = None
    smc_context_payload: Dict[str, Any] | None = None
    session_analysis_diag: Dict[str, Any] | None = None
    zones_enabled = bool(modules_enabled.get("zones", True))
    vwap_enabled = bool(modules_enabled.get("vwap_tpo", True))
    liquidity_enabled = bool(modules_enabled.get("liquidity", True))
    showcases_enabled = bool(modules_enabled.get("showcases", True))

    for tf_key, candles in list(frames.items()):
        stage_name = f"seed.{tf_key}"
        frames[tf_key] = _apply_sanitizer(stage_name, candles)
    symbol = context.symbol
    cfg = AppConfig.load()
    now_dt = context.now
    stream_price = context.stream_price
    stream_ts = context.stream_ts
    now_ms = int(now_dt.timestamp() * 1000)
    minutes_seeded = False
    network_backfill = bool(network_backfill)
    minute_backfill_enabled = bool(network_backfill)

    if trace_ctx is not None:
        trace_ctx.info(
            "ui.click_received",
            scope="ui",
            symbol=symbol,
            strict_window=strict_window,
            network_backfill=network_backfill,
            minute_backfill_enabled=minute_backfill_enabled,
        )
        trace_ctx.info(
            "pipeline.start",
            scope="pipeline",
            symbol=symbol,
            strict_window=strict_window,
            network_backfill=network_backfill,
            minute_backfill_enabled=minute_backfill_enabled,
        )

    if selection_end_ms is None:
        selection_payload = snapshot.get("selection")
        if isinstance(selection_payload, Mapping):
            selection_end_candidate = _safe_int(selection_payload.get("end"))
            if selection_end_candidate is not None:
                selection_end_ms = selection_end_candidate

    if not frames:
        frames = {}

    base_window_hours = window_hours if window_hours is not None else hours
    if base_window_hours is None or base_window_hours <= 0:
        base_window_hours = 4
    base_window_hours = int(base_window_hours)

    strict_three_day = bool(strict_window and base_window_hours >= 72)
    strict_last_closed_day: date | None = None

    window_end_guess = _resolve_window_end_ms(
        frames,
        now_ms=now_ms,
        selection_end_ms=selection_end_ms,
        stream_ts=stream_ts,
    )

    interval_ms = MINUTE_INTERVAL_MS
    if strict_three_day:
        reference_dt = now_dt
        last_closed_day = reference_dt.date() - timedelta(days=1)
        strict_last_closed_day = last_closed_day
        closed_end_dt = datetime.combine(last_closed_day + timedelta(days=1), dtime.min, tzinfo=timezone.utc) - timedelta(minutes=1)
        closed_end_ms = int(closed_end_dt.timestamp() * 1000)
        window_end_guess = min(window_end_guess, closed_end_ms)
        window_end_guess = max(_align_to_interval(window_end_guess, interval_ms), interval_ms)
        minutes_expected = max(int(round(base_window_hours * 60)), 1)
        window_start_candidate = window_end_guess - (minutes_expected - 1) * interval_ms
        window_start_ms = max(0, _align_to_interval(window_start_candidate, interval_ms))
    else:
        window_start_ms = max(
            0,
            _align_to_interval(window_end_guess - base_window_hours * MS_IN_HOUR, interval_ms),
        )
        minutes_expected = _expected_minutes(interval_ms, window_start_ms, window_end_guess)

    if trace_ctx is not None:
        trace_ctx.info(
            "pipeline.contract_ok",
            scope="pipeline",
            base_window_hours=base_window_hours,
            strict_three_day=strict_three_day,
            window_end_ms=window_end_guess,
        )
    existing_minutes = frames.get("1m")
    existing_count = len(existing_minutes) if isinstance(existing_minutes, Sequence) else 0

    minutes_found = existing_count
    ensure_sources: List[str] = []
    ensure_df = None

    if existing_count >= minutes_expected:
        minutes_seeded = True
        ensure_sources = ["snapshot"]
    elif not minute_backfill_enabled:
        minutes_seeded = False
        ensure_sources = ["snapshot"]
    else:
        minute_backfill_unavailable = False
        try:
            ensure_result = await ensure_window_real(
                symbol=symbol,
                start_ts=window_start_ms,
                end_ts=window_end_guess,
                interval="1m",
            )
        except InsufficientCoverageError as exc:
            note = str(exc)
            if note and note not in notes:
                notes.append(note)
            LOGGER.warning(
                "ensure_window.partial_coverage",
                extra={
                    "symbol": symbol,
                    "window_hours": round(base_window_hours, 2),
                    "minutes_expected": exc.expected,
                    "minutes_found": exc.found,
                    "coverage_pct": exc.coverage_pct,
                },
            )
            ensure_sources = ["snapshot"]
            minutes_found = existing_count
            ensure_df = None
            minutes_seeded = False
            minute_backfill_unavailable = True
        else:
            minutes_found = int(ensure_result.minutes_found)
            ensure_sources = list(ensure_result.source_sequence)
            ensure_df = ensure_result.frame

            if isinstance(ensure_df, pd.DataFrame) and not ensure_df.empty:
                frames["1m"] = [
                    {
                        "t": int(row.ts_open),
                        "o": float(row.open),
                        "h": float(row.high),
                        "l": float(row.low),
                        "c": float(row.close),
                        "v": float(row.volume),
                        "takerBuyBase": float(getattr(row, "taker_buy_vol", getattr(row, "takerBuyBase", 0.0))),
                        "takerBuyQuote": float(
                            getattr(row, "taker_buy_quote", getattr(row, "takerBuyQuote", 0.0))
                        ),
                        "trades": int(getattr(row, "trades", 0)),
                    }
                    for row in ensure_df.itertuples(index=False)
                ]
            else:
                frames["1m"] = []

            minutes_seeded = True

        if minute_backfill_unavailable:
            enforce_minute_coverage = False
            minute_backfill_enabled = False

    if ensure_sources:
        source_note = f"Real data sources: {', '.join(ensure_sources)}"
        if source_note not in notes:
            notes.append(source_note)

    if minutes_seeded:
        minute_backfill_enabled = False

    vision_ingest_summary: Dict[str, Any] | None = None
    vision_required_hours: int | None = None
    if network_backfill:
        required_hours = max(
            base_window_hours,
            pipeline_preset.summary_window_hours,
            pipeline_preset.orderflow.window_hours,
            pipeline_preset.zones.focus_window_hours,
        )
        vision_required_hours = required_hours
        ingest_end_ms = now_ms
        ingest_start_ms = max(0, ingest_end_ms - required_hours * MS_IN_HOUR)
        vision_ingest_summary = await _maybe_ingest_vision_data(
            symbol,
            ingest_start_ms,
            ingest_end_ms,
            preset=pipeline_preset,
            allow_network=network_backfill,
            trace=trace_ctx,
        )

    vision_missing_datasets: Set[str] = set()
    metrics_fallback_used = False
    offline_orderflow_start_ms: int | None = None
    offline_orderflow_end_ms: int | None = None
    if vision_ingest_summary is not None:
        missing_entries = vision_ingest_summary.get("missing", [])
        if isinstance(missing_entries, Sequence):
            for entry in missing_entries:
                if isinstance(entry, Mapping):
                    dataset_name = entry.get("dataset")
                    if dataset_name:
                        vision_missing_datasets.add(str(dataset_name))
        ingested_block = vision_ingest_summary.get("ingested")
        if isinstance(ingested_block, Mapping):
            open_interest_entry = ingested_block.get("openInterest")
            if isinstance(open_interest_entry, Mapping) and open_interest_entry.get("source") == "metrics":
                metrics_fallback_used = True
                vision_missing_datasets.discard("openInterest")
                fallback_note = (
                    "Open interest reconstructed from metrics dataset (openInterest archive unavailable)."
                )
                if fallback_note not in notes:
                    notes.append(fallback_note)

    if strict_three_day and not network_backfill:
        orderflow_allow_network = False
        reference_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        last_closed_day = reference_dt.date() - timedelta(days=1)
        start_day = last_closed_day - timedelta(days=2)
        start_dt = datetime.combine(start_day, dtime.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(last_closed_day + timedelta(days=1), dtime.min, tzinfo=timezone.utc) - timedelta(minutes=1)
        offline_orderflow_start_ms = int(start_dt.timestamp() * 1000)
        offline_orderflow_end_ms = int(end_dt.timestamp() * 1000)
        orderflow_window_ms = max(
            MINUTE_INTERVAL_MS,
            offline_orderflow_end_ms - offline_orderflow_start_ms + MINUTE_INTERVAL_MS,
        )
        orderflow_window_minutes = max(offline_orderflow_end_ms - offline_orderflow_start_ms, 0) // MINUTE_INTERVAL_MS + 1
        orderflow_required_hours = max(1, orderflow_window_minutes // 60)
        offline_note = "Orderflow window set to last three closed days (snapshot)."
        if offline_note not in notes:
            notes.append(offline_note)
    elif vision_missing_datasets and "bookDepth" in vision_missing_datasets and liquidity_enabled:
        liquidity_enabled = False
        modules_enabled["liquidity"] = False
        note = "Liquidity module disabled: Binance Vision bookDepth dataset not available."
        if note not in notes:
            notes.append(note)

    if vision_missing_datasets:
        if "liquidationOrders" in vision_missing_datasets:
            note = "Liquidation data missing from Binance Vision archives; showcases may omit sweep metrics."
            if note not in notes:
                notes.append(note)
        if "fundingRate" in vision_missing_datasets:
            note = "Funding rate history unavailable in Binance Vision archives for the requested window."
            if note not in notes:
                notes.append(note)

    summary_result: "CollectionSummary" | None = None
    if strict_three_day and not minutes_seeded:
        summary_end_ms = window_end_guess
        summary_start_ms = max(
            0,
            summary_end_ms - 72 * MS_IN_HOUR,
        )
        summary_trace = (
            trace_ctx.child(stage="summary_collect") if trace_ctx is not None else None
        )
        summary_start = time.perf_counter()
        if trace_ctx is not None:
            trace_ctx.info(
                "fetch.batch_start",
                scope="summary.1m",
                window={"from": summary_start_ms, "to": summary_end_ms},
                details="collect_recent_summary",
            )
        try:
            from . import summary_collector  # local import to avoid circular deps

            summary_result = await summary_collector.collect_recent_summary(
                symbol,
                start_ms=summary_start_ms,
                end_ms=summary_end_ms,
                intervals=("1m",),
                trace=summary_trace,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception(
                "Failed to collect strict 3-day summary", extra={"symbol": symbol}
            )
            if trace_ctx is not None:
                trace_ctx.error(
                    "error.summary_collection",
                    scope="summary.1m",
                    details=str(exc),
                    window={"from": summary_start_ms, "to": summary_end_ms},
                )
            expected_count = int(
                (summary_end_ms - summary_start_ms) // MINUTE_INTERVAL_MS + 1
            )
            missing = MinuteDataUnavailable(
                symbol=symbol,
                start_ms=summary_start_ms,
                end_ms=summary_end_ms,
                missing_count=expected_count,
                expected_count=expected_count,
                coverage_pct=0.0,
                gaps=[(summary_start_ms, summary_end_ms)],
            )
            minute_payload = _build_minute_missing_payload(
                context,
                now=now_dt,
                missing=missing,
            )
            if trace_ctx is not None:
                trace_ctx.info(
                    "output.prepare_payload",
                    scope="output",
                    status="minute_missing",
                    missing_fields=0,
                )
            return await _finish(
                _finalise_payload(
                    minute_payload,
                    status="minute_missing",
                    pipeline_start=pipeline_start,
                    fetch_ms=fetch_ms,
                    db_ms=db_ms,
                    trace_ctx=trace_ctx,
                )
            )
        summary_elapsed_ms = (time.perf_counter() - summary_start) * 1000.0
        fetch_ms += summary_elapsed_ms
        if trace_ctx is not None:
            trace_ctx.info(
                "fetch.batch_done",
                scope="summary.1m",
                window={"from": summary_start_ms, "to": summary_end_ms},
                metrics={
                    "requests": getattr(summary_result, "requests", 0),
                    "ms": round(summary_elapsed_ms, 2),
                    "candles_written": getattr(summary_result, "candles_written", 0),
                },
                details="collect_recent_summary",
            )

    if not minutes_seeded:
        repo_window_ms = max(MINUTE_INTERVAL_MS, base_window_hours * MS_IN_HOUR)
        if minute_backfill_enabled:
            repo_window_ms = max(REPOSITORY_LOOKBACK_MS, repo_window_ms)
        repo_start_ms = max(0, _align_to_interval(window_start_ms, MINUTE_INTERVAL_MS))
        db_start = time.perf_counter()
        repository_minutes: List[Dict[str, Any]] = []
        repo_error: Exception | None = None
        if trace_ctx is not None:
            trace_ctx.info(
                "fetch.batch_start",
                scope="repository.1m",
                window={"from": repo_start_ms, "to": window_end_guess},
                details="repository.fetch_candles",
            )
        try:
            repository_minutes = await _load_repository_candles(
                symbol,
                "1m",
                repo_start_ms,
                window_end_guess,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            repo_error = exc
            LOGGER.debug(
                "Failed to seed minutes from repository",
                exc_info=exc,
                extra={"symbol": symbol, "repo_start_ms": repo_start_ms, "repo_end_ms": window_end_guess},
            )
            repository_minutes = []
        finally:
            repo_elapsed_ms = (time.perf_counter() - db_start) * 1000.0
            db_ms += repo_elapsed_ms
            if trace_ctx is not None:
                event_level = "warn" if repo_error else "info"
                emitter = getattr(trace_ctx, event_level)
                emitter(
                    "fetch.batch_done",
                    scope="repository.1m",
                    window={"from": repo_start_ms, "to": window_end_guess},
                    metrics={
                        "candles": len(repository_minutes),
                        "ms": round(repo_elapsed_ms, 2),
                    },
                    details="repository.fetch_candles",
                    error=str(repo_error) if repo_error else None,
                )
        if repository_minutes:
            frames["1m"] = _merge_candle_collections(frames.get("1m", []), repository_minutes)

    try:
        budget.raise_if_exceeded("seed_1m_backfill")
    except _TimeBudgetExceeded as exc:
        LOGGER.warning(
            "Time budget exceeded before seeding minute frame",
            extra={"stage": exc.stage, "symbol": symbol},
        )
        return await _finish(_insufficient_from_context(context, now_override=now_dt))

    if not strict_three_day and not minutes_seeded:
        if trace_ctx is not None:
            trace_ctx.info(
                "fetch.batch_start",
                scope="rest.1m",
                window={"from": repo_start_ms, "to": window_end_guess},
                details="backfill_timeframe",
            )
        fetch_start = time.perf_counter()
        backfilled = await _backfill_timeframe_with_rest(
            frames,
            symbol=symbol,
            timeframe="1m",
            window_end_ms=window_end_guess,
            window_hours=base_window_hours,
            allow_network=minute_backfill_enabled,
        )
        elapsed_ms = (time.perf_counter() - fetch_start) * 1000.0
        fetch_ms += elapsed_ms
        if trace_ctx is not None:
            trace_ctx.info(
                "fetch.batch_done",
                scope="rest.1m",
                window={"from": repo_start_ms, "to": window_end_guess},
                metrics={"fetched": int(backfilled), "ms": round(elapsed_ms, 2)},
                details="backfill_timeframe",
            )
    frames["1m"] = _apply_sanitizer("rest.1m", frames.get("1m", []))

    minute_seed = frames.get("1m", [])
    if minute_seed:
        frames["1m"] = _deduplicate_sorted(minute_seed)

    primary_key = _primary_frame_key(snapshot, frames)
    if not primary_key and "1m" in frames:
        primary_key = "1m"
    if not primary_key:
        return await _finish(_insufficient_from_context(context, now_override=now_dt))

    if not frames.get(primary_key):
        try:
            budget.raise_if_exceeded("seed_primary_backfill")
        except _TimeBudgetExceeded as exc:
            LOGGER.warning(
                "Time budget exceeded before seeding primary frame",
                extra={"stage": exc.stage, "symbol": symbol, "primary_key": primary_key},
            )
            return await _finish(_insufficient_from_context(context, now_override=now_dt))
        fetch_start = time.perf_counter()
        await _backfill_timeframe_with_rest(
            frames,
            symbol=symbol,
            timeframe=primary_key,
            window_end_ms=window_end_guess,
            window_hours=base_window_hours,
            allow_network=minute_backfill_enabled,
        )
        fetch_ms += (time.perf_counter() - fetch_start) * 1000.0
        frames[primary_key] = _apply_sanitizer(f"rest.{primary_key}", frames.get(primary_key, []))

    primary_candles = frames.get(primary_key, [])
    if not primary_candles:
        return await _finish(_insufficient_from_context(context, now_override=now_dt))

    # Drop unused granularities to keep the payload focused on the requested set.
    frames.pop("3m", None)
    frames.pop("5m", None)

    minute_candles = _deduplicate_sorted(frames.get("1m", []))
    frames["1m"] = minute_candles

    rollup_payload: Dict[str, Dict[str, Any]] = {}
    await progress_tracker.start(
        "rollups",
        timeframes=list(rollup_timeframes),
    )
    if minute_candles:
        lookback_days = max(1, int(math.ceil(base_window_hours / 24)))
        try:
            rollup_payload = await build_multi_tf_ohlcv(
                symbol,
                lookback_days,
                timeframes=rollup_timeframes,
                seed_minutes=minute_candles,
                trace=trace_ctx.child(stage="rollups.seed") if trace_ctx is not None else None,
            )
        except MinuteDataUnavailable as exc:
            if strict_window and base_window_hours >= 72:
                status = "minute_missing"
                insufficient_reason = "minute_missing"
                if trace_ctx is not None:
                    trace_ctx.error(
                        "error.minute_missing",
                        missing_minutes=exc.missing_count,
                        window={"from": exc.start_ms, "to": exc.end_ms},
                        coverage_pct=round(exc.coverage_pct, 3),
                    )
                minute_payload = _build_minute_missing_payload(
                    context,
                    now=now_dt,
                    missing=exc,
                )
                fail_bar_counts, fail_agg_counts, fail_completeness = _build_gap_metrics(
                    exc.missing_count,
                    0,
                    False,
                )
                await progress_tracker.fail(
                    "rollups",
                    bar_counts=fail_bar_counts,
                    agg_counts=fail_agg_counts,
                    completeness=fail_completeness,
                )
                return await _finish(
                    _finalise_payload(
                        minute_payload,
                        status=status,
                        pipeline_start=pipeline_start,
                        fetch_ms=fetch_ms,
                        db_ms=db_ms,
                        trace_ctx=trace_ctx,
                    )
                )
            if trace_ctx is not None:
                trace_ctx.warn(
                    "availability.checked",
                    scope="ohlcv.1m",
                    missing_minutes=exc.missing_count,
                    coverage_pct=round(exc.coverage_pct, 3),
                )
            rollup_payload = build_multi_timeframe_ohlcv(minute_candles, symbol=symbol)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to build local OHLCV rollups for %s: %s", symbol, exc)
            raise
        for tf_key in rollup_timeframes:
            if tf_key == "1m":
                continue
            existing_series = frames.get(tf_key)
            if existing_series:
                continue
            tf_payload = rollup_payload.get(tf_key)
            if not isinstance(tf_payload, Mapping):
                continue
            normalised_series = _normalise_external_candles(tf_payload)
            if normalised_series:
                frames[tf_key] = normalised_series

    if resample_when_sparse and minute_candles:
        for tf_key, series in list(frames.items()):
            if tf_key == "1m" or not series:
                continue
            if _series_needs_resample(series, tf_key):
                resampled = _resample_minutes_to_tf(minute_candles, tf_key)
                if resampled:
                    LOGGER.debug(
                        "Resampled timeframe to fix minute-resolution leak",
                        extra={
                            "symbol": symbol,
                            "timeframe": tf_key,
                            "candles_before": len(series),
                            "candles_after": len(resampled),
                        },
                    )
                    frames[tf_key] = resampled

    rollup_counts = {
        tf_key: len((rollup_payload.get(tf_key) or {}).get("candles", []))
        for tf_key in combined_ohlcv_tfs
    }
    rollup_counts["seed_minutes"] = len(minute_candles)
    await progress_tracker.complete(
        "rollups",
        bar_counts=rollup_counts,
        agg_counts={"lookback_days": max(1, int(math.ceil(base_window_hours / 24))) if minute_candles else 0},
        completeness={"has_seed": bool(minute_candles), "source": "remote" if rollup_payload else "local"},
    )

    profile_config = resolve_profile_config(symbol, raw_meta)
    profile_meta: Dict[str, Any] = {}
    sessions = list(session_windows)
    profile_tpo: List[Dict[str, Any]] = []
    profile_flat: List[Dict[str, float]] = []
    profile_zones: List[Dict[str, Any]] = []
    profile_level_map: Dict[str, Dict[str, float]] = {}
    detected_zones: Dict[str, Any] = {
        "symbol": symbol,
        "zones": {
            "fvg": [],
            "fvl": [],
            "ob": [],
            "mb": [],
            "bb": [],
            "rb": [],
            "pb": [],
            "sr": [],
            "profile_levels": [],
        },
    }
    zone_cfg = _apply_zone_threshold_overrides(ZonesConfig(tick_size=profile_config.get("tick_size")))

    await progress_tracker.start(
        "vwap_tpo",
        sessions=[session for session, _, _ in sessions],
    )

    requested_tf = str(profile_config.get("target_tf_key", "1m") or "1m")
    target_tf_key = requested_tf
    base_candidates = frames.get(target_tf_key, [])
    if base_candidates and _series_needs_resample(base_candidates, target_tf_key):
        resampled = _resample_minutes_to_tf(minute_candles, target_tf_key)
        if resampled:
            base_candidates = resampled
    if not base_candidates:
        if target_tf_key != "1m" and minute_candles:
            resampled = _resample_minutes_to_tf(minute_candles, target_tf_key)
            if resampled:
                base_candidates = resampled
        if not base_candidates and minute_candles:
            base_candidates = minute_candles
            target_tf_key = "1m"
    if not base_candidates and frames:
        for fallback_tf in _EXPECTED_OHLCV_TFS:
            candidate = frames.get(fallback_tf)
            if candidate:
                base_candidates = candidate
                target_tf_key = fallback_tf
                break
        if not base_candidates:
            for fallback_tf, candidate in frames.items():
                if candidate:
                    base_candidates = candidate
                    target_tf_key = fallback_tf
                    break
    base_candles = _deduplicate_sorted(base_candidates)
    if base_candles:
        frames[target_tf_key] = base_candles
    else:
        frames[target_tf_key] = []
    if target_tf_key != requested_tf:
        LOGGER.info(
            "profile target timeframe fallback",
            extra={
                "symbol": symbol,
                "requested_tf": requested_tf,
                "resolved_tf": target_tf_key,
                "minute_seed": len(minute_candles),
            },
        )

    if primary_key == "1m":
        primary_candles = minute_candles
    elif primary_key == target_tf_key:
        primary_candles = base_candles
    else:
        primary_candles = _deduplicate_sorted(frames.get(primary_key, []))
        frames[primary_key] = primary_candles
        if not primary_candles:
            primary_candles = base_candles

    if profile_config.get("preset") and base_candles and sessions:
        cache_token = (
            "check_all",
            snapshot.get("id"),
            symbol,
            target_tf_key,
        )
        (profile_tpo, profile_flat, profile_zones) = await asyncio.to_thread(
            build_profile_package,
            base_candles,
            sessions=sessions,
            last_n=int(profile_config.get("last_n", 3)),
            tick_size=profile_config.get("tick_size"),
            adaptive_bins=bool(profile_config.get("adaptive_bins", True)),
            value_area_pct=float(profile_config.get("value_area_pct", 0.7)),
            atr_multiplier=float(profile_config.get("atr_multiplier", 0.5)),
            target_bins=int(profile_config.get("target_bins", 80)),
            clip_threshold=float(profile_config.get("clip_threshold", 0.0)),
            smooth_window=int(profile_config.get("smooth_window", 1)),
            cache_token=cache_token,
            tf_key=target_tf_key,
            invalid_ts_handler=_register_invalid_candle,
            meta_out=profile_meta,
        )
        profile_level_map = _build_profile_level_map(profile_tpo)
        sessions_empty_flag = bool(profile_meta.get("sessions_empty"))

    snapshot_selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    selection_start = selection_start_ms or _safe_int(snapshot_selection.get("start")) if snapshot_selection else None
    selection_end = selection_end_ms or _safe_int(snapshot_selection.get("end")) if snapshot_selection else None

    if selection_start is None:
        selection_start = primary_candles[0]["t"]
    if selection_end is None:
        selection_end = primary_candles[-1]["t"]

    if selection_start > selection_end:
        selection_start, selection_end = selection_end, selection_start

    if context.has_now_override:
        window_end_ms = _align_to_interval(now_ms, MINUTE_INTERVAL_MS) - MINUTE_INTERVAL_MS
    else:
        window_end_ms = minute_candles[-1]["t"] if minute_candles else None

    target_interval_ms = _timeframe_interval_ms(target_tf_key) or MINUTE_INTERVAL_MS

    if window_end_ms is None and base_candles:
        window_end_ms = base_candles[-1]["t"] + max(target_interval_ms - MINUTE_INTERVAL_MS, 0)

    if window_end_ms is None and primary_candles:
        primary_interval = _timeframe_interval_ms(primary_key) or MINUTE_INTERVAL_MS
        window_end_ms = primary_candles[-1]["t"] + max(primary_interval - MINUTE_INTERVAL_MS, 0)

    if window_end_ms is None:
        return await _finish(_insufficient_from_context(context, now_override=now_dt))

    window_end_ms = max(0, _align_to_interval(window_end_ms, MINUTE_INTERVAL_MS))

    if window_hours is not None:
        try:
            hours_candidate = int(window_hours)
        except (TypeError, ValueError):
            hours_candidate = 0
        hours_window = max(1, hours_candidate)
    else:
        hours_window = hours if hours in VALID_HOUR_WINDOWS else min(VALID_HOUR_WINDOWS)

    override_start_ms: int | None = None
    if window_start_override_ms is not None:
        try:
            override_start_ms = int(window_start_override_ms)
        except (TypeError, ValueError):
            override_start_ms = None
        if override_start_ms is not None:
            override_start_ms = max(0, override_start_ms)

    span_ms: int
    if override_start_ms is not None:
        window_start_ms = max(0, _align_to_interval(override_start_ms, MINUTE_INTERVAL_MS))
        if window_start_ms >= window_end_ms:
            window_start_ms = max(0, window_end_ms - MINUTE_INTERVAL_MS)
        if target_interval_ms > MINUTE_INTERVAL_MS and not strict_window:
            aligned_target = _align_to_interval(window_start_ms, target_interval_ms)
            if aligned_target < window_start_ms:
                aligned_target += target_interval_ms
            if aligned_target >= window_end_ms:
                aligned_target = max(0, window_end_ms - target_interval_ms)
            window_start_ms = max(0, aligned_target)
        span_ms = max(MINUTE_INTERVAL_MS, window_end_ms - window_start_ms + MINUTE_INTERVAL_MS)
        hours_window = max(1, int(math.ceil(span_ms / MS_IN_HOUR)))
    else:
        raw_window_start = window_end_ms - hours_window * MS_IN_HOUR + MINUTE_INTERVAL_MS
        window_start_ms = max(0, _align_to_interval(raw_window_start, MINUTE_INTERVAL_MS))
        if target_interval_ms > MINUTE_INTERVAL_MS:
            window_start_ms = max(0, _align_to_interval(window_start_ms, target_interval_ms))
        span_ms = max(MINUTE_INTERVAL_MS, window_end_ms - window_start_ms + MINUTE_INTERVAL_MS)

    if strict_three_day:
        if 'summary_start_ms' in locals():
            window_start_ms = max(window_start_ms, summary_start_ms)
        if 'summary_end_ms' in locals():
            window_end_ms = min(window_end_ms, summary_end_ms)

    if base_window_hours >= 24 and minute_candles:
        earliest_ts = _safe_int(minute_candles[0].get("t"))
        if earliest_ts is not None and earliest_ts > window_start_ms:
            window_start_ms = earliest_ts
            span_ms = max(MINUTE_INTERVAL_MS, window_end_ms - window_start_ms + MINUTE_INTERVAL_MS)

    window_span_hours = span_ms / MS_IN_HOUR
    enforce_minute_coverage = bool(strict_three_day or window_span_hours >= 24)

    minute_index_all = {candle["t"]: candle for candle in minute_candles}
    minute_window_index = {
        ts: candle
        for ts, candle in minute_index_all.items()
        if window_start_ms <= ts <= window_end_ms
    }

    await progress_tracker.start(
        "gaps",
        window={"start_ms": window_start_ms, "end_ms": window_end_ms},
        enforce=enforce_minute_coverage,
        strict=strict_three_day,
    )

    expected_minutes = _build_expected_times(window_start_ms, window_end_ms, MINUTE_INTERVAL_MS)
    expected_count = len(expected_minutes)
    time_gaps = _summarise_missing_times(expected_minutes, minute_window_index)
    minute_missing_before = sum(gap["count"] for gap in time_gaps)

    if trace_ctx is not None and time_gaps:
        trace_ctx.info(
            "gaps.detected",
            scope="ohlcv.1m",
            tf="1m",
            count=len(time_gaps),
            window={"from": window_start_ms, "to": window_end_ms},
        )

    fetched_unique = 0
    relaxed_minute_gap = False
    if time_gaps:
        if strict_three_day:
            coverage_pct = (
                ((expected_count - minute_missing_before) / expected_count) * 100.0
                if expected_count
                else 100.0
            )
            missing = MinuteDataUnavailable(
                symbol=symbol,
                start_ms=window_start_ms,
                end_ms=window_end_ms,
                missing_count=minute_missing_before,
                expected_count=expected_count,
                coverage_pct=round(coverage_pct, 3),
                gaps=[(int(gap.get("from", window_start_ms)), int(gap.get("to", window_start_ms))) for gap in time_gaps],
            )
            if trace_ctx is not None:
                trace_ctx.warn(
                    "availability.checked",
                    scope="ohlcv.1m",
                    missing_minutes=minute_missing_before,
                    coverage_pct=round(coverage_pct, 3),
                    window={"from": window_start_ms, "to": window_end_ms},
                )
            minute_payload = _build_minute_missing_payload(
                context,
                now=now_dt,
                missing=missing,
            )
            fail_bar_counts, fail_agg_counts, fail_completeness = _build_gap_metrics(
                minute_missing_before,
                0,
                False,
            )
            await progress_tracker.fail(
                "gaps",
                bar_counts=fail_bar_counts,
                agg_counts=fail_agg_counts,
                completeness=fail_completeness,
            )
            if trace_ctx is not None:
                trace_ctx.info(
                    "output.prepare_payload",
                    scope="output",
                    status="minute_missing",
                    missing_fields=0,
                )
            return await _finish(
                _finalise_payload(
                    minute_payload,
                    status="minute_missing",
                    pipeline_start=pipeline_start,
                    fetch_ms=fetch_ms,
                    db_ms=db_ms,
                    trace_ctx=trace_ctx,
                )
            )
        if not minute_backfill_enabled:
            data_quality = {
                "tf": target_tf_key,
                "window": {"start_ms": window_start_ms, "end_ms": window_end_ms},
                "minute_missing_before": minute_missing_before,
                "minute_missing_after": minute_missing_before,
                "fetched_1m_count": 0,
                "tf_missing_before": 0,
                "tf_missing_after": 0,
                "time_gaps": time_gaps,
            }
            if enforce_minute_coverage:
                fail_bar_counts, fail_agg_counts, fail_completeness = _build_gap_metrics(
                    minute_missing_before,
                    0,
                    False,
                )
                await progress_tracker.fail(
                    "gaps",
                    bar_counts=fail_bar_counts,
                    agg_counts=fail_agg_counts,
                    completeness=fail_completeness,
                )
                raise DataQualityError(data_quality)
            relaxed_minute_gap = True
            LOGGER.info(
                "Relaxed minute coverage without network backfill",
                extra={
                    "symbol": symbol,
                    "window_hours": round(window_span_hours, 2),
                    "gaps": len(time_gaps),
                },
            )
        else:
            try:
                budget.raise_if_exceeded("download_missing_minutes_start")
                downloaded_minutes = await _call_download_missing_minutes_async(
                    symbol,
                    window_start_ms,
                    window_end_ms,
                    time_gaps,
                    budget=budget,
                    allow_network=minute_backfill_enabled,
                )
                downloaded_minutes = _apply_sanitizer(
                    "rest.1m.download", downloaded_minutes
                )
            except BinanceDownloadError as exc:
                detail = {
                    "tf": target_tf_key,
                    "window": {"start_ms": window_start_ms, "end_ms": window_end_ms},
                    "minute_missing_before": minute_missing_before,
                    "minute_missing_after": minute_missing_before,
                    "fetched_1m_count": exc.downloaded,
                    "tf_missing_before": 0,
                    "tf_missing_after": 0,
                    "time_gaps": time_gaps,
                    "downloaded": exc.downloaded,
                }
                if enforce_minute_coverage:
                    fail_bar_counts, fail_agg_counts, fail_completeness = _build_gap_metrics(
                        minute_missing_before,
                        exc.downloaded,
                        False,
                    )
                    await progress_tracker.fail(
                        "gaps",
                        bar_counts=fail_bar_counts,
                        agg_counts=fail_agg_counts,
                        completeness=fail_completeness,
                    )
                    raise DataQualityError(detail) from exc
                relaxed_minute_gap = True
                LOGGER.info(
                    "Relaxed minute coverage after download failure",
                    extra={
                        "symbol": symbol,
                        "window_hours": round(window_span_hours, 2),
                        "downloaded": exc.downloaded,
                    },
                )
                downloaded_minutes = []
            except _TimeBudgetExceeded as exc:
                LOGGER.warning(
                    "Time budget exceeded while downloading missing 1m candles",
                    extra={
                        "stage": exc.stage,
                        "symbol": symbol,
                        "window_start_ms": window_start_ms,
                        "window_end_ms": window_end_ms,
                        "time_gaps": time_gaps,
                    },
                )
                if enforce_minute_coverage:
                    fail_bar_counts, fail_agg_counts, fail_completeness = _build_gap_metrics(
                        minute_missing_before,
                        fetched_unique,
                        False,
                    )
                    await progress_tracker.fail(
                        "gaps",
                        bar_counts=fail_bar_counts,
                        agg_counts=fail_agg_counts,
                        completeness=fail_completeness,
                    )
                    return await _finish(_insufficient_from_context(context, now_override=now_dt))
                relaxed_minute_gap = True
                downloaded_minutes = []
            for candle in downloaded_minutes:
                ts = candle["t"]
                if ts < window_start_ms or ts > window_end_ms:
                    continue
                if ts not in minute_window_index:
                    fetched_unique += 1
                minute_window_index[ts] = candle
                minute_index_all[ts] = candle

    minute_missing_after = sum(1 for ts in expected_minutes if ts not in minute_window_index)
    data_quality = {
        "tf": target_tf_key,
        "window": {"start_ms": window_start_ms, "end_ms": window_end_ms},
        "minute_missing_before": minute_missing_before,
        "minute_missing_after": minute_missing_after,
        "fetched_1m_count": fetched_unique,
        "tf_missing_before": 0,
        "tf_missing_after": 0,
        "time_gaps": time_gaps,
    }

    if minute_missing_before and not relaxed_minute_gap and not enforce_minute_coverage:
        relaxed_minute_gap = True

    if minute_missing_after > 0:
        data_quality["downloaded"] = fetched_unique
        if enforce_minute_coverage:
            raise DataQualityError(data_quality)
        LOGGER.info(
            "Proceeding with relaxed minute coverage",
            extra={
                "symbol": symbol,
                "window_hours": round(window_span_hours, 2),
                "missing_after": minute_missing_after,
            },
        )
        notes.append(
            f"Minute coverage relaxed: missing {minute_missing_after} candles across ~{round(window_span_hours, 2)}h window."
        )

    if relaxed_minute_gap and not any(
        note.startswith("Minute coverage relaxed") for note in notes
    ):
        notes.append(
            f"Minute coverage relaxed for ~{round(window_span_hours, 2)}h window."
        )

    if data_quality:
        minute_coverage_diag = dict(data_quality)

    gap_metrics_bar, gap_metrics_agg, gap_completeness = _build_gap_metrics(
        minute_missing_after,
        fetched_unique,
        relaxed_minute_gap,
    )
    await progress_tracker.complete(
        "gaps",
        bar_counts=gap_metrics_bar,
        agg_counts=gap_metrics_agg,
        completeness=gap_completeness,
    )

    frames["1m"] = [minute_index_all[ts] for ts in sorted(minute_index_all)]
    minute_candles = frames["1m"]

    try:
        budget.raise_if_exceeded("zones_preparation")
    except _TimeBudgetExceeded as exc:
        LOGGER.warning(
            "Time budget exceeded before zones preparation",
            extra={"stage": exc.stage, "symbol": symbol},
        )
        return await _finish(_insufficient_from_context(context, now_override=now_dt))

    zones_window_hours = max(1, hours_window)
    if not strict_window:
        zones_window_hours = max(48, zones_window_hours)
    fifteen_min_ms = TIMEFRAME_TO_MS.get("15m") or 15 * MINUTE_INTERVAL_MS
    if strict_window:
        zones_window_start_ms = max(0, _align_to_interval(window_start_ms, MINUTE_INTERVAL_MS))
    else:
        zone_window_ms = zones_window_hours * MS_IN_HOUR
        raw_zone_start = max(0, window_end_ms - zone_window_ms)
        if fifteen_min_ms:
            raw_zone_start = max(0, _align_to_interval(raw_zone_start, fifteen_min_ms))
        zones_window_start_ms = raw_zone_start

    if strict_three_day and zone_min_bars_strict:
        min_bars_per_tf = dict(zone_min_bars_strict)
    else:
        min_bars_per_tf = dict(zone_min_bars_default)
    required_bars_per_tf: Dict[str, int] = {}
    warmup_required_per_tf: Dict[str, int] = dict(zone_warmup_bars)
    max_history_span_ms = 0
    max_warmup_span_ms = 0
    for tf_key, baseline in min_bars_per_tf.items():
        baseline_required = max(1, int(baseline))
        required_bars_per_tf[tf_key] = baseline_required
        warmup_required = int(warmup_required_per_tf.get(tf_key, baseline_required))
        warmup_required = max(0, warmup_required)
        warmup_required_per_tf[tf_key] = warmup_required
        interval_ms_tf = TIMEFRAME_TO_MS.get(tf_key)
        if not interval_ms_tf:
            continue
        span_required = max(baseline_required, warmup_required) * interval_ms_tf
        warmup_span = warmup_required * interval_ms_tf
        if span_required > max_history_span_ms:
            max_history_span_ms = span_required
        if warmup_span > max_warmup_span_ms:
            max_warmup_span_ms = warmup_span
    if not minute_backfill_enabled:
        warmup_cap_ms = max(MINUTE_INTERVAL_MS, base_window_hours * MS_IN_HOUR)
        if max_warmup_span_ms > warmup_cap_ms:
            max_warmup_span_ms = warmup_cap_ms
        if max_history_span_ms > warmup_cap_ms:
            max_history_span_ms = warmup_cap_ms
    if strict_window:
        zones_history_start_ms = zones_window_start_ms - max_warmup_span_ms
    else:
        history_candidate = zones_window_start_ms - max_history_span_ms
        history_candidate = min(history_candidate, zones_window_start_ms - MS_IN_DAY)
        zones_history_start_ms = history_candidate
    zones_history_start_ms = max(0, _align_to_interval(zones_history_start_ms, MINUTE_INTERVAL_MS))
    if not minute_backfill_enabled and minute_index_all:
        earliest_available = min(minute_index_all)
        if earliest_available > zones_history_start_ms:
            zones_history_start_ms = max(0, _align_to_interval(earliest_available, MINUTE_INTERVAL_MS))

    zone_expected_minutes = _build_expected_times(
        zones_history_start_ms, window_end_ms, MINUTE_INTERVAL_MS
    )
    zone_history_gaps = _summarise_missing_times(zone_expected_minutes, minute_index_all)
    if trace_ctx is not None and zone_history_gaps:
        trace_ctx.info(
            "gaps.detected",
            tf="1m",
            scope="zones_history",
            count=len(zone_history_gaps),
            window_start=zones_history_start_ms,
            window_end=window_end_ms,
        )
    warmup_missing_minutes = 0
    warmup_gap_only: List[Dict[str, Any]] = []
    blocking_gaps: List[Dict[str, Any]] = []
    if zone_history_gaps:
        for gap in zone_history_gaps:
            gap_from = int(gap.get("from", zones_history_start_ms))
            gap_to = int(gap.get("to", gap_from))
            if gap_to >= zones_window_start_ms:
                blocking_gaps.append(gap)
            else:
                warmup_gap_only.append(gap)
        warmup_missing_minutes = sum(int(gap.get("count", 0)) for gap in warmup_gap_only)
        blocking_missing = sum(int(gap.get("count", 0)) for gap in blocking_gaps)
        if blocking_gaps:
            zone_expected_minutes = _build_expected_times(
                zones_history_start_ms,
                window_end_ms,
                MINUTE_INTERVAL_MS,
            )
            zone_expected_count = len(zone_expected_minutes)
            if strict_three_day:
                coverage_pct = (
                    ((zone_expected_count - blocking_missing) / zone_expected_count) * 100.0
                    if zone_expected_count
                    else 100.0
                )
                missing = MinuteDataUnavailable(
                    symbol=symbol,
                    start_ms=zones_history_start_ms,
                    end_ms=window_end_ms,
                    missing_count=blocking_missing,
                    expected_count=zone_expected_count,
                    coverage_pct=round(coverage_pct, 3),
                    gaps=[
                        (
                            int(gap.get("from", zones_history_start_ms)),
                            int(gap.get("to", zones_history_start_ms)),
                        )
                        for gap in blocking_gaps
                    ],
                )
                if trace_ctx is not None:
                    trace_ctx.warn(
                        "availability.checked",
                        scope="ohlcv.1m.zones",
                        missing_minutes=blocking_missing,
                        window={"from": zones_history_start_ms, "to": window_end_ms},
                    )
                minute_payload = _build_minute_missing_payload(
                    context,
                    now=now_dt,
                    missing=missing,
                )
                if trace_ctx is not None:
                    trace_ctx.info(
                        "output.prepare_payload",
                        scope="output",
                        status="minute_missing",
                        missing_fields=0,
                    )
                return await _finish(
                    _finalise_payload(
                        minute_payload,
                        status="minute_missing",
                        pipeline_start=pipeline_start,
                        fetch_ms=fetch_ms,
                        db_ms=db_ms,
                        trace_ctx=trace_ctx,
                    )
                )
            if not minute_backfill_enabled:
                relaxed_minute_gap = True
                LOGGER.info(
                    "Relaxed zone history coverage without network backfill",
                    extra={
                        "symbol": symbol,
                        "window_hours": round(zones_window_hours, 2),
                        "missing_minutes": blocking_missing,
                        "gaps": len(blocking_gaps),
                    },
                )
                note = (
                    f"Minute coverage relaxed for zone history: missing {blocking_missing} of "
                    f"{zone_expected_count} minutes without network backfill"
                )
                if note not in notes:
                    notes.append(note)
                blocking_gaps = []
                blocking_missing = 0
            else:
                try:
                    budget.raise_if_exceeded("zones_history_backfill_start")
                    zone_downloaded_minutes = await _call_download_missing_minutes_async(
                        symbol,
                        zones_history_start_ms,
                        window_end_ms,
                        blocking_gaps,
                        budget=budget,
                    allow_network=minute_backfill_enabled,
                    )
                except BinanceDownloadError as exc:
                    detail = {
                        "tf": target_tf_key,
                        "window": {"start_ms": zones_history_start_ms, "end_ms": window_end_ms},
                        "stage": "zones_history",
                        "minute_missing_before": blocking_missing,
                        "fetched_1m_count": exc.downloaded,
                        "time_gaps": blocking_gaps,
                    }
                    raise DataQualityError(detail) from exc
                except _TimeBudgetExceeded as exc:
                    LOGGER.warning(
                        "Time budget exceeded while seeding zone history",
                        extra={
                            "stage": exc.stage,
                            "symbol": symbol,
                            "zones_history_start_ms": zones_history_start_ms,
                            "window_end_ms": window_end_ms,
                            "time_gaps": blocking_gaps,
                        },
                    )
                    return await _finish(_insufficient_from_context(context, now_override=now_dt))
                for candle in zone_downloaded_minutes:
                    ts = candle["t"]
                    if ts < zones_history_start_ms or ts > window_end_ms:
                        continue
                    minute_index_all[ts] = candle
        if warmup_gap_only and trace_ctx is not None:
            trace_ctx.info(
                "availability.warmup_partial",
                scope="ohlcv.1m.zones",
                missing_minutes=warmup_missing_minutes,
                window={"from": zones_history_start_ms, "to": zones_window_start_ms},
            )

    frames["1m"] = [minute_index_all[ts] for ts in sorted(minute_index_all)]
    minute_candles = frames["1m"]

    minute_zone_history = _filter_candles(
        minute_candles, start_ms=zones_history_start_ms, end_ms=window_end_ms
    )
    minute_zone_window = [
        candle for candle in minute_zone_history if int(candle["t"]) >= zones_window_start_ms
    ]

    def _build_zone_frames(source: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        bundle: Dict[str, List[Dict[str, Any]]] = {}
        if not source:
            return bundle
        ordered = sorted(source, key=lambda candle: int(candle["t"]))
        bundle["1m"] = [dict(item) for item in ordered]
        for tf_key in ("3m", "5m", "15m", "1h", "4h", "1d"):
            if not zone_resample:
                continue
            interval_ms_tf = TIMEFRAME_TO_MS.get(tf_key)
            if interval_ms_tf is None:
                continue
            aggregated = resample_ohlcv(ordered, interval_ms_tf)
            if not aggregated:
                continue
            normalised: List[Dict[str, Any]] = []
            for item in aggregated:
                if not isinstance(item, Mapping):
                    continue
                ts = _safe_int(item.get("t"))
                if ts is None or ts > window_end_ms:
                    continue
                normalised.append(
                    {
                        "t": ts,
                        "o": _coerce_float(item.get("o")),
                        "h": _coerce_float(item.get("h")),
                        "l": _coerce_float(item.get("l")),
                        "c": _coerce_float(item.get("c")),
                        "v": _coerce_float(item.get("v")),
                    }
                )
            if normalised:
                normalised.sort(key=lambda candle: candle["t"])
                bundle[tf_key] = normalised
        return bundle

    zone_frames_full = _build_zone_frames(minute_zone_history)
    zone_frames_window: Dict[str, List[Dict[str, Any]]] = {}
    zone_frames_window["1m"] = [dict(item) for item in minute_zone_window]
    for tf_key, series in zone_frames_full.items():
        if tf_key == "1m":
            continue
        window_series = [
            dict(item)
            for item in series
            if (ts := _safe_int(item.get("t"))) is not None and ts >= zones_window_start_ms
        ]
        if tf_key not in zone_frames_window or window_series:
            zone_frames_window[tf_key] = window_series

    zone_tf_lengths = {
        tf: len(zone_frames_full.get(tf, [])) for tf in ("15m", "1h", "4h", "1d")
    }
    warmup_bars_per_tf: Dict[str, int] = {}
    warmup_diag: Dict[str, Dict[str, Any]] = {}
    for tf in ("15m", "1h", "4h", "1d"):
        total = len(zone_frames_full.get(tf, []))
        window_count = len(zone_frames_window.get(tf, []))
        warmup_actual = max(0, total - window_count)
        warmup_bars_per_tf[tf] = warmup_actual
        warmup_required = int(warmup_required_per_tf.get(tf, warmup_actual))
        warmup_adjusted = warmup_required
        if not minute_backfill_enabled and warmup_actual < warmup_required:
            warmup_adjusted = warmup_actual
        warmup_diag[tf] = {
            "required": warmup_required,
            "adjusted_required": warmup_adjusted,
            "actual": warmup_actual,
            "partial": bool(warmup_adjusted and warmup_actual < warmup_adjusted),
        }
    zone_availability: Dict[str, Dict[str, Any]] = {}
    if strict_window:
        for tf_key, required in required_bars_per_tf.items():
            baseline_required = max(1, int(required))
            available = len(zone_frames_window.get(tf_key, []))
            total_history = len(zone_frames_full.get(tf_key, []))
            adjusted_required = baseline_required
            if available > 0:
                adjusted_required = min(baseline_required, available)
            ok = available >= max(1, adjusted_required)
            zone_availability[tf_key] = {
                "required": baseline_required,
                "adjusted_required": int(adjusted_required),
                "available": int(available),
                "total": int(total_history),
                "ok": bool(ok),
            }
    pivot_overrides: Dict[str, int] = dict(zone_pivot_overrides)
    for tf_key in ("15m", "1h"):
        available = len(zone_frames_window.get(tf_key, []))
        baseline_required = required_bars_per_tf.get(tf_key)
        if baseline_required is None:
            continue
        baseline_required = max(1, int(baseline_required))
        if available <= 0:
            pivot_overrides.setdefault(tf_key, 1 if tf_key == "15m" else 2)
            continue
        if available < baseline_required:
            pivot_overrides.setdefault(tf_key, 1 if tf_key == "15m" else 2)
    if pivot_overrides:
        zone_cfg.pivot_overrides = dict(pivot_overrides)
    zones_diag = {
        "tf_lengths": zone_tf_lengths,
        "atr_period": zone_cfg.atr_period,
        "warmup_bars_per_tf": warmup_bars_per_tf,
        "warmup_required_per_tf": {tf: int(val) for tf, val in warmup_required_per_tf.items()},
        "warmup_diag": warmup_diag,
        "warmup_missing_minutes": int(warmup_missing_minutes),
        "warmup_partial": bool(any(entry.get("partial") for entry in warmup_diag.values()) or warmup_missing_minutes),
        "window_hours": zones_window_hours,
        "tick_size": zone_cfg.tick_size,
    }
    if zone_availability:
        zones_diag["availability"] = zone_availability
        zones_diag["available_tfs"] = [tf for tf, info in zone_availability.items() if int(info.get("available", 0)) > 0]
        zones_diag["ok_tfs"] = [tf for tf, info in zone_availability.items() if bool(info.get("ok"))]
    else:
        zones_diag.setdefault("available_tfs", [])
        zones_diag.setdefault("ok_tfs", [])
    zone_cfg.zones_window_start_ms = zones_window_start_ms
    zone_cfg.window_end_ms_prev_closed = window_end_ms
    zone_cfg.allow_base_fallback = True
    liquidity_equal_levels = build_equal_liquidity_levels(
        zone_frames_full,
        max_levels=zone_top_n,
    )

    base_index_all = {candle["t"]: candle for candle in base_candles}

    if target_interval_ms <= MINUTE_INTERVAL_MS:
        expected_tf_times = expected_minutes
    else:
        expected_tf_times: List[int] = []
        cursor = window_start_ms
        while True:
            last_minute = cursor + target_interval_ms - MINUTE_INTERVAL_MS
            if last_minute > window_end_ms:
                break
            expected_tf_times.append(cursor)
            cursor += target_interval_ms

    tf_missing_before = sum(1 for ts in expected_tf_times if ts not in base_index_all)
    aggregated_added = 0
    if tf_missing_before:
        for open_ts in expected_tf_times:
            if open_ts in base_index_all:
                continue
            aggregated = _aggregate_from_minutes(minute_window_index, open_ts, target_interval_ms)
            if aggregated is None:
                continue
            base_index_all[open_ts] = aggregated
            aggregated_added += 1

    tf_missing_after = sum(1 for ts in expected_tf_times if ts not in base_index_all)
    data_quality["tf_missing_before"] = tf_missing_before
    data_quality["tf_missing_after"] = tf_missing_after

    if tf_missing_after > 0:
        data_quality["downloaded"] = fetched_unique
        if target_interval_ms > MINUTE_INTERVAL_MS:
            raise DataQualityError(data_quality)

    frames[target_tf_key] = [base_index_all[ts] for ts in sorted(base_index_all)]
    base_candles = frames[target_tf_key]
    if primary_key == target_tf_key:
        primary_candles = base_candles

    selection_payload: Dict[str, Any] = {
        "start": selection_start,
        "end": selection_end,
    }
    htf_section, htf_quality = build_htf_section(
        symbol,
        frames,
        selection_payload,
        allow_network=bool(network_backfill),
    )

    liquidity_config = raw_meta.get("liquidity") if isinstance(raw_meta, Mapping) else None

    reference_ts = window_end_ms + MINUTE_INTERVAL_MS
    reference_iso = _isoformat_utc(reference_ts)
    detailed_start_ts = window_start_ms

    movement_anchor_ts = detailed_start_ts
    movement_start_ts = min(selection_start, movement_anchor_ts)
    movement_end_ts = max(selection_start, movement_anchor_ts)
    if movement_end_ts > window_end_ms:
        movement_end_ts = window_end_ms

    detailed_start_iso = _isoformat_utc(detailed_start_ts)
    movement_start_iso = _isoformat_utc(movement_start_ts)
    movement_end_iso = _isoformat_utc(movement_end_ts)

    latest_minute_candle = minute_window_index.get(window_end_ms)
    latest_primary_candle = None
    if expected_tf_times:
        latest_primary_candle = base_index_all.get(expected_tf_times[-1])
    elif base_candles:
        latest_primary_candle = base_candles[-1]

    latest_candle_source = (
        latest_minute_candle
        or latest_primary_candle
        or (base_candles[-1] if base_candles else None)
    )
    latest_candle_ts = _safe_int(latest_candle_source.get("t")) if latest_candle_source else None
    if latest_candle_ts is None and base_candles:
        latest_candle_ts = base_candles[-1]["t"]
    if latest_candle_ts is None:
        latest_candle_ts = window_end_ms
    detailed_frames: Dict[str, Any] = {}
    for tf_key, candles in frames.items():
        filtered = _filter_candles(candles, start_ms=detailed_start_ts, end_ms=reference_ts)
        delta_series = _build_delta_series(filtered)
        detailed_frames[tf_key] = {
            "summary": _summarise(filtered),
            "candles": filtered,
            "delta_cvd": delta_series,
            "vwap": _compute_vwap(filtered),
        }

    zones_detailed = _filter_indicator_block(snapshot.get("zones"), detailed_start_ts, reference_ts)
    smt_detailed = _filter_indicator_block(snapshot.get("smt"), detailed_start_ts, reference_ts)
    agg_trades_detailed = _filter_agg_trades(
        snapshot.get("agg_trades"),
        start_ms=detailed_start_ts,
        end_ms=reference_ts,
        include_trades=True,
    )
    daily_vwap_detailed = _build_daily_vwap(frames, start_ms=detailed_start_ts, end_ms=reference_ts)

    detailed_section = {
        "hours": hours_window,
        "range": {
            "start_utc": detailed_start_iso,
            "end_utc": reference_iso,
        },
        "frames": detailed_frames,
        "indicators": {
            "zones": zones_detailed,
            "smt": smt_detailed,
            "delta_cvd": {
                tf: details["delta_cvd"]
                for tf, details in detailed_frames.items()
                if isinstance(details, Mapping) and "delta_cvd" in details
            },
            "vwap_daily": daily_vwap_detailed,
            "agg_trades": agg_trades_detailed,
        },
    }

    movement_frames: Dict[str, Dict[str, Any]] = {}
    delta_summaries: Dict[str, Dict[str, Any]] = {}
    vwap_summaries: Dict[str, Dict[str, Any]] = {}
    for tf_key in ("4h", "1d"):
        candles = frames.get(tf_key)
        if not candles:
            continue
        filtered = _filter_candles(candles, start_ms=movement_start_ts, end_ms=movement_end_ts)
        if not filtered:
            continue
        delta_series = _build_delta_series(filtered)
        movement_frames[tf_key] = {
            "summary": _summarise(filtered),
            "first_candle_utc": _isoformat_utc(filtered[0]["t"]),
            "last_candle_utc": _isoformat_utc(filtered[-1]["t"]),
        }
        delta_summaries[tf_key] = _summarise_delta_series(delta_series)
        vwap_summaries[tf_key] = {
            "value": _compute_vwap(filtered),
            "summary": _summarise(filtered),
        }

    zones_movement = _select_indicator_timeframes(
        _filter_indicator_block(snapshot.get("zones"), movement_start_ts, movement_end_ts),
        ("4h", "1d"),
    )
    smt_movement = _select_indicator_timeframes(
        _filter_indicator_block(snapshot.get("smt"), movement_start_ts, movement_end_ts),
        ("4h", "1d"),
    )
    agg_trades_movement = _filter_agg_trades(
        snapshot.get("agg_trades"),
        start_ms=movement_start_ts,
        end_ms=movement_end_ts,
        include_trades=False,
    )

    movement_days = 0
    if selection_start is not None and selection_end is not None:
        movement_days = max(0, int((selection_end - selection_start) // MS_IN_DAY))

    movement_section = {
        "days": movement_days,
        "range": {
            "start_utc": movement_start_iso,
            "end_utc": movement_end_iso,
        },
        "frames": movement_frames,
        "indicators": {
            "zones": zones_movement,
            "smt": smt_movement,
            "delta_cvd": delta_summaries,
            "vwap": vwap_summaries,
            "agg_trades": agg_trades_movement,
        },
    }

    tick_size_value = profile_config.get("tick_size") if isinstance(profile_config, Mapping) else None
    tick_size_numeric: float | None = None
    if isinstance(tick_size_value, (int, float)) and tick_size_value > 0:
        tick_size_numeric = float(tick_size_value)
    tick_size_source: str | None = "preset" if tick_size_numeric is not None else None

    liquidity_frames: Dict[str, Dict[str, Any]] = {}
    liquidity_trim_meta: Dict[str, Any] | None = None

    def _clean_series(series: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(series, Sequence):
            return []
        return [c for c in series if isinstance(c, Mapping)]  # type: ignore[list-item]

    LOGGER.info(
        "Liquidity module flag",
        extra={
            "symbol": symbol,
            "enabled": liquidity_enabled,
            "strict_three_day": strict_three_day,
            "network_backfill": network_backfill,
        },
    )
    liquidity_config_overrides: Dict[str, Any] = {}

    if liquidity_enabled:
        minute_full = _clean_series(frames.get("1m"))
        liquidity_minutes = minute_full

        if liquidity_minutes:
            liquidity_frames["1m"] = {"candles": liquidity_minutes, "source": "minute"}

        liquidity_start_ms: int | None = None
        liquidity_end_ms: int | None = None
        if liquidity_minutes:
            liquidity_start_ms = _safe_int(liquidity_minutes[0].get("t"))
            liquidity_end_ms = _safe_int(liquidity_minutes[-1].get("t"))

        def _limit_series(series: Sequence[Mapping[str, Any]] | None) -> List[Dict[str, Any]]:
            if not series:
                return []
            if liquidity_start_ms is None and liquidity_end_ms is None:
                return _clean_series(series)
            return _filter_candles(
                [item for item in series if isinstance(item, Mapping)],
                start_ms=liquidity_start_ms,
                end_ms=liquidity_end_ms,
            )

        htf_candles = htf_section.get("candles") if isinstance(htf_section, Mapping) else None
        if isinstance(htf_candles, Mapping):
            for tf_key in ("15m", "1h", "1d"):
                series = htf_candles.get(tf_key)
                cleaned = _limit_series(series if isinstance(series, Sequence) else None)
                if cleaned:
                    liquidity_frames[tf_key] = {"candles": cleaned, "source": "htf"}

        for tf_key in ("15m", "1h"):
            if tf_key in liquidity_frames:
                continue
            if not liquidity_minutes:
                continue
            interval_ms = TIMEFRAME_TO_MS.get(tf_key)
            if not interval_ms:
                continue
            aggregated = resample_ohlcv(liquidity_minutes, interval_ms)
            if not aggregated:
                continue
            liquidity_frames[tf_key] = {"candles": aggregated, "source": "aggregated"}

        if "1d" not in liquidity_frames:
            raw_daily = frames.get("1d")
            daily_series = _limit_series(raw_daily if isinstance(raw_daily, Sequence) else None)
            if daily_series:
                liquidity_frames["1d"] = {"candles": daily_series, "source": "short_window"}

        tick_inference_frames: Dict[str, Sequence[Mapping[str, Any]]] = {}
        for tf_key, payload in liquidity_frames.items():
            candles = payload.get("candles") if isinstance(payload, Mapping) else None
            if isinstance(candles, Sequence):
                tick_inference_frames[tf_key] = [c for c in candles if isinstance(c, Mapping)]  # type: ignore[list-item]

        tick_size_numeric, tick_size_source = resolve_liquidity_tick_size(
            symbol,
            tick_size_value,
            tick_inference_frames,
            meta=raw_meta,
            logger=logging.getLogger(__name__),
        )

        logging.getLogger(__name__).debug(
            "Liquidity tick size resolved for check-all",
            extra={
                "symbol": symbol,
                "normalized_symbol": normalise_symbol_for_tick(symbol) or "UNKNOWN",
                "tick_size": tick_size_numeric,
                "tick_size_source": tick_size_source,
            },
        )

        if trace_ctx is not None:
            trace_ctx.info(
                "compute.liquidity.start",
                scope="liquidity",
                frames={tf: len(payload.get("candles", [])) for tf, payload in liquidity_frames.items()},
            )
        liquidity_start = time.perf_counter()
        liquidity_payload = await asyncio.to_thread(
            build_liquidity_snapshot,
            liquidity_frames,
            symbol=symbol,
            tick_size=tick_size_numeric,
            tick_source_hint=tick_size_source,
            meta=raw_meta,
            selection=selection_payload,
            config=liquidity_config_overrides,
        )
        if trace_ctx is not None:
            trace_ctx.info(
                "compute.liquidity",
                scope="liquidity",
                duration_ms=round((time.perf_counter() - liquidity_start) * 1000.0, 2),
                frames={tf: len(payload.get("candles", [])) for tf, payload in liquidity_frames.items()},
            )
    else:
        liquidity_trim_meta = None
        liquidity_payload = {
            "eqh": [],
            "eql": [],
            "pdh": None,
            "pdl": None,
            "sweeps": [],
            "candidates": {"eqh": [], "eql": []},
            "diagnostics": {"status": "disabled"},
        }
        if tick_size_numeric is None:
            tick_size_source = "disabled"
        else:
            tick_size_source = tick_size_source or "disabled"
        disabled_note = "Liquidity module skipped for offline strict run."
        if disabled_note not in notes:
            notes.append(disabled_note)
        LOGGER.info("Liquidity module disabled for strict offline run", extra={"symbol": symbol})

    (
        last_price_value,
        last_iso_ts,
        last_tf,
        snapshot_age_sec,
        insufficient_reason,
        last_price_source,
        last_price_diag,
    ) = _resolve_last_price(
        frames,
        now=now_dt,
        stream_price=stream_price,
        stream_ts=stream_ts,
        tick_size=tick_size_numeric,
    )

    if snapshot_age_sec is not None:
        LOGGER.info("Snapshot age evaluated", extra={"snapshot_age_sec": snapshot_age_sec})

    if tick_size_numeric and isinstance(tick_size_numeric, (int, float)):
        zone_cfg.tick_size = float(tick_size_numeric)
    zones_diag["tick_size"] = zone_cfg.tick_size

    analysis_entry_price = None
    analysis_block = snapshot.get("analysis") if isinstance(snapshot.get("analysis"), Mapping) else None
    if isinstance(analysis_block, Mapping):
        trade_block = analysis_block.get("trade")
        if isinstance(trade_block, Mapping):
            analysis_entry_price = _safe_float(trade_block.get("entry_price"))

    if (
        analysis_entry_price is not None
        and last_price_value is not None
        and not math.isclose(analysis_entry_price, last_price_value, rel_tol=1e-9, abs_tol=1e-6)
    ):
        LOGGER.warning(
            "Analysis entry price differs from resolved last price",
            extra={
                "analysis_entry_price": analysis_entry_price,
                "last_price": last_price_value,
                "last_tf": last_tf,
                "last_ts_utc": last_iso_ts,
            },
        )

    await progress_tracker.start(
        "zones",
        availability=zone_availability,
        warmup=warmup_diag,
    )

    detect_elapsed_ms: float | None = None
    if zone_frames_full:
        try:
            detect_start = time.perf_counter()
            detected_zones = await asyncio.to_thread(
                detect_zones,
                frames=zone_frames_full,
                profile_levels=profile_level_map,
                liquidity_levels=liquidity_equal_levels,
                config=zone_cfg,
            )
            detect_elapsed_ms = (time.perf_counter() - detect_start) * 1000.0
        except Exception:  # pragma: no cover - defensive logging guard
            logging.getLogger(__name__).exception(
                "Failed to detect zones for check-all payload",
                extra={
                    "snapshot_id": snapshot.get("id"),
                    "symbol": symbol,
                    "timeframe": target_tf_key,
                },
            )
            detected_zones = {
                "zones": {
                    "fvg": [],
                    "fvl": [],
                    "ob": [],
                    "mb": [],
                    "bb": [],
                    "rb": [],
                    "pb": [],
                    "sr": [],
                    "profile_levels": [],
                },
                "meta": {},
            }
    if detect_elapsed_ms is not None and trace_ctx is not None:
        trace_ctx.info(
            "compute.zones",
            scope="zones",
            duration_ms=round(detect_elapsed_ms, 2),
            bars=len(zone_frames_full.get("1m", [])),
        )

    zones_container = detected_zones.get("zones") if isinstance(detected_zones, Mapping) else None
    if isinstance(zones_container, MutableMapping) and profile_level_map:
        if not zones_container.get("profile_levels"):
            zones_container["profile_levels"] = [
                {"type": level, "price": price, "session": session}
                for session, level_map in profile_level_map.items()
                for level, price in level_map.items()
            ]

    if isinstance(detected_zones, MutableMapping):
        meta_block = detected_zones.setdefault("meta", {})
        if isinstance(meta_block, MutableMapping):
            meta_block["zones_diag"] = zones_diag
            detection_diag = meta_block.get("diagnostics")
            if isinstance(detection_diag, Mapping):
                timeframes_diag = detection_diag.get("timeframes")
                structure_map: Dict[str, Any] = {}
                rb_counts: Dict[str, int] = {}
                rb_raw_counts: Dict[str, int] = {}
                rb_flow_map: Dict[str, Any] = {}
                rb_reject_map: Dict[str, Any] = {}
                rb_fallback_map: Dict[str, bool] = {}
                if isinstance(timeframes_diag, Sequence):
                    for frame_entry in timeframes_diag:
                        if not isinstance(frame_entry, Mapping):
                            continue
                        tf_name = str(frame_entry.get("tf") or "")
                        structure_info = frame_entry.get("structure_diag")
                        if isinstance(structure_info, Mapping) and tf_name:
                            structure_map[tf_name] = {
                                "pivots": int(structure_info.get("pivots", 0)),
                                "bos_up": int(structure_info.get("bos_up", 0)),
                                "bos_down": int(structure_info.get("bos_down", 0)),
                                "choch": int(structure_info.get("choch", 0)),
                            }
                        rb_info = frame_entry.get("rb")
                        if isinstance(rb_info, Mapping) and tf_name:
                            rb_counts[tf_name] = int(rb_info.get("count", 0))
                            stats_payload = rb_info.get("stats")
                            if isinstance(stats_payload, Mapping):
                                rb_raw_counts[tf_name] = int(stats_payload.get("rb_raw_count", 0))
                            flow_payload = rb_info.get("flow")
                            if isinstance(flow_payload, Mapping):
                                rb_flow_map[tf_name] = {
                                    str(key): int(value)
                                    for key, value in flow_payload.items()
                                    if isinstance(value, (int, float))
                                }
                            reject_payload = rb_info.get("reject")
                            if isinstance(reject_payload, Mapping):
                                rb_reject_map[tf_name] = {
                                    str(key): int(value)
                                    for key, value in reject_payload.items()
                                    if isinstance(value, (int, float))
                                }
                            if "base_fallback_used" in rb_info:
                                rb_fallback_map[tf_name] = bool(rb_info.get("base_fallback_used"))
                if structure_map:
                    zones_diag["structure_diag"] = structure_map
                if rb_raw_counts:
                    zones_diag["rb_raw_count"] = rb_raw_counts
                if rb_counts:
                    zones_diag["rb_count"] = rb_counts
                if rb_flow_map:
                    zones_diag["rb_flow"] = rb_flow_map
                if rb_reject_map:
                    zones_diag["rb_reject"] = rb_reject_map
                if rb_fallback_map:
                    zones_diag["base_fallback_used"] = rb_fallback_map
            zones_diag["detection"] = detection_diag
    zone_type_timeframes = {
        "fvg": ("15m", "1h", "4h"),
        "fvl": ("15m", "1h", "4h"),
        "ob": ("15m", "1h", "4h"),
        "mb": ("1h", "4h"),
        "bb": ("1h", "4h"),
        "rb": ("15m", "1h", "4h"),
        "pb": ("15m", "1h", "4h"),
        "sr": ("1h", "4h"),
    }

    if isinstance(zones_container, MutableMapping):
        timestamp_filters = {
            "fvg": "created_utc",
            "fvl": "created_utc",
            "ob": "origin_utc",
            "mb": "origin_utc",
            "bb": "origin_utc",
            "rb": "origin_utc",
            "pb": "origin_utc",
            "sr": "ts",
        }
        for key, field in timestamp_filters.items():
            series = zones_container.get(key)
            if not isinstance(series, Sequence):
                continue
            filtered: List[Dict[str, Any]] = []
            for item in series:
                if not isinstance(item, Mapping):
                    continue
                ts_ms = _iso_to_ms(item.get(field))
                if ts_ms is None or ts_ms >= zones_window_start_ms:
                    filtered.append(dict(item))
            zones_container[key] = filtered

        fvg_series = zones_container.get("fvg")
        if isinstance(fvg_series, Sequence) and not fvg_series:
            meta_block = (
                detected_zones.get("meta") if isinstance(detected_zones, Mapping) else None
            )
            fvg_stats: Dict[str, Any] = {}
            if isinstance(meta_block, Mapping):
                stats_payload = meta_block.get("fvg_stats")
                if isinstance(stats_payload, Mapping):
                    for tf_key, tf_stats in stats_payload.items():
                        tf_name = str(tf_key)
                        if isinstance(tf_stats, Mapping):
                            fvg_stats[tf_name] = {
                                str(metric): int(value)
                                for metric, value in tf_stats.items()
                                if isinstance(value, (int, float))
                            }
                        else:
                            fvg_stats[tf_name] = tf_stats
            logging.getLogger(__name__).info(
                "FVG detection returned no zones for window",
                extra={
                    "symbol": symbol,
                    "fvg_stats": fvg_stats,
                    "zones_window_start": zones_window_start_ms,
                    "window_hours": zones_window_hours,
                },
            )

    gating_diag: Dict[str, Dict[str, Any]] = {}
    zone_min_required_map = {
        "fvg": 1,
        "fvl": 1,
        "ob": 1,
        "rb": 1,
        "pb": 1,
        "mb": 1,
        "bb": 1,
    }
    if strict_window and zone_availability and isinstance(zones_container, MutableMapping):
        for zone_key, tf_candidates in zone_type_timeframes.items():
            if zone_key not in zones_container:
                continue
            requested = [tf for tf in tf_candidates if tf in zone_availability]
            if not requested:
                continue
            ok_list = [tf for tf in requested if zone_availability.get(tf, {}).get("ok", False)]
            missing_list = [tf for tf in requested if not zone_availability.get(tf, {}).get("ok", False)]
            min_required = zone_min_required_map.get(zone_key, len(requested)) or 0
            gating_diag[zone_key] = {
                "requested_tfs": requested,
                "ok_tfs": ok_list,
                "missing_tfs": missing_list,
                "min_required": int(min_required),
            }
            if len(ok_list) < max(1, min_required):
                zones_container[zone_key] = []
        if gating_diag:
            zones_diag["gating"] = gating_diag

    if "active_session_atr" not in locals():
        active_session_atr = None  # type: ignore[assignment]
    if "active_session_name" not in locals():
        active_session_name = None  # type: ignore[assignment]

    if isinstance(liquidity_config, Mapping):
        liquidity_config_overrides.update(liquidity_config)

    eq_settings = get_settings().eql_settings
    eq_ratios: List[float] = []
    if isinstance(getattr(eq_settings, "tolerance_bps_by_tf", None), Mapping):
        for value in eq_settings.tolerance_bps_by_tf.values():
            ratio = _bps_to_ratio(value, default=0.0)
            if ratio > 0:
                eq_ratios.append(ratio)
    base_ratio = _bps_to_ratio(getattr(eq_settings, "tolerance_bps", 0.0), default=0.0)
    if base_ratio > 0:
        eq_ratios.append(base_ratio)
    if eq_ratios:
        resolved_ratio = min(eq_ratios)
        if resolved_ratio > 0 and "tolerance_percent" not in liquidity_config_overrides:
            liquidity_config_overrides.setdefault("tolerance_mode", "percent")
            liquidity_config_overrides["tolerance_percent"] = resolved_ratio
    min_separations: List[int] = []
    if isinstance(getattr(eq_settings, "min_separation_bars_by_tf", None), Mapping):
        for value in eq_settings.min_separation_bars_by_tf.values():
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                min_separations.append(numeric)
    try:
        base_separation = int(eq_settings.min_separation_bars)
    except (TypeError, ValueError):
        base_separation = 0
    if base_separation > 0:
        min_separations.append(base_separation)
    if min_separations:
        resolved_min_sep = max(1, min(min_separations))
        liquidity_config_overrides.setdefault("min_distance_bars", resolved_min_sep)

    session_atr_pct: float | None = None
    if active_session_atr is not None:
        liquidity_config_overrides["session_atr_value"] = active_session_atr
        if last_price_value is not None and last_price_value > 0:
            session_atr_pct = max(active_session_atr / last_price_value, 0.0)
            liquidity_config_overrides["session_atr_pct"] = session_atr_pct
    if active_session_name:
        liquidity_config_overrides["session_label"] = active_session_name

    atr_pct_cap = float(atr_adaptation.get("pct_cap", 0.05))
    cluster_min = float(atr_adaptation.get("cluster_min", 0.00005))
    cluster_multiplier = float(atr_adaptation.get("cluster_multiplier", 3.5))
    cluster_max = float(atr_adaptation.get("cluster_max", 0.002))
    sweep_base = float(atr_adaptation.get("sweep_base", 0.3))
    sweep_multiplier = float(atr_adaptation.get("sweep_multiplier", 6.0))
    sweep_min = float(atr_adaptation.get("sweep_min", 0.25))
    sweep_max = float(atr_adaptation.get("sweep_max", 0.75))
    min_move_base = float(atr_adaptation.get("min_move_base", 0.25))
    min_move_multiplier = float(atr_adaptation.get("min_move_multiplier", 10.0))
    min_move_min = float(atr_adaptation.get("min_move_min", 0.2))
    min_move_max = float(atr_adaptation.get("min_move_max", 1.0))
    epsilon_multiplier = float(atr_adaptation.get("epsilon_multiplier", 2.5))
    epsilon_min = float(atr_adaptation.get("epsilon_min", 0.0002))
    epsilon_max = float(atr_adaptation.get("epsilon_max", 0.02))

    if (
        active_session_atr is not None
        and tick_size_numeric is not None
        and tick_size_numeric > 0
    ):
        atr_ticks = max(active_session_atr / tick_size_numeric, 0.0)
        eqh_ticks = max(1.0, min(atr_ticks * 0.5, 15.0))
        liquidity_config_overrides["tolerance_eqh_ticks"] = eqh_ticks
        liquidity_config_overrides["tolerance_eql_ticks"] = eqh_ticks

    if session_atr_pct is not None:
        bounded_pct = min(session_atr_pct, atr_pct_cap)
        cluster_pct = max(cluster_min, min(bounded_pct * cluster_multiplier, cluster_max))
        liquidity_config_overrides["cluster_price_window_pct"] = cluster_pct
        sweep_mult = max(sweep_min, min(sweep_max, sweep_base + bounded_pct * sweep_multiplier))
        min_move_atr = max(min_move_min, min(min_move_max, min_move_base + bounded_pct * min_move_multiplier))
        epsilon_pct = max(epsilon_min, min(epsilon_max, bounded_pct * epsilon_multiplier))
        liquidity_config_overrides["sweep_atr_multiplier"] = sweep_mult
        liquidity_config_overrides["sweep_min_move_atr"] = min_move_atr
        liquidity_config_overrides["sweep_epsilon_pct"] = epsilon_pct
    if liquidity_trim_meta is not None:
        liquidity_config_overrides.setdefault("window_trim_hours", liquidity_trim_meta.get("window_hours"))
        liquidity_config_overrides.setdefault("window_trim_from", liquidity_trim_meta.get("trimmed_from"))
        liquidity_config_overrides.setdefault("window_trim_to", liquidity_trim_meta.get("trimmed_to"))
        trim_note = f"Liquidity window trimmed to last {int(liquidity_trim_meta['window_hours'])}h for offline strict run."
        if trim_note not in notes:
            notes.append(trim_note)

    liquidity_diagnostics = (
        liquidity_payload.pop("diagnostics", None)
        if isinstance(liquidity_payload, MutableMapping)
        else None
    )
    liquidity_config_payload = (
        liquidity_diagnostics.get("config")
        if isinstance(liquidity_diagnostics, Mapping)
        else None
    )
    if isinstance(liquidity_config_payload, Mapping):
        liquidity_payload["config"] = dict(liquidity_config_payload)

    sweeps_series = liquidity_payload.get("sweeps") if isinstance(liquidity_payload, Mapping) else None
    deduped_sweeps = _dedupe_sweeps(sweeps_series, tick_size=tick_size_numeric)
    linked_sweeps = _link_sweeps_to_zones(
        deduped_sweeps,
        zones_container if isinstance(zones_container, Mapping) else None,
        tick_size=tick_size_numeric,
    )
    if isinstance(liquidity_payload, MutableMapping):
        liquidity_payload["sweeps"] = linked_sweeps

    zones_recent_focus = _build_zone_focus(
        zones_container if isinstance(zones_container, Mapping) else None,
        last_price=last_price_value,
        reference_ms=window_end_ms,
        window_hours=zone_focus_window_hours,
        top_limits=zone_top_n,
    )
    if isinstance(zones_recent_focus.get("meta"), MutableMapping):
        if active_session_name:
            zones_recent_focus["meta"]["session"] = active_session_name
    if active_session_atr is not None:
        zones_recent_focus["meta"]["session_atr"] = active_session_atr
    if session_atr_pct is not None:
        zones_recent_focus["meta"]["session_atr_pct"] = session_atr_pct
    zones_diag["recent_focus"] = copy.deepcopy(zones_recent_focus.get("meta", {}))
    if isinstance(detected_zones, MutableMapping):
        meta_block = detected_zones.setdefault("meta", {})
        if isinstance(meta_block, MutableMapping):
            meta_block["recent_focus"] = copy.deepcopy(zones_recent_focus)

    zone_counts_summary: Dict[str, int] = {}
    zones_block = detected_zones.get("zones") if isinstance(detected_zones, Mapping) else None
    if isinstance(zones_block, Mapping):
        for key in ("fvg", "fvl", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels"):
            series = zones_block.get(key)
            zone_counts_summary[key] = len(series) if isinstance(series, Sequence) else 0
    zone_focus_counts = {}
    focus_meta = zones_recent_focus.get("meta") if isinstance(zones_recent_focus, Mapping) else None
    if isinstance(focus_meta, Mapping):
        counts_payload = focus_meta.get("counts")
        if isinstance(counts_payload, Mapping):
            zone_focus_counts = dict(counts_payload)
    await progress_tracker.complete(
        "zones",
        bar_counts=zone_counts_summary,
        agg_counts={"focus_counts": zone_focus_counts},
        completeness={"availability": zone_availability, "warmup": warmup_diag},
    )

    minute_htf_source: List[Mapping[str, Any]] = []
    minute_frame_present = "1m" in frames
    if minute_frame_present:
        minute_htf_source = [
            minute_window_index[ts]
            for ts in sorted(minute_window_index)
            if ts in minute_window_index
        ]

    orderflow_config = _resolve_orderflow_config(raw_meta)
    orderflow_end_ms = window_end_ms
    orderflow_start_ms = max(0, orderflow_end_ms - orderflow_window_ms + MINUTE_INTERVAL_MS)
    if offline_orderflow_start_ms is not None:
        orderflow_start_ms = max(orderflow_start_ms, offline_orderflow_start_ms)
    if offline_orderflow_end_ms is not None:
        orderflow_end_ms = min(orderflow_end_ms, offline_orderflow_end_ms)
    if base_window_hours >= 24:
        orderflow_start_ms = max(orderflow_start_ms, window_start_ms)
    if minute_window_index:
        earliest_minute = min(minute_window_index)
        latest_minute = max(minute_window_index)
        orderflow_start_ms = max(orderflow_start_ms, earliest_minute)
        orderflow_end_ms = min(orderflow_end_ms, latest_minute)
        if orderflow_end_ms < orderflow_start_ms:
            orderflow_end_ms = orderflow_start_ms
    target_length = _per_bar_target_length(orderflow_window_minutes)
    proxy_minute_source: Sequence[Mapping[str, Any]] = frames.get("1m", [])
    if strict_three_day and minute_htf_source:
        # Use the fully reconstructed minute window for strict offline runs, as snapshot
        # frames often only contain the last few hours.
        proxy_minute_source = minute_htf_source
    if orderflow_enabled:
        await progress_tracker.start(
            "delta",
            window={"start_ms": orderflow_start_ms, "end_ms": orderflow_end_ms},
        )
        orderflow_start = time.perf_counter()
        try:
            allow_orderflow_network = bool(network_backfill or orderflow_allow_network)
            orderflow_block, orderflow_diag = await _build_orderflow_block(
                proxy_minute_source,
                snapshot.get("agg_trades"),
                orderflow_start_ms=orderflow_start_ms,
                orderflow_end_ms=orderflow_end_ms,
                config=orderflow_config,
                symbol=symbol,
                allow_network=allow_orderflow_network,
                trace_ctx=trace_ctx,
                budget=budget,
                orderflow_source=snapshot.get("orderflow"),
                window_minutes=orderflow_window_minutes,
                page_span_ms=agg_trades_page_ms,
                target_timeframes=orderflow_timeframes,
                aggregated_timeframes=aggregated_orderflow_timeframes,
                minute_interval_ms=MINUTE_INTERVAL_MS,
                timeframe_to_ms=TIMEFRAME_TO_MS,
                target_length=target_length,
                fetch_override=_fetch_binance_agg_trades,
                budget_exceeded_exc=_TimeBudgetExceeded,
            )
            if trace_ctx is not None:
                agg_payload = snapshot.get("agg_trades")
                trace_ctx.info(
                    "compute.orderflow",
                    scope="orderflow",
                    duration_ms=round((time.perf_counter() - orderflow_start) * 1000.0, 2),
                    trades=len(agg_payload.get("agg", [])) if isinstance(agg_payload, Mapping) else None,
                    minutes=len(proxy_minute_source),
                )
        except Exception as exc:
            await progress_tracker.fail(
                "delta",
                bar_counts={},
                agg_counts={},
                completeness={"error": str(exc)},
            )
            raise
        orderflow_counts = {
            tf: len((orderflow_block.get(tf) or {}).get("per_bar", []))
            for tf in orderflow_timeframes
        }
        series_lengths = orderflow_diag.get("series_lengths") if isinstance(orderflow_diag, Mapping) else {}
        coverage_info = orderflow_diag.get("coverage") if isinstance(orderflow_diag, Mapping) else {}
        await progress_tracker.complete(
            "delta",
            bar_counts=orderflow_counts,
            agg_counts=series_lengths if isinstance(series_lengths, Mapping) else {},
            completeness=coverage_info if isinstance(coverage_info, Mapping) else {},
        )
        delta_source = orderflow_diag.get("delta_source") if isinstance(orderflow_diag, Mapping) else None
        if delta_source == "proxy":
            LOGGER.info(
                "Orderflow proxy built",
                extra={"symbol": symbol, "bars_1m": orderflow_counts.get("1m", 0)},
            )
            notes.append("Orderflow metrics reconstructed from minute candles (no agg-trades available).")
        elif delta_source == "external":
            notes.append("Orderflow metrics sourced from snapshot orderflow payload.")
    else:
        await progress_tracker.start("delta")
        orderflow_block = {
            tf: {"per_bar": [], "summary": {}}
            for tf in orderflow_timeframes
        }
        orderflow_diag = {"disabled": True}
        await progress_tracker.complete(
            "delta",
            bar_counts={tf: 0 for tf in orderflow_timeframes},
            agg_counts={},
            completeness={"disabled": True},
        )
    if rollup_payload:
        ohlcv_block = rollup_payload
    else:
        ohlcv_block = build_multi_timeframe_ohlcv(minute_htf_source, symbol=symbol)
    if trace_ctx is not None:
        trace_ctx.info(
            "compute.rollups",
            scope="rollups",
            frames=len(ohlcv_block) if isinstance(ohlcv_block, Mapping) else 0,
        )
    hourly_htf = aggregate_1m_to_1h(minute_htf_source) if minute_frame_present else []
    htf_blocks: List[Dict[str, Any]] = []
    if minute_frame_present:
        htf_blocks.append({"tf": "1h", "candles": hourly_htf})

    smc_blocks: List[Mapping[str, Any]] = []
    smc_config = _resolve_smc_config(raw_meta if isinstance(raw_meta, Mapping) else None)
    structure_events = _extract_structure_events(snapshot)
    ob_candidates: List[Mapping[str, Any]] = []
    zones_payload = detected_zones.get("zones") if isinstance(detected_zones, Mapping) else None
    if isinstance(zones_payload, Mapping):
        existing_ob = zones_payload.get("ob")
        if isinstance(existing_ob, Sequence):
            for entry in existing_ob:
                if isinstance(entry, Mapping):
                    ob_candidates.append(entry)
    _collect_ob_candidates(snapshot.get("zones"), ob_candidates)
    _collect_ob_candidates(snapshot.get("smt"), ob_candidates)
    liquidity_source: Mapping[str, Any] | None = None
    if isinstance(liquidity_payload, Mapping):
        liquidity_source = liquidity_payload
    elif isinstance(snapshot.get("liquidity"), Mapping):
        liquidity_source = snapshot.get("liquidity")  # type: ignore[assignment]
    smc_timeout = SMC_TIMEOUT_SECONDS if strict_three_day else None
    try:
        smc_future = asyncio.to_thread(
            detect_smc_blocks,
            hourly_htf,
            timeframe="1h",
            structure_flags=structure_events,
            ob_zones=ob_candidates,
            liquidity_levels=liquidity_source,
            config=smc_config,
        )
        if smc_timeout is not None and smc_timeout > 0:
            smc_blocks_list, smc_stats = await asyncio.wait_for(smc_future, timeout=smc_timeout)
        else:
            smc_blocks_list, smc_stats = await smc_future
    except asyncio.TimeoutError:
        smc_blocks_list = []
        smc_stats = {"status": "timeout"}
        timeout_note = "SMC block detection timed out; skipping SMC augmentation."
        if timeout_note not in notes:
            notes.append(timeout_note)
        if trace_ctx is not None:
            trace_ctx.warn(
                "compute.smc.timeout",
                scope="smc",
                timeout_s=smc_timeout,
                candles=len(hourly_htf),
            )
    smc_blocks = smc_blocks_list
    if smc_blocks and isinstance(zones_payload, MutableMapping):
        existing_ob = zones_payload.get("ob")
        merged = [dict(item) for item in existing_ob] if isinstance(existing_ob, list) else []
        merged.extend(dict(block) for block in smc_blocks)
        zones_payload["ob"] = merged

    if smc_blocks:
        _inject_smc_blocks(detailed_section.get("indicators"), smc_blocks)
        _inject_smc_blocks(movement_section.get("indicators"), smc_blocks)

    minute_series = frames.get("1m", [])
    session_series = minute_series
    session_source_tf = "1m"
    session_interval_ms = MINUTE_INTERVAL_MS
    if not session_series:
        fallback_three_min = _deduplicate_sorted(frames.get("3m", []))
        if fallback_three_min:
            session_series = fallback_three_min
            session_source_tf = "3m"
            session_interval_ms = TIMEFRAME_TO_MS.get("3m", MINUTE_INTERVAL_MS * 3)
    if strict_three_day and strict_last_closed_day is not None:
        daily_start_ms = int(
            datetime.combine(strict_last_closed_day, dtime.min, tzinfo=timezone.utc).timestamp() * 1000
        )
    else:
        daily_start_ms = _start_of_day_ms(window_end_ms)
    composite_day_end_ms = daily_start_ms + MS_IN_DAY - MINUTE_INTERVAL_MS
    if composite_day_end_ms < daily_start_ms:
        composite_day_end_ms = daily_start_ms
    effective_daily_end_ms = min(window_end_ms, composite_day_end_ms)
    session_profiles: Dict[str, Dict[str, Any]] = {}
    session_sigma_blocks: Dict[str, Dict[str, Any]] = {}
    session_boundaries: Dict[str, Dict[str, Any]] = {}

    if strict_three_day:
        session_window_map: Dict[str, Tuple[int, int, int]] = {}
        session_day = strict_last_closed_day or (
            safe_datetime_from_ms(window_end_ms, UTC).date()
            if safe_datetime_from_ms(window_end_ms, UTC) is not None
            else datetime.fromtimestamp(window_end_ms / 1000, timezone.utc).date()
        )
        session_tz = VWAP_SESSION_TZ or timezone.utc
        for session_name, session_start, session_end in sessions:
            session_start_ms, session_end_ms, session_close_ms = _session_window_for_day(
                session_day,
                session_start,
                session_end,
                session_tz=session_tz,
            )
            session_start_ms = max(session_start_ms, window_start_ms)
            session_end_ms = min(session_end_ms, window_end_ms)
            session_close_ms = min(session_close_ms, window_end_ms + MINUTE_INTERVAL_MS)
            session_window_map[session_name] = (
                session_start_ms,
                session_end_ms,
                session_close_ms,
            )
        compact_result = build_compact_vwap_profiles(
            session_series,
            daily_window=(daily_start_ms, effective_daily_end_ms),
            composite_window=(daily_start_ms, min(effective_daily_end_ms, composite_day_end_ms)),
            session_windows=session_window_map,
            tick_size=tick_size_numeric,
            value_area_pct=VALUE_AREA_PCT,
            cache_token=("compact_vwap", symbol),
            ib_minutes=60,
            trace_ctx=trace_ctx,
        )
        daily_vwap_profile = compact_result.daily or {
            "vwap": 0.0,
            "sd1": {"minus": None, "plus": None},
            "sd2": {"minus": None, "plus": None},
        }
        if isinstance(daily_vwap_profile, MutableMapping):
            daily_vwap_profile.setdefault("open_utc", _isoformat_utc(daily_start_ms))
            daily_vwap_profile.setdefault("close_utc", _isoformat_utc(effective_daily_end_ms))
        composite_day_profile = compact_result.composite or {}
        session_profiles = {
            name: dict(payload) for name, payload in compact_result.sessions.items()
        }
        session_sigma_blocks = compact_result.session_sigma
        session_boundaries = compact_result.session_boundaries
        for session_name, profile_entry in session_profiles.items():
            boundary = session_boundaries.get(session_name, {})
            start_ms = boundary.get("start_ms")
            close_ms = boundary.get("close_ms")
            if isinstance(profile_entry, MutableMapping):
                profile_entry.setdefault("open_utc", _isoformat_utc(start_ms))
                profile_entry.setdefault("close_utc", _isoformat_utc(close_ms))
            start_ms_int = _safe_int(boundary.get("start_ms"))
            end_ms_int = _safe_int(boundary.get("end_ms"))
            if start_ms_int is not None and end_ms_int is not None:
                session_atr = _session_atr_value(
                    session_series,
                    start_ms=start_ms_int,
                    end_ms=end_ms_int,
                    period=max(1, int(zone_cfg.atr_period or 14)),
                )
                if session_atr is not None:
                    boundary["atr"] = session_atr
                    if isinstance(profile_entry, MutableMapping):
                        profile_entry.setdefault("session_atr", session_atr)
        vwap_sigma_payload = {
            "daily": compact_result.daily_sigma,
            "sessions": session_sigma_blocks,
        }
    else:
        daily_filtered_minutes = _filter_candles(
            session_series, start_ms=daily_start_ms, end_ms=window_end_ms
        )
        daily_vwap_profile = _build_volume_profile_stats(
            session_series,
            start_ms=daily_start_ms,
            end_ms=window_end_ms,
            tick_size=tick_size_numeric,
            value_area_pct=VALUE_AREA_PCT,
        )
        composite_day_profile = _build_volume_profile_stats(
            session_series,
            start_ms=daily_start_ms,
            end_ms=min(window_end_ms, composite_day_end_ms),
            tick_size=tick_size_numeric,
            value_area_pct=VALUE_AREA_PCT,
        )
        for session_name, session_start, session_end in sessions:
            session_start_ms, session_end_ms, session_close_ms = _session_window(
                window_end_ms, session_start, session_end
            )
            session_filtered = _filter_candles(
                session_series, start_ms=session_start_ms, end_ms=session_end_ms
            )
            ib_high, ib_low = _compute_initial_balance_extrema(
                session_filtered, session_start_ms=session_start_ms
            )
            profile_entry = _build_volume_profile_stats(
                session_series,
                start_ms=session_start_ms,
                end_ms=session_end_ms,
                tick_size=tick_size_numeric,
                value_area_pct=VALUE_AREA_PCT,
            )
            if isinstance(profile_entry, MutableMapping):
                if "session_high" in profile_entry and "high" not in profile_entry:
                    profile_entry["high"] = profile_entry.get("session_high")
                if "session_low" in profile_entry and "low" not in profile_entry:
                    profile_entry["low"] = profile_entry.get("session_low")
                profile_entry["open_utc"] = _isoformat_utc(session_start_ms)
                profile_entry["close_utc"] = _isoformat_utc(session_close_ms)
                profile_entry["ib_high"] = ib_high
                profile_entry["ib_low"] = ib_low
            session_atr = _session_atr_value(
                session_series,
                start_ms=session_start_ms,
                end_ms=session_end_ms,
                period=max(1, int(zone_cfg.atr_period or 14)),
            )
            if session_atr is not None:
                if isinstance(profile_entry, MutableMapping):
                    profile_entry["session_atr"] = session_atr
            session_profiles[session_name] = profile_entry
            session_sigma_blocks[session_name] = _build_vwap_sigma_block(
                session_filtered, basis="session"
            )
            session_boundaries[session_name] = {
                "start_ms": session_start_ms,
                "end_ms": session_end_ms,
                "close_ms": session_close_ms,
                "ib_high": ib_high,
                "ib_low": ib_low,
            }
            if session_atr is not None:
                session_boundaries[session_name]["atr"] = session_atr
        vwap_sigma_payload = {
            "daily": _build_vwap_sigma_block(daily_filtered_minutes, basis="daily"),
            "sessions": session_sigma_blocks,
        }

    active_session_name: str | None = None
    active_session_atr: float | None = None

    session_completeness: Dict[str, Dict[str, Any]] = {}
    for session_name, boundary in session_boundaries.items():
        if not isinstance(boundary, Mapping):
            continue
        start_ms = _safe_int(boundary.get("start_ms"))
        end_ms = _safe_int(boundary.get("end_ms"))
        close_ms = _safe_int(boundary.get("close_ms"))
        if start_ms is None or end_ms is None or close_ms is None:
            continue
        session_filtered = _filter_candles(
            session_series,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        completeness = _compute_session_completeness(
            session_filtered,
            start_ms=start_ms,
            end_ms=end_ms,
            close_ms=close_ms,
            interval_ms=session_interval_ms,
            timeframe=session_source_tf,
        )
        if isinstance(boundary, MutableMapping):
            boundary["completeness"] = completeness
        session_completeness[session_name] = completeness

    vwap_payload = {
        "daily": daily_vwap_profile,
        "sessions": session_profiles,
    }

    session_time_lookup = {
        str(name).lower(): (start_time, end_time)
        for name, start_time, end_time in sessions
    }
    if session_boundaries:
        for name, boundary in session_boundaries.items():
            if not isinstance(boundary, Mapping):
                continue
            start_ms = _safe_int(boundary.get("start_ms"))
            close_ms = _safe_int(boundary.get("close_ms"))
            if start_ms is None or close_ms is None:
                continue
            if start_ms <= window_end_ms <= close_ms:
                active_session_name = name
                active_session_atr = _safe_float(boundary.get("atr"))
                break
        if active_session_name is None:
            latest_entry = None
            latest_close = -1
            for name, boundary in session_boundaries.items():
                if not isinstance(boundary, Mapping):
                    continue
                close_ms = _safe_int(boundary.get("close_ms"))
                if close_ms is None:
                    continue
                if close_ms > latest_close:
                    latest_close = close_ms
                    latest_entry = (name, boundary)
            if latest_entry is not None:
                active_session_name = latest_entry[0]
                active_session_atr = _safe_float(latest_entry[1].get("atr"))

    for entry in profile_tpo:
        if not isinstance(entry, MutableMapping):
            continue
        session_label = entry.get("session")
        if not isinstance(session_label, str) or session_label.lower() == "daily":
            continue
        schedule = session_time_lookup.get(session_label.lower())
        if not schedule:
            continue
        date_str = entry.get("date")
        session_date = None
        if isinstance(date_str, str) and date_str:
            try:
                session_date = datetime.fromisoformat(date_str).date()
            except ValueError:
                session_date = None
        if session_date is None:
            continue
        start_time, end_time = schedule
        tz = VWAP_SESSION_TZ or UTC
        start_local = datetime.combine(session_date, start_time, tzinfo=tz)
        end_local = datetime.combine(session_date, end_time, tzinfo=tz)
        if end_time <= start_time:
            end_local += timedelta(days=1)
        start_dt = start_local.astimezone(UTC)
        end_dt = end_local.astimezone(UTC)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) - MINUTE_INTERVAL_MS
        session_candles = _filter_candles(
            session_series, start_ms=start_ms, end_ms=end_ms
        )
        ib_high, ib_low = _compute_initial_balance_extrema(
            session_candles, session_start_ms=start_ms
        )
        if "session_high" in entry and "high" not in entry:
            entry["high"] = entry.get("session_high")
        if "session_low" in entry and "low" not in entry:
            entry["low"] = entry.get("session_low")
        entry["open_utc"] = _isoformat_utc(start_ms)
        entry["close_utc"] = _isoformat_utc(int(end_dt.timestamp() * 1000))
        entry["ib_high"] = ib_high
        entry["ib_low"] = ib_low

    def _sigma_levels_map(block: Mapping[str, Any] | None) -> Dict[int, Dict[str, float | None]]:
        levels: Dict[int, Dict[str, float | None]] = {}
        if not isinstance(block, Mapping):
            return levels
        sigma_entries = block.get("sigma")
        if not isinstance(sigma_entries, Sequence):
            return levels
        for entry in sigma_entries:
            if not isinstance(entry, Mapping):
                continue
            try:
                key = int(entry.get("k"))
            except (TypeError, ValueError):
                continue
            minus_val = _safe_float(entry.get("price_minus"))
            plus_val = _safe_float(entry.get("price_plus"))
            levels[key] = {"minus": minus_val, "plus": plus_val}
        return levels

    def _sd_payload(levels: Mapping[int, Mapping[str, float | None]], order: int) -> Dict[str, float | None]:
        payload = levels.get(order, {}) if isinstance(levels, Mapping) else {}
        minus_value = payload.get("minus") if isinstance(payload, Mapping) else None
        plus_value = payload.get("plus") if isinstance(payload, Mapping) else None
        return {"minus": minus_value, "plus": plus_value}

    daily_sigma_levels = _sigma_levels_map(vwap_sigma_payload.get("daily"))
    vwap_tpo_daily = None
    if isinstance(daily_vwap_profile, Mapping) and daily_vwap_profile:
        vwap_tpo_daily = {
            "open_utc": _isoformat_utc(daily_start_ms),
            "vwap": daily_vwap_profile.get("vwap"),
            "sd1": _sd_payload(daily_sigma_levels, 1),
            "sd2": _sd_payload(daily_sigma_levels, 2),
        }

    vwap_tpo_sessions: Dict[str, Dict[str, Any]] = {}
    session_sigma_levels: Dict[str, Dict[int, Dict[str, float | None]]] = {
        name: _sigma_levels_map(block)
        for name, block in session_sigma_blocks.items()
    }
    for session_name, profile_entry in session_profiles.items():
        boundary = session_boundaries.get(session_name, {})
        sigma_levels = session_sigma_levels.get(session_name, {})
        open_ms = boundary.get("start_ms")
        close_ms = boundary.get("close_ms")
        ib_high = boundary.get("ib_high")
        ib_low = boundary.get("ib_low")
        completeness = session_completeness.get(session_name)

        poc_value: float | None = None
        high_value: float | None = None
        low_value: float | None = None
        if isinstance(profile_entry, Mapping):
            poc_value = _safe_float(profile_entry.get("poc"))
            high_value = _safe_float(profile_entry.get("high"))
            if high_value is None:
                high_value = _safe_float(profile_entry.get("session_high"))
            low_value = _safe_float(profile_entry.get("low"))
            if low_value is None:
                low_value = _safe_float(profile_entry.get("session_low"))
            if (
                poc_value is not None
                and high_value is not None
                and low_value is not None
            ):
                upper = max(high_value, low_value)
                lower = min(high_value, low_value)
                poc_value = min(max(poc_value, lower), upper)
            elif poc_value is not None:
                poc_value = None
            if isinstance(profile_entry, MutableMapping):
                profile_entry["poc"] = poc_value
                if high_value is not None:
                    profile_entry.setdefault("high", high_value)
                if low_value is not None:
                    profile_entry.setdefault("low", low_value)

        session_payload = {
            "open_utc": _isoformat_utc(open_ms) if open_ms is not None else None,
            "close_utc": _isoformat_utc(close_ms) if close_ms is not None else None,
            "vwap": profile_entry.get("vwap") if isinstance(profile_entry, Mapping) else None,
            "sd1": _sd_payload(sigma_levels, 1),
            "sd2": _sd_payload(sigma_levels, 2),
            "poc": poc_value,
            "vah": profile_entry.get("vah") if isinstance(profile_entry, Mapping) else None,
            "val": profile_entry.get("val") if isinstance(profile_entry, Mapping) else None,
            "ib_high": ib_high,
            "ib_low": ib_low,
            "high": high_value,
            "low": low_value,
            "completeness": completeness,
        }
        if isinstance(profile_entry, Mapping):
            raw_high = _safe_float(profile_entry.get("high"))
            raw_low = _safe_float(profile_entry.get("low"))
            if raw_high is not None:
                session_payload["high"] = raw_high
            if raw_low is not None:
                session_payload["low"] = raw_low
        vwap_tpo_sessions[session_name] = session_payload

    composite_day_payload = None
    if isinstance(composite_day_profile, Mapping):
        composite_day_payload = {
            "poc": composite_day_profile.get("poc"),
            "vah": composite_day_profile.get("vah"),
            "val": composite_day_profile.get("val"),
        }

    vwap_tpo_block = {
        "daily": vwap_tpo_daily,
        "sessions": vwap_tpo_sessions,
    }

    vwap_stage_counts = {
        name: session_completeness.get(name, {}).get("bars_observed")
        for name in vwap_tpo_sessions.keys()
    }
    vwap_stage_completeness = {
        name: {
            "status": session_completeness.get(name, {}).get("status"),
            "coverage_ratio": session_completeness.get(name, {}).get("coverage_ratio"),
        }
        for name in vwap_tpo_sessions.keys()
    }
    await progress_tracker.complete(
        "vwap_tpo",
        bar_counts=vwap_stage_counts,
        agg_counts={"sessions": len(vwap_tpo_sessions)},
        completeness=vwap_stage_completeness,
    )

    if trace_ctx is not None:
        trace_ctx.info(
            "compute.vwap_tpo",
            scope="vwap_tpo",
            sessions=len(vwap_tpo_sessions),
        )

    prev_day_block = _build_prev_day_block(
        session_series,
        daily_start_ms=daily_start_ms,
        tick_size=tick_size_numeric,
    )

    def _float_or_none(value: Any) -> float | None:
        return _safe_float(value)

    composite_day_public = {
        "poc": _float_or_none((composite_day_payload or {}).get("poc")),
        "vah": _float_or_none((composite_day_payload or {}).get("vah")),
        "val": _float_or_none((composite_day_payload or {}).get("val")),
    }

    def _normalise_sd(sd_block: Mapping[str, Any] | None) -> Dict[str, float | None]:
        if not isinstance(sd_block, Mapping):
            return {"minus": None, "plus": None}
        return {
            "minus": _float_or_none(sd_block.get("minus")),
            "plus": _float_or_none(sd_block.get("plus")),
        }

    daily_sd1 = _normalise_sd((vwap_tpo_daily or {}).get("sd1") if isinstance(vwap_tpo_daily, Mapping) else None)
    daily_sd2 = _normalise_sd((vwap_tpo_daily or {}).get("sd2") if isinstance(vwap_tpo_daily, Mapping) else None)
    daily_open = None
    daily_vwap_value = None
    if isinstance(vwap_tpo_daily, Mapping):
        daily_open = vwap_tpo_daily.get("open_utc")
        daily_vwap_value = _float_or_none(vwap_tpo_daily.get("vwap"))
    if daily_open is None:
        daily_open = _isoformat_utc(daily_start_ms)

    vwap_tpo_daily_public = {
        "open_utc": daily_open,
        "vwap": daily_vwap_value,
        "sd1": daily_sd1,
        "sd2": daily_sd2,
    }

    ordered_sessions: Dict[str, Dict[str, Any]] = {}
    for session_name, _, _ in sessions:
        raw_payload = vwap_tpo_sessions.get(session_name, {})
        open_utc = raw_payload.get("open_utc") if isinstance(raw_payload, Mapping) else None
        close_utc = raw_payload.get("close_utc") if isinstance(raw_payload, Mapping) else None
        completeness_raw = raw_payload.get("completeness") if isinstance(raw_payload, Mapping) else None

        completeness_public: Dict[str, Any] | None = None
        if isinstance(completeness_raw, Mapping):
            completeness_public = {
                "status": completeness_raw.get("status") if isinstance(completeness_raw.get("status"), str) else None,
                "coverage_ratio": _float_or_none(completeness_raw.get("coverage_ratio")),
                "bars_expected": _safe_int(completeness_raw.get("bars_expected")),
                "bars_observed": _safe_int(completeness_raw.get("bars_observed")),
                "missing_bars": _safe_int(completeness_raw.get("missing_bars")),
                "tf": completeness_raw.get("tf") if isinstance(completeness_raw.get("tf"), str) else None,
            }
            interval_value = _safe_int(completeness_raw.get("interval_ms"))
            if interval_value is not None:
                completeness_public["interval_ms"] = interval_value
            observed_window = completeness_raw.get("observed_window")
            if isinstance(observed_window, Mapping):
                completeness_public["observed_window"] = {
                    "start_ms": _safe_int(observed_window.get("start_ms")),
                    "end_ms": _safe_int(observed_window.get("end_ms")),
                }

        ordered_sessions[session_name] = {
            "open_utc": open_utc,
            "close_utc": close_utc,
            "vwap": _float_or_none(raw_payload.get("vwap")) if isinstance(raw_payload, Mapping) else None,
            "sd1": _normalise_sd(raw_payload.get("sd1") if isinstance(raw_payload, Mapping) else None),
            "sd2": _normalise_sd(raw_payload.get("sd2") if isinstance(raw_payload, Mapping) else None),
            "poc": _float_or_none(raw_payload.get("poc")) if isinstance(raw_payload, Mapping) else None,
            "vah": _float_or_none(raw_payload.get("vah")) if isinstance(raw_payload, Mapping) else None,
            "val": _float_or_none(raw_payload.get("val")) if isinstance(raw_payload, Mapping) else None,
            "ib_high": _float_or_none(raw_payload.get("ib_high")) if isinstance(raw_payload, Mapping) else None,
            "ib_low": _float_or_none(raw_payload.get("ib_low")) if isinstance(raw_payload, Mapping) else None,
            "high": _float_or_none(raw_payload.get("high")) if isinstance(raw_payload, Mapping) else None,
            "low": _float_or_none(raw_payload.get("low")) if isinstance(raw_payload, Mapping) else None,
            "completeness": completeness_public,
        }

    vwap_tpo_public = {
        "daily": vwap_tpo_daily_public,
        "sessions": ordered_sessions,
    }

    ohlcv_public: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for tf in combined_ohlcv_tfs:
        tf_payload = ohlcv_block.get(tf) if isinstance(ohlcv_block, Mapping) else None
        candles: List[Dict[str, Any]] = []
        if isinstance(tf_payload, Mapping):
            raw_candles = tf_payload.get("candles")
            if isinstance(raw_candles, Sequence):
                candles = [dict(candle) for candle in raw_candles if isinstance(candle, Mapping)]
        ohlcv_public[tf] = {"candles": candles}

    orderflow_public: Dict[str, Dict[str, Any]] = {}
    for tf in orderflow_timeframes:
        tf_payload = orderflow_block.get(tf) if isinstance(orderflow_block, Mapping) else None
        per_bar: List[Dict[str, Any]] = []
        summary_payload: Dict[str, Any] = {}
        if isinstance(tf_payload, Mapping):
            raw_series = tf_payload.get("per_bar")
            if isinstance(raw_series, Sequence):
                per_bar = [dict(entry) for entry in raw_series if isinstance(entry, Mapping)]
            summary_source = tf_payload.get("summary")
            if isinstance(summary_source, Mapping):
                summary_payload = dict(summary_source)
        orderflow_public[tf] = {"per_bar": per_bar, "summary": summary_payload}

    footprint_payload = orderflow_block.get("footprint") if isinstance(orderflow_block, Mapping) else None
    footprint_per_bar: List[Dict[str, Any]] = []
    footprint_summary: Dict[str, Any] = {}
    if isinstance(footprint_payload, Mapping):
        raw_series = footprint_payload.get("per_bar")
        if isinstance(raw_series, Sequence):
            footprint_per_bar = [dict(entry) for entry in raw_series if isinstance(entry, Mapping)]
        summary_source = footprint_payload.get("summary")
        if isinstance(summary_source, Mapping):
            footprint_summary = dict(summary_source)
    orderflow_public["footprint"] = {"per_bar": footprint_per_bar, "summary": footprint_summary}

    orderflow_public["diag"] = orderflow_diag

    zones_container = detected_zones.get("zones") if isinstance(detected_zones, Mapping) else None
    zone_keys = ("fvg", "fvl", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels")
    zones_public: Dict[str, Any] = {key: [] for key in zone_keys}
    if isinstance(zones_container, Mapping):
        for key in zone_keys:
            raw_zone = zones_container.get(key)
            if isinstance(raw_zone, Sequence):
                zones_public[key] = [
                    dict(item) for item in raw_zone if isinstance(item, Mapping)
                ]

    allowed_zone_statuses = ZONE_ALLOWED_STATUSES
    for key in zone_keys:
        series = zones_public.get(key)
        if not isinstance(series, list):
            continue
        filtered_series = []
        for entry in series:
            if not isinstance(entry, Mapping):
                continue
            entry_copy = dict(entry)
            if key in ("fvg", "ob"):
                normalised_status = _normalise_zone_status(entry_copy, key)
                entry_copy["status"] = normalised_status
                if normalised_status and normalised_status not in allowed_zone_statuses:
                    continue
            filtered_series.append(entry_copy)
        zones_public[key] = filtered_series

    zones_public["recent"] = zones_recent_focus

    gating_public = zones_diag.get("gating") if isinstance(zones_diag.get("gating"), Mapping) else {}
    zones_public["diag"] = zones_diag
    if strict_window and zone_availability:
        for zone_key, tf_candidates in zone_type_timeframes.items():
            if zones_public.get(zone_key):
                continue
            if not tf_candidates:
                continue
            gating_info = gating_public.get(zone_key) if isinstance(gating_public, Mapping) else None
            requested = gating_info.get("requested_tfs") if isinstance(gating_info, Mapping) else []
            ok_list = gating_info.get("ok_tfs") if isinstance(gating_info, Mapping) else []
            missing_list = gating_info.get("missing_tfs") if isinstance(gating_info, Mapping) else []
            if requested and ok_list:
                continue
            if not requested:
                continue
            details = []
            for tf_name in requested:
                info = zone_availability.get(tf_name, {})
                available = int(info.get("available", len(zone_frames_window.get(tf_name, []))))
                required = int(info.get("adjusted_required", info.get("required", 0)))
                details.append(f"{tf_name}: {available}/{required}")
            message = (
                f"?? ?????????????????? ???????????? ???????????? ?????? {zone_key.upper()} ?????? "
                f"(???????????????? {', '.join(details)})."
            )
            zones_public[zone_key] = [{"message": message, "period": "topup"}]

    if trace_ctx is not None:
        trace_ctx.info(
            "compute.poi.topN",
            scope="zones",
            counts={key: len(zones_public.get(key, [])) for key in zone_keys},
        )


    liquidity_public = {
        "eqh": list(liquidity_equal_levels.get("eqh", [])),
        "eql": list(liquidity_equal_levels.get("eql", [])),
    }
    if isinstance(liquidity_payload, Mapping):
        sweeps_public = liquidity_payload.get("sweeps")
        if isinstance(sweeps_public, Sequence):
            liquidity_public["sweeps"] = [
                dict(entry) for entry in sweeps_public if isinstance(entry, Mapping)
            ]
        config_snapshot = liquidity_payload.get("config")
        if isinstance(config_snapshot, Mapping):
            liquidity_public["config"] = dict(config_snapshot)

    risk_prefs_public = {"rr_min": 2.5, "risk_per_trade_pct": 1.0}

    context_meta = raw_meta.get("context") if isinstance(raw_meta, Mapping) else None
    raw_bias: str | None = None
    raw_narrative: str | None = None
    raw_open_opposite: Any = None
    if isinstance(context_meta, Mapping):
        for key in ("globalBias", "global_bias"):
            value = context_meta.get(key)
            if isinstance(value, str):
                raw_bias = value.lower()
                break
        narrative_value = context_meta.get("narrative")
        if isinstance(narrative_value, str):
            raw_narrative = narrative_value
        raw_open_opposite = context_meta.get("openOppositeZones")
        if raw_open_opposite is None:
            raw_open_opposite = context_meta.get("open_opposite_zones")

    allowed_bias = {"bull", "bear", "neutral"}
    context_public = {
        "globalBias": raw_bias if raw_bias in allowed_bias else "neutral",
        "narrative": raw_narrative or "",
        "openOppositeZones": bool(raw_open_opposite) if isinstance(raw_open_opposite, bool) else False,
    }

    timeframe_order: Tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")
    zone_keys_summary: Tuple[str, ...] = ("eqh", "eql", "fvg", "ob")

    def _normalise_tf_name(value: Any) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return str(value).strip().lower()

    tf_zone_counts: Dict[str, Dict[str, int]] = {
        tf: {zone_key: 0 for zone_key in zone_keys_summary} for tf in timeframe_order
    }
    sweeps_counts: Dict[str, int] = {tf: 0 for tf in timeframe_order}
    atr_values_map: Dict[str, float | None] = {tf: None for tf in timeframe_order}

    if isinstance(liquidity_payload, Mapping):
        for zone_key in ("eqh", "eql"):
            zone_series = liquidity_payload.get(zone_key)
            if not isinstance(zone_series, Sequence):
                continue
            for entry in zone_series:
                if not isinstance(entry, Mapping):
                    continue
                tf_name = _normalise_tf_name(entry.get("tf"))
                if tf_name in tf_zone_counts:
                    tf_zone_counts[tf_name][zone_key] += 1
        sweep_series = liquidity_payload.get("sweeps")
        if isinstance(sweep_series, Sequence):
            for entry in sweep_series:
                if not isinstance(entry, Mapping):
                    continue
                tf_name = _normalise_tf_name(entry.get("tf"))
                if tf_name in sweeps_counts:
                    sweeps_counts[tf_name] += 1

    if isinstance(zones_container, Mapping):
        for zone_key in ("fvg", "ob"):
            zone_series = zones_container.get(zone_key)
            if not isinstance(zone_series, Sequence):
                continue
            for entry in zone_series:
                if not isinstance(entry, Mapping):
                    continue
                tf_name = _normalise_tf_name(entry.get("tf"))
                if tf_name in tf_zone_counts:
                    tf_zone_counts[tf_name][zone_key] += 1

    if isinstance(liquidity_diagnostics, Mapping):
        for tf in timeframe_order:
            tf_diag = liquidity_diagnostics.get(tf)
            if not isinstance(tf_diag, Mapping):
                continue
            atr_stats = tf_diag.get("atr_stats")
            if not isinstance(atr_stats, Mapping):
                continue
            atr_mean = _safe_float(atr_stats.get("atr_mean"))
            if atr_mean is not None and math.isfinite(atr_mean):
                atr_values_map[tf] = float(atr_mean)

    vwap_sessions_public = (
        vwap_tpo_public.get("sessions")
        if isinstance(vwap_tpo_public, Mapping)
        else {}
    )
    session_names: Tuple[str, ...] = ("asia", "london", "ny")
    session_presence_template: Dict[str, bool] = {
        name: bool(vwap_sessions_public.get(name))
        for name in session_names
    }

    timeframes_summary: Dict[str, Dict[str, Any]] = {}
    availability_timeframes: Dict[str, Dict[str, Any]] = {}
    for tf in timeframe_order:
        zones_snapshot = {zone_key: tf_zone_counts[tf].get(zone_key, 0) for zone_key in zone_keys_summary}
        timeframes_summary[tf] = {
            "zones": zones_snapshot,
            "sweeps": sweeps_counts.get(tf, 0),
            "atr": atr_values_map.get(tf),
            "vwap_sessions": dict(session_presence_template),
        }
        availability_timeframes[tf] = {
            "zones": {zone_key: zones_snapshot[zone_key] > 0 for zone_key in zone_keys_summary},
            "sweeps": sweeps_counts.get(tf, 0),
            "atr": atr_values_map.get(tf) is not None,
            "vwap_sessions": dict(session_presence_template),
        }

    smc_summary_lines: List[str] = []
    smc_blocks_public: List[Dict[str, Any]] = []
    if smc_blocks:
        for block in smc_blocks:
            if not isinstance(block, Mapping):
                continue
            public_block = {key: value for key, value in block.items() if key != "_created_idx"}
            smc_blocks_public.append(public_block)
            created_at_ms = _safe_int(block.get("created_at"))
            created_iso = _isoformat_utc(created_at_ms) if created_at_ms is not None else ""
            tf_label = str(block.get("tf") or "").lower()
            direction_label = str(block.get("type") or "").lower()
            kind_label = str(block.get("kind") or "").upper()
            price_range = block.get("range")
            low_value = _safe_float(price_range[0]) if isinstance(price_range, Sequence) and price_range else None
            high_value = _safe_float(price_range[1]) if isinstance(price_range, Sequence) and len(price_range) > 1 else None
            low_text = f"{low_value:.4f}" if low_value is not None else "n/a"
            high_text = f"{high_value:.4f}" if high_value is not None else "n/a"
            status_label = str(block.get("status") or "").lower() or "unknown"
            parts = [
                created_iso,
                tf_label,
                direction_label,
                kind_label,
                f"{low_text}-{high_text}",
                f"status={status_label}",
            ]
            score_value = block.get("score")
            if isinstance(score_value, (int, float)):
                parts.append(f"score={float(score_value):.2f}")
            confidence_value = block.get("confidence")
            if isinstance(confidence_value, (int, float)):
                parts.append(f"conf={float(confidence_value):.2f}")
            smc_summary_lines.append(" | ".join(part for part in parts if part))

    data_payload = {
        "symbol": symbol,
        "ohlcv": ohlcv_public,
        "orderflow": orderflow_public,
        "vwap_tpo": vwap_tpo_public,
        "tpo": {"composite_day": composite_day_public},
        "prev_day": prev_day_block,
        "zones": zones_public,
        "zones_confirmed": [],
        "liquidity": liquidity_public,
        "risk_prefs": risk_prefs_public,
        "context": context_public,
        "timeframes": timeframes_summary,
    }
    if smc_summary_lines or smc_blocks_public:
        data_payload["smc"] = {
            "summary": smc_summary_lines,
            "blocks": smc_blocks_public,
        }

    showcase_window = ZONE_FOCUS_WINDOW_HOURS
    if isinstance(zones_recent_focus, Mapping):
        window_candidate = zones_recent_focus.get("window_hours")
        window_candidate_int = _safe_int(window_candidate)
        if window_candidate_int is not None and window_candidate_int > 0:
            showcase_window = window_candidate_int

    await progress_tracker.start("vitrines")
    showcases_block, showcases_diag = _build_final_showcases(
        zones_recent_focus if isinstance(zones_recent_focus, Mapping) else None,
        window_hours=showcase_window,
        last_price=last_price_value,
        sessions=vwap_tpo_public.get("sessions") if isinstance(vwap_tpo_public, Mapping) else None,
        orderflow=orderflow_public,
        liquidity=liquidity_public,
    )
    data_payload["showcases"] = showcases_block
    vitrines_bar_counts = {
        "A": len(showcases_block.get("A", {}).get("items", [])),
        "B": len(showcases_block.get("B", {}).get("items", [])),
    }
    vitrines_completeness = {
        key: value.get("completeness") if isinstance(value, Mapping) else {}
        for key, value in showcases_block.items()
    }
    await progress_tracker.complete(
        "vitrines",
        bar_counts=vitrines_bar_counts,
        agg_counts={"diagnostics": showcases_diag},
        completeness=vitrines_completeness,
    )

    session_analysis_diag = {
        "enabled": bool(network_backfill),
        "session_status": "skipped",
        "context_status": "skipped",
    }
    if network_backfill:
        session_end_ts = max(window_end_ms, window_start_ms)
        try:
            session_end_dt = datetime.fromtimestamp(session_end_ts / 1000, tz=UTC)
        except (OSError, OverflowError, ValueError):
            session_end_dt = datetime.now(UTC)
        session_timeout = max(SMC_TIMEOUT_SECONDS, 12.0)
        context_timeout = max(SMC_TIMEOUT_SECONDS * 2.0, 18.0)
        session_date = session_end_dt.date()
        try:
            session_result = await asyncio.wait_for(
                build_session_snapshot(
                    symbol,
                    session_date=session_date,
                    cfg=smc_detection_cfg,
                ),
                timeout=session_timeout,
            )
        except asyncio.TimeoutError:
            session_analysis_diag["session_status"] = "timeout"
        except Exception as exc:  # pragma: no cover - defensive
            session_analysis_diag["session_status"] = "error"
            session_analysis_diag["session_error"] = str(exc)
            LOGGER.debug("SMC session snapshot failed for %s", symbol, exc_info=exc)
        else:
            if session_result:
                smc_session_payload = session_result
                session_analysis_diag["session_status"] = "ok"
            else:
                session_analysis_diag["session_status"] = "empty"

        try:
            context_result = await asyncio.wait_for(
                build_72h_context(
                    symbol,
                    hours=72,
                    cfg=smc_detection_cfg,
                ),
                timeout=context_timeout,
            )
        except asyncio.TimeoutError:
            session_analysis_diag["context_status"] = "timeout"
        except Exception as exc:  # pragma: no cover - defensive
            session_analysis_diag["context_status"] = "error"
            session_analysis_diag["context_error"] = str(exc)
            LOGGER.debug("SMC 72h context build failed for %s", symbol, exc_info=exc)
        else:
            if context_result:
                smc_context_payload = context_result
                session_analysis_diag["context_status"] = "ok"
            else:
                session_analysis_diag["context_status"] = "empty"
    else:
        session_analysis_diag["reason"] = "network_disabled"

    if smc_session_payload:
        data_payload["smc_session"] = smc_session_payload
    if smc_context_payload:
        data_payload["smc_context"] = smc_context_payload

    availability: Dict[str, Any] = {"timeframes": dict(availability_timeframes)}
    missing_fields: Set[str] = set()

    if trace_ctx is not None:
        trace_ctx.info(
            "availability.checked",
            scope="payload",
            blocks=list(availability.keys()),
        )

    ohlcv_core_available = {"15m": False, "1h": False}
    orderflow_core_available = {"15m": False, "1h": False}
    session_ready_names: List[str] = []
    session_status_map: Dict[str, str | None] = {}

    for tf in combined_ohlcv_tfs:
        candles_payload = ohlcv_public.get(tf, {})
        candles = candles_payload.get("candles") if isinstance(candles_payload, Mapping) else []
        count = len(candles) if isinstance(candles, Sequence) else 0
        if count == 0:
            missing_fields.add(f"ohlcv.{tf}")
        if tf in ohlcv_core_available and count > 0:
            ohlcv_core_available[tf] = True

    sessions_public = vwap_tpo_public.get("sessions", {}) if isinstance(vwap_tpo_public, Mapping) else {}
    for session_name in ("asia", "london", "ny"):
        raw_session = sessions_public.get(session_name) if isinstance(sessions_public, Mapping) else None
        session_present = isinstance(raw_session, Mapping) and bool(raw_session)
        metrics_presence: Dict[str, bool] = {}
        if not session_present:
            missing_fields.add(f"vwap_tpo.sessions.{session_name}")
        for metric in ("poc", "vah", "val", "ib_high", "ib_low"):
            metric_value = raw_session.get(metric) if isinstance(raw_session, Mapping) else None
            has_metric = metric_value is not None
            metrics_presence[metric] = has_metric
            if not has_metric:
                missing_fields.add(f"vwap_tpo.sessions.{session_name}.{metric}")
        completeness_payload = raw_session.get("completeness") if isinstance(raw_session, Mapping) else None
        status_value: str | None = None
        if isinstance(completeness_payload, Mapping):
            raw_status = completeness_payload.get("status")
            if isinstance(raw_status, str):
                status_value = raw_status.strip().lower() or None
        session_status_map[session_name] = status_value
        if status_value in {"complete", "partial"}:
            session_ready_names.append(session_name)

    for zone_key, series in zones_public.items():
        if zone_key == "diag":
            continue
        count = len(series) if isinstance(series, Sequence) else 0
        if count == 0:
            missing_fields.add(f"zones.{zone_key}")

    orderflow_metrics = {"footprint": False, "delta": False, "cvd": False}
    orderflow_source = snapshot.get("orderflow") if isinstance(snapshot.get("orderflow"), Mapping) else None
    if isinstance(orderflow_source, Mapping):
        footprint_payload = orderflow_source.get("footprint")
        if isinstance(footprint_payload, Sequence) and footprint_payload:
            orderflow_metrics["footprint"] = True

    for tf, payload in orderflow_public.items():
        if tf in {"diag", "footprint"}:
            continue
        per_bar = payload.get("per_bar") if isinstance(payload, Mapping) else []
        series = per_bar if isinstance(per_bar, Sequence) else []
        series_list = [entry for entry in series if isinstance(entry, Mapping)]
        if not series_list:
            missing_fields.add(f"orderflow.{tf}")
        if series_list:
            if any(entry.get("delta") is not None for entry in series_list):
                orderflow_metrics["delta"] = True
            if any(entry.get("cvd") is not None for entry in series_list):
                orderflow_metrics["cvd"] = True
            if tf in orderflow_core_available:
                orderflow_core_available[tf] = True

    footprint_public = orderflow_public.get("footprint", {})
    footprint_series = footprint_public.get("per_bar") if isinstance(footprint_public, Mapping) else []
    footprint_list = [entry for entry in footprint_series if isinstance(entry, Mapping)] if isinstance(footprint_series, Sequence) else []
    if footprint_list:
        orderflow_metrics["footprint"] = True
    else:
        missing_fields.add("orderflow.footprint")

    for metric, present in orderflow_metrics.items():
        if not present:
            missing_fields.add(f"orderflow.{metric}")

    readiness_missing: List[str] = []
    readiness_details = {
        "ohlcv": {tf: bool(value) for tf, value in ohlcv_core_available.items()},
        "orderflow": {tf: bool(value) for tf, value in orderflow_core_available.items()},
        "sessions": dict(session_status_map),
        "sessions_ready": list(session_ready_names),
    }

    if not any(ohlcv_core_available.values()):
        readiness_missing.append("readiness.ohlcv")
    if not any(orderflow_core_available.values()):
        readiness_missing.append("readiness.orderflow")
    if not session_ready_names:
        readiness_missing.append("readiness.sessions")

    readiness_summary = {
        "ready": not readiness_missing,
        "missing": list(readiness_missing),
        "requirements": readiness_details,
    }

    if readiness_missing:
        missing_fields.update(readiness_missing)
        if status == "ok":
            status = "insufficient_data"
        readiness_note = "Readiness requirements unmet: " + ", ".join(readiness_missing)
        if readiness_note not in notes:
            notes.append(readiness_note)
    elif status == "ok":
        status = "ready"

    missing_fields_list = sorted(missing_fields)
    if missing_fields_list:
        LOGGER.info("Missing fields detected", extra={"missing_fields": missing_fields_list})

    meta_block: Dict[str, Any] = {
        "symbol": symbol,
        "tz": "Europe/Berlin",
        "last_price": last_price_value,
        "last_ts_utc": last_iso_ts,
        "last_tf": last_tf,
        "last_price_source": last_price_source,
    }
    meta_block["pipeline_preset"] = pipeline_preset.name
    meta_block["modules"] = dict(modules_enabled)
    meta_block["window_hours"] = int(base_window_hours)
    meta_block["strict_window"] = bool(strict_window)
    meta_block["strict_three_day"] = bool(strict_three_day)
    meta_block["minute_backfill_enabled"] = bool(minute_backfill_enabled)
    if vision_ingest_summary is not None:
        ingest_snapshot = dict(vision_ingest_summary)
        if vision_required_hours is not None:
            ingest_snapshot["required_hours"] = int(vision_required_hours)
        window_payload = ingest_snapshot.get("window")
        if isinstance(window_payload, Mapping):
            ingest_snapshot["window"] = dict(window_payload)
        meta_block["vision_ingest"] = ingest_snapshot
    if snapshot_age_sec is not None:
        meta_block["snapshot_age_sec"] = snapshot_age_sec
    if insufficient_reason:
        meta_block["insufficient_reason"] = insufficient_reason
    if status == "insufficient_data" and readiness_missing and not meta_block.get("insufficient_reason"):
        meta_block["insufficient_reason"] = "readiness_requirements"
    meta_block["stale"] = bool(
        isinstance(insufficient_reason, str) and insufficient_reason.startswith("stale_snapshot")
    )
    if last_price_diag.get("mismatch"):
        meta_block["stream_vs_ohlcv_mismatch"] = True

    invalid_total = invalid_ts_total + invalid_ohlc_total
    meta_block["invalid_candles_count"] = invalid_total
    meta_block["invalid_ts_count"] = invalid_ts_total
    meta_block["invalid_ohlc_count"] = invalid_ohlc_total
    meta_block["sanitized"] = sanitized_applied
    meta_block["sessions_empty"] = sessions_empty_flag

    stage_breakdown: Dict[str, Dict[str, int]] = {}
    for stage, counts in sorted(invalid_candle_stages.items()):
        filtered_counts = {key: value for key, value in counts.items() if value}
        if filtered_counts:
            stage_breakdown[stage] = filtered_counts
    meta_block["invalid_candle_stages"] = stage_breakdown

    if invalid_ts_total:
        if stage_breakdown:
            ts_stages = {stage: counts.get("invalid_ts", 0) for stage, counts in stage_breakdown.items()}
            ts_summary = ", ".join(
                f"{stage}:{count}" for stage, count in ts_stages.items() if count
            )
        else:
            ts_summary = ""
        if ts_summary:
            notes.append(
                f"Filtered {invalid_ts_total} candles with invalid timestamps ({ts_summary})"
            )
        else:
            notes.append(
                f"Filtered {invalid_ts_total} candles with invalid timestamps"
            )
    if invalid_ohlc_total:
        notes.append(
            f"Filtered {invalid_ohlc_total} candles with invalid OHLC ranges"
        )
    meta_block["readiness"] = readiness_summary

    expected_total_minutes = len(expected_minutes)
    coverage_targets: Tuple[str, ...] = combined_ohlcv_tfs
    coverage_diag: Dict[str, Any] = {}
    coverage_trace: List[Dict[str, Any]] = []

    for tf in coverage_targets:
        interval_ms = TIMEFRAME_TO_MS.get(tf)
        tf_payload = ohlcv_public.get(tf) if isinstance(ohlcv_public, Mapping) else {}
        raw_candles = tf_payload.get("candles") if isinstance(tf_payload, Mapping) else []
        tf_candles = [dict(candle) for candle in raw_candles if isinstance(candle, Mapping)]
        deduped_series = _deduplicate_sorted(tf_candles) if tf_candles else []
        series_index: Dict[int, Dict[str, Any]] = {}
        for candle in deduped_series:
            ts = _safe_int(candle.get("t"))
            if ts is None:
                continue
            series_index[ts] = candle

        coverage_start = window_start_ms
        coverage_end = window_end_ms
        expected_times: List[int] = []
        filtered_index: Dict[int, Dict[str, Any]] = series_index

        if interval_ms:
            aligned_start = _align_to_interval(window_start_ms, interval_ms)
            if aligned_start < window_start_ms:
                aligned_start += interval_ms
            aligned_end = _align_to_interval(window_end_ms, interval_ms)
            if aligned_end < aligned_start:
                aligned_end = aligned_start
            coverage_start = aligned_start
            coverage_end = aligned_end
            expected_times = _build_expected_times(coverage_start, coverage_end, interval_ms)
            filtered_index = {
                ts: series_index[ts]
                for ts in series_index
                if coverage_start <= ts <= coverage_end
            }

        present_bars = len(filtered_index)
        expected_bars = len(expected_times) if expected_times else present_bars
        gaps = (
            _summarise_missing_times(expected_times, filtered_index)
            if expected_times
            else []
        )
        missing_bars = sum(int(gap.get("count", 0)) for gap in gaps)
        gap_count = len(gaps)

        coverage_state: Dict[str, Any] = {
            "interval_ms": interval_ms,
            "window": {"start_ms": coverage_start, "end_ms": coverage_end},
            "expected_bars": expected_bars,
            "present_bars": present_bars,
            "missing_bars": missing_bars,
            "gap_count": gap_count,
        }
        if gaps:
            coverage_state["time_gaps"] = gaps
        coverage_diag[tf] = coverage_state
        coverage_trace.append(
            {
                "tf": tf,
                "interval_ms": interval_ms,
                "expected_bars": expected_bars,
                "present_bars": present_bars,
                "gap_count": gap_count,
            }
        )

    minute_state = coverage_diag.get("1m")
    if minute_state is not None:
        minute_gaps = list(time_gaps)
        minute_missing_after = int(minute_coverage_diag.get("minute_missing_after", minute_missing_after)) if isinstance(minute_coverage_diag, Mapping) else minute_missing_after  # type: ignore[has-type]
        minute_missing_before_current = int(minute_coverage_diag.get("minute_missing_before", minute_missing_before)) if isinstance(minute_coverage_diag, Mapping) else minute_missing_before  # type: ignore[has-type]
        fetched_after = int(minute_coverage_diag.get("fetched_1m_count", fetched_unique)) if isinstance(minute_coverage_diag, Mapping) else fetched_unique  # type: ignore[has-type]
        minute_state.update(
            {
                "window": {"start_ms": window_start_ms, "end_ms": window_end_ms},
                "expected_bars": expected_total_minutes,
                "present_bars": max(0, expected_total_minutes - minute_missing_after),
                "missing_bars": max(0, minute_missing_after),
                "gap_count": len(minute_gaps),
                "missing_before": minute_missing_before_current,
                "missing_after": minute_missing_after,
                "fetched": fetched_after,
                "time_gaps": minute_gaps,
            }
        )
        for trace_entry in coverage_trace:
            if trace_entry.get("tf") == "1m":
                trace_entry["expected_bars"] = minute_state["expected_bars"]
                trace_entry["present_bars"] = minute_state["present_bars"]
                trace_entry["gap_count"] = minute_state["gap_count"]
                break

    api_diag_block = {
        "requests": 0,
        "retries": 0,
        "rate_limit_hits": 0,
        "backoffs": 0,
    }

    sessions_diag: Dict[str, Any] = {}
    if isinstance(vwap_sessions_public, Mapping):
        for session_name in session_names:
            session_payload = vwap_sessions_public.get(session_name)
            if isinstance(session_payload, Mapping):
                sessions_diag[session_name] = {
                    "present": bool(session_payload),
                    "open_utc": session_payload.get("open_utc"),
                    "close_utc": session_payload.get("close_utc"),
                }
            else:
                sessions_diag[session_name] = {"present": False}
    else:
        for session_name in session_names:
            sessions_diag[session_name] = {"present": False}

    diagnostics_block: Dict[str, Any] = {}
    if liquidity_diagnostics:
        diagnostics_block["liquidity"] = dict(liquidity_diagnostics)
    if zones_diag:
        diagnostics_block["zones"] = dict(zones_diag)
    if liquidity_equal_levels:
        diagnostics_block["liquidity_levels"] = dict(liquidity_equal_levels)
    diagnostics_block["orderflow"] = copy.deepcopy(orderflow_public)
    diagnostics_block["vwap_tpo"] = copy.deepcopy(vwap_tpo_block)
    diagnostics_block["movement"] = movement_section
    diagnostics_block["showcases"] = showcases_diag
    diagnostics_block["readiness"] = readiness_summary
    diagnostics_block["coverage"] = coverage_diag
    diagnostics_block["api"] = api_diag_block
    diagnostics_block["sessions"] = sessions_diag
    diagnostics_block["coverage_trace"] = coverage_trace
    diagnostics_block["daily"] = {
        "available": bool(
            isinstance(ohlcv_public.get("1d"), Mapping)
            and ohlcv_public.get("1d", {}).get("candles")
        )
    }
    if session_analysis_diag is not None:
        diagnostics_block["session_analysis"] = session_analysis_diag
    meta_block["diagnostics"] = diagnostics_block

    if trace_ctx is not None:
        trace_ctx.info(
            "output.prepare_payload",
            scope="output",
            status=status,
            missing_fields=len(missing_fields_list),
        )

    final_payload = {
        "status": status,
        "meta": meta_block,
        "data": data_payload,
        "availability": availability,
        "missing_fields": missing_fields_list,
        "notes": notes,
    }

    return await _finish(
        _finalise_payload(
            final_payload,
            status=status,
            pipeline_start=pipeline_start,
            fetch_ms=fetch_ms,
            db_ms=db_ms,
            trace_ctx=trace_ctx,
        )
    )
