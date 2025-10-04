"""Snapshot builder for the inspection check-all endpoint."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime
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

import httpx

import src.services.inspection as inspection
from .binance import BINANCE_FAPI_REST
from .candles_repository import get_repository
from .inspection import build_htf_section
from .liquidity import (
    build_liquidity_snapshot,
    normalise_symbol_for_tick,
    resolve_liquidity_tick_size,
)
from .presets import resolve_profile_config
from .profile import build_profile_package
from .ohlc_sanitizer import SanitizedCandles, sanitize_candles
from .smc import SMCConfig, detect_smc_blocks
from .zones import Config as ZonesConfig, detect_zones
from .timeutils import safe_datetime_from_ms

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .summary_collector import CollectionSummary
UTC = timezone.utc
LOGGER = logging.getLogger(__name__)
MS_IN_HOUR = 3_600_000
MS_IN_DAY = 86_400_000
VALID_HOUR_WINDOWS = {1, 2, 3, 4}
VALUE_AREA_PCT = 0.70
REPOSITORY_LOOKBACK_MS = 5 * MS_IN_DAY

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

VWAP_TPO_SESSIONS: Tuple[Tuple[str, dtime, dtime], ...] = (
    ("asia", dtime(hour=0, minute=0), dtime(hour=3, minute=0)),
    ("london", dtime(hour=7, minute=0), dtime(hour=10, minute=0)),
    ("ny", dtime(hour=13, minute=30), dtime(hour=16, minute=30)),
)

_RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}
_MAX_RETRIES = 5

_BUILD_TIMEOUT_SECONDS = 12.0
_ASYNC_BUILD_TIMEOUT_SECONDS = 5.0

_EXPECTED_OHLCV_TFS: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")
_EXPECTED_ORDERFLOW_TFS: Tuple[str, ...] = ("15m", "1h")
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
    **kwargs: Any,
) -> Dict[str, Any] | None:
    """Execute ``build_check_all_datas`` with a timeout that respects cancellation."""

    context = _prepare_snapshot_context(snapshot, kwargs.get("now_utc"))
    build_kwargs = dict(kwargs)
    build_kwargs.setdefault("network_backfill", network_backfill)
    if trace is not None:
        build_kwargs.setdefault("trace", trace)

    async def _invoke() -> Dict[str, Any] | None:
        return await build_check_all_datas(snapshot, **build_kwargs)

    try:
        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(_invoke(), timeout)
        return await _invoke()
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
        return _insufficient_from_context(context)


def _isoformat_utc(timestamp_ms: int) -> str:
    """Return a stable Z-suffixed ISO string for a millisecond timestamp."""

    clamped_ms = max(0, int(timestamp_ms))
    dt = safe_datetime_from_ms(clamped_ms, UTC)
    if dt is None:
        return "1970-01-01T00:00:00Z"
    return dt.isoformat().replace("+00:00", "Z")


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


def _align_to_interval(value: int, interval_ms: int) -> int:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (value // interval_ms) * interval_ms


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
    }

    availability_payload = {
        "ohlcv": {tf: {"candles": 0, "has_data": False} for tf in _EXPECTED_OHLCV_TFS},
        "vwap_sessions": {
            session: {
                "present": False,
                "metrics": {
                    metric: False
                    for metric in ("poc", "vah", "val", "ib_high", "ib_low")
                },
            }
            for session in ("asia", "london", "ny")
        },
        "zones": {key: {"count": 0} for key in _EXPECTED_ZONE_KEYS},
        "orderflow": {
            "timeframes": {
                tf: {"bars": 0, "has_data": False} for tf in _EXPECTED_ORDERFLOW_TFS
            },
            "metrics": {metric: False for metric in _EXPECTED_ORDERFLOW_METRICS},
        },
    }

    missing_fields: Set[str] = set()
    missing_fields.update(f"ohlcv.{tf}" for tf in _EXPECTED_OHLCV_TFS)
    for session in ("asia", "london", "ny"):
        missing_fields.add(f"vwap_tpo.sessions.{session}")
        for metric in ("poc", "vah", "val", "ib_high", "ib_low"):
            missing_fields.add(f"vwap_tpo.sessions.{session}.{metric}")
    missing_fields.update(f"zones.{key}" for key in _EXPECTED_ZONE_KEYS)
    missing_fields.update(f"orderflow.{tf}" for tf in _EXPECTED_ORDERFLOW_TFS)
    missing_fields.update(f"orderflow.{metric}" for metric in _EXPECTED_ORDERFLOW_METRICS)

    payload = {
        "status": "insufficient_data",
        "meta": meta_block,
        "data": data_payload,
        "availability": availability_payload,
        "missing_fields": sorted(missing_fields),
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


def _build_expected_times(start_ms: int, end_ms: int, interval_ms: int) -> List[int]:
    if end_ms < start_ms:
        return []
    steps = ((end_ms - start_ms) // interval_ms) + 1
    return [start_ms + index * interval_ms for index in range(steps)]


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
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
    budget: _TimeBudget | None = None,
) -> List[Sequence[object]]:
    params = {
        "symbol": symbol.upper(),
        "interval": "1m",
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": str(limit),
    }

    delay = 0.5
    for attempt in range(_MAX_RETRIES):
        if budget is not None:
            budget.raise_if_exceeded("request_binance_minutes")
        try:
            response = await client.get(BINANCE_FAPI_REST, params=params)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data  # type: ignore[return-value]
            return []
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                if budget is not None:
                    budget.raise_if_exceeded("request_binance_minutes_backoff")
                    remaining = budget.remaining()
                    if remaining is not None and remaining <= 0:
                        raise
                    await asyncio.sleep(min(delay, max(0.0, remaining)))
                else:
                    await asyncio.sleep(delay)
                delay *= 2
                continue
            raise
        except httpx.RequestError:
            if attempt < _MAX_RETRIES - 1:
                if budget is not None:
                    budget.raise_if_exceeded("request_binance_minutes_retry")
                    remaining = budget.remaining()
                    if remaining is not None and remaining <= 0:
                        raise
                    await asyncio.sleep(min(delay, max(0.0, remaining)))
                else:
                    await asyncio.sleep(delay)
                delay *= 2
                continue
            raise
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
        timeout = httpx.Timeout(25.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
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
                    raw_rows = await _request_binance_minutes_async(
                        client,
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
    except (httpx.HTTPError, httpx.TransportError) as exc:  # pragma: no cover - defensive
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
            return None
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


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _pivot_radius_for_equal_levels(tf: str) -> int:
    return _EQUAL_LIQUIDITY_PIVOT_RADIUS.get(tf, 2)


def _minimum_separation_for_equal_levels(tf: str) -> int:
    return _EQUAL_LIQUIDITY_MIN_SEPARATION.get(tf, 5)


def _relative_tolerance_for_equal_levels(tf: str) -> float:
    return _EQUAL_LIQUIDITY_REL_TOLERANCE.get(tf, 0.0003)


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
    frames: Mapping[str, Sequence[Mapping[str, Any]]]
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


@dataclass(slots=True)
class OrderflowConfig:
    """Configuration overrides for orderflow-derived metrics."""

    imbalance_ratio: float = 1.8
    absorption_ratio: float = 2.0
    atr_period: int = 14
    atr_band_k: float = 0.2
    large_trade_min_qty: float = 0.0
    large_trade_lookback_minutes: int = 2880
    large_trade_percentile: float = 0.99
    epsilon: float = 1e-9


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


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 1:
        return max(values)
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = index - lower
    return lower_value * (1 - weight) + upper_value * weight


def _compute_atr_series(
    candles: Sequence[Mapping[str, Any]],
    *,
    period: int,
) -> Dict[int, float]:
    atr_values: Dict[int, float] = {}
    prev_close: float | None = None
    recent_tr: List[float] = []
    sorted_candles: List[Tuple[int, Mapping[str, Any]]] = []
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        sorted_candles.append((ts, candle))
    sorted_candles.sort(key=lambda item: item[0])

    if not sorted_candles:
        return atr_values

    period = max(1, int(period))

    for ts, candle in sorted_candles:
        high = float(candle.get("h", 0.0))
        low = float(candle.get("l", 0.0))
        close = float(candle.get("c", 0.0))
        range_high_low = high - low
        if prev_close is None:
            true_range = range_high_low
        else:
            true_range = max(
                range_high_low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
        recent_tr.append(true_range)
        if len(recent_tr) > period:
            recent_tr.pop(0)
        atr = sum(recent_tr) / len(recent_tr)
        atr_values[ts] = atr
        prev_close = close

    return atr_values


def _extract_trades(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    trades_raw = payload.get("agg")
    trades: List[Dict[str, Any]] = []
    if not isinstance(trades_raw, Sequence):
        return trades
    for entry in trades_raw:
        if not isinstance(entry, Mapping):
            continue
        ts = _safe_int(entry.get("t"))
        if ts is None:
            continue
        qty = _coerce_float(entry.get("q"))
        if not math.isfinite(qty):
            continue
        side = str(entry.get("side", "")).strip().lower()
        if side not in {"buy", "sell"}:
            continue
        trades.append({"t": ts, "q": float(qty), "side": side})
    return trades


def _bucket_trades_by_minute(
    trades: Sequence[Mapping[str, Any]],
    *,
    minute_interval: int,
    start_ms: int | None,
    end_ms: int | None,
) -> Dict[int, List[Mapping[str, Any]]]:
    buckets: Dict[int, List[Mapping[str, Any]]] = {}
    for trade in trades:
        ts = _safe_int(trade.get("t"))
        if ts is None:
            continue
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts >= end_ms:
            continue
        bucket = _align_to_interval(ts, minute_interval)
        buckets.setdefault(bucket, []).append(trade)
    return buckets


def _compute_large_trade_threshold(
    trades: Sequence[Mapping[str, Any]],
    *,
    cutoff_ts: int | None,
    lookback_minutes: int,
    minute_interval: int,
    base_threshold: float,
    percentile: float,
) -> float:
    if not trades:
        return max(0.0, base_threshold)

    reference_cutoff = None
    if cutoff_ts is not None:
        reference_cutoff = cutoff_ts - max(0, lookback_minutes - 1) * minute_interval

    quantities: List[float] = []
    for trade in trades:
        ts = _safe_int(trade.get("t"))
        if ts is None:
            continue
        if reference_cutoff is not None and ts < reference_cutoff:
            continue
        qty = _coerce_float(trade.get("q"))
        if math.isfinite(qty):
            quantities.append(float(qty))

    if not quantities:
        for trade in trades:
            qty = _coerce_float(trade.get("q"))
            if math.isfinite(qty):
                quantities.append(float(qty))

    percentile_value = _percentile(quantities, percentile) if quantities else None
    threshold = max(0.0, base_threshold)
    if percentile_value is not None:
        threshold = max(threshold, float(percentile_value))
    return threshold


def _build_orderflow_per_bar(
    minute_candles: Sequence[Mapping[str, Any]],
    trades_by_minute: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    config: OrderflowConfig,
    large_trade_threshold: float,
) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    atr_series = _compute_atr_series(minute_candles, period=config.atr_period)

    running_cvd = 0.0
    for candle in sorted(minute_candles, key=lambda item: _safe_int(item.get("t")) or 0):
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        trades = trades_by_minute.get(ts, [])
        ask_volume = sum(float(trade.get("q", 0.0)) for trade in trades if str(trade.get("side")).lower() == "buy")
        bid_volume = sum(float(trade.get("q", 0.0)) for trade in trades if str(trade.get("side")).lower() == "sell")

        delta = ask_volume - bid_volume
        running_cvd += delta

        atr_value = atr_series.get(ts)
        if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
            atr_value = max(float(candle.get("h", 0.0)) - float(candle.get("l", 0.0)), 0.0)
        band = atr_value * config.atr_band_k

        close_price = float(candle.get("c", 0.0))
        high_price = float(candle.get("h", 0.0))
        low_price = float(candle.get("l", 0.0))

        close_near_high = abs(high_price - close_price) <= band if band > 0 else math.isclose(high_price, close_price)
        close_near_low = abs(close_price - low_price) <= band if band > 0 else math.isclose(low_price, close_price)

        imbalance_buy = False
        imbalance_sell = False
        if ask_volume > 0:
            imbalance_buy = (ask_volume / max(bid_volume, config.epsilon)) >= config.imbalance_ratio
        if bid_volume > 0:
            imbalance_sell = (bid_volume / max(ask_volume, config.epsilon)) >= config.imbalance_ratio

        absorption_high = (
            delta < 0
            and close_near_high
            and bid_volume > 0
            and (bid_volume / max(ask_volume, config.epsilon)) >= config.absorption_ratio
        )
        absorption_low = (
            delta > 0
            and close_near_low
            and ask_volume > 0
            and (ask_volume / max(bid_volume, config.epsilon)) >= config.absorption_ratio
        )

        large_count = 0
        if trades:
            threshold = max(0.0, large_trade_threshold)
            large_count = sum(1 for trade in trades if float(trade.get("q", 0.0)) >= threshold)

        series.append(
            {
                "ts": ts,
                "delta": delta,
                "cvd": running_cvd,
                "bid_vol": bid_volume,
                "ask_vol": ask_volume,
                "large_trades_count": int(large_count),
                "absorption_high": bool(absorption_high),
                "absorption_low": bool(absorption_low),
                "imbalance_buy": bool(imbalance_buy),
                "imbalance_sell": bool(imbalance_sell),
            }
        )

    return series


def _aggregate_orderflow_series(
    series: Sequence[Mapping[str, Any]],
    *,
    interval_ms: int,
    minute_interval: int,
) -> List[Dict[str, Any]]:
    if not series:
        return []

    per_minute = {int(item["ts"]): item for item in series if isinstance(item, Mapping) and "ts" in item}
    bucket_times = sorted({_align_to_interval(ts, interval_ms) for ts in per_minute})
    expected = max(1, interval_ms // minute_interval)

    aggregated: List[Dict[str, Any]] = []
    running_cvd = 0.0

    for bucket_start in bucket_times:
        bucket_entries: List[Mapping[str, Any]] = []
        for index in range(expected):
            minute_ts = bucket_start + index * minute_interval
            entry = per_minute.get(minute_ts)
            if entry is None:
                bucket_entries = []
                break
            bucket_entries.append(entry)
        if not bucket_entries:
            continue

        ask_volume = sum(float(entry.get("ask_vol", 0.0)) for entry in bucket_entries)
        bid_volume = sum(float(entry.get("bid_vol", 0.0)) for entry in bucket_entries)
        delta = ask_volume - bid_volume
        running_cvd += delta

        aggregated.append(
            {
                "ts": bucket_start,
                "delta": delta,
                "cvd": running_cvd,
                "bid_vol": bid_volume,
                "ask_vol": ask_volume,
                "large_trades_count": int(
                    sum(int(entry.get("large_trades_count", 0)) for entry in bucket_entries)
                ),
                "absorption_high": any(bool(entry.get("absorption_high")) for entry in bucket_entries),
                "absorption_low": any(bool(entry.get("absorption_low")) for entry in bucket_entries),
                "imbalance_buy": any(bool(entry.get("imbalance_buy")) for entry in bucket_entries),
                "imbalance_sell": any(bool(entry.get("imbalance_sell")) for entry in bucket_entries),
            }
        )

    return aggregated


def _build_orderflow_block(
    minute_candles: Sequence[Mapping[str, Any]],
    trades_payload: Any,
    *,
    config: OrderflowConfig,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    minute_interval = MINUTE_INTERVAL_MS
    if not minute_candles or minute_interval <= 0:
        return {tf: {"per_bar": []} for tf in ("1m", "3m", "5m", "15m")}

    trades = _extract_trades(trades_payload)
    minute_ts: List[int] = []
    for candle in minute_candles:
        ts = _safe_int(candle.get("t"))
        if ts is not None:
            minute_ts.append(ts)
    minute_ts.sort()
    start_ms = minute_ts[0] if minute_ts else None
    end_ms = (minute_ts[-1] + minute_interval) if minute_ts else None

    trades_by_minute = _bucket_trades_by_minute(
        trades,
        minute_interval=minute_interval,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    last_ts = minute_ts[-1] if minute_ts else None
    threshold = _compute_large_trade_threshold(
        trades,
        cutoff_ts=last_ts,
        lookback_minutes=config.large_trade_lookback_minutes,
        minute_interval=minute_interval,
        base_threshold=config.large_trade_min_qty,
        percentile=config.large_trade_percentile,
    )

    minute_series = _build_orderflow_per_bar(
        minute_candles,
        trades_by_minute,
        config=config,
        large_trade_threshold=threshold,
    )

    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for tf in ("15m", "1h"):
        interval_ms = TIMEFRAME_TO_MS.get(tf)
        if not interval_ms or interval_ms <= minute_interval:
            result[tf] = {"per_bar": minute_series[:] if interval_ms == minute_interval else []}
            continue
        aggregated = _aggregate_orderflow_series(
            minute_series,
            interval_ms=interval_ms,
            minute_interval=minute_interval,
        )
        result[tf] = {"per_bar": aggregated}

    return result


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
) -> tuple[int, int, int]:
    anchor_aligned = _align_to_interval(anchor_ms, MINUTE_INTERVAL_MS)
    anchor_dt = safe_datetime_from_ms(anchor_aligned, UTC)
    if anchor_dt is None:
        anchor_dt = datetime(1970, 1, 1, tzinfo=UTC)
    day_start = datetime(anchor_dt.year, anchor_dt.month, anchor_dt.day, tzinfo=UTC)
    session_start_dt = datetime.combine(day_start.date(), start_time, tzinfo=UTC)
    session_end_dt = datetime.combine(day_start.date(), end_time, tzinfo=UTC)

    if end_time <= start_time:
        session_end_dt += timedelta(days=1)

    start_ms = int(session_start_dt.timestamp() * 1000)
    end_boundary_ms = int(session_end_dt.timestamp() * 1000)
    raw_end_ms = end_boundary_ms - MINUTE_INTERVAL_MS
    if raw_end_ms < start_ms:
        raw_end_ms = start_ms

    end_ms = min(raw_end_ms, anchor_aligned)
    close_ms = end_boundary_ms

    if end_ms < start_ms:
        end_ms = start_ms
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

    result: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "summary": summary,
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

    for tf_key, candles in list(frames.items()):
        stage_name = f"seed.{tf_key}"
        frames[tf_key] = _apply_sanitizer(stage_name, candles)
    raw_meta = context.raw_meta
    symbol = context.symbol
    now_dt = context.now
    stream_price = context.stream_price
    stream_ts = context.stream_ts
    now_ms = int(now_dt.timestamp() * 1000)

    if trace_ctx is not None:
        trace_ctx.info(
            "ui.click_received",
            scope="ui",
            symbol=symbol,
            strict_window=strict_window,
            network_backfill=network_backfill,
        )
        trace_ctx.info(
            "pipeline.start",
            scope="pipeline",
            symbol=symbol,
            strict_window=strict_window,
            network_backfill=network_backfill,
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
    if strict_three_day and network_backfill:
        network_backfill = False

    window_end_guess = _resolve_window_end_ms(
        frames,
        now_ms=now_ms,
        selection_end_ms=selection_end_ms,
        stream_ts=stream_ts,
    )

    if strict_three_day:
        aligned_end = _align_to_interval(window_end_guess, MINUTE_INTERVAL_MS)
        if aligned_end > window_end_guess:
            aligned_end -= MINUTE_INTERVAL_MS
        window_end_guess = max(aligned_end, MINUTE_INTERVAL_MS)

    if trace_ctx is not None:
        trace_ctx.info(
            "pipeline.contract_ok",
            scope="pipeline",
            base_window_hours=base_window_hours,
            strict_three_day=strict_three_day,
            window_end_ms=window_end_guess,
        )

    summary_result: "CollectionSummary" | None = None
    if strict_three_day:
        summary_end_ms = window_end_guess
        summary_start_ms = max(
            0,
            summary_end_ms - 72 * MS_IN_HOUR + MINUTE_INTERVAL_MS,
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
            return _finalise_payload(
                minute_payload,
                status="minute_missing",
                pipeline_start=pipeline_start,
                fetch_ms=fetch_ms,
                db_ms=db_ms,
                trace_ctx=trace_ctx,
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

    repo_window_ms = max(REPOSITORY_LOOKBACK_MS, base_window_hours * MS_IN_HOUR)
    repo_start_ms = max(0, _align_to_interval(window_end_guess - repo_window_ms, MINUTE_INTERVAL_MS))
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
        return _insufficient_from_context(context, now_override=now_dt)

    if not strict_three_day:
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
            allow_network=network_backfill,
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
        return _insufficient_from_context(context, now_override=now_dt)

    if not frames.get(primary_key):
        try:
            budget.raise_if_exceeded("seed_primary_backfill")
        except _TimeBudgetExceeded as exc:
            LOGGER.warning(
                "Time budget exceeded before seeding primary frame",
                extra={"stage": exc.stage, "symbol": symbol, "primary_key": primary_key},
            )
            return _insufficient_from_context(context, now_override=now_dt)
        fetch_start = time.perf_counter()
        await _backfill_timeframe_with_rest(
            frames,
            symbol=symbol,
            timeframe=primary_key,
            window_end_ms=window_end_guess,
            window_hours=base_window_hours,
            allow_network=network_backfill,
        )
        fetch_ms += (time.perf_counter() - fetch_start) * 1000.0
        frames[primary_key] = _apply_sanitizer(f"rest.{primary_key}", frames.get(primary_key, []))

    primary_candles = frames.get(primary_key, [])
    if not primary_candles:
        return _insufficient_from_context(context, now_override=now_dt)

    # Drop unused granularities to keep the payload focused on the requested set.
    frames.pop("3m", None)
    frames.pop("5m", None)

    minute_candles = _deduplicate_sorted(frames.get("1m", []))
    frames["1m"] = minute_candles

    rollup_payload: Dict[str, Dict[str, Any]] = {}
    if minute_candles:
        lookback_days = max(1, int(math.ceil(base_window_hours / 24)))
        try:
            rollup_payload = await build_multi_tf_ohlcv(
                symbol,
                lookback_days,
                timeframes=("1m", "3m", "5m", "15m", "1h", "4h", "1d"),
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
                return _finalise_payload(
                    minute_payload,
                    status=status,
                    pipeline_start=pipeline_start,
                    fetch_ms=fetch_ms,
                    db_ms=db_ms,
                    trace_ctx=trace_ctx,
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
        for tf_key in ("3m", "5m", "15m", "1h", "4h", "1d"):
            existing_series = frames.get(tf_key)
            if existing_series:
                continue
            tf_payload = rollup_payload.get(tf_key)
            if not isinstance(tf_payload, Mapping):
                continue
            normalised_series = _normalise_external_candles(tf_payload)
            if normalised_series:
                frames[tf_key] = normalised_series

    profile_config = resolve_profile_config(symbol, raw_meta)
    profile_meta: Dict[str, Any] = {}
    sessions = list(VWAP_TPO_SESSIONS)
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
    zone_cfg = ZonesConfig(tick_size=profile_config.get("tick_size"))

    target_tf_key = profile_config.get("target_tf_key", "1m")
    base_candidates = frames.get(target_tf_key, [])
    if not base_candidates:
        base_candidates = minute_candles
    if not base_candidates and frames:
        base_candidates = next(iter(frames.values()))
    base_candles = _deduplicate_sorted(base_candidates)
    frames[target_tf_key] = base_candles

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
        return _insufficient_from_context(context, now_override=now_dt)

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
        raw_window_start = window_end_ms - hours_window * MS_IN_HOUR
        window_start_ms = max(0, _align_to_interval(raw_window_start, MINUTE_INTERVAL_MS))
        if target_interval_ms > MINUTE_INTERVAL_MS:
            window_start_ms = max(0, _align_to_interval(window_start_ms, target_interval_ms))

    minute_index_all = {candle["t"]: candle for candle in minute_candles}
    minute_window_index = {
        ts: candle
        for ts, candle in minute_index_all.items()
        if window_start_ms <= ts <= window_end_ms
    }

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
            if trace_ctx is not None:
                trace_ctx.info(
                    "output.prepare_payload",
                    scope="output",
                    status="minute_missing",
                    missing_fields=0,
                )
            return _finalise_payload(
                minute_payload,
                status="minute_missing",
                pipeline_start=pipeline_start,
                fetch_ms=fetch_ms,
                db_ms=db_ms,
                trace_ctx=trace_ctx,
            )
        if not network_backfill:
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
            raise DataQualityError(data_quality)
        try:
            budget.raise_if_exceeded("download_missing_minutes_start")
            downloaded_minutes = await _call_download_missing_minutes_async(
                symbol,
                window_start_ms,
                window_end_ms,
                time_gaps,
                budget=budget,
                allow_network=network_backfill,
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
            raise DataQualityError(detail) from exc
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
            return _insufficient_from_context(context, now_override=now_dt)
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

    if minute_missing_after > 0:
        data_quality["downloaded"] = fetched_unique
        raise DataQualityError(data_quality)

    frames["1m"] = [minute_index_all[ts] for ts in sorted(minute_index_all)]
    minute_candles = frames["1m"]

    try:
        budget.raise_if_exceeded("zones_preparation")
    except _TimeBudgetExceeded as exc:
        LOGGER.warning(
            "Time budget exceeded before zones preparation",
            extra={"stage": exc.stage, "symbol": symbol},
        )
        return _insufficient_from_context(context, now_override=now_dt)

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
    warmup_bars_base = zone_cfg.atr_period + 10
    min_bars_per_tf = {"15m": 200, "1h": 60, "4h": 24}
    required_bars_per_tf: Dict[str, int] = {}
    history_candidate = zones_window_start_ms
    for tf_key, baseline in min_bars_per_tf.items():
        required = max(baseline, warmup_bars_base)
        required_bars_per_tf[tf_key] = required
        if strict_window:
            continue
        interval_ms_tf = TIMEFRAME_TO_MS.get(tf_key)
        if not interval_ms_tf:
            continue
        candidate = zones_window_start_ms - required * interval_ms_tf
        if candidate < history_candidate:
            history_candidate = candidate
    if strict_window:
        zones_history_start_ms = zones_window_start_ms
    else:
        history_candidate = min(history_candidate, zones_window_start_ms - MS_IN_DAY)
        zones_history_start_ms = max(0, history_candidate)
    zones_history_start_ms = max(0, _align_to_interval(zones_history_start_ms, MINUTE_INTERVAL_MS))

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
    if zone_history_gaps:
        gap_missing = sum(gap["count"] for gap in zone_history_gaps)
        zone_expected_minutes = _build_expected_times(
            zones_history_start_ms,
            window_end_ms,
            MINUTE_INTERVAL_MS,
        )
        zone_expected_count = len(zone_expected_minutes)
        if strict_three_day:
            coverage_pct = (
                ((zone_expected_count - gap_missing) / zone_expected_count) * 100.0
                if zone_expected_count
                else 100.0
            )
            missing = MinuteDataUnavailable(
                symbol=symbol,
                start_ms=zones_history_start_ms,
                end_ms=window_end_ms,
                missing_count=gap_missing,
                expected_count=zone_expected_count,
                coverage_pct=round(coverage_pct, 3),
                gaps=[(int(gap.get("from", zones_history_start_ms)), int(gap.get("to", zones_history_start_ms))) for gap in zone_history_gaps],
            )
            if trace_ctx is not None:
                trace_ctx.warn(
                    "availability.checked",
                    scope="ohlcv.1m.zones",
                    missing_minutes=gap_missing,
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
            return _finalise_payload(
                minute_payload,
                status="minute_missing",
                pipeline_start=pipeline_start,
                fetch_ms=fetch_ms,
                db_ms=db_ms,
                trace_ctx=trace_ctx,
            )
        if not network_backfill:
            detail = {
                "tf": target_tf_key,
                "window": {"start_ms": zones_history_start_ms, "end_ms": window_end_ms},
                "stage": "zones_history",
                "minute_missing_before": gap_missing,
                "fetched_1m_count": 0,
                "time_gaps": zone_history_gaps,
            }
            raise DataQualityError(detail)
        try:
            budget.raise_if_exceeded("zones_history_backfill_start")
            zone_downloaded_minutes = await _call_download_missing_minutes_async(
                symbol,
                zones_history_start_ms,
                window_end_ms,
                zone_history_gaps,
                budget=budget,
                allow_network=network_backfill,
            )
        except BinanceDownloadError as exc:
            detail = {
                "tf": target_tf_key,
                "window": {"start_ms": zones_history_start_ms, "end_ms": window_end_ms},
                "stage": "zones_history",
                "minute_missing_before": gap_missing,
                "fetched_1m_count": exc.downloaded,
                "time_gaps": zone_history_gaps,
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
                    "time_gaps": zone_history_gaps,
                },
            )
            return _insufficient_from_context(context, now_override=now_dt)
        for candle in zone_downloaded_minutes:
            ts = candle["t"]
            if ts < zones_history_start_ms or ts > window_end_ms:
                continue
            minute_index_all[ts] = candle

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
    zone_frames_window = _build_zone_frames(minute_zone_window)

    zone_tf_lengths = {
        tf: len(zone_frames_full.get(tf, [])) for tf in ("15m", "1h", "4h", "1d")
    }
    warmup_bars_per_tf: Dict[str, int] = {}
    for tf in ("15m", "1h", "4h", "1d"):
        total = len(zone_frames_full.get(tf, []))
        window_count = len(zone_frames_window.get(tf, []))
        warmup_bars_per_tf[tf] = max(0, total - window_count)
    zone_availability: Dict[str, Dict[str, int | bool]] = {}
    if strict_window:
        for tf_key, required in required_bars_per_tf.items():
            available = len(zone_frames_window.get(tf_key, []))
            zone_availability[tf_key] = {
                "required": int(required),
                "available": int(available),
                "ok": bool(available >= required),
            }
    zones_diag = {
        "tf_lengths": zone_tf_lengths,
        "atr_period": zone_cfg.atr_period,
        "warmup_bars_per_tf": warmup_bars_per_tf,
        "window_hours": zones_window_hours,
        "tick_size": zone_cfg.tick_size,
    }
    if zone_availability:
        zones_diag["availability"] = zone_availability
    zone_cfg.zones_window_start_ms = zones_window_start_ms
    zone_cfg.window_end_ms_prev_closed = window_end_ms
    zone_cfg.allow_base_fallback = True
    liquidity_equal_levels = build_equal_liquidity_levels(zone_frames_full)

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
        raise DataQualityError(data_quality)

    frames[target_tf_key] = [base_index_all[ts] for ts in sorted(base_index_all)]
    base_candles = frames[target_tf_key]
    if primary_key == target_tf_key:
        primary_candles = base_candles

    selection_payload: Dict[str, Any] = {
        "start": selection_start,
        "end": selection_end,
    }
    htf_section, htf_quality = build_htf_section(symbol, frames, selection_payload)

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

    liquidity_frames: Dict[str, Dict[str, Any]] = {}

    def _clean_series(series: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(series, Sequence):
            return []
        return [c for c in series if isinstance(c, Mapping)]  # type: ignore[list-item]

    minute_full = _clean_series(frames.get("1m"))
    if minute_full:
        liquidity_frames["1m"] = {"candles": minute_full, "source": "minute"}

    htf_candles = htf_section.get("candles") if isinstance(htf_section, Mapping) else None
    if isinstance(htf_candles, Mapping):
        for tf_key in ("15m", "1h", "1d"):
            series = htf_candles.get(tf_key)
            cleaned = _clean_series(series)
            if cleaned:
                liquidity_frames[tf_key] = {"candles": cleaned, "source": "htf"}

    for tf_key in ("15m", "1h"):
        if tf_key in liquidity_frames:
            continue
        if not minute_full:
            continue
        interval_ms = TIMEFRAME_TO_MS.get(tf_key)
        if not interval_ms:
            continue
        aggregated = resample_ohlcv(minute_full, interval_ms)
        if not aggregated:
            continue
        liquidity_frames[tf_key] = {"candles": aggregated, "source": "aggregated"}

    if "1d" not in liquidity_frames:
        daily_series = _clean_series(frames.get("1d"))
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
        "Liquidity tick size resolved for check-all",  # contextual debug entry
        extra={
            "symbol": symbol,
            "normalized_symbol": normalise_symbol_for_tick(symbol) or "UNKNOWN",
            "tick_size": tick_size_numeric,
            "tick_size_source": tick_size_source,
        },
    )

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

    if zone_frames_full:
        try:
            detected_zones = await asyncio.to_thread(
                detect_zones,
                frames=zone_frames_full,
                profile_levels=profile_level_map,
                liquidity_levels=liquidity_equal_levels,
                config=zone_cfg,
            )
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

    if strict_window and zone_availability and isinstance(zones_container, MutableMapping):
        for zone_key, tf_candidates in zone_type_timeframes.items():
            if not tf_candidates or zone_key not in zones_container:
                continue
            if any(not zone_availability.get(tf, {}).get("ok", False) for tf in tf_candidates):
                zones_container[zone_key] = []

    liquidity_payload = await asyncio.to_thread(
        build_liquidity_snapshot,
        liquidity_frames,
        symbol=symbol,
        tick_size=tick_size_numeric,
        meta=raw_meta,
        selection=selection_payload,
        config=liquidity_config if isinstance(liquidity_config, Mapping) else None,
    )

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

    minute_htf_source: List[Mapping[str, Any]] = []
    minute_frame_present = "1m" in frames
    if minute_frame_present:
        minute_htf_source = [
            minute_window_index[ts]
            for ts in sorted(minute_window_index)
            if ts in minute_window_index
        ]

    orderflow_config = _resolve_orderflow_config(raw_meta)
    orderflow_block = _build_orderflow_block(
        minute_htf_source,
        snapshot.get("agg_trades"),
        config=orderflow_config,
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
    smc_blocks_list, smc_stats = await asyncio.to_thread(
        detect_smc_blocks,
        hourly_htf,
        timeframe="1h",
        structure_flags=structure_events,
        ob_zones=ob_candidates,
        liquidity_levels=liquidity_source,
        config=smc_config,
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
    daily_start_ms = _start_of_day_ms(window_end_ms)
    daily_filtered_minutes = _filter_candles(
        minute_series, start_ms=daily_start_ms, end_ms=window_end_ms
    )
    daily_vwap_profile = _build_volume_profile_stats(
        minute_series,
        start_ms=daily_start_ms,
        end_ms=window_end_ms,
        tick_size=tick_size_numeric,
        value_area_pct=VALUE_AREA_PCT,
    )

    composite_day_end_ms = daily_start_ms + MS_IN_DAY - MINUTE_INTERVAL_MS
    if composite_day_end_ms < daily_start_ms:
        composite_day_end_ms = daily_start_ms
    composite_day_profile = _build_volume_profile_stats(
        minute_series,
        start_ms=daily_start_ms,
        end_ms=min(window_end_ms, composite_day_end_ms),
        tick_size=tick_size_numeric,
        value_area_pct=VALUE_AREA_PCT,
    )

    session_profiles: Dict[str, Dict[str, Any]] = {}
    session_sigma_blocks: Dict[str, Dict[str, Any]] = {}
    session_boundaries: Dict[str, Dict[str, Any]] = {}
    for session_name, session_start, session_end in sessions:
        (
            session_start_ms,
            session_end_ms,
            session_close_ms,
        ) = _session_window(window_end_ms, session_start, session_end)
        session_filtered = _filter_candles(
            minute_series, start_ms=session_start_ms, end_ms=session_end_ms
        )
        ib_high, ib_low = _compute_initial_balance_extrema(
            session_filtered, session_start_ms=session_start_ms
        )
        profile_entry = _build_volume_profile_stats(
            minute_series,
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

    vwap_payload = {
        "daily": daily_vwap_profile,
        "sessions": session_profiles,
    }

    vwap_sigma_payload = {
        "daily": _build_vwap_sigma_block(daily_filtered_minutes, basis="daily"),
        "sessions": session_sigma_blocks,
    }

    session_time_lookup = {
        str(name).lower(): (start_time, end_time)
        for name, start_time, end_time in sessions
    }
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
        start_dt = datetime.combine(session_date, start_time, tzinfo=UTC)
        end_dt = datetime.combine(session_date, end_time, tzinfo=UTC)
        if end_time <= start_time:
            end_dt += timedelta(days=1)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) - MINUTE_INTERVAL_MS
        session_candles = _filter_candles(
            minute_series, start_ms=start_ms, end_ms=end_ms
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
        session_payload = {
            "open_utc": _isoformat_utc(open_ms) if open_ms is not None else None,
            "close_utc": _isoformat_utc(close_ms) if close_ms is not None else None,
            "vwap": profile_entry.get("vwap") if isinstance(profile_entry, Mapping) else None,
            "sd1": _sd_payload(sigma_levels, 1),
            "sd2": _sd_payload(sigma_levels, 2),
            "poc": profile_entry.get("poc") if isinstance(profile_entry, Mapping) else None,
            "vah": profile_entry.get("vah") if isinstance(profile_entry, Mapping) else None,
            "val": profile_entry.get("val") if isinstance(profile_entry, Mapping) else None,
            "ib_high": ib_high,
            "ib_low": ib_low,
            "high": profile_entry.get("session_high") if isinstance(profile_entry, Mapping) else None,
            "low": profile_entry.get("session_low") if isinstance(profile_entry, Mapping) else None,
        }
        if isinstance(profile_entry, Mapping):
            if profile_entry.get("high") is not None:
                session_payload["high"] = profile_entry.get("high")
            if profile_entry.get("low") is not None:
                session_payload["low"] = profile_entry.get("low")
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

    if trace_ctx is not None:
        trace_ctx.info(
            "compute.vwap_tpo",
            scope="vwap_tpo",
            sessions=len(vwap_tpo_sessions),
        )

    prev_day_block = _build_prev_day_block(
        minute_series,
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
        }

    vwap_tpo_public = {
        "daily": vwap_tpo_daily_public,
        "sessions": ordered_sessions,
    }

    ohlcv_public: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for tf in ("1m", "3m", "5m", "15m", "1h", "4h", "1d"):
        tf_payload = ohlcv_block.get(tf) if isinstance(ohlcv_block, Mapping) else None
        candles: List[Dict[str, Any]] = []
        if isinstance(tf_payload, Mapping):
            raw_candles = tf_payload.get("candles")
            if isinstance(raw_candles, Sequence):
                candles = [dict(candle) for candle in raw_candles if isinstance(candle, Mapping)]
        ohlcv_public[tf] = {"candles": candles}

    orderflow_public: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for tf in ("15m", "1h"):
        tf_payload = orderflow_block.get(tf) if isinstance(orderflow_block, Mapping) else None
        per_bar: List[Dict[str, Any]] = []
        if isinstance(tf_payload, Mapping):
            raw_series = tf_payload.get("per_bar")
            if isinstance(raw_series, Sequence):
                per_bar = [dict(entry) for entry in raw_series if isinstance(entry, Mapping)]
        orderflow_public[tf] = {"per_bar": per_bar}

    zones_container = detected_zones.get("zones") if isinstance(detected_zones, Mapping) else None
    zone_keys = ("fvg", "fvl", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels")
    zones_public: Dict[str, List[Dict[str, Any]]] = {key: [] for key in zone_keys}
    if isinstance(zones_container, Mapping):
        for key in zone_keys:
            raw_zone = zones_container.get(key)
            if isinstance(raw_zone, Sequence):
                zones_public[key] = [
                    dict(item) for item in raw_zone if isinstance(item, Mapping)
                ]

    if strict_window and zone_availability:
        for zone_key, tf_candidates in zone_type_timeframes.items():
            if zones_public.get(zone_key):
                continue
            if not tf_candidates:
                continue
            insufficient: List[str] = []
            tracked = 0
            for tf_name in tf_candidates:
                tracked += 1
                info = zone_availability.get(tf_name)
                required = required_bars_per_tf.get(tf_name, 0)
                available = len(zone_frames_window.get(tf_name, []))
                ok = available >= required if required else False
                if info is not None:
                    required = int(info.get("required", required))
                    available = int(info.get("available", available))
                    ok = bool(info.get("ok", available >= required if required else False))
                if not required:
                    continue
                if not ok:
                    insufficient.append(f"{tf_name}: {available}/{required}")
            if tracked and insufficient and len(insufficient) == tracked:
                message = (
                    f"?? ?????????????????? ???????????? ???????????? ?????? {zone_key.upper()} ?????? "
                    f"(???????????????? {', '.join(insufficient)})."
                )
                zones_public[zone_key] = [{"message": message, "period": "topup"}]

    if trace_ctx is not None:
        trace_ctx.info(
            "compute.poi.topN",
            scope="zones",
            counts={key: len(value) for key, value in zones_public.items()},
        )

    liquidity_public = {
        "eqh": list(liquidity_equal_levels.get("eqh", [])),
        "eql": list(liquidity_equal_levels.get("eql", [])),
    }

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

    data_payload = {
        "symbol": symbol,
        "ohlcv": ohlcv_public,
        "orderflow": orderflow_public,
        "vwap_tpo": vwap_tpo_public,
        "tpo": {"composite_day": composite_day_public},
        "prev_day": prev_day_block,
        "zones": zones_public,
        "liquidity": liquidity_public,
        "risk_prefs": risk_prefs_public,
        "context": context_public,
    }

    availability: Dict[str, Any] = {
        "ohlcv": {},
        "vwap_sessions": {},
        "zones": {},
        "orderflow": {"timeframes": {}, "metrics": {}},
    }
    missing_fields: Set[str] = set()

    if trace_ctx is not None:
        trace_ctx.info(
            "availability.checked",
            scope="payload",
            blocks=list(availability.keys()),
        )

    for tf in ("1m", "3m", "5m", "15m", "1h", "4h", "1d"):
        candles_payload = ohlcv_public.get(tf, {})
        candles = candles_payload.get("candles") if isinstance(candles_payload, Mapping) else []
        count = len(candles) if isinstance(candles, Sequence) else 0
        availability["ohlcv"][tf] = {"candles": count, "has_data": count > 0}
        if count == 0:
            missing_fields.add(f"ohlcv.{tf}")

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
        availability["vwap_sessions"][session_name] = {
            "present": session_present,
            "metrics": metrics_presence,
        }

    for zone_key, series in zones_public.items():
        count = len(series) if isinstance(series, Sequence) else 0
        availability["zones"][zone_key] = {"count": count}
        if count == 0:
            missing_fields.add(f"zones.{zone_key}")

    orderflow_metrics = {"footprint": False, "delta": False, "cvd": False}
    orderflow_source = snapshot.get("orderflow") if isinstance(snapshot.get("orderflow"), Mapping) else None
    if isinstance(orderflow_source, Mapping):
        footprint_payload = orderflow_source.get("footprint")
        if isinstance(footprint_payload, Sequence) and footprint_payload:
            orderflow_metrics["footprint"] = True

    for tf, payload in orderflow_public.items():
        per_bar = payload.get("per_bar") if isinstance(payload, Mapping) else []
        series = per_bar if isinstance(per_bar, Sequence) else []
        series_list = [entry for entry in series if isinstance(entry, Mapping)]
        availability["orderflow"]["timeframes"][tf] = {
            "bars": len(series_list),
            "has_data": len(series_list) > 0,
        }
        if not series_list:
            missing_fields.add(f"orderflow.{tf}")
        if series_list:
            if any(entry.get("delta") is not None for entry in series_list):
                orderflow_metrics["delta"] = True
            if any(entry.get("cvd") is not None for entry in series_list):
                orderflow_metrics["cvd"] = True

    for metric, present in orderflow_metrics.items():
        if not present:
            missing_fields.add(f"orderflow.{metric}")
    availability["orderflow"]["metrics"] = orderflow_metrics

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
    if snapshot_age_sec is not None:
        meta_block["snapshot_age_sec"] = snapshot_age_sec
    if insufficient_reason:
        meta_block["insufficient_reason"] = insufficient_reason
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

    return _finalise_payload(
        final_payload,
        status=status,
        pipeline_start=pipeline_start,
        fetch_ms=fetch_ms,
        db_ms=db_ms,
        trace_ctx=trace_ctx,
    )

