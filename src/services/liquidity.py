"""Liquidity level and sweep detection helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import logging
import math
import re
import statistics
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from ..meta import HARDCODED_TICK_SIZES
from .ohlc import TIMEFRAME_TO_MS, resample_ohlcv

MS_IN_DAY = 86_400_000
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("15m", "1h")
_PRICE_FIELDS: tuple[str, ...] = ("o", "h", "l", "c")
_SYMBOL_SUFFIXES: tuple[str, ...] = ("PERP",)
SWEEP_CAP_PCT = 0.004

LOGGER = logging.getLogger(__name__)
LIQUIDITY_COUNTERS: Counter[str] = Counter()
MIN_FALLBACK_TICK = 1e-6


@dataclass(slots=True)
class LiquidityConfig:
    """Runtime configuration for liquidity detection."""

    swing_window: int = 3
    lookback_swings: int = 30
    r_ticks: int = 5
    atr_period: int = 14
    sweep_atr_multiplier: float = 0.3
    tolerance_mode: str = "ticks"
    tolerance_percent: float = 0.0
    sweep_min_move_atr: float = 0.3
    sweep_epsilon_pct: float = 0.05
    sweep_lookback_bars: int = 200
    sweep_confirm_window: int = 10
    tolerance_eqh_ticks: float | None = None
    tolerance_eql_ticks: float | None = None
    min_points_dynamic: bool = True
    min_points_alpha: float = 2.0
    min_points_floor: int = 3
    merge_clusters: bool = True
    merge_ticks: float = 3.0
    merge_overlap_ratio: float = 0.6
    enable_resample_when_sparse: bool = True
    degraded_window_floor: int = 1
    cluster_price_window_pct: float = 0.0002
    cluster_time_window_bars: int = 10
    min_distance_bars: int = 3
    feature_strict_legacy_mode: bool = False
    feature_relaxed_clustering: bool = True
    feature_extended_resample: bool = True
<<<<<<< Updated upstream
=======
    session_atr_value: float | None = None
    session_atr_pct: float | None = None
    session_label: str | None = None
>>>>>>> Stashed changes


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def _quantise(price: float, tick_size: float | None) -> float:
    if tick_size is None or tick_size <= 0:
        return float(price)
    ticks = round(price / tick_size)
    return ticks * tick_size


def _count_pairs_within_tolerance(
    swings: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    tick_size: float | None,
    tolerance_mode: str = "ticks",
    tolerance_percent: float = 0.0,
) -> int:
    if tolerance <= 0 or not swings:
        return 0
    quantised: List[float] = []
    for swing in swings:
        price_value = _coerce_float(swing.get("price"))
        if price_value is None:
            continue
        quantised.append(_quantise(price_value, tick_size))
    count = 0
    for left in range(len(quantised)):
        base = quantised[left]
        for right in range(left + 1, len(quantised)):
            threshold = tolerance
            if tolerance_mode == "percent" and tolerance_percent > 0:
                dynamic = max(abs(base) * tolerance_percent, tick_size or 0.0)
                threshold = max(threshold, dynamic)
            if abs(base - quantised[right]) <= threshold + 1e-9:
                count += 1
    return count


def normalise_symbol_for_tick(symbol: str | None) -> str:
    """Normalise symbol identifiers before tick-size lookup."""

    if not symbol:
        return ""

    text = re.sub(r"\s+", "", str(symbol).upper())
    if not text:
        return ""

    split_candidates = [part for part in re.split(r"[:/@ ]", text) if part]
    if not split_candidates:
        split_candidates = [text]
    else:
        split_candidates.append(text)

    preferred: List[str] = []
    fallback: List[str] = []

    for candidate in split_candidates:
        cleaned = re.sub(r"[^A-Z0-9]", "", candidate)
        if not cleaned:
            continue
        for suffix in _SYMBOL_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        if cleaned.endswith("USDTPERP"):
            cleaned = cleaned[: -len("USDTPERP")] + "USDT"
        if cleaned.endswith("USD") or "USDT" in cleaned or "USDC" in cleaned:
            preferred.append(cleaned)
        else:
            fallback.append(cleaned)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return ""


def _infer_tick_size_from_frames(
    frames: Mapping[str, Sequence[Mapping[str, Any]]]
) -> float | None:
    """Infer a plausible tick size from OHLCV data when metadata is missing."""

    samples: List[Decimal] = []
    seen: set[Decimal] = set()
    for candles in frames.values():
        if not isinstance(candles, Sequence):
            continue
        for candle in candles:
            if not isinstance(candle, Mapping):
                continue
            for field in _PRICE_FIELDS:
                price = _coerce_float(candle.get(field))
                if price is None:
                    continue
                try:
                    decimal_value = Decimal(str(price))
                except (InvalidOperation, ValueError):
                    continue
                if decimal_value in seen:
                    continue
                seen.add(decimal_value)
                samples.append(decimal_value)
            if len(samples) >= 5_000:
                break
        if len(samples) >= 5_000:
            break

    if len(samples) < 2:
        return None

    samples.sort()
    min_step: Decimal | None = None
    previous = samples[0]
    for current in samples[1:]:
        diff = current - previous
        if diff > 0:
            if min_step is None or diff < min_step:
                min_step = diff
                if min_step == 0:
                    min_step = None
        previous = current

    if min_step is not None and min_step > 0:
        return float(min_step)

    max_precision = 0
    for value in samples:
        exponent = -value.as_tuple().exponent
        if exponent > max_precision:
            max_precision = exponent

    if max_precision > 0:
        return float(10 ** (-max_precision))
    return None


def _extract_tick_size_from_meta(
    meta: Mapping[str, Any] | None,
    normalized_symbol: str,
) -> float | None:
    """Search snapshot metadata for a matching tick size."""

    if not normalized_symbol or not isinstance(meta, Mapping):
        return None

    visited: set[int] = set()

    def _candidate_from_entry(
        entry: Mapping[str, Any],
        *,
        symbol_hint: str | None,
    ) -> float | None:
        candidate_symbol = entry.get("symbol") or entry.get("pair") or entry.get("instrument")
        if isinstance(candidate_symbol, str):
            candidate_norm = normalise_symbol_for_tick(candidate_symbol)
        else:
            candidate_norm = None
        if not candidate_norm:
            candidate_norm = symbol_hint

        tick_fields = (
            entry.get("tickSize"),
            entry.get("tick_size"),
            entry.get("ticksize"),
        )
        for value in tick_fields:
            numeric = _coerce_float(value)
            if numeric and numeric > 0 and candidate_norm == normalized_symbol:
                return float(numeric)

        filters = entry.get("filters")
        if isinstance(filters, Sequence):
            for filt in filters:
                if not isinstance(filt, Mapping):
                    continue
                filter_type = filt.get("filterType") or filt.get("filter_type")
                if isinstance(filter_type, str) and filter_type.upper() not in {
                    "PRICE_FILTER",
                    "LOT_SIZE",
                }:
                    continue
                tick_value = _coerce_float(
                    filt.get("tickSize")
                    or filt.get("tick_size")
                    or filt.get("ticksize")
                )
                if tick_value and tick_value > 0 and candidate_norm == normalized_symbol:
                    return float(tick_value)
        return None

    def _visit(node: Any, symbol_hint: str | None = None) -> float | None:
        if isinstance(node, Mapping):
            node_id = id(node)
            if node_id in visited:
                return None
            visited.add(node_id)
            direct = _candidate_from_entry(node, symbol_hint=symbol_hint)
            if direct is not None:
                return direct

            for key, value in node.items():
                next_hint = symbol_hint
                if isinstance(key, str):
                    key_norm = normalise_symbol_for_tick(key)
                    if key_norm:
                        next_hint = key_norm
                result = _visit(value, symbol_hint=next_hint)
                if result is not None:
                    return result
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                result = _visit(item, symbol_hint=symbol_hint)
                if result is not None:
                    return result
        return None

    for key in ("exchange_info", "exchangeInfo", "exchange", "symbol_info", "symbolInfo"):
        candidate = meta.get(key)
        value = _visit(candidate)
        if value is not None:
            return value

    return _visit(meta)


def resolve_liquidity_tick_size(
    symbol: str,
    profile_tick_size: Any,
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    meta: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[float, str]:
    """Resolve a positive tick size for liquidity detection with fallbacks."""

    log = logger or LOGGER
    normalized_symbol = normalise_symbol_for_tick(symbol)
    symbol_label = symbol or "UNKNOWN"
    base_extra = {
        "symbol": symbol_label,
        "normalized_symbol": normalized_symbol or "UNKNOWN",
    }

    profile_numeric = _coerce_float(profile_tick_size)
    if profile_numeric is not None and profile_numeric <= 0:
        profile_numeric = None

    if profile_numeric is not None:
        base_extra["profile_tick_size"] = profile_numeric

    exchange_tick = _extract_tick_size_from_meta(meta, normalized_symbol)
    hardcoded_tick = HARDCODED_TICK_SIZES.get(normalized_symbol)
    inferred_tick = _infer_tick_size_from_frames(frames)

    tick_size: float | None = None
    tick_source = "unknown"

    if profile_numeric is not None and profile_numeric > 0:
        tick_size = float(profile_numeric)
        tick_source = "param"
        if exchange_tick and not math.isclose(float(exchange_tick), tick_size, rel_tol=1e-9, abs_tol=1e-12):
            log.debug(
                "Exchange tick size differs from explicit parameter",
                extra={
                    **base_extra,
                    "tick_size_source": tick_source,
                    "tick_size": tick_size,
                    "exchange_tick_size": float(exchange_tick),
                },
            )
    else:
        if isinstance(hardcoded_tick, (int, float)) and hardcoded_tick > 0:
            tick_size = float(hardcoded_tick)
            tick_source = "hardcoded"
            if exchange_tick and not math.isclose(float(exchange_tick), tick_size, rel_tol=1e-9, abs_tol=1e-12):
                log.debug(
                    "Exchange tick size differs from curated fallback",
                    extra={
                        **base_extra,
                        "tick_size_source": tick_source,
                        "tick_size": tick_size,
                        "exchange_tick_size": float(exchange_tick),
                    },
                )
        elif exchange_tick is not None and exchange_tick > 0:
            tick_size = float(exchange_tick)
            tick_source = "exchange"
        elif inferred_tick is not None and inferred_tick > 0:
            tick_size = float(inferred_tick)
            tick_source = "inferred"

    if tick_size is None or tick_size <= 0:
        fallback_tick = inferred_tick if inferred_tick and inferred_tick > 0 else None
        if fallback_tick is None or fallback_tick <= 0:
            fallback_tick = MIN_FALLBACK_TICK
        tick_size = float(fallback_tick)
        tick_source = "fallback_min"
        LIQUIDITY_COUNTERS["tick_size_unresolved"] += 1
        log.warning(
            "Unable to resolve positive liquidity tick size, using fallback",
            extra={**base_extra, "tick_size_source": tick_source, "tick_size": tick_size},
        )

    if (
        profile_numeric is not None
        and tick_source != "param"
        and not math.isclose(profile_numeric, tick_size, rel_tol=1e-12, abs_tol=1e-12)
    ):
        log.warning(
            "Profile tick size overridden by authoritative source",
            extra={
                **base_extra,
                "tick_size_source": tick_source,
                "tick_size": tick_size,
                "profile_tick_size": profile_numeric,
            },
        )
    elif tick_source == "param":
        base_extra["tick_size_source"] = tick_source
        base_extra["tick_size"] = tick_size
        log.debug("Using explicit liquidity tick size", extra=base_extra)
    else:
        log.debug(
            "Resolved liquidity tick size",
            extra={**base_extra, "tick_size_source": tick_source, "tick_size": tick_size},
        )

    return tick_size, tick_source


def _sample_swing_pairs(
    swings: Sequence[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return up to ``limit`` swing pairs sorted by ascending price delta."""

    samples: List[Dict[str, Any]] = []
    for left in range(len(swings)):
        left_price = _coerce_float(swings[left].get("price"))
        left_time = swings[left].get("t")
        if left_price is None or not isinstance(left_time, (int, float)):
            continue
        for right in range(left + 1, len(swings)):
            right_price = _coerce_float(swings[right].get("price"))
            right_time = swings[right].get("t")
            if right_price is None or not isinstance(right_time, (int, float)):
                continue
            delta = abs(left_price - right_price)
            samples.append(
                {
                    "delta": delta,
                    "left": {"t": int(left_time), "price": left_price},
                    "right": {"t": int(right_time), "price": right_price},
                }
            )

    samples.sort(key=lambda entry: entry["delta"])
    if limit <= 0:
        return samples
    return samples[:limit]


def _append_reason(
    sink: List[Dict[str, Any]] | None,
    reason: str,
    **context: Any,
) -> None:
    """Record a structured diagnostic reason if a sink is provided."""

    if sink is None:
        return
    entry: Dict[str, Any] = {"reason": reason}
    for key, value in context.items():
        if value is None:
            continue
        entry[key] = value
    sink.append(entry)


def _resolve_config(raw: Mapping[str, Any] | None) -> LiquidityConfig:
    config = LiquidityConfig()

    if not isinstance(raw, Mapping):
        return config

    def _positive_int(value: Any, default: int, *, lower: int = 1, upper: int | None = None) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        if numeric < lower:
            return default
        if upper is not None and numeric > upper:
            return upper
        return numeric

    def _positive_float(value: Any, default: float, *, lower: float = 0.0, upper: float | None = None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if numeric < lower:
            return default
        if upper is not None and numeric > upper:
            return upper
        return numeric

    config.swing_window = _positive_int(raw.get("swing_window"), config.swing_window, lower=1, upper=10)
    config.lookback_swings = _positive_int(raw.get("lookback"), config.lookback_swings, lower=5, upper=200)
    config.r_ticks = _positive_int(raw.get("r_ticks"), config.r_ticks, lower=1, upper=20)
    config.atr_period = _positive_int(raw.get("atr_period"), config.atr_period, lower=1, upper=200)

    multiplier = raw.get("sweep_atr_multiplier")
    try:
        numeric = float(multiplier)
    except (TypeError, ValueError):
        numeric = config.sweep_atr_multiplier
    if math.isfinite(numeric) and numeric >= 0:
        config.sweep_atr_multiplier = numeric

    tolerance_section = raw.get("tolerance")
    if isinstance(tolerance_section, Mapping):
        mode_value = tolerance_section.get("mode") or tolerance_section.get("type")
        if isinstance(mode_value, str):
            config.tolerance_mode = mode_value.lower()
        percent_value = tolerance_section.get("percent") or tolerance_section.get("pct")
        if percent_value is not None:
            config.tolerance_percent = _positive_float(percent_value, config.tolerance_percent, lower=0.0, upper=0.2)
        eqh = tolerance_section.get("eqh")
        eql = tolerance_section.get("eql")
        if eqh is not None:
            config.tolerance_eqh_ticks = _positive_float(
                eqh, config.tolerance_eqh_ticks or float(config.r_ticks), lower=0.0, upper=100.0
            )
        if eql is not None:
            config.tolerance_eql_ticks = _positive_float(
                eql, config.tolerance_eql_ticks or float(config.r_ticks), lower=0.0, upper=100.0
            )
    else:
        if "tolerance_eqh_ticks" in raw:
            config.tolerance_eqh_ticks = _positive_float(
                raw.get("tolerance_eqh_ticks"), float(config.r_ticks), lower=0.0, upper=100.0
            )
        if "tolerance_eql_ticks" in raw:
            config.tolerance_eql_ticks = _positive_float(
                raw.get("tolerance_eql_ticks"), float(config.r_ticks), lower=0.0, upper=100.0
            )
        if "tolerance_mode" in raw:
            mode_value = raw.get("tolerance_mode")
            if isinstance(mode_value, str):
                config.tolerance_mode = mode_value.lower()
        if "tolerance_percent" in raw:
            config.tolerance_percent = _positive_float(
                raw.get("tolerance_percent"), config.tolerance_percent, lower=0.0, upper=0.2
            )

    min_points_section = raw.get("min_points")
    if isinstance(min_points_section, Mapping):
        if "dynamic" in min_points_section:
            config.min_points_dynamic = bool(min_points_section.get("dynamic"))
        coefficient = (
            min_points_section.get("coefficient")
            or min_points_section.get("alpha")
            or min_points_section.get("coef")
        )
        if coefficient is not None:
            config.min_points_alpha = _positive_float(
                coefficient, config.min_points_alpha, lower=0.1, upper=10.0
            )
        floor_value = min_points_section.get("min") or min_points_section.get("floor")
        if floor_value is not None:
            config.min_points_floor = _positive_int(floor_value, config.min_points_floor, lower=2, upper=10)
        merge_ticks_value = min_points_section.get("merge_ticks")
        if merge_ticks_value is not None:
            config.merge_ticks = _positive_float(
                merge_ticks_value, config.merge_ticks, lower=0.0, upper=50.0
            )
        overlap_value = (
            min_points_section.get("merge_overlap_ratio")
            or min_points_section.get("merge_overlap")
            or min_points_section.get("merge_share")
        )
        if overlap_value is not None:
            config.merge_overlap_ratio = _positive_float(
                overlap_value, config.merge_overlap_ratio, lower=0.0, upper=1.0
            )
        if "merge_clusters" in min_points_section:
            config.merge_clusters = bool(min_points_section.get("merge_clusters"))
    else:
        if "min_points_dynamic" in raw:
            config.min_points_dynamic = bool(raw.get("min_points_dynamic"))
        if "min_points_coefficient" in raw:
            config.min_points_alpha = _positive_float(
                raw.get("min_points_coefficient"), config.min_points_alpha, lower=0.1, upper=10.0
            )
        if "min_points_floor" in raw:
            config.min_points_floor = _positive_int(
                raw.get("min_points_floor"), config.min_points_floor, lower=2, upper=10
            )
        if "merge_ticks" in raw:
            config.merge_ticks = _positive_float(raw.get("merge_ticks"), config.merge_ticks, lower=0.0, upper=50.0)
        if "merge_overlap_ratio" in raw:
            config.merge_overlap_ratio = _positive_float(
                raw.get("merge_overlap_ratio"), config.merge_overlap_ratio, lower=0.0, upper=1.0
            )
        if "merge_clusters" in raw:
            config.merge_clusters = bool(raw.get("merge_clusters"))

    if "enable_resample_when_sparse" in raw:
        config.enable_resample_when_sparse = bool(raw.get("enable_resample_when_sparse"))

    if "degraded_window_floor" in raw:
        config.degraded_window_floor = _positive_int(
            raw.get("degraded_window_floor"), config.degraded_window_floor, lower=1, upper=5
        )

    cluster_section = raw.get("cluster")
    if isinstance(cluster_section, Mapping):
        price_pct = cluster_section.get("price_window_pct") or cluster_section.get("price_pct")
        if price_pct is not None:
            config.cluster_price_window_pct = _positive_float(
                price_pct, config.cluster_price_window_pct, lower=0.0, upper=0.01
            )
        time_bars = cluster_section.get("time_window_bars") or cluster_section.get("bars")
        if time_bars is not None:
            config.cluster_time_window_bars = _positive_int(
                time_bars, config.cluster_time_window_bars, lower=1, upper=50
            )
    if "cluster_price_window_pct" in raw:
        config.cluster_price_window_pct = _positive_float(
            raw.get("cluster_price_window_pct"), config.cluster_price_window_pct, lower=0.0, upper=0.01
        )
    if "cluster_time_window_bars" in raw:
        config.cluster_time_window_bars = _positive_int(
            raw.get("cluster_time_window_bars"), config.cluster_time_window_bars, lower=1, upper=50
        )
    if "min_distance_bars" in raw:
        config.min_distance_bars = _positive_int(
            raw.get("min_distance_bars"), config.min_distance_bars, lower=1, upper=50
        )

    sweep_section = raw.get("sweep") or raw.get("sweeps")
    if isinstance(sweep_section, Mapping):
        if "atr_min_move" in sweep_section or "min_move_atr" in sweep_section:
            config.sweep_min_move_atr = _positive_float(
                sweep_section.get("atr_min_move") or sweep_section.get("min_move_atr"),
                config.sweep_min_move_atr,
                lower=0.0,
                upper=5.0,
            )
        if "epsilon_pct" in sweep_section:
            config.sweep_epsilon_pct = _positive_float(
                sweep_section.get("epsilon_pct"), config.sweep_epsilon_pct, lower=0.0, upper=0.5
            )
        if "lookback_bars" in sweep_section:
            config.sweep_lookback_bars = _positive_int(
                sweep_section.get("lookback_bars"), config.sweep_lookback_bars, lower=10, upper=1000
            )
        if "confirm_window" in sweep_section:
            config.sweep_confirm_window = _positive_int(
                sweep_section.get("confirm_window"), config.sweep_confirm_window, lower=1, upper=100
            )
    if "sweep_min_move_atr" in raw:
        config.sweep_min_move_atr = _positive_float(
            raw.get("sweep_min_move_atr"), config.sweep_min_move_atr, lower=0.0, upper=5.0
        )
    if "sweep_epsilon_pct" in raw:
        config.sweep_epsilon_pct = _positive_float(
            raw.get("sweep_epsilon_pct"), config.sweep_epsilon_pct, lower=0.0, upper=0.5
        )
    if "sweep_lookback_bars" in raw:
        config.sweep_lookback_bars = _positive_int(
            raw.get("sweep_lookback_bars"), config.sweep_lookback_bars, lower=10, upper=1000
        )
    if "sweep_confirm_window" in raw:
        config.sweep_confirm_window = _positive_int(
            raw.get("sweep_confirm_window"), config.sweep_confirm_window, lower=1, upper=100
        )

    feature_section = raw.get("feature") or raw.get("features")
    if isinstance(feature_section, Mapping):
        if "strict_legacy_mode" in feature_section:
            config.feature_strict_legacy_mode = bool(feature_section.get("strict_legacy_mode"))
        if "eql_relaxed_clustering" in feature_section:
            config.feature_relaxed_clustering = bool(feature_section.get("eql_relaxed_clustering"))
        if "liquidity_relaxed_clustering" in feature_section:
            config.feature_relaxed_clustering = bool(feature_section.get("liquidity_relaxed_clustering"))
        if "extended_resample" in feature_section:
            config.feature_extended_resample = bool(feature_section.get("extended_resample"))

    if config.feature_strict_legacy_mode:
        config.feature_relaxed_clustering = False
        config.feature_extended_resample = False
        config.min_points_dynamic = False
        config.merge_clusters = False
        config.enable_resample_when_sparse = False
        config.min_points_floor = max(2, config.min_points_floor)
        config.min_points_alpha = max(2.0, config.min_points_alpha)
        config.tolerance_eqh_ticks = None
        config.tolerance_eql_ticks = None
        config.tolerance_mode = "ticks"
        config.tolerance_percent = 0.0
        config.cluster_price_window_pct = 0.0
        config.cluster_time_window_bars = 1
        config.min_distance_bars = 1
        config.sweep_min_move_atr = config.sweep_atr_multiplier
        config.sweep_epsilon_pct = 0.0
        config.sweep_lookback_bars = max(10, config.sweep_lookback_bars)
        config.sweep_confirm_window = max(1, config.sweep_confirm_window)

    if not config.feature_relaxed_clustering:
        config.merge_clusters = False
        config.min_points_dynamic = False
        config.min_points_floor = 2

    if not config.feature_extended_resample:
        config.enable_resample_when_sparse = False

<<<<<<< Updated upstream
=======
    if "session_atr_value" in raw:
        try:
            session_atr_value = float(raw.get("session_atr_value"))
        except (TypeError, ValueError):
            session_atr_value = None
        if session_atr_value is not None and math.isfinite(session_atr_value) and session_atr_value >= 0:
            config.session_atr_value = session_atr_value

    if "session_atr_pct" in raw:
        try:
            session_atr_pct = float(raw.get("session_atr_pct"))
        except (TypeError, ValueError):
            session_atr_pct = None
        if session_atr_pct is not None and math.isfinite(session_atr_pct) and session_atr_pct >= 0:
            config.session_atr_pct = session_atr_pct

    if "session_label" in raw:
        session_label = raw.get("session_label")
        if isinstance(session_label, str) and session_label:
            config.session_label = session_label

>>>>>>> Stashed changes
    return config


def _frame_source(frame: Mapping[str, Any] | None) -> str | None:
    if not isinstance(frame, Mapping):
        return None
    source = frame.get("source")
    if isinstance(source, str):
        return source
    return None


def _extract_candles(frame: Mapping[str, Any] | None) -> List[Mapping[str, Any]]:
    if not isinstance(frame, Mapping):
        return []
    candles = frame.get("candles")
    if isinstance(candles, Sequence):
        return [c for c in candles if isinstance(c, Mapping)]  # type: ignore[list-item]
    return []


def _augment_supported_frames(
    frames: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Mapping[str, Any]]:
    augmented: Dict[str, Mapping[str, Any]] = {
        key: dict(value) if isinstance(value, Mapping) else {"candles": []}
        for key, value in frames.items()
    }

    for timeframe in frames:
        if timeframe not in SUPPORTED_TIMEFRAMES and timeframe != "1m":
            LOGGER.debug(
                "Skipping unsupported timeframe for liquidity",
                extra={"tf": timeframe, "reason": "tf_skipped"},
            )

    minute_candles = _extract_candles(augmented.get("1m"))
    LOGGER.debug(
        "Liquidity minute seed stats",
        extra={
            "tf": "1m",
            "candles": len(minute_candles),
            "used_source": _frame_source(augmented.get("1m")) or "unknown",
        },
    )

    if not minute_candles:
        return augmented

    for target_tf in SUPPORTED_TIMEFRAMES:
        frame_payload = augmented.get(target_tf)
        existing = _extract_candles(frame_payload)
        if existing:
            LOGGER.debug(
                "Liquidity timeframe already present",
                extra={
                    "tf": target_tf,
                    "candles": len(existing),
                    "used_source": _frame_source(frame_payload) or "unknown",
                },
            )
            continue
        interval_ms = TIMEFRAME_TO_MS.get(target_tf)
        if not interval_ms:
            continue
        aggregated = resample_ohlcv(minute_candles, interval_ms)
        LOGGER.debug(
            "Liquidity generated higher timeframe",
            extra={
                "tf": target_tf,
                "candles": len(aggregated),
                "interval_ms": interval_ms,
                "used_source": "aggregated",
            },
        )
        augmented[target_tf] = {"candles": aggregated, "source": "aggregated"}

    return augmented


def _detect_swings(
    candles: Sequence[Mapping[str, Any]],
    *,
    window: int,
    kind: str,
    tick_size: float,
) -> List[MutableMapping[str, Any]]:
    swings: List[MutableMapping[str, Any]] = []
    if window <= 0:
        return swings
    size = len(candles)
    for index in range(window, size - window):
        candle = candles[index]
        ts = candle.get("t")
        if not isinstance(ts, (int, float)):
            continue
        if kind == "high":
            price = _coerce_float(candle.get("h"))
            if price is None:
                continue
            price = _quantise(price, tick_size)
            higher = True
            for offset in range(1, window + 1):
                left = _coerce_float(candles[index - offset].get("h"))
                right = _coerce_float(candles[index + offset].get("h"))
                if left is None or right is None:
                    higher = False
                    break
                left = _quantise(left, tick_size)
                right = _quantise(right, tick_size)
                if price <= left or price <= right:
                    higher = False
                    break
            if higher:
                swings.append({"t": int(ts), "price": price, "idx": index})
        else:
            price = _coerce_float(candle.get("l"))
            if price is None:
                continue
            price = _quantise(price, tick_size)
            lower = True
            for offset in range(1, window + 1):
                left = _coerce_float(candles[index - offset].get("l"))
                right = _coerce_float(candles[index + offset].get("l"))
                if left is None or right is None:
                    lower = False
                    break
                left = _quantise(left, tick_size)
                right = _quantise(right, tick_size)
                if price >= left or price >= right:
                    lower = False
                    break
            if lower:
                swings.append({"t": int(ts), "price": price, "idx": index})
    return swings


def _cluster_swings(
    swings: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    tick_size: float | None,
    level_type: str,
    timeframe: str,
    min_points: int,
    tolerance_mode: str,
    tolerance_value: float,
    cluster_price_pct: float,
    cluster_time_window_bars: int,
    min_distance_bars: int,
    reason_sink: List[Dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], int]:
    if not swings:
        LOGGER.debug(
            "Skipping swing clustering",
            extra={
                "tf": timeframe,
                "level_type": level_type,
                "reason": "no_swings",
            },
        )
        _append_reason(reason_sink, "no_swings")
        return [], [], {}, 0

    ordered = sorted(swings, key=lambda item: (item.get("t", 0), item.get("idx", 0)))
    clusters: List[Dict[str, Any]] = []
    merge_operations = 0
    for swing in ordered:
        price_value = _coerce_float(swing.get("price"))
        time_value = swing.get("t")
        idx_value = swing.get("idx")
        if price_value is None or not isinstance(time_value, (int, float)):
            continue
        try:
            idx = int(idx_value)
        except (TypeError, ValueError):
            idx = len(clusters)
        quantised_price = _quantise(price_value, tick_size)
        matched = False
        for cluster in clusters:
            anchor = cluster["anchor"]
            price_window = max(
                cluster_price_pct * max(abs(anchor), abs(quantised_price)),
                tick_size or 0.0,
            )
            if abs(anchor - quantised_price) > price_window + 1e-12:
                continue
            indices = cluster["indices"]
            if indices:
                if min(abs(idx - existing) for existing in indices) > cluster_time_window_bars:
                    continue
            cluster["prices"].append(quantised_price)
            cluster["swings"].append(int(time_value))
            cluster["indices"].append(idx)
            if not cluster.get("merged"):
                cluster["merged"] = True
                merge_operations += 1
            median_price = statistics.median(cluster["prices"]) if cluster["prices"] else quantised_price
            cluster["anchor"] = _quantise(median_price, tick_size)
            matched = True
            break
        if not matched:
            clusters.append(
                {
                    "anchor": quantised_price,
                    "prices": [quantised_price],
                    "swings": [int(time_value)],
                    "indices": [idx],
                    "merged": False,
                }
            )

    tolerance_mode = (tolerance_mode or "ticks").lower()
    payload: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    reason_hist: Counter[str] = Counter()

    for cluster in clusters:
        indices_sorted = sorted(set(cluster["indices"]))
        filtered_indices: List[int] = []
        for candidate_idx in indices_sorted:
            if not filtered_indices or candidate_idx - filtered_indices[-1] >= min_distance_bars:
                filtered_indices.append(candidate_idx)
        cluster["filtered_indices"] = filtered_indices

    for cluster in clusters:
        swings_ts = sorted(set(cluster["swings"]))
        indices_sorted = cluster.get("filtered_indices", [])
        prices = cluster.get("prices", [])
        candidate_entry: Dict[str, Any] = {
            "type": level_type,
            "tf": timeframe,
            "swings": swings_ts,
            "indices": indices_sorted,
            "tolerance": None,
            "rejected": False,
            "reason": None,
            "details": {},
        }

        if len(indices_sorted) < max(2, min_points):
            _append_reason(
                reason_sink,
                "min_points_fail",
                swings=swings_ts,
                indices=indices_sorted,
                required=min_points,
            )
            candidate_entry["rejected"] = True
            candidate_entry["reason"] = "min_points_fail"
            candidate_entry["details"] = {
                "required": min_points,
                "observed": len(indices_sorted),
            }
            candidates.append(candidate_entry)
            reason_hist["min_points_fail"] += 1
            continue

        if not prices:
            candidate_entry["rejected"] = True
            candidate_entry["reason"] = "no_prices"
            candidate_entry["details"] = {"swings": swings_ts}
            candidates.append(candidate_entry)
            reason_hist["no_prices"] += 1
            continue

        level_price = _quantise(statistics.median(prices), tick_size)
        intra_range = max(prices) - min(prices) if prices else 0.0
        intra_ticks = intra_range / (tick_size or 1.0) if tick_size else 0.0

        tolerance_price = tolerance
        if tolerance_mode == "percent" and tolerance_value > 0:
            tolerance_price = max(level_price * tolerance_value, tick_size or 0.0)

        candidate_entry["price"] = level_price
        candidate_entry["tolerance"] = tolerance_price
        candidate_entry["reason"] = "kept"
        candidate_entry["details"] = {
            "intra_range": intra_range,
            "intra_range_ticks": intra_ticks,
            "merged": bool(cluster.get("merged")),
            "cluster_members": len(prices),
            "cluster_window_pct": cluster_price_pct,
        }
        candidates.append(candidate_entry)
        reason_hist["kept"] += 1
        payload.append(
            {
                "type": level_type,
                "tf": timeframe,
                "price": level_price,
                "swings": swings_ts,
                "indices": indices_sorted,
                "tolerance": tolerance_price,
                "cluster": {
                    "members": len(prices),
                    "intra_range": intra_range,
                    "intra_range_ticks": intra_ticks,
                    "merged": bool(cluster.get("merged")),
                },
            }
        )

    if not payload:
        _append_reason(
            reason_sink,
            "no_clusters",
            swings=len(swings),
            tolerance=tolerance,
            tick_size=tick_size,
        )
        if not reason_hist:
            reason_hist["no_clusters"] += 1

    return payload, candidates, dict(reason_hist), merge_operations


def _compute_atr_series(
    candles: Sequence[Mapping[str, Any]],
    *,
    period: int,
) -> List[float]:
    """Compute Wilder's ATR sequence for the supplied timeframe."""

    atr_values: List[float] = []
    if period <= 0:
        return [0.0 for _ in candles]

    prev_close: float | None = None
    true_ranges: List[float] = []

    for index, candle in enumerate(candles):
        high = _coerce_float(candle.get("h"))
        low = _coerce_float(candle.get("l"))
        close = _coerce_float(candle.get("c"))

        if high is None or low is None or close is None:
            atr_values.append(0.0)
            prev_close = close
            continue

        tr_candidates = [high - low]
        if prev_close is not None:
            tr_candidates.extend((abs(high - prev_close), abs(low - prev_close)))
        true_range = max(tr_candidates) if tr_candidates else 0.0
        true_ranges.append(true_range)

        if index == 0:
            atr = true_range
        elif len(true_ranges) < period:
            atr = sum(true_ranges) / len(true_ranges)
        elif len(true_ranges) == period:
            atr = sum(true_ranges[-period:]) / period
        else:
            prev_atr = atr_values[-1]
            atr = ((prev_atr * (period - 1)) + true_range) / period

        atr_values.append(atr)
        prev_close = close

    return atr_values


def _prepare_levels(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    tick_size: float | None,
    config: LiquidityConfig,
) -> tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, Any]]:
    eqh_levels: List[Dict[str, Any]] = []
    eql_levels: List[Dict[str, Any]] = []

    diagnostics: Dict[str, Any] = {}
    summary_raw: Dict[str, int] = {"eqh": 0, "eql": 0}
    summary_filtered: Dict[str, int] = {"eqh": 0, "eql": 0}
    summary_reasons: Dict[str, Counter[str]] = {"eqh": Counter(), "eql": Counter()}
    degraded_timeframes: List[str] = []

    tick_value = float(tick_size) if tick_size and tick_size > 0 else 0.0
    tolerance_mode = (config.tolerance_mode or "ticks").lower()
    tolerance_percent_value = config.tolerance_percent if config.tolerance_percent > 0 else 0.0
    if tolerance_mode == "percent" and tolerance_percent_value <= 0.0:
        tolerance_percent_value = 0.0004

    eqh_ticks = float(config.tolerance_eqh_ticks) if config.tolerance_eqh_ticks else float(config.r_ticks or 1)
    if eqh_ticks <= 0:
        eqh_ticks = float(config.r_ticks or 1)
    if config.tolerance_eql_ticks:
        eql_ticks = float(config.tolerance_eql_ticks)
    elif config.feature_relaxed_clustering:
        eql_ticks = max(1.0, eqh_ticks - 1.0)
    else:
        eql_ticks = eqh_ticks

    def _ticks_to_tolerance(ticks_value: float) -> float:
        if tick_value <= 0.0 or ticks_value <= 0.0:
            return tick_value
        return max(tick_value, ticks_value * tick_value)

    eqh_tolerance = _ticks_to_tolerance(eqh_ticks)
    eql_tolerance = _ticks_to_tolerance(eql_ticks)
    cluster_price_pct = max(0.0, config.cluster_price_window_pct)
    cluster_time_window_bars = max(1, int(config.cluster_time_window_bars))
    min_distance_bars = max(1, int(config.min_distance_bars))
    minute_seed = _extract_candles(frames.get("1m"))

    LOGGER.debug(
        "Liquidity swing detection config",
        extra={
            "tick_size": tick_size,
            "r_ticks": config.r_ticks,
            "tolerance_eqh": eqh_tolerance,
            "tolerance_eql": eql_tolerance,
            "tolerance_mode": tolerance_mode,
            "tolerance_percent": tolerance_percent_value,
            "swing_window": config.swing_window,
            "lookback": config.lookback_swings,
            "min_points_dynamic": config.min_points_dynamic,
            "cluster_price_pct": cluster_price_pct,
            "cluster_time_window_bars": cluster_time_window_bars,
            "min_distance_bars": min_distance_bars,
        },
    )

    base_minimum_required = 2 * config.swing_window + 1

    for timeframe in SUPPORTED_TIMEFRAMES:
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        candle_count = len(candles)

        frame_diag: Dict[str, Any] = {
            "used_source": source_label,
            "n_bars_total": candle_count,
            "tick_size": tick_size,
            "swing_window_target": config.swing_window,
            "effective_window": config.swing_window,
            "lookback": config.lookback_swings,
            "atr_period": config.atr_period,
            "atr_mult": config.sweep_atr_multiplier,
            "min_points_dynamic": config.min_points_dynamic,
            "degraded_mode": False,
            "degraded_reasons": [],
            "reasons": [],
            "eqh": {
                "swing_count": 0,
                "cluster_count": 0,
                "pairs_within_tol": 0,
                "pairs_within_tol_before_cluster": 0,
                "sample_pairs_top10": [],
                "reasons": [],
                "raw_count": 0,
                "filtered_count": 0,
                "reason_histogram": {},
                "merge_operations": 0,
                "min_points_required": 0,
                "tolerance": eqh_tolerance,
            },
            "eql": {
                "swing_count": 0,
                "cluster_count": 0,
                "pairs_within_tol": 0,
                "pairs_within_tol_before_cluster": 0,
                "sample_pairs_top10": [],
                "reasons": [],
                "raw_count": 0,
                "filtered_count": 0,
                "reason_histogram": {},
                "merge_operations": 0,
                "min_points_required": 0,
                "tolerance": eql_tolerance,
            },
        }
        diagnostics[timeframe] = frame_diag

        LOGGER.debug(
            "Evaluating liquidity swings",
            extra={
                "tf": timeframe,
                "candles": candle_count,
                "used_source": source_label,
            },
        )

        effective_window = config.swing_window
        minimum_required = base_minimum_required
        degraded_notes: List[str] = []
        degrade_mode = False

        if candle_count < minimum_required and config.enable_resample_when_sparse:
            interval_ms = TIMEFRAME_TO_MS.get(timeframe)
            if minute_seed and interval_ms:
                aggregated = resample_ohlcv(minute_seed, interval_ms)
                if len(aggregated) > candle_count:
                    candles = aggregated
                    candle_count = len(candles)
                    degrade_mode = True
                    degraded_notes.append("resampled_from_1m")
                    frame_diag["used_source"] = "aggregated"
                    LOGGER.debug(
                        "Liquidity timeframe resampled from minute seed",
                        extra={
                            "tf": timeframe,
                            "candles": candle_count,
                            "reason": "resample_sparse",
                        },
                    )

        if candle_count < minimum_required:
            max_window = max(config.degraded_window_floor, (candle_count - 1) // 2)
            if max_window < effective_window and max_window >= config.degraded_window_floor:
                effective_window = max_window
                minimum_required = max(1, 2 * effective_window + 1)
                degrade_mode = True
                degraded_notes.append("window_shrunk")
        if effective_window <= 0:
            effective_window = 1
            minimum_required = max(1, 2 * effective_window + 1)

        frame_diag["effective_window"] = effective_window

        if candle_count < minimum_required:
            LOGGER.debug(
                "Skipping liquidity timeframe",
                extra={
                    "tf": timeframe,
                    "reason": "too_few_bars",
                    "candles": candle_count,
                    "required": minimum_required,
                    "used_source": frame_diag["used_source"],
                },
            )
            _append_reason(
                frame_diag["reasons"],
                "too_few_bars",
                candles=candle_count,
                required=minimum_required,
            )
            frame_diag["degraded_mode"] = degrade_mode
            frame_diag["degraded_reasons"] = degraded_notes
            if degrade_mode:
                degraded_timeframes.append(timeframe)
            continue

        swings_high = _detect_swings(
            candles,
            window=effective_window,
            kind="high",
            tick_size=tick_value,
        )
        swings_low = _detect_swings(
            candles,
            window=effective_window,
            kind="low",
            tick_size=tick_value,
        )

        if config.lookback_swings > 0:
            swings_high = swings_high[-config.lookback_swings :]
            swings_low = swings_low[-config.lookback_swings :]

        frame_diag["eqh"]["swing_count"] = len(swings_high)
        frame_diag["eql"]["swing_count"] = len(swings_low)

        eqh_pairs = _count_pairs_within_tolerance(
            swings_high,
            tolerance=eqh_tolerance,
            tick_size=tick_size,
            tolerance_mode=tolerance_mode,
            tolerance_percent=tolerance_percent_value,
        )
        eql_pairs = _count_pairs_within_tolerance(
            swings_low,
            tolerance=eql_tolerance,
            tick_size=tick_size,
            tolerance_mode=tolerance_mode,
            tolerance_percent=tolerance_percent_value,
        )
        frame_diag["eqh"]["pairs_within_tol_before_cluster"] = eqh_pairs
        frame_diag["eqh"]["pairs_within_tol"] = eqh_pairs
        frame_diag["eql"]["pairs_within_tol_before_cluster"] = eql_pairs
        frame_diag["eql"]["pairs_within_tol"] = eql_pairs
        frame_diag["eqh"]["sample_pairs_top10"] = _sample_swing_pairs(swings_high)
        frame_diag["eql"]["sample_pairs_top10"] = _sample_swing_pairs(swings_low)

        frame_diag["eqh"]["tolerance_mode"] = tolerance_mode
        frame_diag["eql"]["tolerance_mode"] = tolerance_mode
        if tolerance_mode == "percent":
            frame_diag["eqh"]["tolerance_percent"] = tolerance_percent_value
            frame_diag["eql"]["tolerance_percent"] = tolerance_percent_value
        else:
            frame_diag["eqh"].pop("tolerance_percent", None)
            frame_diag["eql"].pop("tolerance_percent", None)
        frame_diag["eqh"]["cluster_price_pct"] = cluster_price_pct
        frame_diag["eql"]["cluster_price_pct"] = cluster_price_pct

        baseline_min_points = 3 if timeframe in {"1m", "5m"} else 2
        if config.min_points_dynamic:
            alpha = config.min_points_alpha if config.min_points_alpha > 0 else 1.0
            min_points_required = max(
                config.min_points_floor,
                math.ceil(effective_window / alpha),
            )
        else:
            min_points_required = max(2, config.min_points_floor)
        min_points_required = max(baseline_min_points, min_points_required)

        frame_diag["eqh"]["min_points_required"] = min_points_required
        frame_diag["eql"]["min_points_required"] = min_points_required

        eqh_cluster, eqh_candidates, eqh_hist, eqh_merges = _cluster_swings(
            swings_high,
            tolerance=eqh_tolerance,
            tick_size=tick_size,
            level_type="eqh",
            timeframe=timeframe,
            min_points=min_points_required,
            tolerance_mode=tolerance_mode,
            tolerance_value=tolerance_percent_value,
            cluster_price_pct=cluster_price_pct,
            cluster_time_window_bars=cluster_time_window_bars,
            min_distance_bars=min_distance_bars,
            reason_sink=frame_diag["eqh"]["reasons"],
        )
        eql_cluster, eql_candidates, eql_hist, eql_merges = _cluster_swings(
            swings_low,
            tolerance=eql_tolerance,
            tick_size=tick_size,
            level_type="eql",
            timeframe=timeframe,
            min_points=min_points_required,
            tolerance_mode=tolerance_mode,
            tolerance_value=tolerance_percent_value,
            cluster_price_pct=cluster_price_pct,
            cluster_time_window_bars=cluster_time_window_bars,
            min_distance_bars=min_distance_bars,
            reason_sink=frame_diag["eql"]["reasons"],
        )

        frame_diag["eqh"]["cluster_count"] = len(eqh_cluster)
        frame_diag["eqh"]["raw_count"] = len(eqh_candidates)
        frame_diag["eqh"]["filtered_count"] = len(eqh_cluster)
        frame_diag["eqh"]["reason_histogram"] = eqh_hist
        frame_diag["eqh"]["merge_operations"] = eqh_merges
        frame_diag["eqh"]["candidates"] = eqh_candidates

        frame_diag["eql"]["cluster_count"] = len(eql_cluster)
        frame_diag["eql"]["raw_count"] = len(eql_candidates)
        frame_diag["eql"]["filtered_count"] = len(eql_cluster)
        frame_diag["eql"]["reason_histogram"] = eql_hist
        frame_diag["eql"]["merge_operations"] = eql_merges
        frame_diag["eql"]["candidates"] = eql_candidates

        eqh_levels.extend(eqh_cluster)
        eql_levels.extend(eql_cluster)

        summary_raw["eqh"] += len(eqh_candidates)
        summary_raw["eql"] += len(eql_candidates)
        summary_filtered["eqh"] += len(eqh_cluster)
        summary_filtered["eql"] += len(eql_cluster)
        summary_reasons["eqh"].update(eqh_hist)
        summary_reasons["eql"].update(eql_hist)

        if degrade_mode:
            frame_diag["degraded_mode"] = True
            frame_diag["degraded_reasons"] = degraded_notes
            degraded_timeframes.append(timeframe)
        else:
            frame_diag["degraded_mode"] = False

        LOGGER.debug(
            "Liquidity timeframe summary",
            extra={
                "tf": timeframe,
                "n_bars_total": candle_count,
                "swing_highs": len(swings_high),
                "swing_lows": len(swings_low),
                "eqh_clusters": len(eqh_cluster),
                "eql_clusters": len(eql_cluster),
                "pairs_within_tol_high": frame_diag["eqh"]["pairs_within_tol"],
                "pairs_within_tol_low": frame_diag["eql"]["pairs_within_tol"],
                "tick_size": tick_size,
                "tolerance_eqh": eqh_tolerance,
                "tolerance_eql": eql_tolerance,
                "min_points_required": min_points_required,
                "used_source": frame_diag["used_source"],
                "merge_operations_eqh": eqh_merges,
                "merge_operations_eql": eql_merges,
            },
        )

    diagnostics["_summary"] = {
        "raw": {key: summary_raw[key] for key in summary_raw},
        "filtered": {key: summary_filtered[key] for key in summary_filtered},
        "reason_histogram": {
            key: dict(counter) for key, counter in summary_reasons.items()
        },
        "degraded_timeframes": degraded_timeframes,
    }

    if not eqh_levels:
        LOGGER.debug("No EQH clusters formed", extra={"reason": "no_clusters"})
    if not eql_levels:
        LOGGER.debug("No EQL clusters formed", extra={"reason": "no_clusters"})

    return {"eqh": eqh_levels, "eql": eql_levels}, diagnostics


def _resolve_previous_day(
    candles: Sequence[Mapping[str, Any]],
    *,
    reference_end_ms: int | None,
) -> tuple[Dict[str, Dict[str, Any] | None], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "candles": len(candles),
        "reasons": [],
    }
    if not candles:
        LOGGER.debug(
            "Unable to resolve previous day levels",
            extra={"reason": "no_daily_candles"},
        )
        _append_reason(diagnostics["reasons"], "no_daily_candles")
        return {"pdh": None, "pdl": None}, diagnostics

    if reference_end_ms is None:
        last_ts = candles[-1].get("t")
        reference_end_ms = int(last_ts) + MS_IN_DAY if isinstance(last_ts, (int, float)) else None

    if reference_end_ms is None:
        LOGGER.debug(
            "Unable to resolve previous day levels",
            extra={"reason": "no_reference_time"},
        )
        _append_reason(diagnostics["reasons"], "no_reference_time")
        return {"pdh": None, "pdl": None}, diagnostics

    reference_day = datetime.fromtimestamp(reference_end_ms / 1000, tz=timezone.utc).date()
    previous_day = reference_day - timedelta(days=1)

    target_high: Dict[str, Any] | None = None
    target_low: Dict[str, Any] | None = None
    for candle in reversed(candles):
        open_ts = candle.get("t")
        if not isinstance(open_ts, (int, float)):
            continue
        candle_day = datetime.fromtimestamp(open_ts / 1000, tz=timezone.utc).date()
        if candle_day != previous_day:
            continue
        high = _coerce_float(candle.get("h"))
        low = _coerce_float(candle.get("l"))
        if high is None or low is None:
            continue
        day_start = datetime.combine(previous_day, datetime.min.time(), tzinfo=timezone.utc)
        target_high = {"t": int(day_start.timestamp() * 1000), "price": high}
        target_low = {"t": int(day_start.timestamp() * 1000), "price": low}
        break

    if target_high is None or target_low is None:
        LOGGER.debug(
            "Previous day levels unavailable",
            extra={"reason": "no_daily_candle_prev_utc", "day": str(previous_day)},
        )
        _append_reason(
            diagnostics["reasons"],
            "no_daily_candle_prev_utc",
            day=str(previous_day),
        )

    diagnostics["found"] = bool(target_high and target_low)
    diagnostics["day"] = str(previous_day)

    return {"pdh": target_high, "pdl": target_low}, diagnostics


def _detect_sweeps(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    tick_size: float | None,
    config: LiquidityConfig,
    eqh: Sequence[Mapping[str, Any]],
    eql: Sequence[Mapping[str, Any]],
    pdh: Mapping[str, Any] | None,
    pdl: Mapping[str, Any] | None,
    normalized_symbol: str | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sweeps: List[Dict[str, Any]] = []
    tick_min = tick_size if tick_size and tick_size > 0 else 0.0

    LOGGER.debug(
        "Liquidity sweep detection config",
        extra={
            "atr_period": config.atr_period,
            "atr_mult": config.sweep_atr_multiplier,
            "tick_size": tick_size,
        },
    )

    upper_levels_by_tf: Dict[str, List[Mapping[str, Any]]] = {tf: [] for tf in SUPPORTED_TIMEFRAMES}
    lower_levels_by_tf: Dict[str, List[Mapping[str, Any]]] = {tf: [] for tf in SUPPORTED_TIMEFRAMES}

    for level in eqh:
        tf = str(level.get("tf"))
        if tf in upper_levels_by_tf:
            upper_levels_by_tf[tf].append(level)
    for level in eql:
        tf = str(level.get("tf"))
        if tf in lower_levels_by_tf:
            lower_levels_by_tf[tf].append(level)

    for tf in SUPPORTED_TIMEFRAMES:
        if pdh:
            upper_levels_by_tf[tf].append({"type": "pdh", "price": pdh["price"], "t": pdh["t"]})  # type: ignore[index]
        if pdl:
            lower_levels_by_tf[tf].append({"type": "pdl", "price": pdl["price"], "t": pdl["t"]})  # type: ignore[index]

    diagnostics: Dict[str, Any] = {}

    def _build_swing_fallback_levels(
        candles: Sequence[Mapping[str, Any]],
        *,
        kind: str,
    ) -> List[Dict[str, Any]]:
        if not candles:
            return []
        swing_window = max(1, config.swing_window)
        fallback_swings = _detect_swings(
            candles,
            window=swing_window,
            kind="high" if kind == "high" else "low",
            tick_size=tick_min,
        )
        if config.lookback_swings > 0:
            fallback_swings = fallback_swings[-config.lookback_swings :]
        seen_prices: set[float] = set()
        fallback_levels: List[Dict[str, Any]] = []
        for swing in fallback_swings:
            price_value = _coerce_float(swing.get("price"))
            ts_value = swing.get("t")
            if price_value is None or ts_value is None:
                continue
            price_quant = _quantise(price_value, tick_size)
            key = round(price_quant, 8)
            if key in seen_prices:
                continue
            seen_prices.add(key)
            fallback_levels.append(
                {
                    "type": "eqh_fallback" if kind == "high" else "eql_fallback",
                    "price": price_quant,
                    "t": int(ts_value),
                    "swings": [int(ts_value)],
                    "source": "swing_fallback",
                }
            )
        return fallback_levels

    for timeframe in SUPPORTED_TIMEFRAMES:
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        upper_levels = upper_levels_by_tf.get(timeframe, [])
        lower_levels = lower_levels_by_tf.get(timeframe, [])

        fallback_upper_used = False
        fallback_lower_used = False
        if not upper_levels:
            fallback_upper = _build_swing_fallback_levels(candles, kind="high")
            if fallback_upper:
                upper_levels = fallback_upper
                fallback_upper_used = True
        if not lower_levels:
            fallback_lower = _build_swing_fallback_levels(candles, kind="low")
            if fallback_lower:
                lower_levels = fallback_lower
                fallback_lower_used = True

        frame_diag = {
            "used_source": source_label,
            "candles": len(candles),
            "atr_period": config.atr_period,
            "atr_mult": config.sweep_atr_multiplier,
        "tick_size": tick_size,
        "upper": {
            "levels": len(upper_levels),
            "events": 0,
            "reasons": [],
            "candidates_before_atr": 0,
            "candidates_after_atr": 0,
            "samples": [],
            "fallback_used": fallback_upper_used,
            "fallback_source": "swing" if fallback_upper_used else None,
        },
        "lower": {
            "levels": len(lower_levels),
            "events": 0,
            "reasons": [],
            "candidates_before_atr": 0,
            "candidates_after_atr": 0,
            "samples": [],
            "fallback_used": fallback_lower_used,
            "fallback_source": "swing" if fallback_lower_used else None,
        },
    }
        diagnostics[timeframe] = frame_diag

        if fallback_upper_used:
            frame_diag["upper"]["fallback_source"] = "swing"
        if fallback_lower_used:
            frame_diag["lower"]["fallback_source"] = "swing"

        if not upper_levels:
            _append_reason(frame_diag["upper"]["reasons"], "no_levels")
        if not lower_levels:
            _append_reason(frame_diag["lower"]["reasons"], "no_levels")

        if not candles:
            LOGGER.debug(
                "Skipping sweep evaluation due to empty frame",
                extra={"tf": timeframe, "reason": "no_candles", "used_source": source_label},
            )
            _append_reason(frame_diag["upper"]["reasons"], "no_candles")
            _append_reason(frame_diag["lower"]["reasons"], "no_candles")
            continue

        valid_bars = [
            candle
            for candle in candles
            if _coerce_float(candle.get("h")) is not None
            and _coerce_float(candle.get("l")) is not None
            and _coerce_float(candle.get("c")) is not None
        ]
        if len(valid_bars) < config.atr_period + 1:
            LOGGER.debug(
                "Skipping sweep evaluation due to insufficient bars for ATR",
                extra={
                    "tf": timeframe,
                    "reason": "too_few_bars",
                    "used_source": source_label,
                    "required": config.atr_period + 1,
                    "available": len(valid_bars),
                },
            )
            _append_reason(frame_diag["upper"]["reasons"], "too_few_bars")
            _append_reason(frame_diag["lower"]["reasons"], "too_few_bars")
            frame_diag["atr_stats"] = {
                "atr_period": config.atr_period,
                "atr_mean": None,
                "atr_median": None,
                "atr_limit": None,
                "epsilon": None,
                "tick_size": tick_size,
            }
            continue

        LOGGER.debug(
            "Evaluating sweep candidates",
            extra={
                "tf": timeframe,
                "direction": "upper",
                "candles": len(candles),
                "levels": len(upper_levels),
                "used_source": source_label,
            },
        )
        atr_values = _compute_atr_series(candles, period=config.atr_period)
        valid_atr_values = [value for value in atr_values if value > 0]
        atr_mean = statistics.fmean(valid_atr_values) if valid_atr_values else 0.0
        atr_median = statistics.median(valid_atr_values) if valid_atr_values else 0.0
        atr_limit_mean = atr_mean * config.sweep_atr_multiplier
        epsilon_preview = max(atr_limit_mean, tick_min)

        frame_diag["atr_stats"] = {
            "atr_period": config.atr_period,
            "atr_mean": atr_mean,
            "atr_median": atr_median,
            "atr_limit": atr_limit_mean,
            "epsilon": epsilon_preview,
            "tick_size": tick_size,
        }

        LOGGER.debug(
            "ATR diagnostics for sweeps",
            extra={
                "tf": timeframe,
                "used_source": source_label,
                "atr_period": config.atr_period,
                "atr_mean": atr_mean,
                "atr_median": atr_median,
                "atr_limit": atr_limit_mean,
                "epsilon_preview": epsilon_preview,
                "tick_size": tick_size,
            },
        )

        if (
            timeframe == "15m"
            and normalized_symbol == "BTCUSDT"
            and atr_mean > 0
            and atr_mean < 5
        ):
            LOGGER.error(
                "ATR too small — check units",
                extra={
                    "tf": timeframe,
                    "normalized_symbol": normalized_symbol,
                    "atr_mean": atr_mean,
                    "atr_period": config.atr_period,
                },
            )
            _append_reason(frame_diag["upper"]["reasons"], "atr_too_small")
            _append_reason(frame_diag["lower"]["reasons"], "atr_too_small")
            continue

        epsilon_samples: List[float] = []
        upper_candidate_samples: List[Dict[str, Any]] = []
        lower_candidate_samples: List[Dict[str, Any]] = []

        lookback_start_index = max(0, len(candles) - max(config.sweep_lookback_bars, 1))
        frame_diag["upper"]["lookback_start"] = lookback_start_index
        frame_diag["lower"]["lookback_start"] = lookback_start_index
        frame_diag["upper"]["min_move_atr"] = config.sweep_min_move_atr
        frame_diag["lower"]["min_move_atr"] = config.sweep_min_move_atr
        frame_diag["upper"]["confirm_window"] = config.sweep_confirm_window
        frame_diag["lower"]["confirm_window"] = config.sweep_confirm_window
        frame_diag["upper"]["epsilon_pct"] = config.sweep_epsilon_pct
        frame_diag["lower"]["epsilon_pct"] = config.sweep_epsilon_pct

        for index, candle in enumerate(candles):
            if index < lookback_start_index:
                continue
            high = _coerce_float(candle.get("h"))
            close = _coerce_float(candle.get("c"))
            ts = candle.get("t")
            if high is None or close is None or not isinstance(ts, (int, float)):
                continue
            high = _quantise(high, tick_size)
            close = _quantise(close, tick_size)
            atr_value = atr_values[index] if index < len(atr_values) else 0.0
            atr_component = atr_value * config.sweep_atr_multiplier
            atr_limit = max(tick_min, atr_component)
            min_move = max(config.sweep_min_move_atr * atr_value, tick_min)
            for level in upper_levels:
                level_price = _coerce_float(level.get("price"))
                if level_price is None:
                    continue
                level_price = _quantise(level_price, tick_size)
                formed_after: int | None = None
                level_type = str(level.get("type"))
                if level_type == "eqh":
                    swings = level.get("swings") if isinstance(level.get("swings"), Sequence) else []
                    swing_ts = [int(value) for value in swings if isinstance(value, (int, float))]
                    if swing_ts:
                        formed_after = max(swing_ts)
                elif level_type == "pdh" and isinstance(level.get("t"), (int, float)):
                    formed_after = int(level["t"]) + MS_IN_DAY
                if formed_after is not None and ts <= formed_after:
                    continue
                overshoot = high - level_price
                if overshoot <= 0:
                    continue
                frame_diag["upper"]["candidates_before_atr"] += 1
                if overshoot < min_move:
                    _append_reason(
                        frame_diag["upper"]["reasons"],
                        "min_move_fail",
                        level_type=level_type,
                        overshoot=overshoot,
                        min_move=min_move,
                    )
                    continue
                epsilon = max(atr_limit, tick_min)
                if config.sweep_epsilon_pct > 0 and level_price > 0:
                    epsilon = max(epsilon, level_price * config.sweep_epsilon_pct)
                price_cap = level_price * SWEEP_CAP_PCT if level_price > 0 else None
                if price_cap is not None and price_cap > 0:
                    epsilon = min(epsilon, max(price_cap, tick_min))
                epsilon_samples.append(epsilon)
                if overshoot <= epsilon:
                    _append_reason(
                        frame_diag["upper"]["reasons"],
                        "tolerance_fail",
                        level_type=level_type,
                        overshoot=overshoot,
                        tolerance=epsilon,
                        epsilon=epsilon,
                    )
                    continue
                if len(upper_candidate_samples) < 5:
                    upper_candidate_samples.append(
                        {
                            "level": level_price,
                            "high": high,
                            "close": close,
                            "overshoot": overshoot,
                            "epsilon": epsilon,
                            "atr_limit": atr_limit,
                            "min_move": min_move,
                        }
                    )
                frame_diag["upper"]["candidates_after_atr"] += 1
                confirm_idx: int | None = None
                confirm_close_val = close
                confirm_limit = min(len(candles), index + config.sweep_confirm_window + 1)
                for confirm in range(index + 1, confirm_limit):
                    confirm_close_raw = _coerce_float(candles[confirm].get("c"))
                    if confirm_close_raw is None:
                        continue
                    confirm_close_candidate = _quantise(confirm_close_raw, tick_size)
                    if confirm_close_candidate <= level_price:
                        confirm_idx = confirm
                        confirm_close_val = confirm_close_candidate
                        break
                if confirm_idx is None:
                    _append_reason(
                        frame_diag["upper"]["reasons"],
                        "no_return_window",
                        level_type=level_type,
                        confirm_window=config.sweep_confirm_window,
                    )
                    continue
                confirm_ts = int(candles[confirm_idx].get("t", ts))
                return_bars = confirm_idx - index
                sweeps.append(
                    {
                        "type": "sweep_top",
                        "tf": timeframe,
                        "level_type": level_type,
                        "level_price": level_price,
                        "t": int(ts),
                        "atr_tolerance": epsilon,
                        "min_move": min_move,
                        "level_source": level.get("source", "eq"),
                        "retest_t": confirm_ts,
                        "retest_close": confirm_close_val,
                        "return_bars": return_bars,
                        "fallback": bool(level.get("source") == "swing_fallback"),
                    }
                )
                frame_diag["upper"]["events"] += 1

        LOGGER.debug(
            "Evaluating sweep candidates",
            extra={
                "tf": timeframe,
                "direction": "lower",
                "candles": len(candles),
                "levels": len(lower_levels),
                "used_source": source_label,
            },
        )
        for index, candle in enumerate(candles):
            if index < lookback_start_index:
                continue
            low = _coerce_float(candle.get("l"))
            close = _coerce_float(candle.get("c"))
            ts = candle.get("t")
            if low is None or close is None or not isinstance(ts, (int, float)):
                continue
            low = _quantise(low, tick_size)
            close = _quantise(close, tick_size)
            atr_value = atr_values[index] if index < len(atr_values) else 0.0
            atr_component = atr_value * config.sweep_atr_multiplier
            atr_limit = max(tick_min, atr_component)
            min_move = max(config.sweep_min_move_atr * atr_value, tick_min)
            for level in lower_levels:
                level_price = _coerce_float(level.get("price"))
                if level_price is None:
                    continue
                level_price = _quantise(level_price, tick_size)
                formed_after: int | None = None
                level_type = str(level.get("type"))
                if level_type == "eql":
                    swings = level.get("swings") if isinstance(level.get("swings"), Sequence) else []
                    swing_ts = [int(value) for value in swings if isinstance(value, (int, float))]
                    if swing_ts:
                        formed_after = max(swing_ts)
                elif level_type == "pdl" and isinstance(level.get("t"), (int, float)):
                    formed_after = int(level["t"]) + MS_IN_DAY
                if formed_after is not None and ts <= formed_after:
                    continue
                overshoot = level_price - low
                if overshoot <= 0:
                    continue
                frame_diag["lower"]["candidates_before_atr"] += 1
                if overshoot < min_move:
                    _append_reason(
                        frame_diag["lower"]["reasons"],
                        "min_move_fail",
                        level_type=level_type,
                        overshoot=overshoot,
                        min_move=min_move,
                    )
                    continue
                epsilon = max(atr_limit, tick_min)
                if config.sweep_epsilon_pct > 0 and level_price > 0:
                    epsilon = max(epsilon, level_price * config.sweep_epsilon_pct)
                price_cap = level_price * SWEEP_CAP_PCT if level_price > 0 else None
                if price_cap is not None and price_cap > 0:
                    epsilon = min(epsilon, max(price_cap, tick_min))
                epsilon_samples.append(epsilon)
                if overshoot <= epsilon:
                    _append_reason(
                        frame_diag["lower"]["reasons"],
                        "tolerance_fail",
                        level_type=level_type,
                        overshoot=overshoot,
                        tolerance=epsilon,
                        epsilon=epsilon,
                    )
                    continue
                if len(lower_candidate_samples) < 5:
                    lower_candidate_samples.append(
                        {
                            "level": level_price,
                            "low": low,
                            "close": close,
                            "overshoot": overshoot,
                            "epsilon": epsilon,
                            "atr_limit": atr_limit,
                            "min_move": min_move,
                        }
                    )
                frame_diag["lower"]["candidates_after_atr"] += 1
                confirm_idx = None
                confirm_close_val = close
                confirm_limit = min(len(candles), index + config.sweep_confirm_window + 1)
                for confirm in range(index + 1, confirm_limit):
                    confirm_close_raw = _coerce_float(candles[confirm].get("c"))
                    if confirm_close_raw is None:
                        continue
                    confirm_close_candidate = _quantise(confirm_close_raw, tick_size)
                    if confirm_close_candidate >= level_price:
                        confirm_idx = confirm
                        confirm_close_val = confirm_close_candidate
                        break
                if confirm_idx is None:
<<<<<<< Updated upstream
                    _append_reason(
                        frame_diag["lower"]["reasons"],
                        "no_return_window",
                        level_type=level_type,
                        confirm_window=config.sweep_confirm_window,
                    )
                    continue
                confirm_ts = int(candles[confirm_idx].get("t", ts))
                return_bars = confirm_idx - index
=======
                    level_origin = str(level.get("source", "eq"))
                    if index >= len(candles) - 1 and level_origin not in {"swing_fallback", "fallback"}:
                        confirm_ts = int(ts)
                        return_bars = 0
                        confirm_close_val = close
                    else:
                        _append_reason(
                            frame_diag["lower"]["reasons"],
                            "no_return_window",
                            level_type=level_type,
                            confirm_window=config.sweep_confirm_window,
                        )
                        continue
                else:
                    confirm_ts = int(candles[confirm_idx].get("t", ts))
                    return_bars = confirm_idx - index
>>>>>>> Stashed changes
                sweeps.append(
                    {
                        "type": "sweep_bottom",
                        "tf": timeframe,
                        "level_type": level_type,
                        "level_price": level_price,
                        "t": int(ts),
                        "atr_tolerance": epsilon,
                        "min_move": min_move,
                        "level_source": level.get("source", "eq"),
                        "retest_t": confirm_ts,
                        "retest_close": confirm_close_val,
                        "return_bars": return_bars,
                        "fallback": bool(level.get("source") == "swing_fallback"),
                    }
                )
                frame_diag["lower"]["events"] += 1

        if upper_candidate_samples:
            frame_diag["upper"]["samples"] = upper_candidate_samples
        if lower_candidate_samples:
            frame_diag["lower"]["samples"] = lower_candidate_samples

        epsilon_stats = {
            "min": min(epsilon_samples) if epsilon_samples else None,
            "max": max(epsilon_samples) if epsilon_samples else None,
        }
        LOGGER.debug(
            "Sweep timeframe summary",
            extra={
                "tf": timeframe,
                "used_source": source_label,
                "candles": len(candles),
                "upper_levels": len(upper_levels),
                "lower_levels": len(lower_levels),
                "upper_events": frame_diag["upper"]["events"],
                "lower_events": frame_diag["lower"]["events"],
                "atr_period": config.atr_period,
                "atr_mult": config.sweep_atr_multiplier,
                "tick_size": tick_size,
                "atr_last": atr_values[-1] if atr_values else None,
                "epsilon_min": epsilon_stats["min"],
                "epsilon_max": epsilon_stats["max"],
                "candidates_upper": frame_diag["upper"]["candidates_before_atr"],
                "candidates_lower": frame_diag["lower"]["candidates_before_atr"],
                "candidates_upper_after_atr": frame_diag["upper"]["candidates_after_atr"],
                "candidates_lower_after_atr": frame_diag["lower"]["candidates_after_atr"],
            },
        )

    sweeps.sort(key=lambda item: item.get("t", 0))
    return sweeps, diagnostics


def build_liquidity_snapshot(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    symbol: str | None = None,
    tick_size: float | None,
    tick_source_hint: str | None = None,
    meta: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute liquidity levels and sweeps for the inspection payload."""

    normalized_symbol = normalise_symbol_for_tick(symbol)

    frame_sequences: Dict[str, List[Mapping[str, Any]]] = {}
    for tf, payload in frames.items():
        candles = _extract_candles(payload)
        if candles:
            frame_sequences[tf] = candles

    resolved_tick: float
    tick_source: str
    if tick_source_hint and tick_size is not None and tick_size > 0:
        resolved_tick = float(tick_size)
        tick_source = tick_source_hint or "param"
    else:
        resolved_tick, inferred_source = resolve_liquidity_tick_size(
            symbol or normalized_symbol,
            None,
            frame_sequences,
            meta=meta,
            logger=LOGGER,
        )
        tick_source = inferred_source
        if tick_size is not None and tick_size > 0:
            tick_source = "param"
    if resolved_tick <= 0:
        LOGGER.error(
            "Liquidity detection aborted due to non-positive tick size",
            extra={
                "symbol": symbol or "UNKNOWN",
                "normalized_symbol": normalized_symbol or "UNKNOWN",
                "tick_size": resolved_tick,
                "tick_size_source": tick_source,
            },
        )
        raise ValueError("Liquidity detection requires a positive tick size")

    resolved_config = _resolve_config(config)

    augmented_frames = _augment_supported_frames(frames)
    levels, level_diagnostics = _prepare_levels(
        augmented_frames,
        tick_size=resolved_tick,
        config=resolved_config,
    )
    level_summary = level_diagnostics.get("_summary", {})
    levels_by_tf = {key: value for key, value in level_diagnostics.items() if key != "_summary"}

<<<<<<< Updated upstream
=======
    def _fallback_levels(
        candles: Sequence[Mapping[str, Any]] | None,
        *,
        side: str,
        tf_label: str,
    ) -> List[Dict[str, Any]]:
        if not candles:
            return []
        prices: List[float] = []
        for candle in candles:
            if not isinstance(candle, Mapping):
                continue
            price_value = _coerce_float(candle.get("h" if side == "eqh" else "l"))
            if price_value is None:
                continue
            prices.append(price_value)
        if not prices:
            return []
        if side == "eqh":
            price = max(prices)
        else:
            price = min(prices)
        tolerance = float(resolved_config.r_ticks) * resolved_tick
        return [
            {
                "price": price,
                "tf": tf_label,
                "tolerance": tolerance,
                "tolerance_ticks": float(resolved_config.r_ticks),
                "source": "fallback",
            }
        ]

>>>>>>> Stashed changes
    raw_candidates: Dict[str, List[Dict[str, Any]]] = {"eqh": [], "eql": []}
    for tf_key, tf_diag in levels_by_tf.items():
        if not isinstance(tf_diag, Mapping):
            continue
        eqh_diag = tf_diag.get("eqh")
        if isinstance(eqh_diag, Mapping):
            candidates = eqh_diag.get("candidates")
            if isinstance(candidates, Sequence):
                raw_candidates["eqh"].extend(dict(candidate) for candidate in candidates if isinstance(candidate, Mapping))
        eql_diag = tf_diag.get("eql")
        if isinstance(eql_diag, Mapping):
            candidates = eql_diag.get("candidates")
            if isinstance(candidates, Sequence):
                raw_candidates["eql"].extend(dict(candidate) for candidate in candidates if isinstance(candidate, Mapping))
<<<<<<< Updated upstream
=======

    if not levels["eqh"]:
        fallback_eqh = _fallback_levels(
            frame_sequences.get("15m"),
            side="eqh",
            tf_label="15m",
        )
        if fallback_eqh:
            levels["eqh"] = fallback_eqh
            LIQUIDITY_COUNTERS["eqh_fallback"] += 1
            raw_candidates["eqh"].extend(dict(entry) for entry in fallback_eqh)

    if not levels["eql"]:
        fallback_eql = _fallback_levels(
            frame_sequences.get("15m"),
            side="eql",
            tf_label="15m",
        )
        if fallback_eql:
            levels["eql"] = fallback_eql
            LIQUIDITY_COUNTERS["eql_fallback"] += 1
            raw_candidates["eql"].extend(dict(entry) for entry in fallback_eql)
>>>>>>> Stashed changes

    daily_candles = _extract_candles(augmented_frames.get("1d"))
    selection_end = None
    if isinstance(selection, Mapping):
        end_value = selection.get("end")
        if isinstance(end_value, (int, float)):
            selection_end = int(end_value)
    daily_levels, daily_diagnostics = _resolve_previous_day(
        daily_candles,
        reference_end_ms=selection_end,
    )

    pdh_level = daily_levels.get("pdh")
    if isinstance(pdh_level, MutableMapping):
        price = _coerce_float(pdh_level.get("price"))
        if price is not None:
            pdh_level["price"] = _quantise(price, resolved_tick)
    pdl_level = daily_levels.get("pdl")
    if isinstance(pdl_level, MutableMapping):
        price = _coerce_float(pdl_level.get("price"))
        if price is not None:
            pdl_level["price"] = _quantise(price, resolved_tick)

    sweeps, sweep_diagnostics = _detect_sweeps(
        augmented_frames,
        tick_size=resolved_tick,
        config=resolved_config,
        eqh=levels["eqh"],
        eql=levels["eql"],
        pdh=daily_levels["pdh"],
        pdl=daily_levels["pdl"],
        normalized_symbol=normalized_symbol,
    )

    LOGGER.debug(
        "Liquidity detection summary",
        extra={
            "eqh": len(levels["eqh"]),
            "eql": len(levels["eql"]),
            "sweeps": len(sweeps),
        },
    )

    summary_raw = level_summary.get("raw", {}) if isinstance(level_summary, Mapping) else {}
    summary_filtered = level_summary.get("filtered", {}) if isinstance(level_summary, Mapping) else {}
    reason_hist = level_summary.get("reason_histogram", {}) if isinstance(level_summary, Mapping) else {}
    degraded_timeframes = level_summary.get("degraded_timeframes", []) if isinstance(level_summary, Mapping) else []

    def _ordered_reasons(histogram: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(histogram, Mapping):
            return []
        items: List[Dict[str, Any]] = []
        for reason_code, count in histogram.items():
            try:
                numeric = int(count)
            except (TypeError, ValueError):
                continue
            items.append({"reason": str(reason_code), "count": numeric})
        items.sort(key=lambda entry: (-entry["count"], entry["reason"]))
        return items

    eqh_reasons = _ordered_reasons(reason_hist.get("eqh") if isinstance(reason_hist, Mapping) else {})
    eql_reasons = _ordered_reasons(reason_hist.get("eql") if isinstance(reason_hist, Mapping) else {})
    if not eqh_reasons and not levels["eqh"]:
        eqh_reasons = [{"reason": "no_candidates", "count": 1}]
    if not eql_reasons and not levels["eql"]:
        eql_reasons = [{"reason": "no_candidates", "count": 1}]

<<<<<<< Updated upstream
    metrics_block = {
        "tick_size_unresolved": 1 if tick_source == "fallback_min" else 0,
        "raw_candidates_eqh": int(summary_raw.get("eqh", 0)) if isinstance(summary_raw, Mapping) else 0,
        "raw_candidates_eql": int(summary_raw.get("eql", 0)) if isinstance(summary_raw, Mapping) else 0,
=======
    eqh_raw_count = int(summary_raw.get("eqh", 0)) if isinstance(summary_raw, Mapping) else 0
    eql_raw_count = int(summary_raw.get("eql", 0)) if isinstance(summary_raw, Mapping) else 0
    if eqh_raw_count == 0 and levels["eqh"]:
        eqh_raw_count = len(levels["eqh"])
    if eql_raw_count == 0 and levels["eql"]:
        eql_raw_count = len(levels["eql"])

    metrics_block = {
        "tick_size_unresolved": 1 if tick_source == "fallback_min" else 0,
        "raw_candidates_eqh": eqh_raw_count,
        "raw_candidates_eql": eql_raw_count,
>>>>>>> Stashed changes
        "filtered_eqh": len(levels["eqh"]),
        "filtered_eql": len(levels["eql"]),
    }

<<<<<<< Updated upstream
=======
    if not eqh_reasons and levels["eqh"]:
        eqh_reasons = [{"reason": "fallback_generated", "count": len(levels["eqh"]) }]
    if not eql_reasons and levels["eql"]:
        eql_reasons = [{"reason": "fallback_generated", "count": len(levels["eql"]) }]

    summary_block = {
        "eqh": len(levels["eqh"]),
        "eql": len(levels["eql"]),
        "sweeps": len(sweeps),
        "eqh_filtered": len(levels["eqh"]),
        "eql_filtered": len(levels["eql"]),
        "eqh_raw": eqh_raw_count,
        "eql_raw": eql_raw_count,
        "eqh_reasons": eqh_reasons,
        "eql_reasons": eql_reasons,
        "reason_histogram": {
            "eqh": dict(reason_hist.get("eqh", {})) if isinstance(reason_hist, Mapping) else {},
            "eql": dict(reason_hist.get("eql", {})) if isinstance(reason_hist, Mapping) else {},
        },
        "degraded_timeframes": degraded_timeframes,
        "has_degraded": bool(degraded_timeframes),
    }

>>>>>>> Stashed changes
    diagnostics_payload = {
        "config": {
            "swing_window": resolved_config.swing_window,
            "lookback": resolved_config.lookback_swings,
            "r_ticks": resolved_config.r_ticks,
            "atr_period": resolved_config.atr_period,
            "sweep_atr_multiplier": resolved_config.sweep_atr_multiplier,
            "tick_size": resolved_tick,
            "tolerance_eqh_ticks": resolved_config.tolerance_eqh_ticks or float(resolved_config.r_ticks),
            "tolerance_eql_ticks": resolved_config.tolerance_eql_ticks or float(resolved_config.r_ticks),
            "min_points_dynamic": resolved_config.min_points_dynamic,
            "min_points_alpha": resolved_config.min_points_alpha,
            "min_points_floor": resolved_config.min_points_floor,
            "merge_ticks": resolved_config.merge_ticks,
            "merge_overlap_ratio": resolved_config.merge_overlap_ratio,
            "enable_resample_when_sparse": resolved_config.enable_resample_when_sparse,
            "feature_relaxed_clustering": resolved_config.feature_relaxed_clustering,
            "feature_extended_resample": resolved_config.feature_extended_resample,
            "feature_strict_legacy_mode": resolved_config.feature_strict_legacy_mode,
        },
        "tick_size": {
            "symbol": symbol,
            "normalized_symbol": normalized_symbol or "UNKNOWN",
            "source": tick_source,
            "value": resolved_tick,
        },
        "levels": levels_by_tf,
        "levels_summary": level_summary,
        "daily": daily_diagnostics,
        "sweeps": sweep_diagnostics,
        "metrics": metrics_block,
<<<<<<< Updated upstream
        "summary": {
            "eqh": len(levels["eqh"]),
            "eql": len(levels["eql"]),
            "sweeps": len(sweeps),
            "eqh_filtered": len(levels["eqh"]),
            "eql_filtered": len(levels["eql"]),
            "eqh_raw": int(summary_raw.get("eqh", 0)) if isinstance(summary_raw, Mapping) else 0,
            "eql_raw": int(summary_raw.get("eql", 0)) if isinstance(summary_raw, Mapping) else 0,
            "eqh_reasons": eqh_reasons,
            "eql_reasons": eql_reasons,
            "reason_histogram": {
                "eqh": dict(reason_hist.get("eqh", {})) if isinstance(reason_hist, Mapping) else {},
                "eql": dict(reason_hist.get("eql", {})) if isinstance(reason_hist, Mapping) else {},
            },
            "degraded_timeframes": degraded_timeframes,
            "has_degraded": bool(degraded_timeframes),
        },
=======
        "summary": summary_block,
>>>>>>> Stashed changes
    }

    if resolved_config.session_atr_value is not None:
        diagnostics_payload["config"]["session_atr_value"] = resolved_config.session_atr_value
    if resolved_config.session_atr_pct is not None:
        diagnostics_payload["config"]["session_atr_pct"] = resolved_config.session_atr_pct
    if resolved_config.session_label:
        diagnostics_payload["config"]["session_label"] = resolved_config.session_label

    return {
        "eqh": levels["eqh"],
        "eql": levels["eql"],
        "pdh": daily_levels["pdh"],
        "pdl": daily_levels["pdl"],
        "sweeps": sweeps,
        "candidates": raw_candidates,
        "diagnostics": diagnostics_payload,
    }




async def generate_liquidity_map(
    candles: Sequence[Mapping[str, Any]],
    window_days: int,
) -> Dict[str, object]:
    """Build liquidity map metrics including PDH/PDL and session extremes."""

    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if not candles:
        return {
            "PDH": None,
            "PDL": None,
            "EQH": [],
            "EQL": [],
            "session_highs_lows": [],
            "resting_liquidity": [],
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    daily_highs: Dict[str, float] = {}
    daily_lows: Dict[str, float] = {}
    session_highs_lows: List[Dict[str, object]] = []

    by_session: Dict[str, List[float]] = {"asia": [], "london": [], "ny": []}
    by_session_low: Dict[str, List[float]] = {"asia": [], "london": [], "ny": []}

    eq_highs: List[float] = []
    eq_lows: List[float] = []
    resting: List[Dict[str, float]] = []

    for row in candles:
        if not isinstance(row, Mapping):
            continue
        ts_value = row.get("t") or row.get("time")
        if isinstance(ts_value, str):
            try:
                ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            except ValueError:
                continue
        elif isinstance(ts_value, (int, float)):
            ts = datetime.fromtimestamp(float(ts_value) / 1000, tz=timezone.utc)
        else:
            continue
        if ts < cutoff:
            continue
        date_key = ts.date().isoformat()
        high = _coerce_float(row.get("h")) or 0.0
        low = _coerce_float(row.get("l")) or 0.0
        volume = _coerce_float(row.get("v")) or 0.0
        daily_highs[date_key] = max(daily_highs.get(date_key, high), high)
        if date_key not in daily_lows:
            daily_lows[date_key] = low
        else:
            daily_lows[date_key] = min(daily_lows[date_key], low)
        session_name = "asia"
        hour = ts.hour
        if 8 <= hour < 16:
            session_name = "london"
        elif hour >= 16 or hour < 0:
            session_name = "ny"
        by_session[session_name].append(high)
        by_session_low[session_name].append(low)
        if volume > 0:
            resting.append({"price": float(row.get("c", high)), "volume": volume})
        if abs(high - low) <= 1e-8:
            eq_highs.append(high)
            eq_lows.append(low)

    pdh = max(daily_highs.values()) if daily_highs else None
    pdl = min(daily_lows.values()) if daily_lows else None

    for session_name in ("asia", "london", "ny"):
        highs = by_session.get(session_name, [])
        lows = by_session_low.get(session_name, [])
        if highs and lows:
            session_highs_lows.append(
                {
                    "session": session_name,
                    "high": max(highs),
                    "low": min(lows),
                }
            )

    resting.sort(key=lambda item: item["volume"], reverse=True)
    resting_liquidity = resting[:10]

    return {
        "PDH": pdh,
        "PDL": pdl,
        "EQH": sorted(set(eq_highs))[-10:],
        "EQL": sorted(set(eq_lows))[:10],
        "session_highs_lows": session_highs_lows,
        "resting_liquidity": resting_liquidity,
    }
