"""Liquidity level and sweep detection helpers."""

from __future__ import annotations

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


@dataclass(slots=True)
class LiquidityConfig:
    """Runtime configuration for liquidity detection."""

    swing_window: int = 3
    lookback_swings: int = 30
    r_ticks: int = 5
    atr_period: int = 14
    sweep_atr_multiplier: float = 0.3


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
            if abs(base - quantised[right]) <= tolerance + 1e-9:
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
    tick_source = "auto"

    if isinstance(hardcoded_tick, (int, float)) and hardcoded_tick > 0:
        tick_size = float(hardcoded_tick)
        tick_source = "hardcoded"
    elif profile_numeric is not None and profile_numeric > 0:
        tick_size = float(profile_numeric)
        tick_source = "profile"
    elif exchange_tick is not None and exchange_tick > 0:
        tick_size = float(exchange_tick)
        tick_source = "exchange"
    elif inferred_tick is not None and inferred_tick > 0:
        tick_size = float(inferred_tick)
        tick_source = "auto"

    if tick_size is None or tick_size <= 0:
        log.error(
            "Unable to resolve positive liquidity tick size",
            extra={**base_extra, "tick_size_source": "auto"},
        )
        raise ValueError("Unable to resolve positive liquidity tick size")

    if (
        profile_numeric is not None
        and tick_source != "profile"
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
    if not isinstance(raw, Mapping):
        return LiquidityConfig()

    config = LiquidityConfig()

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
                swings.append({"t": int(ts), "price": price})
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
                swings.append({"t": int(ts), "price": price})
    return swings


def _cluster_swings(
    swings: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    tick_size: float | None,
    level_type: str,
    timeframe: str,
    reason_sink: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
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
        return []

    ordered = sorted(swings, key=lambda item: item.get("t", 0))
    clusters: List[Dict[str, Any]] = []
    for swing in ordered:
        price_value = _coerce_float(swing.get("price"))
        time_value = swing.get("t")
        if price_value is None or not isinstance(time_value, (int, float)):
            continue
        quantised_price = _quantise(price_value, tick_size)
        matched = False
        for cluster in clusters:
            if abs(cluster["anchor"] - quantised_price) <= tolerance:
                cluster["prices"].append(quantised_price)
                cluster["swings"].append(int(time_value))
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
                }
            )

    payload: List[Dict[str, Any]] = []
    for cluster in clusters:
        swings_ts = sorted(set(cluster["swings"]))
        if len(swings_ts) < 2:
            LOGGER.debug(
                "Skipping swing cluster due to size",
                extra={
                    "reason": "min_points_fail",
                    "tf": timeframe,
                    "level_type": level_type,
                    "swings": swings_ts,
                    "tolerance": tolerance,
                    "tick_size": tick_size,
                },
            )
            _append_reason(
                reason_sink,
                "min_points_fail",
                swings=swings_ts,
                tolerance=tolerance,
                tick_size=tick_size,
            )
            continue
        prices = cluster["prices"]
        if not prices:
            continue
        level_price = statistics.median(prices)
        level_price = _quantise(level_price, tick_size)
        payload.append(
            {
                "type": level_type,
                "tf": timeframe,
                "price": level_price,
                "swings": swings_ts,
                "tolerance": tolerance,
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
    return payload


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

    tolerance = 0.0
    if tick_size and tick_size > 0:
        tolerance = max(config.r_ticks * tick_size, tick_size)

    diagnostics: Dict[str, Any] = {}

    LOGGER.debug(
        "Liquidity swing detection config",
        extra={
            "tick_size": tick_size,
            "r_ticks": config.r_ticks,
            "tolerance": tolerance,
            "swing_window": config.swing_window,
            "lookback": config.lookback_swings,
        },
    )

    minimum_required = 2 * config.swing_window + 1

    for timeframe in SUPPORTED_TIMEFRAMES:
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        candle_count = len(candles)

        frame_diag = {
            "used_source": source_label,
            "n_bars_total": candle_count,
            "tick_size": tick_size,
            "r_ticks": config.r_ticks,
            "tolerance": tolerance,
            "swing_window": config.swing_window,
            "lookback": config.lookback_swings,
            "atr_period": config.atr_period,
            "atr_mult": config.sweep_atr_multiplier,
            "reasons": [],
            "eqh": {
                "swing_count": 0,
                "cluster_count": 0,
                "pairs_within_tol": 0,
                "pairs_within_tol_before_cluster": 0,
                "sample_pairs_top10": [],
                "reasons": [],
            },
            "eql": {
                "swing_count": 0,
                "cluster_count": 0,
                "pairs_within_tol": 0,
                "pairs_within_tol_before_cluster": 0,
                "sample_pairs_top10": [],
                "reasons": [],
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
        if candle_count < minimum_required:
            LOGGER.debug(
                "Skipping liquidity timeframe",
                extra={
                    "tf": timeframe,
                    "reason": "too_few_bars",
                    "candles": candle_count,
                    "used_source": source_label,
                },
            )
            _append_reason(
                frame_diag["reasons"],
                "too_few_bars",
                candles=candle_count,
                required=minimum_required,
            )
            continue
        swings_high = _detect_swings(
            candles,
            window=config.swing_window,
            kind="high",
            tick_size=tick_size or 0.0,
        )
        swings_low = _detect_swings(
            candles,
            window=config.swing_window,
            kind="low",
            tick_size=tick_size or 0.0,
        )
        if config.lookback_swings > 0:
            swings_high = swings_high[-config.lookback_swings :]
            swings_low = swings_low[-config.lookback_swings :]
        frame_diag["eqh"]["swing_count"] = len(swings_high)
        frame_diag["eql"]["swing_count"] = len(swings_low)
        eqh_pairs = _count_pairs_within_tolerance(
            swings_high,
            tolerance=tolerance,
            tick_size=tick_size,
        )
        frame_diag["eqh"]["pairs_within_tol_before_cluster"] = eqh_pairs
        frame_diag["eqh"]["pairs_within_tol"] = eqh_pairs
        frame_diag["eqh"]["sample_pairs_top10"] = _sample_swing_pairs(swings_high)
        if eqh_pairs == 0 and len(swings_high) >= 2:
            LOGGER.debug(
                "No swing high pairs within tolerance prior to clustering",
                extra={
                    "tf": timeframe,
                    "reason": "pairs_within_tol_before_cluster==0",
                    "tick_size": tick_size,
                    "tolerance": tolerance,
                    "swings": len(swings_high),
                },
            )
        eql_pairs = _count_pairs_within_tolerance(
            swings_low,
            tolerance=tolerance,
            tick_size=tick_size,
        )
        frame_diag["eql"]["pairs_within_tol_before_cluster"] = eql_pairs
        frame_diag["eql"]["pairs_within_tol"] = eql_pairs
        frame_diag["eql"]["sample_pairs_top10"] = _sample_swing_pairs(swings_low)
        if eql_pairs == 0 and len(swings_low) >= 2:
            LOGGER.debug(
                "No swing low pairs within tolerance prior to clustering",
                extra={
                    "tf": timeframe,
                    "reason": "pairs_within_tol_before_cluster==0",
                    "tick_size": tick_size,
                    "tolerance": tolerance,
                    "swings": len(swings_low),
                },
            )
        LOGGER.debug(
            "Liquidity swings detected",
            extra={
                "tf": timeframe,
                "swing_highs": len(swings_high),
                "swing_lows": len(swings_low),
                "pairs_within_tol_high": frame_diag["eqh"]["pairs_within_tol"],
                "pairs_within_tol_low": frame_diag["eql"]["pairs_within_tol"],
                "used_source": source_label,
            },
        )
        eqh_cluster = _cluster_swings(
            swings_high,
            tolerance=tolerance,
            tick_size=tick_size,
            level_type="eqh",
            timeframe=timeframe,
            reason_sink=frame_diag["eqh"]["reasons"],
        )
        frame_diag["eqh"]["cluster_count"] = len(eqh_cluster)
        if not eqh_cluster:
            LOGGER.debug(
                "No EQH clusters on timeframe",
                extra={"tf": timeframe, "reason": "no_clusters"},
            )
        eqh_levels.extend(eqh_cluster)

        eql_cluster = _cluster_swings(
            swings_low,
            tolerance=tolerance,
            tick_size=tick_size,
            level_type="eql",
            timeframe=timeframe,
            reason_sink=frame_diag["eql"]["reasons"],
        )
        frame_diag["eql"]["cluster_count"] = len(eql_cluster)
        if not eql_cluster:
            LOGGER.debug(
                "No EQL clusters on timeframe",
                extra={"tf": timeframe, "reason": "no_clusters"},
            )
        eql_levels.extend(eql_cluster)

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
                "r_ticks": config.r_ticks,
                "tolerance": tolerance,
                "atr_period": config.atr_period,
                "atr_mult": config.sweep_atr_multiplier,
                "used_source": source_label,
            },
        )

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

    for timeframe in SUPPORTED_TIMEFRAMES:
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        upper_levels = upper_levels_by_tf.get(timeframe, [])
        lower_levels = lower_levels_by_tf.get(timeframe, [])

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
            },
            "lower": {
                "levels": len(lower_levels),
                "events": 0,
                "reasons": [],
                "candidates_before_atr": 0,
                "candidates_after_atr": 0,
                "samples": [],
            },
        }
        diagnostics[timeframe] = frame_diag

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

        for index, candle in enumerate(candles):
            high = _coerce_float(candle.get("h"))
            close = _coerce_float(candle.get("c"))
            ts = candle.get("t")
            if high is None or close is None or not isinstance(ts, (int, float)):
                continue
            high = _quantise(high, tick_size)
            close = _quantise(close, tick_size)
            atr_component = atr_values[index] * config.sweep_atr_multiplier
            atr_limit = max(tick_min, atr_component)
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
                price_cap = level_price * SWEEP_CAP_PCT if level_price > 0 else None
                epsilon = max(atr_limit, tick_min)
                if price_cap is not None and price_cap > 0:
                    epsilon = min(epsilon, max(price_cap, tick_min))
                epsilon_samples.append(epsilon)
                if overshoot <= epsilon:
                    LOGGER.debug(
                        "Skipping sweep candidate due to tolerance",
                        extra={
                            "reason": "tolerance_fail",
                            "tf": timeframe,
                            "level_type": level_type,
                            "overshoot": overshoot,
                            "tolerance": epsilon,
                            "epsilon": epsilon,
                            "used_source": source_label,
                        },
                    )
                    _append_reason(
                        frame_diag["upper"]["reasons"],
                        "tolerance_fail",
                        level_type=level_type,
                        overshoot=overshoot,
                        tolerance=epsilon,
                        epsilon=epsilon,
                    )
                    continue
                frame_diag["upper"]["candidates_before_atr"] += 1
                if len(upper_candidate_samples) < 5:
                    upper_candidate_samples.append(
                        {
                            "level": level_price,
                            "high": high,
                            "close": close,
                            "overshoot": overshoot,
                            "epsilon": epsilon,
                            "atr_limit": atr_limit,
                        }
                    )
                atr_cap = max(epsilon * 2.0, tick_min)
                if overshoot > atr_cap + 1e-9:
                    LOGGER.debug(
                        "Skipping sweep candidate due to ATR breach",
                        extra={
                            "reason": "atr_breach",
                            "tf": timeframe,
                            "level_type": level_type,
                            "overshoot": overshoot,
                            "atr_limit": atr_cap,
                            "epsilon": epsilon,
                            "used_source": source_label,
                        },
                    )
                    _append_reason(
                        frame_diag["upper"]["reasons"],
                        "atr_breach",
                        level_type=level_type,
                        overshoot=overshoot,
                        atr_limit=atr_cap,
                        epsilon=epsilon,
                    )
                    continue
                frame_diag["upper"]["candidates_after_atr"] += 1
                if close >= level_price:
                    _append_reason(
                        frame_diag["upper"]["reasons"],
                        "no_return_close",
                        level_type=level_type,
                        close=close,
                        level=level_price,
                    )
                    continue
                sweeps.append(
                    {
                        "type": "sweep_top",
                        "level_type": level_type,
                        "level_price": level_price,
                        "t": int(ts),
                        "atr_tolerance": epsilon,
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
            low = _coerce_float(candle.get("l"))
            close = _coerce_float(candle.get("c"))
            ts = candle.get("t")
            if low is None or close is None or not isinstance(ts, (int, float)):
                continue
            low = _quantise(low, tick_size)
            close = _quantise(close, tick_size)
            atr_component = atr_values[index] * config.sweep_atr_multiplier
            atr_limit = max(tick_min, atr_component)
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
                price_cap = level_price * SWEEP_CAP_PCT if level_price > 0 else None
                epsilon = max(atr_limit, tick_min)
                if price_cap is not None and price_cap > 0:
                    epsilon = min(epsilon, max(price_cap, tick_min))
                epsilon_samples.append(epsilon)
                if overshoot <= epsilon:
                    LOGGER.debug(
                        "Skipping sweep candidate due to tolerance",
                        extra={
                            "reason": "tolerance_fail",
                            "tf": timeframe,
                            "level_type": level_type,
                            "overshoot": overshoot,
                            "tolerance": epsilon,
                            "epsilon": epsilon,
                            "used_source": source_label,
                        },
                    )
                    _append_reason(
                        frame_diag["lower"]["reasons"],
                        "tolerance_fail",
                        level_type=level_type,
                        overshoot=overshoot,
                        tolerance=epsilon,
                        epsilon=epsilon,
                    )
                    continue
                frame_diag["lower"]["candidates_before_atr"] += 1
                if len(lower_candidate_samples) < 5:
                    lower_candidate_samples.append(
                        {
                            "level": level_price,
                            "low": low,
                            "close": close,
                            "overshoot": overshoot,
                            "epsilon": epsilon,
                            "atr_limit": atr_limit,
                        }
                    )
                atr_cap = max(epsilon * 2.0, tick_min)
                if overshoot > atr_cap + 1e-9:
                    LOGGER.debug(
                        "Skipping sweep candidate due to ATR breach",
                        extra={
                            "reason": "atr_breach",
                            "tf": timeframe,
                            "level_type": level_type,
                            "overshoot": overshoot,
                            "atr_limit": atr_cap,
                            "epsilon": epsilon,
                            "used_source": source_label,
                        },
                    )
                    _append_reason(
                        frame_diag["lower"]["reasons"],
                        "atr_breach",
                        level_type=level_type,
                        overshoot=overshoot,
                        atr_limit=atr_cap,
                        epsilon=epsilon,
                    )
                    continue
                frame_diag["lower"]["candidates_after_atr"] += 1
                if close <= level_price:
                    _append_reason(
                        frame_diag["lower"]["reasons"],
                        "no_return_close",
                        level_type=level_type,
                        close=close,
                        level=level_price,
                    )
                    continue
                sweeps.append(
                    {
                        "type": "sweep_bottom",
                        "level_type": level_type,
                        "level_price": level_price,
                        "t": int(ts),
                        "atr_tolerance": epsilon,
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

    resolved_tick, tick_source = resolve_liquidity_tick_size(
        symbol or normalized_symbol,
        tick_size,
        frame_sequences,
        meta=meta,
        logger=LOGGER,
    )
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

    diagnostics_payload = {
        "config": {
            "swing_window": resolved_config.swing_window,
            "lookback": resolved_config.lookback_swings,
            "r_ticks": resolved_config.r_ticks,
            "atr_period": resolved_config.atr_period,
            "sweep_atr_multiplier": resolved_config.sweep_atr_multiplier,
            "tick_size": resolved_tick,
        },
        "tick_size": {
            "symbol": symbol,
            "normalized_symbol": normalized_symbol or "UNKNOWN",
            "source": tick_source,
            "value": resolved_tick,
        },
        "levels": level_diagnostics,
        "daily": daily_diagnostics,
        "sweeps": sweep_diagnostics,
        "summary": {
            "eqh": len(levels["eqh"]),
            "eql": len(levels["eql"]),
            "sweeps": len(sweeps),
        },
    }

    return {
        "eqh": levels["eqh"],
        "eql": levels["eql"],
        "pdh": daily_levels["pdh"],
        "pdl": daily_levels["pdl"],
        "sweeps": sweeps,
        "diagnostics": diagnostics_payload,
    }

