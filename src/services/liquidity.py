"""Liquidity level and sweep detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import statistics
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import logging

from .ohlc import TIMEFRAME_TO_MS, resample_ohlcv

MS_IN_DAY = 86_400_000
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("15m", "1h")

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LiquidityConfig:
    """Runtime configuration for liquidity detection."""

    swing_window: int = 3
    lookback_swings: int = 30
    r_ticks: int = 4
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
                "used_source": "fallback",
            },
        )
        augmented[target_tf] = {"candles": aggregated, "source": "fallback"}

    return augmented


def _detect_swings(
    candles: Sequence[Mapping[str, Any]],
    *,
    window: int,
    kind: str,
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
            higher = True
            for offset in range(1, window + 1):
                left = _coerce_float(candles[index - offset].get("h"))
                right = _coerce_float(candles[index + offset].get("h"))
                if left is None or right is None or price <= left or price <= right:
                    higher = False
                    break
            if higher:
                swings.append({"t": int(ts), "price": price})
        else:
            price = _coerce_float(candle.get("l"))
            if price is None:
                continue
            lower = True
            for offset in range(1, window + 1):
                left = _coerce_float(candles[index - offset].get("l"))
                right = _coerce_float(candles[index + offset].get("l"))
                if left is None or right is None or price >= left or price >= right:
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
                cluster["anchor"] = statistics.fmean(cluster["prices"])
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
            continue
        prices = cluster["prices"]
        if not prices:
            continue
        level_price = statistics.fmean(prices)
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
    return payload


def _compute_atr_series(
    candles: Sequence[Mapping[str, Any]],
    *,
    period: int,
) -> List[float]:
    atr_values: List[float] = []
    if period <= 0:
        return [0.0 for _ in candles]
    tr_window: List[float] = []
    prev_close: float | None = None
    for candle in candles:
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
        tr_window.append(true_range)
        if len(tr_window) > period:
            tr_window.pop(0)
        atr = sum(tr_window) / len(tr_window) if tr_window else 0.0
        atr_values.append(atr)
        prev_close = close
    return atr_values


def _prepare_levels(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    tick_size: float | None,
    config: LiquidityConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    eqh_levels: List[Dict[str, Any]] = []
    eql_levels: List[Dict[str, Any]] = []

    tolerance = (config.r_ticks * tick_size) if tick_size and tick_size > 0 else 0.0

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

    for timeframe in SUPPORTED_TIMEFRAMES:
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        candle_count = len(candles)
        LOGGER.debug(
            "Evaluating liquidity swings",
            extra={
                "tf": timeframe,
                "candles": candle_count,
                "used_source": source_label,
            },
        )
        if candle_count < 2 * config.swing_window + 1:
            LOGGER.debug(
                "Skipping liquidity timeframe",
                extra={
                    "tf": timeframe,
                    "reason": "too_few_bars",
                    "candles": candle_count,
                    "used_source": source_label,
                },
            )
            continue
        swings_high = _detect_swings(candles, window=config.swing_window, kind="high")
        swings_low = _detect_swings(candles, window=config.swing_window, kind="low")
        if config.lookback_swings > 0:
            swings_high = swings_high[-config.lookback_swings :]
            swings_low = swings_low[-config.lookback_swings :]
        LOGGER.debug(
            "Liquidity swings detected",
            extra={
                "tf": timeframe,
                "swing_highs": len(swings_high),
                "swing_lows": len(swings_low),
                "used_source": source_label,
            },
        )
        eqh_cluster = _cluster_swings(
            swings_high,
            tolerance=tolerance,
            tick_size=tick_size,
            level_type="eqh",
            timeframe=timeframe,
        )
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
        )
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

    return {"eqh": eqh_levels, "eql": eql_levels}


def _resolve_previous_day(
    candles: Sequence[Mapping[str, Any]],
    *,
    reference_end_ms: int | None,
) -> Dict[str, Dict[str, Any] | None]:
    if not candles:
        LOGGER.debug(
            "Unable to resolve previous day levels",
            extra={"reason": "no_daily_candles"},
        )
        return {"pdh": None, "pdl": None}

    if reference_end_ms is None:
        last_ts = candles[-1].get("t")
        reference_end_ms = int(last_ts) + MS_IN_DAY if isinstance(last_ts, (int, float)) else None

    if reference_end_ms is None:
        LOGGER.debug(
            "Unable to resolve previous day levels",
            extra={"reason": "no_reference_time"},
        )
        return {"pdh": None, "pdl": None}

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

    return {"pdh": target_high, "pdl": target_low}


def _detect_sweeps(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    tick_size: float | None,
    config: LiquidityConfig,
    eqh: Sequence[Mapping[str, Any]],
    eql: Sequence[Mapping[str, Any]],
    pdh: Mapping[str, Any] | None,
    pdl: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
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

    for timeframe, levels in upper_levels_by_tf.items():
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        if not candles:
            LOGGER.debug(
                "Skipping sweep evaluation due to empty frame",
                extra={"tf": timeframe, "reason": "no_candles", "used_source": source_label},
            )
            continue
        LOGGER.debug(
            "Evaluating sweep candidates",
            extra={
                "tf": timeframe,
                "direction": "upper",
                "candles": len(candles),
                "levels": len(levels),
                "used_source": source_label,
            },
        )
        atr_values = _compute_atr_series(candles, period=config.atr_period)
        for index, candle in enumerate(candles):
            high = _coerce_float(candle.get("h"))
            close = _coerce_float(candle.get("c"))
            ts = candle.get("t")
            if high is None or close is None or not isinstance(ts, (int, float)):
                continue
            atr_component = atr_values[index] * config.sweep_atr_multiplier
            epsilon = max(tick_min, atr_component)
            min_required = tick_min if tick_min and tick_min > 0 else epsilon
            atr_cap = atr_component if atr_component > 0 else None
            for level in levels:
                level_price = _coerce_float(level.get("price"))
                if level_price is None:
                    continue
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
                if min_required > 0 and overshoot < min_required:
                    LOGGER.debug(
                        "Skipping sweep candidate due to tolerance",
                        extra={
                            "reason": "tolerance_fail",
                            "tf": timeframe,
                            "level_type": level_type,
                            "overshoot": overshoot,
                            "tolerance": min_required,
                            "epsilon": epsilon,
                            "used_source": source_label,
                        },
                    )
                    continue
                if atr_cap is not None and overshoot > atr_cap + 1e-9:
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
                    continue
                if close >= level_price:
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

    for timeframe, levels in lower_levels_by_tf.items():
        frame_payload = frames.get(timeframe)
        source_label = _frame_source(frame_payload) or "unknown"
        candles = _extract_candles(frame_payload)
        if not candles:
            LOGGER.debug(
                "Skipping sweep evaluation due to empty frame",
                extra={"tf": timeframe, "reason": "no_candles", "used_source": source_label},
            )
            continue
        LOGGER.debug(
            "Evaluating sweep candidates",
            extra={
                "tf": timeframe,
                "direction": "lower",
                "candles": len(candles),
                "levels": len(levels),
                "used_source": source_label,
            },
        )
        atr_values = _compute_atr_series(candles, period=config.atr_period)
        for index, candle in enumerate(candles):
            low = _coerce_float(candle.get("l"))
            close = _coerce_float(candle.get("c"))
            ts = candle.get("t")
            if low is None or close is None or not isinstance(ts, (int, float)):
                continue
            atr_component = atr_values[index] * config.sweep_atr_multiplier
            epsilon = max(tick_min, atr_component)
            min_required = tick_min if tick_min and tick_min > 0 else epsilon
            atr_cap = atr_component if atr_component > 0 else None
            for level in levels:
                level_price = _coerce_float(level.get("price"))
                if level_price is None:
                    continue
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
                if min_required > 0 and overshoot < min_required:
                    LOGGER.debug(
                        "Skipping sweep candidate due to tolerance",
                        extra={
                            "reason": "tolerance_fail",
                            "tf": timeframe,
                            "level_type": level_type,
                            "overshoot": overshoot,
                            "tolerance": min_required,
                            "epsilon": epsilon,
                            "used_source": source_label,
                        },
                    )
                    continue
                if atr_cap is not None and overshoot > atr_cap + 1e-9:
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
                    continue
                if close <= level_price:
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

    sweeps.sort(key=lambda item: item.get("t", 0))
    return sweeps


def build_liquidity_snapshot(
    frames: Mapping[str, Mapping[str, Any]],
    *,
    tick_size: float | None,
    selection: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute liquidity levels and sweeps for the inspection payload."""

    resolved_config = _resolve_config(config)

    augmented_frames = _augment_supported_frames(frames)
    levels = _prepare_levels(augmented_frames, tick_size=tick_size, config=resolved_config)

    daily_candles = _extract_candles(augmented_frames.get("1d"))
    selection_end = None
    if isinstance(selection, Mapping):
        end_value = selection.get("end")
        if isinstance(end_value, (int, float)):
            selection_end = int(end_value)
    daily_levels = _resolve_previous_day(daily_candles, reference_end_ms=selection_end)

    sweeps = _detect_sweeps(
        augmented_frames,
        tick_size=tick_size,
        config=resolved_config,
        eqh=levels["eqh"],
        eql=levels["eql"],
        pdh=daily_levels["pdh"],
        pdl=daily_levels["pdl"],
    )

    LOGGER.debug(
        "Liquidity detection summary",
        extra={
            "eqh": len(levels["eqh"]),
            "eql": len(levels["eql"]),
            "sweeps": len(sweeps),
        },
    )

    return {
        "eqh": levels["eqh"],
        "eql": levels["eql"],
        "pdh": daily_levels["pdh"],
        "pdl": daily_levels["pdl"],
        "sweeps": sweeps,
    }

