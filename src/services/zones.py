"""Structured zone detection for Fair Value Gaps, Order Blocks and derivatives.

This module implements the zone taxonomy described in the specification for
Задача 6.  The implementation focuses on deterministic, testable logic that can
be evaluated purely from OHLCV candles without relying on external state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from decimal import Decimal
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .ohlc import TIMEFRAME_TO_MS, resample_ohlcv
from .smc import SMCConfig, detect_smc_blocks

Timeframe = str
Candle = Mapping[str, Any]


@dataclass(slots=True)
class Config:
    """Configuration bundle controlling the detectors."""

    tick_size: float | None = None
    atr_period: int = 14
    k_impulse: float = 0.25
    w_swing: int = 2
    r_zone_pct: float = 0.15
    displacement_body: float = 0.6
    displacement_range: float = 1.1
    displacement_body_floor: float = 0.25
    displacement_range_floor: float = 0.5
    base_min_bars: int = 1
    base_max_bars: int = 4
    base_max_atr: float = 0.8
    base_min_overlap: float = 0.5
    impulse_min_cover: float = 0.6
    ob_body_max_atr: float = 1.0
    ob_overlap_ratio: float = 0.75
    ob_distance_atr: float = 0.25
    min_block_ratio: float = 0.2
    epsilon_ticks: float = 1.0
    liquidity_window: int = 3
    sr_merge_pct: float = 0.0002
    zones_window_start_ms: int | None = None
    window_end_ms_prev_closed: int | None = None
    allow_base_fallback: bool = True
    base_fallback_max_age: int = 200
    base_fallback_max_distance_atr: float = 3.0
    min_gap_atr_ratio: float = 0.08
    min_gap_tick_multiple: float = 2.0
    min_gap_pct: float = 0.0003
    m_wick_atr: float = 3.0


_PIVOT_WINDOWS: Dict[str, int] = {"15m": 2, "1h": 3, "4h": 4}
_WARMUP_REQUIREMENTS: Dict[str, int] = {"15m": 24, "1h": 48, "4h": 12}


def _ms_to_iso(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _infer_tick_size(candles: Sequence[Candle]) -> float | None:
    prices: List[float] = []
    for candle in candles:
        for key in ("o", "h", "l", "c"):
            value = candle.get(key)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price):
                prices.append(price)
    if len(prices) < 2:
        return None
    prices = sorted(set(prices))
    min_diff = math.inf
    for left, right in zip(prices, prices[1:]):
        diff = right - left
        if diff <= 0:
            continue
        if diff < min_diff:
            min_diff = diff
    if not math.isfinite(min_diff) or min_diff <= 0:
        return None
    return float(round(min_diff, 12))


def _round_tick(value: float, tick_size: float | None) -> float:
    if tick_size is None or tick_size <= 0:
        return float(value)
    return round(value / tick_size) * tick_size


def _tick_size_from_price(price: float) -> float | None:
    if not math.isfinite(price) or price <= 0:
        return None
    decimal_price = Decimal(str(price)).normalize()
    exponent = decimal_price.as_tuple().exponent
    decimals = max(0, -exponent)
    tick = 10 ** (-decimals)
    return float(tick)


def _last_price_from_frames(frames: Mapping[str, Sequence[Candle]]) -> float | None:
    if not isinstance(frames, Mapping):
        return None
    ordered_frames = sorted(frames.keys(), key=lambda tf: TIMEFRAME_TO_MS.get(tf, math.inf))
    for tf in ordered_frames:
        candles = frames.get(tf)
        if not candles:
            continue
        for candle in reversed(candles):
            for key in ("c", "o", "h", "l"):
                value = candle.get(key)
                try:
                    price = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(price) and price > 0:
                    return price
    return None


def _resolve_tick_size(cfg: Config, frames: Mapping[str, Sequence[Candle]], timeframes: Mapping[str, Sequence[Candle]]) -> float | None:
    tick = cfg.tick_size
    if tick is not None and tick > 0:
        return float(tick)

    last_price = _last_price_from_frames(frames)
    if last_price is None:
        last_price = _last_price_from_frames(timeframes)

    tick_from_price = _tick_size_from_price(last_price) if last_price is not None else None
    if tick_from_price is not None and tick_from_price > 0:
        cfg.tick_size = float(tick_from_price)
        return float(tick_from_price)

    inferred = _infer_tick_size(frames.get("1m", []))
    if inferred is None or inferred <= 0:
        inferred = _infer_tick_size(timeframes.get("15m", []))
    if inferred is not None and inferred > 0:
        cfg.tick_size = float(inferred)
        return float(inferred)

    return None


def _rolling_return_sigma(
    candles: Sequence[Candle],
    *,
    window: int = 20,
) -> List[float]:
    """Compute rolling standard deviation of percentage returns."""

    if window <= 1 or len(candles) < 2:
        return [math.nan] * len(candles)

    normalised_window = max(2, int(window))
    result: List[float] = [math.nan] * len(candles)
    values: Deque[float | None] = deque()
    window_sum = 0.0
    window_sum_sq = 0.0
    window_count = 0

    for idx in range(len(candles)):
        if idx == 0:
            values.append(None)
            continue

        previous_close = float(candles[idx - 1].get("c", 0.0))
        current_close = float(candles[idx].get("c", 0.0))
        if not math.isfinite(previous_close) or previous_close == 0.0:
            pct_return = math.nan
        else:
            pct_return = (current_close - previous_close) / previous_close
        if math.isfinite(pct_return):
            values.append(pct_return)
            window_sum += pct_return
            window_sum_sq += pct_return * pct_return
            window_count += 1
        else:
            values.append(None)

        if len(values) > normalised_window:
            old = values.popleft()
            if old is not None:
                window_sum -= old
                window_sum_sq -= old * old
                window_count -= 1

        if window_count >= 2:
            mean = window_sum / window_count
            variance = max(0.0, (window_sum_sq / window_count) - mean * mean)
            result[idx] = math.sqrt(variance)
        else:
            result[idx] = math.nan

    return result


def _rolling_std_close(
    candles: Sequence[Candle],
    *,
    window: int = 20,
) -> List[float]:
    """Compute rolling standard deviation of close prices."""

    if window <= 1 or not candles:
        return [math.nan] * len(candles)

    normalised_window = max(2, int(window))
    values: Deque[float | None] = deque()
    result: List[float] = [math.nan] * len(candles)
    window_sum = 0.0
    window_sum_sq = 0.0
    window_count = 0

    for idx, candle in enumerate(candles):
        close_raw = candle.get("c")
        try:
            close_value = float(close_raw)
        except (TypeError, ValueError):
            close_value = math.nan
        if math.isfinite(close_value):
            values.append(close_value)
            window_sum += close_value
            window_sum_sq += close_value * close_value
            window_count += 1
        else:
            values.append(None)

        if len(values) > normalised_window:
            old = values.popleft()
            if old is not None:
                window_sum -= old
                window_sum_sq -= old * old
                window_count -= 1

        if window_count >= normalised_window:
            mean = window_sum / window_count
            variance = max(0.0, (window_sum_sq / window_count) - mean * mean)
            result[idx] = math.sqrt(variance)
        else:
            result[idx] = math.nan

    return result


def _true_range(current: Candle, previous: Candle) -> float:
    high = float(current["h"])
    low = float(current["l"])
    prev_close = float(previous["c"])
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr(candles: Sequence[Candle], period: int) -> List[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not candles:
        return []
    atr: List[float] = [math.nan] * len(candles)
    true_ranges: List[float] = [0.0] * len(candles)
    for i in range(1, len(candles)):
        true_ranges[i] = _true_range(candles[i], candles[i - 1])
    if len(candles) <= period:
        return atr
    window_sum = sum(true_ranges[1 : period + 1])
    atr[period] = window_sum / period
    for i in range(period + 1, len(candles)):
        prev_atr = atr[i - 1]
        atr[i] = (prev_atr * (period - 1) + true_ranges[i]) / period
    return atr


def _body_range(candle: Candle) -> Tuple[float, float]:
    open_price = float(candle.get("o", 0.0))
    close_price = float(candle.get("c", 0.0))
    low, high = (open_price, close_price) if open_price <= close_price else (close_price, open_price)
    return low, high


def _candle_range(candle: Candle) -> Tuple[float, float]:
    low = float(candle.get("l", 0.0))
    high = float(candle.get("h", 0.0))
    return (low, high) if low <= high else (high, low)


def _candle_wicks(candle: Candle) -> Tuple[float, float]:
    """Return the lower and upper wick lengths for a candle."""

    open_price = float(candle.get("o", 0.0))
    close_price = float(candle.get("c", 0.0))
    low = float(candle.get("l", 0.0))
    high = float(candle.get("h", 0.0))
    body_low, body_high = (open_price, close_price)
    if open_price > close_price:
        body_low, body_high = close_price, open_price
    lower_wick = max(0.0, body_low - low)
    upper_wick = max(0.0, high - body_high)
    return lower_wick, upper_wick


def _resolve_fvg_gap(
    prev_candle: Candle,
    next_candle: Candle,
    *,
    cfg: Config,
    atr_value: float,
) -> tuple[str, float, float, str] | None:
    """Determine the dominant gap between the outer candles of a triplet."""

    prev_low_wick, prev_high_wick = _candle_range(prev_candle)
    prev_body_low, prev_body_high = _body_range(prev_candle)
    next_low_wick, next_high_wick = _candle_range(next_candle)
    next_body_low, next_body_high = _body_range(next_candle)

    atr_ok = math.isfinite(atr_value) and atr_value > 0
    wick_limit = None
    if atr_ok and cfg.m_wick_atr and cfg.m_wick_atr > 0:
        wick_limit = cfg.m_wick_atr * float(atr_value)

    if wick_limit is not None:
        prev_lower_wick, prev_upper_wick = _candle_wicks(prev_candle)
        next_lower_wick, next_upper_wick = _candle_wicks(next_candle)
        if prev_upper_wick > wick_limit:
            prev_high_wick = prev_body_high
        if prev_lower_wick > wick_limit:
            prev_low_wick = prev_body_low
        if next_upper_wick > wick_limit:
            next_high_wick = next_body_high
        if next_lower_wick > wick_limit:
            next_low_wick = next_body_low

    bullish_candidates = [
        ("wick_wick", prev_high_wick, next_low_wick),
        ("wick_body", prev_high_wick, next_body_low),
        ("body_wick", prev_body_high, next_low_wick),
        ("body_body", prev_body_high, next_body_low),
    ]
    bearish_candidates = [
        ("wick_wick", next_high_wick, prev_low_wick),
        ("wick_body", next_body_high, prev_low_wick),
        ("body_wick", next_high_wick, prev_body_low),
        ("body_body", next_body_high, prev_body_low),
    ]

    combo_priority = {
        "wick_body": 0,
        "wick_wick": 1,
        "body_wick": 2,
        "body_body": 3,
    }

    def _select_candidate(candidates: list[tuple[str, float, float]]) -> tuple[float, float, str, float] | None:
        best: tuple[float, float, str, float] | None = None
        for label, bottom, top in candidates:
            gap = top - bottom
            if gap <= 0:
                continue
            if best is None:
                best = (bottom, top, label, gap)
                continue
            _, _, best_label, best_gap = best
            if gap < best_gap - 1e-9:
                best = (bottom, top, label, gap)
                continue
            if abs(gap - best_gap) <= 1e-9:
                priority_new = combo_priority.get(label, 99)
                priority_best = combo_priority.get(best_label, 99)
                if priority_new < priority_best:
                    best = (bottom, top, label, gap)
        return best

    bullish_best = _select_candidate(bullish_candidates)
    if bullish_best is not None:
        bottom, top, label, _ = bullish_best
        return "up", bottom, top, label

    bearish_best = _select_candidate(bearish_candidates)
    if bearish_best is not None:
        bottom, top, label, _ = bearish_best
        return "down", bottom, top, label

    return None


def _pivot_span(tf: str) -> int:
    return _PIVOT_WINDOWS.get(tf, 2)


def _series_value(series: Sequence[float] | None, index: int) -> float:
    if not series or index < 0 or index >= len(series):
        return math.nan
    value = series[index]
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(value_f):
        return math.nan
    return value_f


def _combine_atr_series(
    primary: Sequence[float],
    fallback: Sequence[float] | None,
    length: int,
) -> Tuple[List[float], int]:
    combined: List[float] = []
    fallback_used = 0
    for idx in range(length):
        primary_value = _series_value(primary, idx)
        if math.isfinite(primary_value) and primary_value > 0:
            combined.append(primary_value)
            continue
        fallback_value = _series_value(fallback, idx)
        if math.isfinite(fallback_value) and fallback_value > 0:
            combined.append(fallback_value)
            fallback_used += 1
        else:
            combined.append(math.nan)
    return combined, fallback_used


def _epsilon(cfg: Config, tick_size: float | None) -> float:
    if tick_size is None or tick_size <= 0:
        return 0.0
    return cfg.epsilon_ticks * tick_size


def _detect_pivots(candles: Sequence[Candle], span: int) -> List[Dict[str, Any]]:
    if span <= 0 or len(candles) < 2 * span + 1:
        return []
    pivots: List[Dict[str, Any]] = []
    for idx in range(span, len(candles) - span):
        window = candles[idx - span : idx + span + 1]
        center = candles[idx]
        high = float(center["h"])
        low = float(center["l"])
        if all(high >= float(item["h"]) for item in window):
            pivots.append({"type": "ph", "idx": idx, "price": high, "t": int(center["t"])})
        if all(low <= float(item["l"]) for item in window):
            pivots.append({"type": "pl", "idx": idx, "price": low, "t": int(center["t"])})
    pivots.sort(key=lambda item: item["idx"])
    return pivots


def _detect_structure(
    candles: Sequence[Candle],
    *,
    tf: str,
    tick_size: float | None,
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pivots = _detect_pivots(candles, _pivot_span(tf))
    bos_events: List[Dict[str, Any]] = []
    choch_events: List[Dict[str, Any]] = []
    trend: str | None = None
    last_ph: Dict[str, Any] | None = None
    last_pl: Dict[str, Any] | None = None
    epsilon = _epsilon(cfg, tick_size)
    for idx, candle in enumerate(candles):
        for pivot in pivots:
            if pivot["idx"] == idx:
                if pivot["type"] == "ph":
                    last_ph = pivot
                elif pivot["type"] == "pl":
                    last_pl = pivot
        close_price = float(candle.get("c", 0.0))
        bos_direction: str | None = None
        if last_ph and idx > last_ph["idx"] and close_price > last_ph["price"] + epsilon:
            bos_direction = "up"
        if last_pl and idx > last_pl["idx"] and close_price < last_pl["price"] - epsilon:
            bos_direction = "down"
        if bos_direction is None:
            continue
        bos_events.append(
            {
                "idx": idx,
                "direction": bos_direction,
                "t": int(candle["t"]),
                "price": close_price,
                "kind": "bos",
            }
        )
        if trend is None:
            trend = bos_direction
            continue
        if bos_direction != trend:
            choch_events.append(
                {
                    "idx": idx,
                    "direction": bos_direction,
                    "t": int(candle["t"]),
                    "price": close_price,
                    "kind": "choch",
                }
            )
        trend = bos_direction
    return pivots, bos_events, choch_events


def _derive_fvg_reason(stats: Mapping[str, int]) -> str:
    rejections = {
        str(key): int(value)
        for key, value in stats.items()
        if str(key).startswith("fvg_reject_") and int(value) > 0
    }
    if rejections:
        return max(rejections.items(), key=lambda item: item[1])[0]
    if int(stats.get("fvg_triplets", 0)) > 0:
        return "no_valid_zones_found"
    return "no_candidates"


def _fvgs_for_tf(
    candles: Sequence[Candle],
    *,
    tf: str,
    cfg: Config,
    tick_size: float | None,
    atr: Sequence[float],
    atr_reliable: bool,
    atr_fallback: Sequence[float] | None,
    bos_events: Sequence[Mapping[str, Any]],
    returns_sigma: Sequence[float] | None = None,
    stats: MutableMapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    if len(candles) < 3:
        return zones

    tick = tick_size or _infer_tick_size(candles)
    seen_keys: set[tuple[str, float, float]] = set()

    for i in range(len(candles) - 2):
        if stats is not None:
            stats["fvg_triplets"] = stats.get("fvg_triplets", 0) + 1
        c0, c1, c2 = candles[i], candles[i + 1], candles[i + 2]
        impulse_idx = i + 2
        atr_value = _series_value(atr, impulse_idx)
        gap_info = _resolve_fvg_gap(c0, c2, cfg=cfg, atr_value=atr_value)
        if gap_info is None:
            if stats is not None:
                stats["fvg_reject_no_gap"] = stats.get("fvg_reject_no_gap", 0) + 1
            continue

        direction, bot_raw, top_raw, gap_mode = gap_info
        gap_abs = top_raw - bot_raw
        if gap_abs <= 0:
            if stats is not None:
                stats["fvg_reject_no_gap"] = stats.get("fvg_reject_no_gap", 0) + 1
            continue

        if stats is not None:
            mode_key = f"fvg_gap_mode_{gap_mode}"
            stats[mode_key] = stats.get(mode_key, 0) + 1
        atr_component = (
            cfg.min_gap_atr_ratio * atr_value
            if math.isfinite(atr_value) and atr_value > 0
            else 0.0
        )
        tick_component = (
            cfg.min_gap_tick_multiple * tick
            if tick is not None and tick > 0
            else 0.0
        )
        gap_thresholds = [value for value in (atr_component, tick_component) if value > 0]
        gap_min = max(gap_thresholds) if gap_thresholds else 0.0
        price_mid = (top_raw + bot_raw) / 2.0
        price_mid_abs = abs(price_mid)
        pct_ok = price_mid_abs <= 0 or (gap_abs / price_mid_abs) >= cfg.min_gap_pct
        abs_ok = gap_min <= 0 or gap_abs >= gap_min
        if not (abs_ok and pct_ok):
            if stats is not None:
                stats["fvg_reject_no_gap"] = stats.get("fvg_reject_no_gap", 0) + 1
            continue

        if stats is not None:
            stats["fvg_raw_count"] = stats.get("fvg_raw_count", 0) + 1

        atr_valid_value = math.isfinite(atr_value) and atr_value > 0
        if not atr_valid_value and atr_fallback is not None:
            fallback_value = _series_value(atr_fallback, impulse_idx)
            if math.isfinite(fallback_value) and fallback_value > 0:
                atr_value = fallback_value
                atr_valid_value = True
        if atr_reliable and not atr_valid_value:
            if stats is not None:
                stats["fvg_reject_displacement"] = stats.get(
                    "fvg_reject_displacement", 0
                ) + 1
            continue

        sigma_value = (
            returns_sigma[impulse_idx]
            if returns_sigma is not None and impulse_idx < len(returns_sigma)
            else math.nan
        )
        k_body = cfg.displacement_body
        k_range = cfg.displacement_range
        if atr_reliable and atr_valid_value:
            if sigma_value and math.isfinite(sigma_value) and sigma_value < 0.5 * atr_value:
                k_body = max(cfg.displacement_body_floor, k_body - 0.2)
                k_range = max(cfg.displacement_range_floor, k_range - 0.3)

        body1 = abs(float(c1["c"]) - float(c1["o"]))
        body2 = abs(float(c2["c"]) - float(c2["o"]))
        range1 = float(c1["h"]) - float(c1["l"])
        range2 = float(c2["h"]) - float(c2["l"])
        impulse_body = max(body1, body2)
        impulse_range = max(range1, range2)
        if atr_reliable and atr_valid_value:
            meets_primary = (impulse_body >= k_body * atr_value) or (
                impulse_range >= k_range * atr_value
            )
            meets_floor = (
                impulse_body >= cfg.displacement_body_floor * atr_value
                and impulse_range >= cfg.displacement_range_floor * atr_value
            )
            if not (meets_primary or meets_floor):
                if stats is not None:
                    stats["fvg_reject_displacement"] = stats.get(
                        "fvg_reject_displacement", 0
                    ) + 1
                continue

        created_idx = i + 2
        status = "open"
        fulfil_idx: int | None = None
        for j in range(created_idx + 1, len(candles)):
            low = float(candles[j]["l"])
            high = float(candles[j]["h"])
            if direction == "up" and low <= bot_raw:
                status = "fulfilled"
                fulfil_idx = j
                break
            if direction == "down" and high >= top_raw:
                status = "fulfilled"
                fulfil_idx = j
                break
        if fulfil_idx is not None and fulfil_idx <= created_idx:
            if stats is not None:
                stats["fvg_reject_fulfilled_same_leg"] = stats.get(
                    "fvg_reject_fulfilled_same_leg", 0
                ) + 1
            continue

        if fulfil_idx is not None:
            opposite = "down" if direction == "up" else "up"
            inverted = False
            for event in bos_events:
                if event["idx"] <= fulfil_idx:
                    continue
                if event["direction"] != opposite:
                    continue
                for k in range(event["idx"], len(candles)):
                    candle = candles[k]
                    body_low, body_high = _body_range(candle)
                    close_price = float(candle["c"])
                    if body_low <= top_raw and body_high >= bot_raw:
                        if direction == "up" and close_price < bot_raw:
                            inverted = True
                            break
                        if direction == "down" and close_price > top_raw:
                            inverted = True
                            break
                if inverted:
                    break
            if inverted:
                status = "inverted"

        dedup_key = (direction, round(bot_raw, 8), round(top_raw, 8))
        if dedup_key in seen_keys:
            if stats is not None:
                stats["fvg_reject_dedup"] = stats.get("fvg_reject_dedup", 0) + 1
            continue
        seen_keys.add(dedup_key)

        top_value = _round_tick(top_raw, tick)
        bot_value = _round_tick(bot_raw, tick)
        if tick and top_value <= bot_value:
            if stats is not None:
                stats["fvg_reject_tick_collapse"] = stats.get(
                    "fvg_reject_tick_collapse", 0
                ) + 1
            top_value = top_raw
            bot_value = bot_raw
        mid_seed = (top_value + bot_value) / 2.0
        mid_value = _round_tick(mid_seed, tick) if tick else mid_seed

        raw_status = status
        zone_status = raw_status
        if raw_status == "inverted":
            zone_status = "tapped" if fulfil_idx is not None else "open"
        elif raw_status == "fulfilled":
            zone_status = "tapped"

        zone = {
            "tf": tf,
            "direction": direction,
            "top": float(top_value),
            "bot": float(bot_value),
            "mid": float(mid_value),
            "created_utc": _ms_to_iso(int(c2["t"])),
            "status": zone_status,
        }
        if raw_status == "inverted":
            zone["inverted"] = True
        zones.append(zone)
    return zones


def _evaluate_zone_status(
    candles: Sequence[Candle],
    *,
    start_idx: int,
    zone_range: Tuple[float, float],
    zone_type: str,
    tick: float | None,
    atr: Sequence[float] | None = None,
    false_touch_atr_ratio: float = 0.1,
) -> Tuple[str, List[Tuple[float, float, int]], int | None, int | None]:
    status = "fresh"
    coverage: List[Tuple[float, float, int]] = []
    first_touch: int | None = None
    invalidated_idx: int | None = None
    epsilon = tick or 0.0
    low, high = zone_range
    false_touch_grace_used = False

    def _atr_value(index: int) -> float:
        if not atr:
            return math.nan
        if index < len(atr):
            value = atr[index]
        else:
            value = atr[-1]
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return math.nan
        if not math.isfinite(value_f) or value_f <= 0.0:
            return math.nan
        return value_f

    for idx in range(start_idx + 1, len(candles)):
        candle = candles[idx]
        body_low, body_high = _body_range(candle)
        close_price = float(candle["c"])
        overlap_low = max(low, body_low)
        overlap_high = min(high, body_high)
        if overlap_high > overlap_low:
            coverage.append((overlap_low, overlap_high, idx))
            if first_touch is None:
                first_touch = idx
            if body_low >= low and body_high <= high and status == "fresh":
                status = "tapped"
        atr_value = _atr_value(idx)
        breach_allowance = (
            false_touch_atr_ratio * atr_value
            if atr_value and math.isfinite(atr_value)
            else 0.0
        )
        if zone_type == "demand" and close_price < low - epsilon:
            breach = low - close_price
            if (
                not false_touch_grace_used
                and breach_allowance > 0.0
                and breach <= breach_allowance
            ):
                false_touch_grace_used = True
                if status == "fresh":
                    status = "tapped"
                continue
            status = "invalidated"
            invalidated_idx = idx
            break
        if zone_type == "supply" and close_price > high + epsilon:
            breach = close_price - high
            if (
                not false_touch_grace_used
                and breach_allowance > 0.0
                and breach <= breach_allowance
            ):
                false_touch_grace_used = True
                if status == "fresh":
                    status = "tapped"
                continue
            status = "invalidated"
            invalidated_idx = idx
            break
    return status, coverage, first_touch, invalidated_idx


def _merge_segments(segments: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted(segments, key=lambda item: item[0])
    merged: List[Tuple[float, float]] = []
    for low, high in ordered:
        if not merged:
            merged.append((low, high))
            continue
        last_low, last_high = merged[-1]
        if low <= last_high:
            merged[-1] = (last_low, max(last_high, high))
        else:
            merged.append((low, high))
    return merged


def _complement_segments(
    base: Tuple[float, float],
    segments: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not segments:
        return [base]
    merged = _merge_segments(segments)
    low, high = base
    cursor = low
    leftovers: List[Tuple[float, float]] = []
    for seg_low, seg_high in merged:
        if seg_low > cursor:
            leftovers.append((cursor, min(seg_low, high)))
        cursor = max(cursor, seg_high)
    if cursor < high:
        leftovers.append((cursor, high))
    return [item for item in leftovers if item[1] - item[0] > 0]


def _ob_for_tf(
    candles: Sequence[Candle],
    *,
    tf: str,
    cfg: Config,
    tick_size: float | None,
    atr: Sequence[float],
    atr_reliable: bool,
    atr_fallback: Sequence[float] | None,
    bos_events: Sequence[Mapping[str, Any]],
    choch_events: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    raw_metadata: List[Dict[str, Any]] = []
    smc_payload: Dict[str, Any] = {"structure": [], "liquidity": [], "ob": []}
    tick = tick_size or _infer_tick_size(candles)
    epsilon = tick or 0.0
    pivot_span = _pivot_span(tf)
    pivots = _detect_pivots(candles, pivot_span)
    liquidity_levels: Dict[str, List[Dict[str, Any]]] = {"eqh": [], "eql": [], "pdh": [], "pdl": []}
    for pivot in pivots:
        entry = {"price": float(pivot["price"]), "t": int(candles[pivot["idx"]]["t"])}
        if pivot["type"] == "ph":
            liquidity_levels["eqh"].append(entry)
        else:
            liquidity_levels["eql"].append(entry)
    if len(candles) >= 2:
        previous = candles[:-1]
        last_full_day = previous[-1]
        liquidity_levels["pdh"].append({"price": float(last_full_day["h"]), "t": int(last_full_day["t"])})
        liquidity_levels["pdl"].append({"price": float(last_full_day["l"]), "t": int(last_full_day["t"])})
    smc_payload["liquidity"] = liquidity_levels
    window_start = cfg.zones_window_start_ms
    window_end = cfg.window_end_ms_prev_closed
    if candles:
        if window_start is None:
            window_start = int(candles[0]["t"])
        if window_end is None:
            window_end = int(candles[-1]["t"])
        else:
            window_end = min(window_end, int(candles[-1]["t"]))
    structure_events: List[Dict[str, Any]] = []
    for event in list(bos_events) + list(choch_events):
        event_ts = int(event.get("t", 0))
        if window_start is not None and event_ts < window_start:
            continue
        if window_end is not None and event_ts > window_end:
            continue
        structure_events.append(
            {
                "kind": event.get("kind", ""),
                "direction": event.get("direction"),
                "t": event_ts,
                "price": event.get("price"),
                "idx": event.get("idx"),
                "tf": tf,
            }
        )
    structure_events.sort(key=lambda item: item.get("t", 0))
    smc_payload["structure"] = structure_events
    smc_payload["window"] = {"start": window_start, "end": window_end}
    smc_payload["tf"] = tf
    for event in bos_events:
        direction = event["direction"]
        zone_type = "demand" if direction == "up" else "supply"
        bos_idx = event["idx"]
        impulse_idx = bos_idx - 1
        if impulse_idx <= 0:
            continue
        atr_value = _series_value(atr, impulse_idx)
        atr_valid_value = math.isfinite(atr_value) and atr_value > 0
        if not atr_valid_value and atr_fallback is not None:
            fallback_value = _series_value(atr_fallback, impulse_idx)
            if math.isfinite(fallback_value) and fallback_value > 0:
                atr_value = fallback_value
                atr_valid_value = True
        if atr_reliable and not atr_valid_value:
            continue
        base_idx = None
        for idx in range(max(0, impulse_idx - pivot_span), impulse_idx + 1):
            candle = candles[idx]
            body_low, body_high = _body_range(candle)
            body_span = body_high - body_low
            if body_span <= 0:
                continue
            if atr_reliable and atr_valid_value and body_span > cfg.ob_body_max_atr * atr_value:
                continue
            base_idx = idx
        if base_idx is None:
            continue
        base_candle = candles[base_idx]
        body_low, body_high = _body_range(base_candle)
        zone_low = min(body_low, body_high)
        zone_high = max(body_low, body_high)
        if tick and zone_high - zone_low < 2 * tick:
            continue
        bos_close = float(candles[bos_idx]["c"])
        distance = abs(bos_close - (zone_high if zone_type == "supply" else zone_low))
        if atr_reliable and atr_valid_value and distance < cfg.ob_distance_atr * atr_value:
            continue
        status, coverage, first_touch, invalidated_idx = _evaluate_zone_status(
            candles,
            start_idx=bos_idx,
            zone_range=(zone_low, zone_high),
            zone_type=zone_type,
            tick=tick,
            atr=atr,
        )
        zone = {
            "tf": tf,
            "type": zone_type,
            "open": _round_tick(zone_low, tick),
            "close": _round_tick(zone_high, tick),
            "mean": _round_tick((zone_low + zone_high) / 2.0, tick),
            "origin_utc": _ms_to_iso(int(candles[bos_idx]["t"])),
            "status": status,
            "source": "bos",
            "confirmed_by": "bos",
        }
        zones.append(zone)
        raw_metadata.append(
            {
                "tf": tf,
                "type": zone_type,
                "range": (zone_low, zone_high),
                "bos_idx": bos_idx,
                "coverage": coverage,
                "first_touch": first_touch,
                "invalidated_idx": invalidated_idx,
            }
        )
        smc_payload["ob"].append(
            {
                "tf": tf,
                "type": zone_type,
                "range": [zone_low, zone_high],
                "created_at": int(candles[bos_idx]["t"]),
            }
        )
    return zones, raw_metadata, smc_payload


def _mb_bb_rb_from_smc(
    candles: Sequence[Candle],
    *,
    tf: str,
    cfg: Config,
    tick_size: float | None,
    atr: Sequence[float],
    returns_sigma: Sequence[float] | None,
    smc_data: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    ob_series = smc_data.get("ob") or []
    ob_count = len(ob_series) if isinstance(ob_series, Sequence) else 0
    structure_flags = smc_data.get("structure") or []
    liquidity_levels = {
        "eqh": smc_data.get("liquidity", {}).get("eqh", []),
        "eql": smc_data.get("liquidity", {}).get("eql", []),
        "pdh": smc_data.get("liquidity", {}).get("pdh", []),
        "pdl": smc_data.get("liquidity", {}).get("pdl", []),
    }
    diagnostics: Dict[str, Any] = {
        "ob_candidates": ob_count,
        "structure_flags": len(structure_flags)
        if isinstance(structure_flags, Sequence)
        else 0,
        "liquidity_levels": {
            key: len(series) if isinstance(series, Sequence) else 0
            for key, series in liquidity_levels.items()
        },
    }
    if ob_count == 0:
        diagnostics["reason"] = "missing_ob_zones"
        return [], [], [], diagnostics
    displacement_body = cfg.displacement_body
    displacement_range = cfg.displacement_range
    smc_config = SMCConfig(
        min_block_size=0.0,
        ttl_bars=0,
        zones_window_start_ms=cfg.zones_window_start_ms,
        window_end_ms_prev_closed=cfg.window_end_ms_prev_closed,
        allow_base_fallback=cfg.allow_base_fallback,
        base_fallback_max_age=cfg.base_fallback_max_age,
        base_fallback_max_distance_atr=cfg.base_fallback_max_distance_atr,
        base_min_bars=cfg.base_min_bars,
        base_max_bars=cfg.base_max_bars,
        base_max_atr=cfg.base_max_atr,
        base_min_overlap=cfg.base_min_overlap,
        impulse_min_cover=cfg.impulse_min_cover,
    )
    blocks, smc_stats = detect_smc_blocks(
        candles,
        timeframe=tf,
        structure_flags=structure_flags,
        ob_zones=ob_series,
        liquidity_levels=liquidity_levels,
        config=smc_config,
        atr=atr,
        returns_sigma=returns_sigma,
        displacement_body=displacement_body,
        displacement_range=displacement_range,
        tick_size=tick_size,
    )
    diagnostics["smc_blocks"] = len(blocks)
    diagnostics.update({str(key): value for key, value in smc_stats.items()})
    mb: List[Dict[str, Any]] = []
    bb: List[Dict[str, Any]] = []
    rb: List[Dict[str, Any]] = []
    for block in blocks:
        kind = block.get("kind")
        range_low, range_high = block.get("range", [0.0, 0.0])[:2]
        mean_price = (float(range_low) + float(range_high)) / 2.0
        if kind == "rb":
            bot_value = float(block.get("bot", range_low))
            top_value = float(block.get("top", range_high))
            mid_value = block.get("mid")
            if mid_value is None:
                mid_value = (bot_value + top_value) / 2.0
            entry = {
                "tf": tf,
                "direction": block.get("direction")
                or ("up" if block.get("type") == "demand" else "down"),
                "type": block.get("type"),
                "bot": float(bot_value),
                "top": float(top_value),
                "mid": _round_tick(float(mid_value), tick_size),
                "origin_utc": _ms_to_iso(int(block.get("created_at", 0))),
                "status": block.get("status", "fresh"),
            }
            shadow = block.get("shadowed_by")
            if shadow:
                entry["shadowed_by"] = shadow
            rb.append(entry)
            continue

        entry = {
            "tf": tf,
            "type": block.get("type"),
            "open": float(range_low),
            "close": float(range_high),
            "mean": _round_tick(mean_price, tick_size),
            "origin_utc": _ms_to_iso(int(block.get("created_at", 0))),
            "status": block.get("status", "fresh"),
        }
        shadow = block.get("shadowed_by")
        if shadow:
            entry["shadowed_by"] = shadow
        if kind == "mb":
            mb.append(entry)
        elif kind == "bb":
            bb.append(entry)
    diagnostics["mb_count"] = len(mb)
    diagnostics["bb_count"] = len(bb)
    diagnostics["rb_count"] = len(rb)
    if not blocks:
        diagnostics.setdefault("reason", "no_smc_blocks")
    elif not rb:
        existing_reason = diagnostics.get("reason")
        fallback_reason = "no_reversal_blocks"
        diagnostics.setdefault("reason", existing_reason or fallback_reason)
    return mb, bb, rb, diagnostics


def _pb_for_tf(
    candles: Sequence[Candle],
    *,
    tf: str,
    cfg: Config,
    tick_size: float | None,
    atr: Sequence[float],
    pivots: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    tick = tick_size or _infer_tick_size(candles)
    for pivot in pivots:
        idx = pivot["idx"]
        atr_value = atr[idx] if idx < len(atr) else math.nan
        if not atr_value or math.isnan(atr_value) or atr_value <= 0:
            continue
        window = candles[max(0, idx - 1) : min(len(candles), idx + 2)]
        if len(window) < 2:
            continue
        range_low = min(float(candle["l"]) for candle in window)
        range_high = max(float(candle["h"]) for candle in window)
        if range_high - range_low > 0.6 * atr_value:
            continue
        body_lows: List[float] = []
        body_highs: List[float] = []
        for candle in window:
            low, high = _body_range(candle)
            body_lows.append(low)
            body_highs.append(high)
        block_low = min(body_lows)
        block_high = max(body_highs)
        if tick and block_high - block_low < 2 * tick:
            continue
        direction = "demand" if pivot["type"] == "pl" else "supply"
        retest_idx: int | None = None
        for j in range(idx + 1, len(candles)):
            candle = candles[j]
            low, high = _candle_range(candle)
            close_price = float(candle["c"])
            atr_j = atr[j] if j < len(atr) else atr_value
            if high < block_low or low > block_high:
                continue
            if direction == "demand" and close_price - block_high >= 0.5 * atr_j:
                retest_idx = j
                break
            if direction == "supply" and block_low - close_price >= 0.5 * atr_j:
                retest_idx = j
                break
        if retest_idx is None:
            continue
        status, _, _, _ = _evaluate_zone_status(
            candles,
            start_idx=retest_idx,
            zone_range=(block_low, block_high),
            zone_type=direction,
            tick=tick,
            atr=atr,
        )
        blocks.append(
            {
                "tf": tf,
                "type": direction,
                "open": _round_tick(block_low, tick),
                "close": _round_tick(block_high, tick),
                "mean": _round_tick((block_low + block_high) / 2.0, tick),
                "origin_utc": _ms_to_iso(int(candles[retest_idx]["t"])),
                "status": status,
            }
        )
    return blocks


def _sr_levels(
    candles_4h: Sequence[Candle],
    candles_1d: Sequence[Candle],
    *,
    cfg: Config,
    tick_size: float | None,
) -> List[Dict[str, Any]]:
    tick = tick_size or _infer_tick_size(candles_4h)
    epsilon_pct = cfg.sr_merge_pct
    levels: List[Dict[str, Any]] = []
    pivots = _detect_pivots(candles_4h, _pivot_span("4h"))
    for pivot in pivots[-4:]:
        price = float(pivot["price"])
        level_type = "resistance" if pivot["type"] == "ph" else "support"
        ts_iso = _ms_to_iso(int(candles_4h[pivot["idx"]]["t"]))
        levels.append({"type": level_type, "price": price, "ts": ts_iso, "valid": True, "formed_at_utc": ts_iso, "created_utc": ts_iso, "origin_utc": ts_iso, "last_touched_utc": ts_iso})
    if candles_1d:
        previous = candles_1d[-1]
        ts_iso = _ms_to_iso(int(previous["t"]))
        levels.append(
            {
                "type": "resistance",
                "price": float(previous["h"]),
                "ts": ts_iso,
                "valid": True,
                "formed_at_utc": ts_iso,
                "created_utc": ts_iso,
                "origin_utc": ts_iso,
                "last_touched_utc": ts_iso,
            }
        )
        levels.append(
            {
                "type": "support",
                "price": float(previous["l"]),
                "ts": ts_iso,
                "valid": True,
                "formed_at_utc": ts_iso,
                "created_utc": ts_iso,
                "origin_utc": ts_iso,
                "last_touched_utc": ts_iso,
            }
        )
    merged: List[Dict[str, Any]] = []
    for level in sorted(levels, key=lambda item: item["price"]):
        if not merged:
            merged.append(level)
            continue
        prev = merged[-1]
        pct_diff = abs(level["price"] - prev["price"]) / max(level["price"], prev["price"], tick or 1.0)
        if pct_diff <= epsilon_pct and level["type"] == prev["type"]:
            prev["price"] = (prev["price"] + level["price"]) / 2.0
            prev["ts"] = max(prev["ts"], level["ts"])
            prev["last_touched_utc"] = max(prev["last_touched_utc"], level["last_touched_utc"])
        else:
            merged.append(level)
    return merged


def _profile_levels(
    profile_refs: Mapping[str, Mapping[str, float]] | None,
    *,
    fallback_iso: str | None = None,
) -> List[Dict[str, Any]]:
    if not isinstance(profile_refs, Mapping):
        return []
    levels: List[Dict[str, Any]] = []
    for session, values in profile_refs.items():
        if not isinstance(values, Mapping):
            continue
        for key in ("poc", "vah", "val"):
            if key not in values:
                continue
            try:
                price = float(values[key])
            except (TypeError, ValueError):
                continue
            entry: Dict[str, Any] = {"type": key, "price": price, "session": session}
            if fallback_iso:
                entry.setdefault("formed_at_utc", fallback_iso)
                entry.setdefault("created_utc", fallback_iso)
                entry.setdefault("origin_utc", fallback_iso)
                entry.setdefault("last_touched_utc", fallback_iso)
            levels.append(entry)
    return levels


def _ensure_timeframes(
    frames: Mapping[str, Sequence[Candle]],
    required: Sequence[str],
) -> Dict[str, List[Candle]]:
    result: Dict[str, List[Candle]] = {}
    base_1m = list(frames.get("1m", []))
    for tf in required:
        candles = frames.get(tf)
        if candles:
            result[tf] = list(candles)
            continue
        if not base_1m:
            result[tf] = []
            continue
        interval = TIMEFRAME_TO_MS.get(tf)
        if interval is None:
            result[tf] = []
            continue
        aggregated = resample_ohlcv(base_1m, interval)
        result[tf] = aggregated
    return result


def detect_zones(
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Detect price zones from multi-timeframe candle bundles.

    The public interface expects keyword arguments.  A single positional
    argument is still accepted for backwards compatibility and is interpreted as
    the ``frames`` mapping.
    """

    frames_arg: Any | None = None
    if args:
        if len(args) == 1 and "frames" not in kwargs:
            frames_arg = args[0]
        else:  # pragma: no cover - guard for legacy positional usage
            raise TypeError(
                "detect_zones accepts only the candle frames as a positional "
                "argument; all other inputs must be passed by keyword"
            )

    if frames_arg is None:
        frames_arg = kwargs.get("frames")

    if frames_arg is None:
        raise TypeError("detect_zones() missing required argument: 'frames'")

    if isinstance(frames_arg, Mapping):
        frames: Mapping[str, Sequence[Candle]] = frames_arg
    elif isinstance(frames_arg, Sequence):
        frames = {"1m": list(frames_arg)}
    else:
        raise TypeError("frames must be a mapping of timeframe to candle series")

    profile_levels: Mapping[str, Mapping[str, float]] | None = kwargs.get(
        "profile_levels"
    )
    liquidity_levels: Mapping[str, Sequence[Mapping[str, Any]]] | None = kwargs.get(
        "liquidity_levels"
    )
    cfg_arg = kwargs.get("config") or kwargs.get("cfg")
    cfg: Config
    if cfg_arg is None:
        cfg = Config()
    elif isinstance(cfg_arg, Config):
        cfg = cfg_arg
    else:
        raise TypeError("config must be an instance of Config or None")

    timeframes = _ensure_timeframes(frames, ["15m", "1h", "4h", "1d"])
    tick = _resolve_tick_size(cfg, frames, timeframes)
    external_liquidity: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(liquidity_levels, Mapping):
        for key in ("eqh", "eql", "pdh", "pdl"):
            raw_series = liquidity_levels.get(key)
            if not isinstance(raw_series, Sequence):
                continue
            cleaned: List[Dict[str, Any]] = []
            for item in raw_series:
                if not isinstance(item, Mapping):
                    continue
                price = item.get("price")
                timestamp = item.get("ts")
                try:
                    price_value = float(price)
                except (TypeError, ValueError):
                    continue
                entry: Dict[str, Any] = {"price": price_value}
                if timestamp is not None:
                    entry["ts"] = timestamp
                cleaned.append(entry)
            if cleaned:
                external_liquidity[key] = cleaned
    fvg_all: List[Dict[str, Any]] = []
    ob_all: List[Dict[str, Any]] = []
    mb_all: List[Dict[str, Any]] = []
    bb_all: List[Dict[str, Any]] = []
    rb_all: List[Dict[str, Any]] = []
    pb_all: List[Dict[str, Any]] = []
    sr_levels: List[Dict[str, Any]] = []
    fvg_stats: Dict[str, Dict[str, int]] = {}
    diagnostics_timeframes: List[Dict[str, Any]] = []
    for tf in ("15m", "1h", "4h"):
        candles = timeframes.get(tf, [])
        tf_diag: Dict[str, Any] = {"tf": tf, "candles": len(candles)}
        stats_entry = fvg_stats.setdefault(tf, {})
        if len(candles) < 3:
            stats_entry["skipped"] = len(candles)
            reason = "insufficient_candles"
            for key in ("fvg", "ob", "mb", "bb", "rb", "pb"):
                tf_diag[key] = {"count": 0, "reason": reason}
            tf_diag["skipped"] = reason
            diagnostics_timeframes.append(tf_diag)
            continue
        warmup_required = _WARMUP_REQUIREMENTS.get(tf, 0)
        warmup_actual = min(len(candles), warmup_required)
        warmup_ok = warmup_required == 0 or warmup_actual >= warmup_required
        tf_diag["warmup"] = {
            "required": warmup_required,
            "actual": warmup_actual,
            "ok": warmup_ok,
        }
        atr_primary = compute_atr(candles, cfg.atr_period)
        returns_sigma = _rolling_return_sigma(candles, window=20)
        atr_proxy = _rolling_std_close(candles, window=20)
        valid_atr = sum(
            1
            for value in atr_primary
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        )
        potential_atr = max(0, len(candles) - cfg.atr_period)
        coverage_ratio = (
            valid_atr / potential_atr if potential_atr > 0 else 0.0
        )
        atr_reliable = potential_atr > 0 and coverage_ratio >= 0.8
        atr_effective, atr_proxy_count = _combine_atr_series(
            atr_primary,
            atr_proxy,
            len(candles),
        )
        tf_diag["atr"] = {
            "length": len(atr_primary),
            "valid_values": valid_atr,
            "coverage_ratio": coverage_ratio,
            "reliable": atr_reliable,
            "fallback_replacements": atr_proxy_count,
        }
        if not atr_reliable:
            tf_diag["atr"]["proxy"] = "stdev_close_20"
        pivots, bos_events, choch_events = _detect_structure(candles, tf=tf, tick_size=tick, cfg=cfg)
        tf_window_start = cfg.zones_window_start_ms
        tf_window_end = cfg.window_end_ms_prev_closed
        if candles:
            first_ts = int(candles[0]["t"])
            last_ts = int(candles[-1]["t"])
            if tf_window_start is None:
                tf_window_start = first_ts
            if tf_window_end is None:
                tf_window_end = last_ts
            else:
                tf_window_end = min(tf_window_end, last_ts)
        pivots_in_window: List[Dict[str, Any]] = []
        if pivots:
            for pivot in pivots:
                idx = int(pivot.get("idx", -1))
                if idx < 0 or idx >= len(candles):
                    continue
                pivot_ts = int(candles[idx]["t"])
                if tf_window_start is not None and pivot_ts < tf_window_start:
                    continue
                if tf_window_end is not None and pivot_ts > tf_window_end:
                    continue
                pivots_in_window.append(dict(pivot))
        bos_in_window: List[Dict[str, Any]] = []
        for event in bos_events:
            event_ts = int(event.get("t", 0))
            if tf_window_start is not None and event_ts < tf_window_start:
                continue
            if tf_window_end is not None and event_ts > tf_window_end:
                continue
            bos_in_window.append(event)
        choch_in_window: List[Dict[str, Any]] = []
        for event in choch_events:
            event_ts = int(event.get("t", 0))
            if tf_window_start is not None and event_ts < tf_window_start:
                continue
            if tf_window_end is not None and event_ts > tf_window_end:
                continue
            choch_in_window.append(event)
        bos_up_count = sum(1 for event in bos_in_window if event.get("direction") == "up")
        bos_down_count = sum(1 for event in bos_in_window if event.get("direction") == "down")
        tf_diag["structure"] = {
            "pivots": len(pivots_in_window),
            "bos": len(bos_in_window),
            "bos_up": bos_up_count,
            "bos_down": bos_down_count,
            "choch": len(choch_in_window),
        }
        tf_diag["structure_diag"] = {
            "tf": tf,
            "pivots": len(pivots_in_window),
            "bos_up": bos_up_count,
            "bos_down": bos_down_count,
            "choch": len(choch_in_window),
        }
        for key in (
            "fvg_triplets",
            "fvg_raw_count",
            "fvg_reject_no_gap",
            "fvg_reject_displacement",
            "fvg_reject_fulfilled_same_leg",
            "fvg_reject_tick_collapse",
            "fvg_reject_dedup",
        ):
            stats_entry.setdefault(key, 0)
        fvgs_tf = _fvgs_for_tf(
            candles,
            tf=tf,
            cfg=cfg,
            tick_size=tick,
            atr=atr_effective,
            atr_reliable=atr_reliable,
            atr_fallback=atr_proxy,
            bos_events=bos_events,
            returns_sigma=returns_sigma,
            stats=stats_entry,
        )
        stats_copy = {str(key): int(value) for key, value in stats_entry.items()}
        fvg_entry: Dict[str, Any] = {"count": len(fvgs_tf), "stats": stats_copy}
        if not fvgs_tf:
            fvg_entry["reason"] = _derive_fvg_reason(stats_entry)
        tf_diag["fvg"] = fvg_entry
        fvg_all.extend(fvgs_tf)
        ob_zones, metadata, smc_payload = _ob_for_tf(
            candles,
            tf=tf,
            cfg=cfg,
            tick_size=tick,
            atr=atr_effective,
            atr_reliable=atr_reliable,
            atr_fallback=atr_proxy,
            bos_events=bos_events,
            choch_events=choch_events,
        )
        ob_entry: Dict[str, Any] = {"count": len(ob_zones)}
        if not ob_zones:
            ob_entry["reason"] = "no_bos_events" if not bos_events else "no_order_blocks"
        tf_diag["ob"] = ob_entry
        ob_all.extend(ob_zones)
        if external_liquidity:
            combined = dict(smc_payload.get("liquidity", {}))
            for key, items in external_liquidity.items():
                existing = list(combined.get(key, []))
                existing.extend(items)
                combined[key] = existing
            smc_payload["liquidity"] = combined
        mb, bb, rb, smc_diag = _mb_bb_rb_from_smc(
            candles,
            tf=tf,
            cfg=cfg,
            tick_size=tick,
            atr=atr_effective,
            returns_sigma=returns_sigma,
            smc_data=smc_payload,
        )
        for key in (
            "rb_raw_count",
            "rb_reject_no_eq",
            "rb_reject_no_sweep",
            "rb_reject_no_choch",
            "rb_reject_no_base",
            "rb_reject_no_impulse",
        ):
            smc_diag.setdefault(key, 0)
        smc_diag.setdefault("rb_flow", {})
        smc_diag.setdefault("rb_reject", {})
        smc_diag.setdefault("base_tick_collapse", 0)
        smc_diag.setdefault("base_fallback_used", False)
        smc_diag.setdefault("rb_impulse_ok", False)
        tf_diag["smc"] = smc_diag
        tf_diag["mb"] = {"count": len(mb)}
        if not mb and smc_diag.get("reason"):
            tf_diag["mb"]["reason"] = smc_diag["reason"]
        tf_diag["bb"] = {"count": len(bb)}
        if not bb and smc_diag.get("reason"):
            tf_diag["bb"]["reason"] = smc_diag["reason"]
        rb_stats = {
            key: int(smc_diag.get(key, 0))
            for key in (
                "rb_raw_count",
                "rb_reject_no_eq",
                "rb_reject_no_sweep",
                "rb_reject_no_choch",
                "rb_reject_no_base",
                "rb_reject_no_impulse",
            )
        }
        rb_entry: Dict[str, Any] = {"count": len(rb), "stats": rb_stats}
        if smc_diag.get("reason"):
            rb_entry["reason"] = smc_diag["reason"]
        flow_diag = smc_diag.get("rb_flow")
        if isinstance(flow_diag, Mapping):
            rb_entry["flow"] = {str(k): int(v) for k, v in flow_diag.items() if isinstance(v, (int, float))}
        reject_diag = smc_diag.get("rb_reject")
        if isinstance(reject_diag, Mapping):
            rb_entry["reject"] = {str(k): int(v) for k, v in reject_diag.items() if isinstance(v, (int, float))}
        rb_entry["base_fallback_used"] = bool(smc_diag.get("base_fallback_used"))
        rb_entry["impulse_ok"] = bool(smc_diag.get("rb_impulse_ok"))
        rb_entry["base_tick_collapse"] = int(smc_diag.get("base_tick_collapse", 0))
        tf_diag["rb"] = rb_entry
        mb_all.extend(mb)
        bb_all.extend(bb)
        rb_all.extend(rb)
        pb_tf = _pb_for_tf(
            candles,
            tf=tf,
            cfg=cfg,
            tick_size=tick,
            atr=atr_effective,
            pivots=pivots,
        )
        tf_diag["pb"] = {"count": len(pb_tf)}
        if not pb_tf:
            tf_diag["pb"]["reason"] = "no_pivots" if not pivots else "no_pivot_blocks"
        pb_all.extend(pb_tf)
        diagnostics_timeframes.append(tf_diag)
    sr_levels = _sr_levels(timeframes.get("4h", []), timeframes.get("1d", []), cfg=cfg, tick_size=tick)
    sr_reason: str | None = None
    if not sr_levels:
        if len(timeframes.get("4h", [])) < 3 and len(timeframes.get("1d", [])) < 3:
            sr_reason = "insufficient_candles"
    def _collect_reasons(zone_key: str) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        for frame_diag in diagnostics_timeframes:
            entry = frame_diag.get(zone_key)
            if not isinstance(entry, Mapping):
                continue
            if int(entry.get("count", 0)) > 0:
                continue
            reason_value = entry.get("reason")
            if reason_value:
                reasons.append({"tf": frame_diag.get("tf"), "reason": reason_value})
        return reasons

    profile_level_fallback: str | None = None
    if cfg.window_end_ms_prev_closed is not None:
        profile_level_fallback = _ms_to_iso(int(cfg.window_end_ms_prev_closed))
    else:
        for tf_candidate in ("1h", "4h", "15m", "1d"):
            candidate = timeframes.get(tf_candidate)
            if candidate:
                profile_level_fallback = _ms_to_iso(int(candidate[-1]["t"]))
                break

    diagnostics_summary: Dict[str, Any] = {}
    zone_collections = {
        "fvg": fvg_all,
        "ob": ob_all,
        "mb": mb_all,
        "bb": bb_all,
        "rb": rb_all,
        "pb": pb_all,
    }
    for zone_key, series in zone_collections.items():
        entry: Dict[str, Any] = {"count": len(series)}
        reasons = _collect_reasons(zone_key)
        if not series and reasons:
            entry["reasons"] = reasons
        diagnostics_summary[zone_key] = entry
    sr_entry: Dict[str, Any] = {"count": len(sr_levels)}
    if not sr_levels and sr_reason:
        sr_entry["reasons"] = [{"tf": "sr", "reason": sr_reason}]
    diagnostics_summary["sr"] = sr_entry

    payload = {
        "zones": {
            "fvg": fvg_all,
            "ob": ob_all,
            "mb": mb_all,
            "bb": bb_all,
            "rb": rb_all,
            "pb": pb_all,
            "sr": sr_levels,
            "profile_levels": _profile_levels(profile_levels, fallback_iso=profile_level_fallback),
        }
    }
    payload["meta"] = {
        "fvg_stats": fvg_stats,
        "diagnostics": {
            "timeframes": diagnostics_timeframes,
            "summary": diagnostics_summary,
        },
    }
    warmup_overview: Dict[str, Any] = {}
    for frame_diag in diagnostics_timeframes:
        tf_key = frame_diag.get("tf")
        warmup_entry = frame_diag.get("warmup")
        if tf_key and isinstance(warmup_entry, Mapping):
            warmup_overview[str(tf_key)] = {
                "required": int(warmup_entry.get("required", 0)),
                "actual": int(warmup_entry.get("actual", 0)),
                "ok": bool(warmup_entry.get("ok")),
            }
    if warmup_overview:
        payload["meta"]["diagnostics"]["warmup"] = warmup_overview
    return payload
