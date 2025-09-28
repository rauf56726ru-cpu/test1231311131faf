"""Structured zone detection for Fair Value Gaps, Order Blocks and derivatives.

This module implements the zone taxonomy described in the specification for
Задача 6.  The implementation focuses on deterministic, testable logic that can
be evaluated purely from OHLCV candles without relying on external state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .ohlc import TIMEFRAME_TO_MS, resample_ohlcv
from .smc import SMCConfig, detect_smc_blocks

Timeframe = str
Candle = Mapping[str, Any]


@dataclass(slots=True)
class Config:
    """Configuration bundle controlling the detectors."""

    tick_size: float | None = None
    atr_period: int = 14
    displacement_body: float = 1.0
    displacement_range: float = 1.5
    ob_body_max_atr: float = 0.7
    ob_overlap_ratio: float = 0.6
    ob_distance_atr: float = 0.5
    min_block_ratio: float = 0.2
    epsilon_ticks: float = 1.0
    liquidity_window: int = 3
    sr_merge_pct: float = 0.0002


_PIVOT_WINDOWS: Dict[str, int] = {"15m": 2, "1h": 3, "4h": 4}


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


def _pivot_span(tf: str) -> int:
    return _PIVOT_WINDOWS.get(tf, 2)


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
            {"idx": idx, "direction": bos_direction, "t": int(candle["t"]), "price": close_price}
        )
        if trend is None:
            trend = bos_direction
            continue
        if bos_direction != trend:
            choch_events.append(
                {"idx": idx, "direction": bos_direction, "t": int(candle["t"]), "price": close_price}
            )
        trend = bos_direction
    return pivots, bos_events, choch_events


def _fvgs_for_tf(
    candles: Sequence[Candle],
    *,
    tf: str,
    cfg: Config,
    tick_size: float | None,
    atr: Sequence[float],
    bos_events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    if len(candles) < 3:
        return zones
    tick = tick_size or _infer_tick_size(candles)
    epsilon = tick or 0.0
    bos_by_index = {event["idx"]: event for event in bos_events}
    for i in range(len(candles) - 2):
        c0, c1, c2 = candles[i], candles[i + 1], candles[i + 2]
        low_mid = float(c1["l"])
        high_mid = float(c1["h"])
        low_next = float(c2["l"])
        high_prev = float(c0["h"])
        high_next = float(c2["h"])
        low_prev = float(c0["l"])
        atr_value = atr[i + 1] if i + 1 < len(atr) else math.nan
        if not atr_value or math.isnan(atr_value) or atr_value <= 0:
            continue
        body = abs(float(c1["c"]) - float(c1["o"]))
        range_span = float(c1["h"]) - float(c1["l"])
        if body < cfg.displacement_body * atr_value and range_span < cfg.displacement_range * atr_value:
            continue
        direction: str | None = None
        top: float | None = None
        bot: float | None = None
        if low_mid > high_prev + epsilon:
            direction = "up"
            top = low_mid
            bot = high_prev
        elif high_mid < low_prev - epsilon:
            direction = "down"
            top = low_prev
            bot = high_mid
        if direction is None or top is None or bot is None:
            continue
        width = top - bot
        if width <= 0:
            continue
        if tick and width < tick:
            continue
        created_idx = i + 2
        status = "open"
        fulfil_idx: int | None = None
        for j in range(created_idx + 1, len(candles)):
            low = float(candles[j]["l"])
            high = float(candles[j]["h"])
            if direction == "up" and low <= bot:
                status = "fulfilled"
                fulfil_idx = j
                break
            if direction == "down" and high >= top:
                status = "fulfilled"
                fulfil_idx = j
                break
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
                    if body_low <= top and body_high >= bot:
                        if direction == "up" and close_price < bot:
                            inverted = True
                            break
                        if direction == "down" and close_price > top:
                            inverted = True
                            break
                if inverted:
                    break
            if inverted:
                status = "inverted"
        zone = {
            "tf": tf,
            "direction": direction,
            "top": _round_tick(top, tick),
            "bot": _round_tick(bot, tick),
            "mid": _round_tick((top + bot) / 2.0, tick),
            "created_utc": _ms_to_iso(int(c2["t"])),
            "status": status,
        }
        zones.append(zone)
    return zones


def _evaluate_zone_status(
    candles: Sequence[Candle],
    *,
    start_idx: int,
    zone_range: Tuple[float, float],
    zone_type: str,
    tick: float | None,
) -> Tuple[str, List[Tuple[float, float, int]], int | None, int | None]:
    status = "fresh"
    coverage: List[Tuple[float, float, int]] = []
    first_touch: int | None = None
    invalidated_idx: int | None = None
    epsilon = tick or 0.0
    low, high = zone_range
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
        if zone_type == "demand" and close_price < low - epsilon:
            status = "invalidated"
            invalidated_idx = idx
            break
        if zone_type == "supply" and close_price > high + epsilon:
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
    bos_events: Sequence[Mapping[str, Any]],
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
    smc_payload["structure"] = [
        {"kind": "bos", "direction": event["direction"], "t": event["t"], "price": event["price"]}
        for event in bos_events
    ]
    for event in bos_events:
        direction = event["direction"]
        zone_type = "demand" if direction == "up" else "supply"
        bos_idx = event["idx"]
        impulse_idx = bos_idx - 1
        if impulse_idx <= 0:
            continue
        atr_value = atr[impulse_idx] if impulse_idx < len(atr) else math.nan
        if not atr_value or math.isnan(atr_value) or atr_value <= 0:
            continue
        base_idx = None
        for idx in range(max(0, impulse_idx - pivot_span), impulse_idx + 1):
            candle = candles[idx]
            body_low, body_high = _body_range(candle)
            body_span = body_high - body_low
            if body_span <= 0:
                continue
            if body_span > cfg.ob_body_max_atr * atr_value:
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
        if distance < cfg.ob_distance_atr * atr_value:
            continue
        status, coverage, first_touch, invalidated_idx = _evaluate_zone_status(
            candles,
            start_idx=bos_idx,
            zone_range=(zone_low, zone_high),
            zone_type=zone_type,
            tick=tick,
        )
        zone = {
            "tf": tf,
            "type": zone_type,
            "open": _round_tick(zone_low, tick),
            "close": _round_tick(zone_high, tick),
            "mean": _round_tick((zone_low + zone_high) / 2.0, tick),
            "origin_utc": _ms_to_iso(int(candles[bos_idx]["t"])),
            "status": status,
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
    smc_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not smc_data.get("ob"):
        return [], [], []
    structure_flags = smc_data.get("structure") or []
    liquidity_levels = {
        "eqh": smc_data.get("liquidity", {}).get("eqh", []),
        "eql": smc_data.get("liquidity", {}).get("eql", []),
        "pdh": smc_data.get("liquidity", {}).get("pdh", []),
        "pdl": smc_data.get("liquidity", {}).get("pdl", []),
    }
    blocks = detect_smc_blocks(
        candles,
        timeframe=tf,
        structure_flags=structure_flags,
        ob_zones=smc_data.get("ob"),
        liquidity_levels=liquidity_levels,
        config=SMCConfig(min_block_size=0.0),
    )
    mb: List[Dict[str, Any]] = []
    bb: List[Dict[str, Any]] = []
    rb: List[Dict[str, Any]] = []
    for block in blocks:
        kind = block.get("kind")
        range_low, range_high = block.get("range", [0.0, 0.0])[:2]
        entry = {
            "tf": tf,
            "type": block.get("type"),
            "open": float(range_low),
            "close": float(range_high),
            "mean": (float(range_low) + float(range_high)) / 2.0,
            "origin_utc": _ms_to_iso(int(block.get("created_at", 0))),
            "status": block.get("status", "fresh"),
        }
        if kind == "mb":
            mb.append(entry)
        elif kind == "bb":
            bb.append(entry)
        elif kind == "rb":
            rb.append(entry)
    return mb, bb, rb


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
        levels.append({"type": level_type, "price": price, "ts": ts_iso, "valid": True})
    if candles_1d:
        previous = candles_1d[-1]
        levels.append(
            {
                "type": "resistance",
                "price": float(previous["h"]),
                "ts": _ms_to_iso(int(previous["t"])),
                "valid": True,
            }
        )
        levels.append(
            {
                "type": "support",
                "price": float(previous["l"]),
                "ts": _ms_to_iso(int(previous["t"])),
                "valid": True,
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
        else:
            merged.append(level)
    return merged


def _profile_levels(
    profile_refs: Mapping[str, Mapping[str, float]] | None,
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
            levels.append({"type": key, "price": price, "session": session})
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
    frames: Mapping[str, Sequence[Candle]],
    *,
    symbol: str,
    cfg: Config,
    profile_levels: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, Any]:
    timeframes = _ensure_timeframes(frames, ["15m", "1h", "4h", "1d"])
    tick = cfg.tick_size or _infer_tick_size(frames.get("1m", []))
    fvg_all: List[Dict[str, Any]] = []
    ob_all: List[Dict[str, Any]] = []
    mb_all: List[Dict[str, Any]] = []
    bb_all: List[Dict[str, Any]] = []
    rb_all: List[Dict[str, Any]] = []
    pb_all: List[Dict[str, Any]] = []
    sr_levels: List[Dict[str, Any]] = []
    for tf in ("15m", "1h", "4h"):
        candles = timeframes.get(tf, [])
        if len(candles) < 3:
            continue
        atr = compute_atr(candles, cfg.atr_period)
        pivots, bos_events, choch_events = _detect_structure(candles, tf=tf, tick_size=tick, cfg=cfg)
        fvg_all.extend(
            _fvgs_for_tf(
                candles,
                tf=tf,
                cfg=cfg,
                tick_size=tick,
                atr=atr,
                bos_events=bos_events,
            )
        )
        ob_zones, metadata, smc_payload = _ob_for_tf(
            candles,
            tf=tf,
            cfg=cfg,
            tick_size=tick,
            atr=atr,
            bos_events=bos_events,
        )
        ob_all.extend(ob_zones)
        mb, bb, rb = _mb_bb_rb_from_smc(candles, tf=tf, smc_data=smc_payload)
        mb_all.extend(mb)
        bb_all.extend(bb)
        rb_all.extend(rb)
        pb_all.extend(
            _pb_for_tf(
                candles,
                tf=tf,
                cfg=cfg,
                tick_size=tick,
                atr=atr,
                pivots=pivots,
            )
        )
    sr_levels = _sr_levels(timeframes.get("4h", []), timeframes.get("1d", []), cfg=cfg, tick_size=tick)
    payload = {
        "symbol": symbol,
        "zones": {
            "fvg": fvg_all,
            "ob": ob_all,
            "mb": mb_all,
            "bb": bb_all,
            "rb": rb_all,
            "pb": pb_all,
            "sr": sr_levels,
            "profile_levels": _profile_levels(profile_levels),
        },
    }
    return payload
