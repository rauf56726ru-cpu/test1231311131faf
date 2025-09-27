"""Detection utilities for price zones such as FVG, OB and CISD."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Sequence

from .ohlc import normalise_ohlcv


@dataclass(slots=True)
class Config:
    """Configuration parameters controlling zone detection."""

    min_gap_pct: float | None = None
    tick_size: float | None = None
    atr_period: int = 14
    k_impulse: float = 1.5
    w_swing: int = 3
    r_zone_pct: float = 0.15
    m_wick_atr: float = 0.5


def _default_min_gap_pct(tf: str) -> float:
    tf_key = str(tf or "").lower()
    if tf_key == "1m":
        return 0.0001
    if tf_key == "5m":
        return 0.0005
    return 0.02


def _round_tick(value: float, tick_size: float | None) -> float:
    if tick_size is None or not math.isfinite(value):
        return float(value)
    if tick_size <= 0:
        return float(value)
    return round(value / tick_size) * tick_size


def _infer_tick_size(candles: Sequence[Mapping[str, Any]]) -> float | None:
    """Infer the minimal price increment from the candle series."""

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

    tick = round(min_diff, 12)
    if tick <= 0:
        tick = float(min_diff)
    return float(tick)


def _true_range(current: Mapping[str, float], previous: Mapping[str, float]) -> float:
    high = current["h"]
    low = current["l"]
    prev_close = previous["c"]
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr(candles: Sequence[Mapping[str, Any]], period: int) -> List[float]:
    """Compute Wilder's ATR for the provided candles.

    Returns a list of ATR values aligned with the input candles. Positions with
    insufficient history are filled with ``math.nan``.
    """

    if period <= 0:
        raise ValueError("period must be positive")
    if not candles:
        return []

    atr: List[float] = [math.nan] * len(candles)
    true_ranges: List[float] = [0.0] * len(candles)
    for i in range(1, len(candles)):
        true_ranges[i] = _true_range(candles[i], candles[i - 1])

    window_sum = sum(true_ranges[1 : period + 1]) if len(candles) > period else 0.0
    if len(candles) > period:
        atr[period] = window_sum / period
    else:
        return atr

    for i in range(period + 1, len(candles)):
        prev_atr = atr[i - 1]
        atr[i] = (prev_atr * (period - 1) + true_ranges[i]) / period

    return atr


def _calc_gap_pct(bot: float, top: float) -> float:
    mid = (bot + top) / 2
    if mid == 0:
        return 0.0
    return (top - bot) / mid


def _fvg_merge_threshold(tick_size: float | None) -> float:
    if tick_size is None:
        return 0.0
    return abs(tick_size)


def _overlaps(a_bot: float, a_top: float, b_bot: float, b_top: float, *, threshold: float) -> bool:
    return max(a_bot, b_bot) <= min(a_top, b_top) + threshold


def detect_fvg(candles: Sequence[Mapping[str, Any]], cfg: Config, tf: str) -> List[Dict[str, Any]]:
    normalised = list(candles)
    inferred_tick = _infer_tick_size(normalised)
    effective_cfg = replace(
        cfg,
        tick_size=cfg.tick_size or inferred_tick,
        min_gap_pct=(
            cfg.min_gap_pct if cfg.min_gap_pct is not None else _default_min_gap_pct(tf)
        ),
    )
    zones, _ = _detect_fvg_internal(normalised, effective_cfg, tf)
    return zones


def _detect_fvg_internal(
    candles: Sequence[Mapping[str, Any]], cfg: Config, tf: str
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(candles) < 3:
        return [], []

    min_gap_pct = (
        cfg.min_gap_pct if cfg.min_gap_pct is not None else _default_min_gap_pct(tf)
    )
    inferred_tick = _infer_tick_size(candles)
    effective_tick = cfg.tick_size or inferred_tick
    threshold = _fvg_merge_threshold(effective_tick)
    raw: List[Dict[str, Any]] = []
    for i in range(1, len(candles) - 1):
        prev_candle = candles[i - 1]
        mid_candle = candles[i]
        next_candle = candles[i + 1]

        bull_gap = next_candle["l"] > prev_candle["h"]
        bear_gap = next_candle["h"] < prev_candle["l"]

        if bull_gap:
            bot = prev_candle["h"]
            top = next_candle["l"]
            gap_pct = _calc_gap_pct(bot, top)
            width = top - bot
            if width <= 0:
                continue
            passes_pct = gap_pct >= min_gap_pct if min_gap_pct is not None else True
            passes_tick = (
                effective_tick is not None
                and width + 1e-12 >= max(effective_tick, 0.0)
            )
            if not passes_pct and not passes_tick:
                continue
            raw.append(
                {
                    "dir": "up",
                    "bot": bot,
                    "top": top,
                    "created_at": mid_candle["t"],
                    "tf": tf,
                    "indices": [i],
                }
            )
        elif bear_gap:
            bot = next_candle["h"]
            top = prev_candle["l"]
            gap_pct = _calc_gap_pct(bot, top)
            width = top - bot
            if width <= 0:
                continue
            passes_pct = gap_pct >= min_gap_pct if min_gap_pct is not None else True
            passes_tick = (
                effective_tick is not None
                and width + 1e-12 >= max(effective_tick, 0.0)
            )
            if not passes_pct and not passes_tick:
                continue
            raw.append(
                {
                    "dir": "down",
                    "bot": bot,
                    "top": top,
                    "created_at": mid_candle["t"],
                    "tf": tf,
                    "indices": [i],
                }
            )

    if not raw:
        return [], []

    raw.sort(key=lambda item: item["created_at"])

    merged: List[Dict[str, Any]] = []
    for zone in raw:
        if not merged:
            merged.append(zone)
            continue
        last = merged[-1]
        if last["dir"] == zone["dir"] and _overlaps(
            last["bot"], last["top"], zone["bot"], zone["top"], threshold=threshold
        ):
            last["bot"] = min(last["bot"], zone["bot"])
            last["top"] = max(last["top"], zone["top"])
            last["created_at"] = min(last["created_at"], zone["created_at"])
            last["indices"].extend(zone["indices"])
        else:
            merged.append(zone)

    for zone in merged:
        zone["indices"] = sorted(set(zone["indices"]))
        zone["fvl"] = (zone["bot"] + zone["top"]) / 2
        zone["bot"] = _round_tick(zone["bot"], effective_tick)
        zone["top"] = _round_tick(zone["top"], effective_tick)
        zone["fvl"] = _round_tick(zone["fvl"], effective_tick)

    for zone in merged:
        created_idx = min(zone["indices"])
        status = "open"
        bot = zone["bot"]
        top = zone["top"]
        for j in range(created_idx + 1, len(candles)):
            high = candles[j]["h"]
            low = candles[j]["l"]
            if high >= top and low <= bot:
                status = "closed"
                zone["closed_at"] = candles[j]["t"]
                break
        zone["status"] = status

    export: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    for zone in merged:
        export_zone = {
            "tf": tf,
            "dir": zone["dir"],
            "top": zone["top"],
            "bot": zone["bot"],
            "fvl": zone["fvl"],
            "created_at": zone["created_at"],
            "status": zone["status"],
        }
        export.append(export_zone)
        metadata.append(
            {
                "dir": zone["dir"],
                "bot": zone["bot"],
                "top": zone["top"],
                "created_at": zone["created_at"],
                "created_idx": min(zone["indices"]),
                "closed_at": zone.get("closed_at"),
            }
        )

    return export, metadata


def detect_ob(candles: Sequence[Mapping[str, Any]], cfg: Config, tf: str) -> List[Dict[str, Any]]:
    normalised = list(candles)
    zones, _ = _detect_ob_internal(normalised, cfg, tf)
    return zones


def _detect_ob_internal(
    candles: Sequence[Mapping[str, Any]], cfg: Config, tf: str
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(candles) < 2:
        return [], []

    atr = compute_atr(candles, cfg.atr_period)
    tick = cfg.tick_size
    zones: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []

    for i in range(len(candles) - 1):
        base = candles[i]
        impulse = candles[i + 1]
        atr_value = atr[i + 1]
        if math.isnan(atr_value) or atr_value == 0:
            continue

        impulse_range = impulse["h"] - impulse["l"]
        gap_down = impulse["o"] < base["l"]
        gap_up = impulse["o"] > base["h"]

        if base["c"] > base["o"] and (
            (impulse["c"] < impulse["o"] and impulse_range > cfg.k_impulse * atr_value)
            or gap_down
        ):
            zone_type = "supply"
        elif base["c"] < base["o"] and (
            (impulse["c"] > impulse["o"] and impulse_range > cfg.k_impulse * atr_value)
            or gap_up
        ):
            zone_type = "demand"
        else:
            continue

        zone_low = _round_tick(base["l"], tick)
        zone_high = _round_tick(base["h"], tick)

        status = _ob_status(
            candles,
            start_index=i,
            low=zone_low,
            high=zone_high,
            zone_type=zone_type,
            tick_size=tick,
        )

        zone_dict = {
            "tf": tf,
            "type": zone_type,
            "range": [zone_low, zone_high],
            "status": status,
            "created_at": base["t"],
        }
        zones.append(zone_dict)
        metadata.append(
            {
                "type": zone_type,
                "low": zone_low,
                "high": zone_high,
                "created_at": base["t"],
                "created_idx": i,
                "status": status,
            }
        )

    return zones, metadata


def _ob_status(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
    low: float,
    high: float,
    zone_type: str,
    tick_size: float | None,
) -> str:
    touched = False
    for j in range(start_index + 1, len(candles)):
        candle = candles[j]
        high_j = candle["h"]
        low_j = candle["l"]
        if zone_type == "supply":
            invert_trigger = _breaks_above(candle["c"], high, tick_size)
        else:
            invert_trigger = _breaks_below(candle["c"], low, tick_size)
        if invert_trigger:
            return "inverted"
        if min(high_j, high) >= max(low_j, low):
            touched = True
    return "tapped" if touched else "open"


def _breaks_above(close: float, boundary: float, tick: float | None) -> bool:
    if tick is None:
        return close > boundary
    return close >= boundary + tick


def _breaks_below(close: float, boundary: float, tick: float | None) -> bool:
    if tick is None:
        return close < boundary
    return close <= boundary - tick


def compute_swings(candles: Sequence[Mapping[str, Any]], w: int) -> List[Dict[str, Any]]:
    normalised = list(candles)
    if w <= 0 or len(normalised) < 2 * w + 1:
        return []

    swings: List[Dict[str, Any]] = []
    for idx in range(w, len(normalised) - w):
        window = normalised[idx - w : idx + w + 1]
        center = normalised[idx]
        is_high = all(center["h"] > candle["h"] for candle in window if candle is not center)
        is_low = all(center["l"] < candle["l"] for candle in window if candle is not center)
        if is_high:
            swings.append({"idx": idx, "t": center["t"], "type": "high", "price": center["h"]})
        if is_low:
            swings.append({"idx": idx, "t": center["t"], "type": "low", "price": center["l"]})

    swings.sort(key=lambda item: item["idx"])
    return swings


def detect_inducement(
    candles: Sequence[Mapping[str, Any]],
    zones_fvg: Sequence[Mapping[str, Any]],
    zones_ob: Sequence[Mapping[str, Any]],
    cfg: Config,
    tf: str,
) -> List[Dict[str, Any]]:
    normalised = list(candles)
    if not normalised:
        return []

    atr = compute_atr(normalised, cfg.atr_period)
    swings = compute_swings(normalised, cfg.w_swing)
    if not swings:
        return []

    time_to_index = {candle["t"]: idx for idx, candle in enumerate(normalised)}

    zone_infos: List[Dict[str, Any]] = []
    for zone in zones_fvg:
        created_idx = time_to_index.get(zone["created_at"])
        if created_idx is None:
            continue
        zone_infos.append(
            {
                "type": "fvg",
                "dir": zone["dir"],
                "bot": zone["bot"],
                "top": zone["top"],
                "created_idx": created_idx,
            }
        )
    for zone in zones_ob:
        created_at = zone.get("created_at")
        if created_at is None:
            continue
        created_idx = time_to_index.get(created_at)
        if created_idx is None:
            continue
        zone_infos.append(
            {
                "type": zone["type"],
                "bot": zone["range"][0],
                "top": zone["range"][1],
                "created_idx": created_idx,
            }
        )

    if not zone_infos:
        return []

    for info in zone_infos:
        info["first_touch"] = _zone_first_touch(
            normalised,
            info["created_idx"],
            info["bot"],
            info["top"],
        )

    results: List[Dict[str, Any]] = []
    for swing in swings:
        idx = swing["idx"]
        if idx >= len(atr) or math.isnan(atr[idx]):
            continue
        candidate = _find_nearest_zone(
            zone_infos,
            swing,
            atr_value=atr[idx],
            candles=normalised,
            cfg=cfg,
        )
        if candidate is None:
            continue
        zone_info, boundary = candidate
        zone_height = zone_info["top"] - zone_info["bot"]
        price = boundary

        close = normalised[idx]["c"]
        first_touch = zone_info.get("first_touch")
        if zone_info["bot"] <= close <= zone_info["top"]:
            zone_type = "inside"
        elif first_touch is None or idx <= first_touch:
            zone_type = "before"
        else:
            zone_type = "after"

        results.append({"tf": tf, "type": zone_type, "price": price})

    return results


def _zone_first_touch(
    candles: Sequence[Mapping[str, Any]], start_idx: int, bot: float, top: float
) -> int | None:
    for j in range(start_idx + 1, len(candles)):
        candle = candles[j]
        if min(candle["h"], top) >= max(candle["l"], bot):
            return j
    return None


def _find_nearest_zone(
    zone_infos: Sequence[Mapping[str, Any]],
    swing: Mapping[str, Any],
    *,
    atr_value: float,
    candles: Sequence[Mapping[str, Any]],
    cfg: Config,
) -> tuple[Mapping[str, Any], float] | None:
    idx = swing["idx"]
    candle = candles[idx]
    high = candle["h"]
    low = candle["l"]
    close = candle["c"]

    best_zone: Mapping[str, Any] | None = None
    best_boundary: float | None = None
    best_distance = math.inf

    for zone in zone_infos:
        if zone["created_idx"] > idx:
            continue
        bot = zone["bot"]
        top = zone["top"]
        zone_height = top - bot
        boundary_candidates: List[float] = []
        if swing["type"] == "high":
            if high >= top:
                boundary_candidates.append(top)
            if high >= bot:
                boundary_candidates.append(bot)
        else:
            if low <= bot:
                boundary_candidates.append(bot)
            if low <= top:
                boundary_candidates.append(top)

        for boundary in boundary_candidates:
            distance = abs(swing["price"] - boundary)
            if zone_height <= 0:
                if cfg.tick_size is None or distance > cfg.tick_size:
                    continue
            else:
                if distance > cfg.r_zone_pct * zone_height:
                    continue

            wick_ok = False
            threshold = cfg.m_wick_atr * atr_value
            if swing["type"] == "high":
                if high >= boundary and high - boundary <= threshold and close <= boundary:
                    wick_ok = True
            else:
                if low <= boundary and boundary - low <= threshold and close >= boundary:
                    wick_ok = True
            if not wick_ok:
                continue

            if distance < best_distance:
                best_distance = distance
                best_zone = zone
                best_boundary = boundary

    if best_zone is None or best_boundary is None:
        return None
    return best_zone, best_boundary


def detect_cisd(
    candles: Sequence[Mapping[str, Any]],
    zones_fvg: Sequence[Mapping[str, Any]],
    swings: Sequence[Mapping[str, Any]],
    cfg: Config,
    tf: str,
) -> List[Dict[str, Any]]:
    normalised = list(candles)
    if not normalised:
        return []

    if not swings:
        swings = compute_swings(normalised, cfg.w_swing)

    if not swings:
        return []

    time_to_index = {candle["t"]: idx for idx, candle in enumerate(normalised)}
    fvg_meta = []
    for zone in zones_fvg:
        created_idx = time_to_index.get(zone["created_at"])
        if created_idx is None:
            continue
        fvg_meta.append(
            {
                "dir": zone["dir"],
                "bot": zone["bot"],
                "top": zone["top"],
                "created_idx": created_idx,
                "status": zone["status"],
            }
        )

    if not fvg_meta:
        return []

    closed_bear_zones = [
        meta for meta in fvg_meta if meta["dir"] == "down" and meta["status"] == "closed"
    ]
    closed_bull_zones = [
        meta for meta in fvg_meta if meta["dir"] == "up" and meta["status"] == "closed"
    ]

    if not closed_bear_zones and not closed_bull_zones:
        return []

    tick = cfg.tick_size
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    swing_idx_pointer = 0
    swings_sorted = sorted(swings, key=lambda item: item["idx"])

    results: List[Dict[str, Any]] = []
    for i, candle in enumerate(normalised):
        while swing_idx_pointer < len(swings_sorted) and swings_sorted[swing_idx_pointer][
            "idx"
        ] == i:
            swing = swings_sorted[swing_idx_pointer]
            if swing["type"] == "high":
                swing_highs.append(swing["price"])
            else:
                swing_lows.append(swing["price"])
            swing_idx_pointer += 1

        if i == 0:
            continue

        last_high = swing_highs[-1] if swing_highs else None
        last_low = swing_lows[-1] if swing_lows else None

        close = candle["c"]

        if last_high is not None and _breaks_above(close, last_high, tick):
            if _validate_structure(swing_highs, swing_lows, direction="up") and _has_closed_zone(
                closed_bear_zones, i
            ):
                results.append({"tf": tf, "type": "bull", "delivery_candle": candle["t"]})

        if last_low is not None and _breaks_below(close, last_low, tick):
            if _validate_structure(swing_highs, swing_lows, direction="down") and _has_closed_zone(
                closed_bull_zones, i
            ):
                results.append({"tf": tf, "type": "bear", "delivery_candle": candle["t"]})

    unique: Dict[tuple[str, int], Dict[str, Any]] = {}
    for item in results:
        unique[(item["type"], item["delivery_candle"])] = item
    return list(unique.values())


def _validate_structure(highs: Sequence[float], lows: Sequence[float], *, direction: str) -> bool:
    if direction == "up":
        if len(lows) < 2 or len(highs) < 2:
            return False
        return lows[-2] > lows[-1] and highs[-2] > highs[-1]
    if len(lows) < 2 or len(highs) < 2:
        return False
    return lows[-2] < lows[-1] and highs[-2] < highs[-1]


def _has_closed_zone(zones: Sequence[Mapping[str, Any]], index: int) -> bool:
    for zone in zones:
        if zone["created_idx"] < index:
            return True
    return False


def detect_zones(
    candles: Sequence[Mapping[str, Any]], tf: str, symbol: str, cfg: Config
) -> Dict[str, Any]:
    payload = normalise_ohlcv(symbol, tf, candles, use_full_span=True)
    series = payload.get("candles", []) if isinstance(payload, Mapping) else []
    inferred_tick = _infer_tick_size(series)
    effective_cfg = replace(
        cfg,
        tick_size=cfg.tick_size or inferred_tick,
        min_gap_pct=(
            cfg.min_gap_pct if cfg.min_gap_pct is not None else _default_min_gap_pct(tf)
        ),
    )
    fvg_zones, _ = _detect_fvg_internal(series, effective_cfg, tf)
    ob_zones, _ = _detect_ob_internal(series, effective_cfg, tf)
    swings = compute_swings(series, effective_cfg.w_swing)
    inducement = detect_inducement(series, fvg_zones, ob_zones, effective_cfg, tf)
    cisd = detect_cisd(series, fvg_zones, swings, effective_cfg, tf)

    return {
        "symbol": symbol,
        "zones": {
            "fvg": fvg_zones,
            "ob": ob_zones,
            "inducement": inducement,
            "cisd": cisd,
        },
    }


def flatten_structured(zones_structured: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    fvg_zones = zones_structured.get("zones", {}).get("fvg", [])
    for zone in fvg_zones:
        result.append({
            "type": "fvg_top",
            "price": zone["top"],
            "dir": zone["dir"],
        })
        result.append({
            "type": "fvg_bot",
            "price": zone["bot"],
            "dir": zone["dir"],
        })
        result.append({
            "type": "fvl",
            "price": zone["fvl"],
            "dir": zone["dir"],
        })
    ob_zones = zones_structured.get("zones", {}).get("ob", [])
    for zone in ob_zones:
        result.append({
            "type": "ob_min",
            "price": zone["range"][0],
            "dir": zone.get("type"),
        })
        result.append({
            "type": "ob_max",
            "price": zone["range"][1],
            "dir": zone.get("type"),
        })
    inducements = zones_structured.get("zones", {}).get("inducement", [])
    for zone in inducements:
        result.append({
            "type": "inducement",
            "price": zone.get("price"),
            "dir": zone.get("type"),
        })
    cisd_zones = zones_structured.get("zones", {}).get("cisd", [])
    for zone in cisd_zones:
        result.append({
            "type": f"cisd_{zone.get('type')}",
            "price": None,
            "dir": zone.get("type"),
        })
    return result

