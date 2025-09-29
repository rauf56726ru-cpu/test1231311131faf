"""Smart Money Concepts block detection utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class SMCConfig:
    """Configuration parameters for SMC block detection."""

    min_block_size: float = 0.0
    displacement_factor: float = 1.5
    displacement_lookback: int = 5
    ttl_bars: int = 50
    zones_window_start_ms: int | None = None
    window_end_ms_prev_closed: int | None = None
    allow_base_fallback: bool = False
    base_fallback_max_age: int = 200
    base_fallback_max_distance_atr: float = 3.0


def _candle_body_range(candle: Mapping[str, float]) -> tuple[float, float]:
    open_price = float(candle.get("o", 0.0))
    close_price = float(candle.get("c", 0.0))
    if open_price <= close_price:
        return open_price, close_price
    return close_price, open_price


def _candle_range(candle: Mapping[str, float]) -> tuple[float, float]:
    high = float(candle.get("h", 0.0))
    low = float(candle.get("l", 0.0))
    if low <= high:
        return low, high
    return high, low


def _ensure_sorted_range(values: Iterable[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    return float(low), float(high)


def _range_size(range_pair: tuple[float, float]) -> float:
    return float(range_pair[1] - range_pair[0])


def _overlap_size(left: tuple[float, float], right: tuple[float, float]) -> float:
    low = max(left[0], right[0])
    high = min(left[1], right[1])
    return max(0.0, high - low)


def _eq_tolerance(price: float, tick_size: float | None) -> float:
    base = 0.0005
    adaptive = 0.0
    if tick_size and price:
        adaptive = 0.5 * tick_size / price
    return max(base, adaptive, 0.0002)


def _normalise_direction_label(value: str | None) -> str:
    label = str(value or "").lower()
    synonyms = {
        "bull": "up",
        "bullish": "up",
        "long": "up",
        "buy": "demand",
        "bear": "down",
        "bearish": "down",
        "short": "down",
        "sell": "supply",
    }
    return synonyms.get(label, label)


def _direction_from_zone(zone_type: str | None) -> str:
    label = _normalise_direction_label(zone_type)
    if label in {"supply", "demand"}:
        return label
    if label == "up":
        return "demand"
    if label == "down":
        return "supply"
    return "unknown"


def _opposite_direction(direction: str) -> str:
    if direction == "supply":
        return "demand"
    if direction == "demand":
        return "supply"
    if direction == "up":
        return "down"
    if direction == "down":
        return "up"
    return direction


def _direction_equivalent(value: str | None, target: str | None) -> bool:
    left = _normalise_direction_label(value)
    right = _normalise_direction_label(target)
    if not right:
        return True
    if left == right:
        return True
    mapping = {
        "up": {"demand"},
        "demand": {"up"},
        "down": {"supply"},
        "supply": {"down"},
    }
    return right in mapping.get(left, set())


def _normalise_events(events: Sequence[Mapping[str, object]] | None) -> List[Mapping[str, object]]:
    if not events:
        return []
    normalised: List[Mapping[str, object]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("kind") or event.get("type") or "").lower()
        direction_raw = event.get("direction") or event.get("dir")
        direction = _normalise_direction_label(direction_raw)
        timestamp = event.get("t") or event.get("time") or event.get("timestamp")
        if timestamp is None:
            continue
        try:
            ts_value = int(timestamp)
        except (TypeError, ValueError):
            continue
        payload = dict(event)
        payload["kind"] = kind
        if direction:
            payload["direction"] = direction
        payload["t"] = ts_value
        normalised.append(payload)
    normalised.sort(key=lambda item: item["t"])
    return normalised


def _normalise_ob_zones(
    zones: Sequence[Mapping[str, object]] | None,
) -> List[MutableMapping[str, object]]:
    normalised: List[MutableMapping[str, object]] = []
    if not zones:
        return normalised
    for zone in zones:
        if not isinstance(zone, Mapping):
            continue
        range_values = zone.get("range")
        if isinstance(range_values, Sequence) and len(range_values) >= 2:
            try:
                low = float(range_values[0])
                high = float(range_values[1])
            except (TypeError, ValueError):
                continue
            range_pair = _ensure_sorted_range((low, high))
        else:
            continue
        created_at = zone.get("created_at") or zone.get("t")
        try:
            created_ts = int(created_at)
        except (TypeError, ValueError):
            continue
        direction = _direction_from_zone(zone.get("type") or zone.get("dir"))
        touches = zone.get("touches")
        try:
            touches_count = int(touches) if touches is not None else 0
        except (TypeError, ValueError):
            touches_count = 0
        entry: MutableMapping[str, object] = {
            "type": direction,
            "range": [range_pair[0], range_pair[1]],
            "created_at": created_ts,
            "touches": touches_count,
            "status": zone.get("status", "open"),
        }
        candle_tf = zone.get("tf") or zone.get("timeframe")
        if candle_tf:
            entry["tf"] = candle_tf
        normalised.append(entry)
    normalised.sort(key=lambda item: item["created_at"])
    return normalised


def _liquidity_levels(liquidity: Mapping[str, object] | None) -> List[tuple[str, float]]:
    if not isinstance(liquidity, Mapping):
        return []
    levels: List[tuple[str, float]] = []
    for key in ("eqh", "eql"):
        entries = liquidity.get(key)
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            price = entry.get("price")
            try:
                price_value = float(price)
            except (TypeError, ValueError):
                continue
            levels.append((str(key), price_value))
    for key in ("pdh", "pdl"):
        entry = liquidity.get(key)
        if not isinstance(entry, Mapping):
            continue
        price = entry.get("price")
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        levels.append((str(key), price_value))
    return levels


def _find_candle_index_by_ts(candles: Sequence[Mapping[str, float]], ts: int) -> int | None:
    for idx, candle in enumerate(candles):
        if int(candle.get("t", -1)) == ts:
            return idx
    return None


def _average_body(candles: Sequence[Mapping[str, float]], end_idx: int, lookback: int) -> float:
    if lookback <= 0:
        return 0.0
    start_idx = max(0, end_idx - lookback)
    if start_idx >= end_idx:
        return 0.0
    total = 0.0
    count = 0
    for idx in range(start_idx, end_idx):
        low, high = _candle_body_range(candles[idx])
        total += abs(high - low)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _block_status(
    candles: Sequence[Mapping[str, float]],
    start_idx: int,
    price_range: tuple[float, float],
    direction: str,
) -> str:
    status = "fresh"
    low, high = price_range
    for idx in range(start_idx + 1, len(candles)):
        candle = candles[idx]
        body_low, body_high = _candle_body_range(candle)
        close_price = float(candle.get("c", body_high))
        if status != "invalidated" and status == "fresh":
            if low <= close_price <= high and body_low <= high and body_high >= low:
                status = "tapped"
        if direction == "demand" and close_price < low:
            status = "invalidated"
            break
        if direction == "supply" and close_price > high:
            status = "invalidated"
            break
    return status


def _timeframe_priority(timeframe: str | None) -> float:
    if timeframe is None:
        return 0.0
    label = str(timeframe).strip().lower()
    if not label:
        return 0.0
    # Common short-hands first to avoid parsing ambiguities.
    explicit = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "45m": 45,
        "1h": 60,
        "2h": 120,
        "3h": 180,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "2d": 2880,
        "3d": 4320,
        "1w": 10080,
    }
    if label in explicit:
        return float(explicit[label])
    digits = ""
    suffix = ""
    for char in label:
        if char.isdigit():
            digits += char
        else:
            suffix += char
    if not digits:
        return 0.0
    try:
        magnitude = float(int(digits))
    except ValueError:
        return 0.0
    suffix = suffix or "m"
    unit_weights = {
        "m": 1.0,
        "min": 1.0,
        "minute": 1.0,
        "minutes": 1.0,
        "h": 60.0,
        "hr": 60.0,
        "hour": 60.0,
        "hours": 60.0,
        "d": 1440.0,
        "day": 1440.0,
        "days": 1440.0,
        "w": 10080.0,
        "week": 10080.0,
        "weeks": 10080.0,
    }
    weight = unit_weights.get(suffix, 1.0)
    return magnitude * weight


def _deduplicate_blocks(blocks: List[MutableMapping[str, object]], block: MutableMapping[str, object]) -> None:
    direction = block.get("type")
    new_range = tuple(block.get("range", (0.0, 0.0)))
    new_created = int(block.get("created_at", 0))
    new_len = _range_size((float(new_range[0]), float(new_range[1])))
    new_tf = block.get("tf")
    new_tf_rank = _timeframe_priority(str(new_tf) if new_tf is not None else None)
    for idx, existing in enumerate(blocks):
        if existing.get("type") != direction:
            continue
        if existing.get("kind") != block.get("kind"):
            continue
        existing_range = tuple(existing.get("range", (0.0, 0.0)))
        overlap = _overlap_size(
            (float(existing_range[0]), float(existing_range[1])),
            (float(new_range[0]), float(new_range[1])),
        )
        if overlap <= 0.0:
            continue
        existing_len = _range_size(
            (float(existing_range[0]), float(existing_range[1]))
        )
        coverage = overlap / max(1e-12, min(existing_len, new_len))
        if coverage >= 0.8:
            replace_existing = False
            existing_tf = existing.get("tf")
            existing_tf_rank = _timeframe_priority(
                str(existing_tf) if existing_tf is not None else None
            )
            if new_tf_rank > existing_tf_rank:
                replace_existing = True
            elif abs(new_tf_rank - existing_tf_rank) <= 1e-9:
                if new_len > existing_len + 1e-12:
                    replace_existing = True
                elif abs(new_len - existing_len) <= 1e-12 and new_created >= int(
                    existing.get("created_at", 0)
                ):
                    replace_existing = True
            if replace_existing:
                blocks[idx] = block
            return
    blocks.append(block)


def detect_smc_blocks(
    candles: Sequence[Mapping[str, float]],
    *,
    timeframe: str = "1h",
    structure_flags: Sequence[Mapping[str, object]] | None = None,
    ob_zones: Sequence[Mapping[str, object]] | None = None,
    liquidity_levels: Mapping[str, object] | None = None,
    config: SMCConfig | None = None,
    atr: Sequence[float] | None = None,
    returns_sigma: Sequence[float] | None = None,
    displacement_body: float = 1.0,
    displacement_range: float = 1.5,
    tick_size: float | None = None,
) -> tuple[List[MutableMapping[str, object]], Dict[str, Any]]:
    """Detect breaker, mitigation and reversal blocks on the supplied candles."""

    diagnostics: Dict[str, Any] = {
        "rb_raw_count": 0,
        "rb_reject_no_eq": 0,
        "rb_reject_no_sweep": 0,
        "rb_reject_no_choch": 0,
        "rb_reject_no_base": 0,
        "rb_reject_no_impulse": 0,
        "rb_flow": {"eq_found": 0, "sweep": 0, "choch": 0, "base": 0, "impulse": 0},
        "rb_reject": {"no_eq": 0, "no_sweep": 0, "no_choch": 0, "no_base": 0, "no_impulse": 0},
        "base_tick_collapse": 0,
        "base_fallback_used": False,
        "rb_impulse_ok": False,
    }

    if not candles:
        return [], diagnostics

    cfg = config or SMCConfig()
    structure_events = _normalise_events(structure_flags)
    base_ob_zones = _normalise_ob_zones(ob_zones)
    liquidity = _liquidity_levels(liquidity_levels)

    blocks: List[MutableMapping[str, object]] = []

    block_type_labels = {
        "bb": "breaker block",
        "mb": "mitigation block",
        "rb": "reversal block",
    }

    def _append_block(
        *,
        kind: str,
        price_range: tuple[float, float],
        created_idx: int,
        direction: str,
        created_at: int,
    ) -> None:
        span = _range_size(price_range)
        if span < cfg.min_block_size - 1e-12:
            return
        status = _block_status(candles, created_idx, price_range, direction)
        block: MutableMapping[str, object] = {
            "kind": kind,
            "range": [round(price_range[0], 12), round(price_range[1], 12)],
            "status": status,
            "created_at": created_at,
            "type": direction,
            "tf": timeframe,
            "block_type": block_type_labels.get(kind, kind),
            "_created_idx": created_idx,
        }
        _deduplicate_blocks(blocks, block)

    # Breaker Blocks
    for zone in base_ob_zones:
        direction = _direction_from_zone(zone.get("type"))
        if direction not in {"supply", "demand"}:
            continue
        created_ts = int(zone["created_at"])
        created_idx = _find_candle_index_by_ts(candles, created_ts)
        if created_idx is None:
            continue
        body_range = _candle_body_range(candles[created_idx])
        if _range_size(body_range) <= 0:
            continue
        zone_range = _ensure_sorted_range(zone.get("range", body_range))
        # BOS in opposite direction
        desired_direction = _opposite_direction(direction)
        bos_event = None
        for event in structure_events:
            if event["kind"] != "bos":
                continue
            if event.get("direction") and not _direction_equivalent(event.get("direction"), desired_direction):
                continue
            if int(event["t"]) <= created_ts:
                continue
            close_price = event.get("close") or event.get("price")
            if close_price is None:
                close_idx = _find_candle_index_by_ts(candles, int(event["t"]))
                if close_idx is not None:
                    close_price = candles[close_idx]["c"]
            try:
                close_value = float(close_price)
            except (TypeError, ValueError):
                continue
            if direction == "supply" and close_value <= zone_range[1]:
                continue
            if direction == "demand" and close_value >= zone_range[0]:
                continue
            bos_event = event
            break
        if bos_event is None:
            continue
        bos_ts = int(bos_event["t"])
        bos_idx = _find_candle_index_by_ts(candles, bos_ts) or created_idx
        opposite = _opposite_direction(direction)
        for idx in range(bos_idx + 1, len(candles)):
            candle = candles[idx]
            body_low, body_high = _candle_body_range(candle)
            if body_low <= body_range[1] and body_high >= body_range[0]:
                _append_block(
                    kind="bb",
                    price_range=body_range,
                    created_idx=idx,
                    direction=opposite,
                    created_at=int(candle.get("t", bos_ts)),
                )
                break

    # Mitigation Blocks
    for zone in base_ob_zones:
        direction = _direction_from_zone(zone.get("type"))
        if direction not in {"supply", "demand"}:
            continue
        created_ts = int(zone["created_at"])
        created_idx = _find_candle_index_by_ts(candles, created_ts)
        if created_idx is None:
            continue
        base_candle = candles[created_idx]
        body_range = _candle_body_range(base_candle)
        body_span = _range_size(body_range)
        if body_span <= 0:
            continue
        partial_idx = None
        partial_overlap = 0.0
        partial_low = body_range[0]
        partial_high = body_range[1]
        for idx in range(created_idx + 1, len(candles)):
            candle = candles[idx]
            body_low, body_high = _candle_body_range(candle)
            overlap = _overlap_size(body_range, (body_low, body_high))
            if overlap <= 0.0:
                continue
            if overlap >= 0.8 * body_span:
                break
            partial_idx = idx
            partial_overlap = overlap
            partial_low = max(body_range[0], body_low)
            partial_high = min(body_range[1], body_high)
            break
        if partial_idx is None or partial_overlap <= 0.0:
            continue
        remaining_ratio = 1.0 - partial_overlap / max(body_span, 1e-12)
        if remaining_ratio < 0.2:
            continue
        if direction == "demand":
            leftover_range = (body_range[0], partial_low)
        else:
            leftover_range = (partial_high, body_range[1])
        if leftover_range[1] - leftover_range[0] <= 0:
            continue
        for idx in range(partial_idx + 1, len(candles)):
            candle = candles[idx]
            body_low, body_high = _candle_body_range(candle)
            if body_low <= leftover_range[1] and body_high >= leftover_range[0]:
                _append_block(
                    kind="mb",
                    price_range=_ensure_sorted_range(leftover_range),
                    created_idx=idx,
                    direction=direction,
                    created_at=int(candle.get("t", zone["created_at"])),
                )
                break

    # Reversal Blocks

    def _atr_value(idx: int) -> float:
        if atr is None or idx >= len(atr):
            return math.nan
        return float(atr[idx])

    def _sigma_value(idx: int) -> float:
        if returns_sigma is None or idx >= len(returns_sigma):
            return math.nan
        return float(returns_sigma[idx])

    window_start_ms = cfg.zones_window_start_ms
    window_end_ms = cfg.window_end_ms_prev_closed
    if candles:
        first_ts = int(candles[0]["t"])
        last_ts = int(candles[-1]["t"])
        if window_start_ms is None:
            window_start_ms = first_ts
        if window_end_ms is None:
            window_end_ms = last_ts
        else:
            window_end_ms = min(window_end_ms, last_ts)

    window_indices = [
        idx
        for idx, candle in enumerate(candles)
        if (window_start_ms is None or int(candle["t"]) >= window_start_ms)
        and (window_end_ms is None or int(candle["t"]) <= window_end_ms)
    ]

    flow_counters: Dict[str, int] = diagnostics.setdefault(
        "rb_flow",
        {"eq_found": 0, "sweep": 0, "choch": 0, "base": 0, "impulse": 0},
    )
    reject_counters: Dict[str, int] = diagnostics.setdefault(
        "rb_reject",
        {"no_eq": 0, "no_sweep": 0, "no_choch": 0, "no_base": 0, "no_impulse": 0},
    )
    diagnostics.setdefault("base_tick_collapse", 0)
    diagnostics.setdefault("base_fallback_used", False)
    diagnostics.setdefault("rb_impulse_ok", False)

    structure_events_window: List[Mapping[str, object]] = []
    structure_tfs: set[str] = set()
    for event in structure_events:
        event_ts = int(event.get("t", 0))
        if window_start_ms is not None and event_ts < window_start_ms:
            continue
        if window_end_ms is not None and event_ts > window_end_ms:
            continue
        kind = str(event.get("kind") or "")
        if kind not in {"bos", "choch"}:
            continue
        event_tf = event.get("tf")
        if event_tf:
            structure_tfs.add(str(event_tf))
        structure_events_window.append(event)
    if structure_tfs:
        assert structure_tfs == {timeframe}

    choch_events_window = [
        event for event in structure_events_window if str(event.get("kind")) == "choch"
    ]
    if not choch_events_window:
        diagnostics["rb_reject_no_choch"] = max(diagnostics["rb_reject_no_choch"], 1)
        reject_counters["no_choch"] = max(reject_counters.get("no_choch", 0), 1)

    def _pivot_span_for_tf(label: str) -> int:
        if label == "1h":
            return 3
        if label == "15m":
            return 2
        return 2

    pivot_span = _pivot_span_for_tf(timeframe)
    pivots_window: List[Dict[str, Any]] = []
    if window_indices and len(candles) >= 2 * pivot_span + 1:
        for idx in range(pivot_span, len(candles) - pivot_span):
            ts_idx = int(candles[idx]["t"])
            if window_start_ms is not None and ts_idx < window_start_ms:
                continue
            if window_end_ms is not None and ts_idx > window_end_ms:
                continue
            segment = candles[idx - pivot_span : idx + pivot_span + 1]
            center = candles[idx]
            high = float(center["h"])
            low = float(center["l"])
            if all(high >= float(bar["h"]) for bar in segment):
                pivots_window.append({"type": "ph", "idx": idx, "price": high, "ts": ts_idx})
            if all(low <= float(bar["l"]) for bar in segment):
                pivots_window.append({"type": "pl", "idx": idx, "price": low, "ts": ts_idx})

    separation = 5
    eq_candidates: List[Dict[str, Any]] = []
    sorted_pivots = sorted(pivots_window, key=lambda item: item["idx"])
    for i, pivot in enumerate(sorted_pivots):
        pivot_type = pivot.get("type")
        if pivot_type not in {"ph", "pl"}:
            continue
        for j in range(i + 1, len(sorted_pivots)):
            other = sorted_pivots[j]
            if other.get("type") != pivot_type:
                continue
            if int(other["idx"]) - int(pivot["idx"]) < separation:
                continue
            price_a = float(pivot.get("price", 0.0))
            price_b = float(other.get("price", 0.0))
            reference_price = (price_a + price_b) / 2.0
            tolerance = _eq_tolerance(reference_price or price_a, tick_size)
            if abs(price_a - price_b) <= tolerance:
                eq_candidates.append(
                    {
                        "type": "eqh" if pivot_type == "ph" else "eql",
                        "price": reference_price,
                        "first_idx": int(pivot["idx"]),
                        "second_idx": int(other["idx"]),
                        "first_ts": int(pivot["ts"]),
                        "second_ts": int(other["ts"]),
                    }
                )
                break

    if not eq_candidates:
        diagnostics["rb_reject_no_eq"] = max(diagnostics["rb_reject_no_eq"], 1)
        reject_counters["no_eq"] = max(reject_counters.get("no_eq", 0), 1)

    def _range_block_base(end_idx: int) -> tuple[tuple[float, float], int] | None:
        if end_idx < 0 or not candles:
            return None
        max_length = min(4, end_idx + 1)
        for length in range(max_length, 0, -1):
            start_idx = end_idx - length + 1
            if start_idx < 0:
                continue
            start_ts = int(candles[start_idx]["t"])
            end_ts = int(candles[end_idx]["t"])
            if window_start_ms is not None and end_ts < window_start_ms:
                continue
            if window_end_ms is not None and start_ts > window_end_ms:
                continue
            body_ranges = [
                _candle_body_range(candles[idx]) for idx in range(start_idx, end_idx + 1)
            ]
            spans = [_range_size(body_range) for body_range in body_ranges]
            if any(span <= 0.0 for span in spans):
                continue
            overlap_ok = True
            for left, right in zip(body_ranges, body_ranges[1:]):
                min_span = min(_range_size(left), _range_size(right))
                if min_span <= 0.0 or _overlap_size(left, right) < 0.5 * min_span:
                    overlap_ok = False
                    break
            if not overlap_ok:
                continue
            combined_low = min(body[0] for body in body_ranges)
            combined_high = max(body[1] for body in body_ranges)
            span = combined_high - combined_low
            atr_val = _atr_value(end_idx)
            if math.isfinite(atr_val) and atr_val > 0.0 and span > 0.8 * atr_val + 1e-12:
                continue
            if span < cfg.min_block_size - 1e-12:
                continue
            return ((combined_low, combined_high), end_idx)
        return None

    def _fallback_ob_base(
        direction: str, impulse_idx: int, impulse_ts: int, eq_price: float
    ) -> tuple[tuple[float, float], int] | None:
        atr_val = _atr_value(impulse_idx)
        max_age = max(1, int(cfg.base_fallback_max_age))
        distance_limit = float(cfg.base_fallback_max_distance_atr)
        for zone in reversed(base_ob_zones):
            if _direction_from_zone(zone.get("type")) != direction:
                continue
            created_at = int(zone.get("created_at", 0))
            if created_at >= impulse_ts:
                continue
            range_values = zone.get("range")
            if not isinstance(range_values, Sequence) or len(range_values) < 2:
                continue
            try:
                low = float(range_values[0])
                high = float(range_values[1])
            except (TypeError, ValueError):
                continue
            base_range = _ensure_sorted_range((low, high))
            if _range_size(base_range) < cfg.min_block_size - 1e-12:
                continue
            base_idx = _find_candle_index_by_ts(candles, created_at)
            if base_idx is None or base_idx >= impulse_idx:
                base_idx = max(0, impulse_idx - 1)
            if impulse_idx - base_idx > max_age:
                continue
            if (
                atr_val
                and math.isfinite(atr_val)
                and atr_val > 0.0
                and distance_limit > 0.0
            ):
                distance = min(
                    abs(eq_price - base_range[0]),
                    abs(eq_price - base_range[1]),
                )
                if distance > distance_limit * atr_val + 1e-12:
                    continue
            if base_idx < len(candles):
                base_ts = int(candles[base_idx]["t"])
                if window_start_ms is not None and base_ts < window_start_ms:
                    continue
                if window_end_ms is not None and base_ts > window_end_ms:
                    continue
            return base_range, base_idx
        return None

    epsilon_tick = float(tick_size) if tick_size else 0.0

    for eq_entry in eq_candidates:
        flow_counters["eq_found"] = flow_counters.get("eq_found", 0) + 1
        trend_direction = "up" if eq_entry["type"] == "eql" else "down"
        block_direction = "demand" if trend_direction == "up" else "supply"
        eq_price = float(eq_entry["price"])

        sweep_idx: int | None = None
        recovery_idx: int | None = None
        start_idx = int(eq_entry["second_idx"]) + 1
        for idx in range(start_idx, len(candles)):
            ts_idx = int(candles[idx]["t"])
            if window_end_ms is not None and ts_idx > window_end_ms:
                break
            low = float(candles[idx]["l"])
            high = float(candles[idx]["h"])
            close_val = float(candles[idx]["c"])
            if trend_direction == "up":
                if low >= eq_price - epsilon_tick:
                    continue
                max_recovery_idx = min(len(candles) - 1, idx + 3)
                recovery_candidate: int | None = None
                for rec_idx in range(idx, max_recovery_idx + 1):
                    rec_ts = int(candles[rec_idx]["t"])
                    if window_end_ms is not None and rec_ts > window_end_ms:
                        break
                    atr_val = _atr_value(rec_idx)
                    if not math.isfinite(atr_val) or atr_val <= 0.0:
                        continue
                    close_rec = float(candles[rec_idx]["c"])
                    if close_rec >= eq_price - 0.2 * atr_val:
                        recovery_candidate = rec_idx
                        break
                if recovery_candidate is None:
                    continue
                sweep_idx = idx
                recovery_idx = recovery_candidate
                break
            else:
                if high <= eq_price + epsilon_tick:
                    continue
                max_recovery_idx = min(len(candles) - 1, idx + 3)
                recovery_candidate = None
                for rec_idx in range(idx, max_recovery_idx + 1):
                    rec_ts = int(candles[rec_idx]["t"])
                    if window_end_ms is not None and rec_ts > window_end_ms:
                        break
                    atr_val = _atr_value(rec_idx)
                    if not math.isfinite(atr_val) or atr_val <= 0.0:
                        continue
                    close_rec = float(candles[rec_idx]["c"])
                    if close_rec <= eq_price + 0.2 * atr_val:
                        recovery_candidate = rec_idx
                        break
                if recovery_candidate is None:
                    continue
                sweep_idx = idx
                recovery_idx = recovery_candidate
                break

        if sweep_idx is None or recovery_idx is None:
            diagnostics["rb_reject_no_sweep"] += 1
            reject_counters["no_sweep"] = reject_counters.get("no_sweep", 0) + 1
            continue

        flow_counters["sweep"] = flow_counters.get("sweep", 0) + 1

        min_choch_idx = sweep_idx + 1
        choch_event: Mapping[str, object] | None = None
        choch_idx: int | None = None
        for event in choch_events_window:
            event_dir = _normalise_direction_label(event.get("direction"))
            if trend_direction == "up" and event_dir not in {"up", "demand"}:
                continue
            if trend_direction == "down" and event_dir not in {"down", "supply"}:
                continue
            event_ts = int(event.get("t", 0))
            event_idx = _find_candle_index_by_ts(candles, event_ts)
            if event_idx is None or event_idx < min_choch_idx:
                continue
            if window_end_ms is not None and event_ts > window_end_ms:
                continue
            choch_event = event
            choch_idx = event_idx
            break

        if choch_event is None or choch_idx is None:
            diagnostics["rb_reject_no_choch"] += 1
            reject_counters["no_choch"] = reject_counters.get("no_choch", 0) + 1
            continue

        if window_start_ms is not None:
            assert window_start_ms <= int(choch_event.get("t", 0))
        if window_end_ms is not None:
            assert int(choch_event.get("t", 0)) <= window_end_ms

        flow_counters["choch"] = flow_counters.get("choch", 0) + 1

        impulse_idx: int | None = None
        for offset in (1, 2):
            idx = choch_idx + offset
            if idx >= len(candles):
                break
            ts_idx = int(candles[idx]["t"])
            if window_end_ms is not None and ts_idx > window_end_ms:
                break
            open_price = float(candles[idx]["o"])
            close_price = float(candles[idx]["c"])
            if trend_direction == "up" and close_price <= open_price:
                continue
            if trend_direction == "down" and close_price >= open_price:
                continue
            atr_val = _atr_value(idx)
            if not math.isfinite(atr_val) or atr_val <= 0.0:
                continue
            sigma_val = _sigma_value(idx)
            k_body = 1.0
            k_range = 1.5
            if sigma_val and math.isfinite(sigma_val) and sigma_val < 0.5 * atr_val:
                k_body = 0.8
                k_range = 1.2
            body_sizes = [abs(close_price - open_price)]
            range_sizes = [float(candles[idx]["h"]) - float(candles[idx]["l"])]
            next_idx = idx + 1
            if next_idx < len(candles) and next_idx <= choch_idx + 2:
                body_sizes.append(
                    abs(float(candles[next_idx]["c"]) - float(candles[next_idx]["o"]))
                )
                range_sizes.append(float(candles[next_idx]["h"]) - float(candles[next_idx]["l"]))
            if max(body_sizes) >= k_body * atr_val or max(range_sizes) >= k_range * atr_val:
                impulse_idx = idx
                diagnostics["rb_impulse_ok"] = True
                flow_counters["impulse"] = flow_counters.get("impulse", 0) + 1
                break

        if impulse_idx is None:
            diagnostics["rb_reject_no_impulse"] += 1
            reject_counters["no_impulse"] = reject_counters.get("no_impulse", 0) + 1
            continue

        base_candidate = _range_block_base(impulse_idx - 1)
        impulse_ts = int(candles[impulse_idx].get("t", choch_event.get("t", 0)))
        fallback_used = False
        if base_candidate is None and cfg.allow_base_fallback:
            base_candidate = _fallback_ob_base(
                block_direction, impulse_idx, impulse_ts, eq_price
            )
            if base_candidate is not None:
                fallback_used = True

        if base_candidate is None:
            diagnostics["rb_reject_no_base"] += 1
            reject_counters["no_base"] = reject_counters.get("no_base", 0) + 1
            continue

        flow_counters["base"] = flow_counters.get("base", 0) + 1
        if fallback_used:
            diagnostics["base_fallback_used"] = True

        base_range, base_idx = base_candidate
        if tick_size and abs(base_range[1] - base_range[0]) < float(tick_size) - 1e-12:
            diagnostics["base_tick_collapse"] = int(diagnostics.get("base_tick_collapse", 0)) + 1

        _append_block(
            kind="rb",
            price_range=base_range,
            created_idx=base_idx,
            direction=block_direction,
            created_at=impulse_ts,
        )
        diagnostics["rb_raw_count"] += 1

    if cfg.ttl_bars > 0 and candles:
        latest_idx = len(candles) - 1
        ttl = max(1, int(cfg.ttl_bars))
        filtered: List[MutableMapping[str, object]] = []
        for block in blocks:
            created_idx = int(block.get("_created_idx", latest_idx))
            if latest_idx - created_idx <= ttl:
                filtered.append(block)
        blocks = filtered

    # Clean helper fields
    for block in blocks:
        block.pop("_created_idx", None)

    return blocks, diagnostics
