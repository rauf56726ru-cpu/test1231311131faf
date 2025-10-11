"""Smart Money Concepts block detection utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    base_min_bars: int = 1
    base_max_bars: int = 4
    base_max_atr: float = 0.8
    base_min_overlap: float = 0.5
    impulse_min_cover: float = 0.6
    allow_bos_substitute: bool = True
    sweep_choch_max_bars: dict[str, int] = field(
        default_factory=lambda: {"15m": 12, "1h": 20, "4h": 28}
    )
    sweep_choch_default_max_bars: int = 16
    sweep_choch_extension_factor: float = 1.5
    opposite_bos_max_bars: dict[str, int] = field(
        default_factory=lambda: {"15m": 12, "1h": 18, "4h": 24}
    )
    opposite_bos_default_max_bars: int = 14
    block_dedup_strategy: str = "priority"
    dedup_weight_tf: float = 0.5
    dedup_weight_recency: float = 0.3
    dedup_weight_confidence: float = 0.2
    feature_strict_legacy_mode: bool = False
    feature_extended_windows: bool = True
    feature_block_scored_dedup: bool = True


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


def _round_to_tick(value: float, tick_size: float | None) -> float:
    if not tick_size or tick_size <= 0:
        return float(value)
    return round(float(value) / tick_size) * tick_size


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


def _block_score(
    block: Mapping[str, object],
    *,
    latest_idx: int,
    weights: Mapping[str, float],
) -> float:
    tf_rank = _timeframe_priority(str(block.get("tf") or ""))
    tf_component = 0.0
    if tf_rank > 0.0:
        tf_component = min(tf_rank / 10080.0, 1.0)

    created_idx = int(block.get("_created_idx", latest_idx))
    recency_component = 0.0
    if latest_idx > 0:
        recency_component = 1.0 - min(max(latest_idx - created_idx, 0) / latest_idx, 1.0)

    try:
        confidence_component = float(block.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence_component = 1.0
    confidence_component = max(0.0, min(confidence_component, 1.0))

    tf_weight = max(0.0, float(weights.get("tf", 0.0)))
    recency_weight = max(0.0, float(weights.get("recency", 0.0)))
    confidence_weight = max(0.0, float(weights.get("confidence", 0.0)))
    total_weight = tf_weight + recency_weight + confidence_weight
    if total_weight <= 0.0:
        total_weight = 1.0
        tf_weight = 0.5
        recency_weight = 0.3
        confidence_weight = 0.2

    return (
        tf_weight * tf_component
        + recency_weight * recency_component
        + confidence_weight * confidence_component
    ) / total_weight


def _deduplicate_blocks(
    blocks: List[MutableMapping[str, object]],
    block: MutableMapping[str, object],
    *,
    priorities: Mapping[str, int] | None = None,
    strategy: str = "priority",
    weights: Mapping[str, float] | None = None,
    latest_idx: int = 0,
    trace: List[Dict[str, object]] | None = None,
) -> tuple[bool, str | None]:
    direction = block.get("type")
    new_range = tuple(block.get("range", (0.0, 0.0)))
    new_created = int(block.get("created_at", 0))
    new_len = _range_size((float(new_range[0]), float(new_range[1])))
    new_tf = block.get("tf")
    new_tf_rank = _timeframe_priority(str(new_tf) if new_tf is not None else None)
    new_kind = str(block.get("kind") or "")
    priority_map = priorities or {}
    new_priority = priority_map.get(new_kind, 0)
    for idx, existing in enumerate(blocks):
        if existing.get("type") != direction:
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
        min_span = min(existing_len, new_len)
        if min_span <= 0.0:
            continue
        coverage = overlap / max(1e-12, min_span)
        if coverage < 0.8:
            continue

        existing_kind = str(existing.get("kind") or "")
        existing_priority = priority_map.get(existing_kind, 0)
        kept_label = existing_kind.upper() or existing_kind

        if existing_kind != new_kind:
            if new_priority > existing_priority:
                kept_label = new_kind.upper() or new_kind
                existing["shadowed_by"] = new_kind.upper() or new_kind
                if trace is not None:
                    trace.append(
                        {
                            "type": existing_kind.upper() or existing_kind,
                            "overlap_pct": float(coverage),
                            "kept": kept_label,
                        }
                    )
                continue
            if new_priority < existing_priority:
                block["shadowed_by"] = existing_kind.upper() or existing_kind
                if trace is not None:
                    trace.append(
                        {
                            "type": existing_kind.upper() or existing_kind,
                            "overlap_pct": float(coverage),
                            "kept": kept_label,
                        }
                    )
                return False, "dedup_priority_loss" if new_kind == "rb" else None
            if trace is not None:
                trace.append(
                    {
                        "type": existing_kind.upper() or existing_kind,
                        "overlap_pct": float(coverage),
                        "kept": kept_label,
                    }
                )
            return False, "dedup_priority_loss" if new_kind == "rb" else None

        replace_existing = False
        reject_reason: str | None = None
        if strategy == "scored":
            score_weights = weights or {}
            existing_score = _block_score(existing, latest_idx=latest_idx, weights=score_weights)
            new_score = _block_score(block, latest_idx=latest_idx, weights=score_weights)
            block["score"] = new_score
            existing["score"] = existing_score
            if new_score > existing_score + 1e-9:
                kept_label = new_kind.upper() or new_kind
                replace_existing = True
            elif new_score < existing_score - 1e-9:
                reject_reason = "dedup_score_loss" if new_kind == "rb" else None
            else:
                existing_tf = existing.get("tf")
                existing_tf_rank = _timeframe_priority(
                    str(existing_tf) if existing_tf is not None else None
                )
                if new_tf_rank > existing_tf_rank:
                    kept_label = new_kind.upper() or new_kind
                    replace_existing = True
                elif abs(new_tf_rank - existing_tf_rank) <= 1e-9:
                    if new_len > existing_len + 1e-12:
                        kept_label = new_kind.upper() or new_kind
                        replace_existing = True
                    elif abs(new_len - existing_len) <= 1e-12 and new_created >= int(
                        existing.get("created_at", 0)
                    ):
                        kept_label = new_kind.upper() or new_kind
                        replace_existing = True
        else:
            existing_tf = existing.get("tf")
            existing_tf_rank = _timeframe_priority(
                str(existing_tf) if existing_tf is not None else None
            )
            if new_tf_rank > existing_tf_rank:
                kept_label = new_kind.upper() or new_kind
                replace_existing = True
            elif abs(new_tf_rank - existing_tf_rank) <= 1e-9:
                if new_len > existing_len + 1e-12:
                    kept_label = new_kind.upper() or new_kind
                    replace_existing = True
                elif abs(new_len - existing_len) <= 1e-12 and new_created >= int(
                    existing.get("created_at", 0)
                ):
                    kept_label = new_kind.upper() or new_kind
                    replace_existing = True
        if trace is not None:
            trace.append(
                {
                    "type": existing_kind.upper() or existing_kind,
                    "overlap_pct": float(coverage),
                    "kept": kept_label,
                    "strategy": strategy,
                    "new_score": block.get("score"),
                    "existing_score": existing.get("score"),
                }
            )
        if replace_existing:
            blocks[idx] = block
            return True, None
        if new_kind == "rb":
            reject_reason = "dedup_priority_loss"
        return False, reject_reason

    blocks.append(block)
    return True, None


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
        "rb_impulse_diag": {},
        "bb_flow": {"ob_found": 0, "invalidated": 0, "opposite_bos": 0},
        "bb_reject": {"no_ob": 0, "no_invalidation": 0, "no_opposite_bos": 0, "proximity_fail": 0},
        "pb_metrics": {"attempts": 0, "built": 0, "by_tf": {}},
        "pb_trace": [],
        "sweep_to_choch_bars": [],
        "trace": [],
    }
    rb_debug: Dict[str, Any] = diagnostics.setdefault("rb_debug", {})

    if not candles:
        return [], diagnostics

    cfg = config or SMCConfig()
    if cfg.feature_strict_legacy_mode:
        cfg.feature_extended_windows = False
        cfg.feature_block_scored_dedup = False
        cfg.allow_bos_substitute = False
        cfg.block_dedup_strategy = "priority"
        cfg.sweep_choch_extension_factor = 1.0
    if not cfg.feature_extended_windows:
        cfg.sweep_choch_extension_factor = 1.0
    if not cfg.feature_block_scored_dedup:
        cfg.block_dedup_strategy = "priority"
    dedup_strategy = (cfg.block_dedup_strategy or "priority").lower()
    if cfg.feature_block_scored_dedup and dedup_strategy == "priority":
        dedup_strategy = "scored"
    if dedup_strategy not in {"priority", "scored"}:
        dedup_strategy = "priority"
    structure_events = _normalise_events(structure_flags)
    base_ob_zones = _normalise_ob_zones(ob_zones)
    liquidity = _liquidity_levels(liquidity_levels)

    blocks: List[MutableMapping[str, object]] = []

    latest_idx = len(candles) - 1 if candles else 0
    dedup_weights = {
        "tf": max(0.0, float(cfg.dedup_weight_tf)),
        "recency": max(0.0, float(cfg.dedup_weight_recency)),
        "confidence": max(0.0, float(cfg.dedup_weight_confidence)),
    }

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

    epsilon_tick = float(tick_size) if tick_size else 0.0

    block_type_labels = {
        "bb": "breaker block",
        "mb": "mitigation block",
        "rb": "reversal block",
    }
    block_priorities = {"rb": 50, "bb": 40, "ob": 30, "mb": 20, "pb": 10}

    def _append_block(
        *,
        kind: str,
        price_range: tuple[float, float],
        created_idx: int,
        direction: str,
        created_at: int,
        trace: List[Dict[str, object]] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> tuple[bool, str | None, List[Dict[str, object]] | None]:
        span = _range_size(price_range)
        if span < cfg.min_block_size - 1e-12:
            return False, "min_block_size", None
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
        if extra:
            for key, value in extra.items():
                block[key] = value
        block.setdefault("confidence", float(block.get("confidence", 1.0)))
        trace_list: List[Dict[str, object]] | None = [] if trace is not None else None
        appended, reject = _deduplicate_blocks(
            blocks,
            block,
            priorities=block_priorities,
            strategy=dedup_strategy,
            weights=dedup_weights,
            latest_idx=latest_idx,
            trace=trace_list,
        )
        if appended:
            if trace is not None:
                trace.clear()
                if trace_list:
                    trace.extend(trace_list)
            return True, None, trace_list
        if trace is not None:
            trace.clear()
            if trace_list:
                trace.extend(trace_list)
        return False, reject, trace_list

    # Breaker Blocks
    bb_flow: Dict[str, int] = diagnostics.get("bb_flow", {})
    bb_reject: Dict[str, int] = diagnostics.get("bb_reject", {})
    opposite_limit_base = cfg.opposite_bos_max_bars.get(timeframe) if isinstance(cfg.opposite_bos_max_bars, Mapping) else None
    if opposite_limit_base is None and isinstance(cfg.opposite_bos_max_bars, Mapping):
        opposite_limit_base = cfg.opposite_bos_max_bars.get("default")
    if opposite_limit_base is None:
        opposite_limit_base = cfg.opposite_bos_default_max_bars
    opposite_limit_base = max(1, int(opposite_limit_base))
    opposite_limit_soft = int(round(opposite_limit_base * max(1.0, cfg.sweep_choch_extension_factor)))
    opposite_limit_extended = opposite_limit_soft
    soft_cap = int(round(opposite_limit_extended * 1.5))

    for zone in base_ob_zones:
        direction = _direction_from_zone(zone.get("type"))
        if direction not in {"supply", "demand"}:
            continue
        created_ts = int(zone.get("created_at", 0))
        if window_start_ms is not None and created_ts < window_start_ms:
            continue
        if window_end_ms is not None and created_ts > window_end_ms:
            continue
        created_idx = _find_candle_index_by_ts(candles, created_ts)
        if created_idx is None:
            continue
        body_range = _candle_body_range(candles[created_idx])
        if _range_size(body_range) <= 0:
            continue
        zone_range = _ensure_sorted_range(zone.get("range", body_range))
        bb_flow["ob_found"] = bb_flow.get("ob_found", 0) + 1
        invalidation_idx: int | None = None
        for idx in range(created_idx + 1, len(candles)):
            ts_idx = int(candles[idx]["t"])
            if window_end_ms is not None and ts_idx > window_end_ms:
                break
            close_value = float(candles[idx]["c"])
            if direction == "supply":
                if close_value > zone_range[1] + epsilon_tick:
                    invalidation_idx = idx
                    break
            else:
                if close_value < zone_range[0] - epsilon_tick:
                    invalidation_idx = idx
                    break
        if invalidation_idx is None:
            bb_reject["no_invalidation"] = bb_reject.get("no_invalidation", 0) + 1
            continue
        bb_flow["invalidated"] = bb_flow.get("invalidated", 0) + 1
        invalidation_ts = int(candles[invalidation_idx]["t"])
        desired_direction = _opposite_direction(direction)
        bb_confidence = 1.0
        bos_event: Mapping[str, object] | None = None
        for event in structure_events:
            if str(event.get("kind")) != "bos":
                continue
            if window_end_ms is not None and int(event.get("t", 0)) > window_end_ms:
                continue
            if int(event.get("t", 0)) < invalidation_ts:
                continue
            if event.get("direction") and not _direction_equivalent(
                event.get("direction"), desired_direction
            ):
                continue
            event_idx = _find_candle_index_by_ts(candles, int(event.get("t", 0)))
            if event_idx is None:
                continue
            distance = event_idx - invalidation_idx
            if distance > opposite_limit_extended:
                if cfg.feature_extended_windows and distance <= max(opposite_limit_extended, soft_cap):
                    bb_flow.setdefault("opposite_bos_soft", 0)
                    bb_flow["opposite_bos_soft"] += 1
                    bb_confidence = max(0.2, opposite_limit_extended / max(distance, 1))
                else:
                    bb_reject["proximity_fail"] = bb_reject.get("proximity_fail", 0) + 1
                    continue
            bos_event = event
            break
        if bos_event is None:
            bb_reject["no_opposite_bos"] = bb_reject.get("no_opposite_bos", 0) + 1
            continue
        bb_flow["opposite_bos"] = bb_flow.get("opposite_bos", 0) + 1
        bos_ts = int(bos_event.get("t", invalidation_ts))
        bos_idx = _find_candle_index_by_ts(candles, bos_ts) or invalidation_idx
        _append_block(
            kind="bb",
            price_range=body_range,
            created_idx=bos_idx,
            direction=desired_direction,
            created_at=bos_ts,
            extra={"confidence": bb_confidence},
        )
    if bb_flow.get("ob_found", 0) == 0:
        bb_reject["no_ob"] = max(bb_reject.get("no_ob", 0), 1)

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
        diagnostics.setdefault("pb_trace", []).append({"tf": timeframe, "stage": "eq", "reason_code": "no_eq"})

    stage_counts = {
        "eq": {"in": len(eq_candidates), "out": len(eq_candidates)},
        "sweep": {"in": len(eq_candidates), "out": 0},
        "structure": {"in": 0, "out": 0},
        "base": {"in": 0, "out": 0},
        "impulse": {"in": 0, "out": 0},
    }
    sweep_success = 0
    structure_success = 0
    base_success = 0
    impulse_success = 0
    sweep_limit_base = cfg.sweep_choch_max_bars.get(timeframe) if isinstance(cfg.sweep_choch_max_bars, Mapping) else None
    if sweep_limit_base is None and isinstance(cfg.sweep_choch_max_bars, Mapping):
        sweep_limit_base = cfg.sweep_choch_max_bars.get("default")
    if sweep_limit_base is None:
        sweep_limit_base = cfg.sweep_choch_default_max_bars
    sweep_limit_base = max(1, int(sweep_limit_base))
    sweep_limit_extended = int(round(sweep_limit_base * max(1.0, cfg.sweep_choch_extension_factor)))
    sweep_hist_local: List[Dict[str, Any]] = []
    pb_metrics_block = diagnostics["pb_metrics"]
    pb_trace_list = diagnostics["pb_trace"]
    metrics_by_tf = pb_metrics_block["by_tf"].setdefault(timeframe, {"attempts": 0, "built": 0})

    bos_events_window = [event for event in structure_events_window if str(event.get("kind")) == "bos"]
    if not choch_events_window and not (cfg.allow_bos_substitute and bos_events_window):
        diagnostics["rb_reject_no_choch"] = max(diagnostics["rb_reject_no_choch"], 1)
        reject_counters["no_choch"] = max(reject_counters.get("no_choch", 0), 1)

    def _range_block_base(
        sweep_idx: int, choch_idx: int, direction: str
    ) -> tuple[tuple[float, float], int] | None:
        if sweep_idx is None or choch_idx is None or not candles:
            return None
        min_bars = max(1, int(cfg.base_min_bars))
        max_bars = max(min_bars, int(cfg.base_max_bars))
        lookback_limit = max(0, choch_idx - 49)
        search_start = max(lookback_limit, sweep_idx - 1, 0)
        best_candidate: tuple[tuple[float, float], int] | None = None
        for end_idx in range(choch_idx, search_start - 1, -1):
            base_ts = int(candles[end_idx]["t"])
            if window_start_ms is not None and base_ts < window_start_ms:
                continue
            if window_end_ms is not None and base_ts > window_end_ms:
                continue
            available = end_idx - search_start + 1
            max_length = min(max_bars, available)
            if max_length < min_bars:
                continue
            for length in range(max_length, min_bars - 1, -1):
                start_idx = end_idx - length + 1
                if start_idx < search_start:
                    continue
                body_ranges: List[tuple[float, float]] = []
                support_bar_found = False
                valid_segment = True
                for idx in range(start_idx, end_idx + 1):
                    candle = candles[idx]
                    body_range = _candle_body_range(candle)
                    if _range_size(body_range) <= 0.0:
                        valid_segment = False
                        break
                    body_ranges.append(body_range)
                    open_price = float(candle.get("o", 0.0))
                    close_price = float(candle.get("c", 0.0))
                    if direction == "demand" and close_price < open_price:
                        support_bar_found = True
                    if direction == "supply" and close_price > open_price:
                        support_bar_found = True
                if not valid_segment or not support_bar_found:
                    continue
                overlap_ok = True
                for left, right in zip(body_ranges, body_ranges[1:]):
                    min_span = min(_range_size(left), _range_size(right))
                    if min_span <= 0.0:
                        overlap_ok = False
                        break
                    if _overlap_size(left, right) < cfg.base_min_overlap * min_span:
                        overlap_ok = False
                        break
                if not overlap_ok:
                    continue
                combined_low = min(body[0] for body in body_ranges)
                combined_high = max(body[1] for body in body_ranges)
                span = combined_high - combined_low
                atr_val = _atr_value(end_idx)
                if (
                    math.isfinite(atr_val)
                    and atr_val > 0.0
                    and span > cfg.base_max_atr * atr_val + 1e-12
                ):
                    continue
                if span < cfg.min_block_size - 1e-12:
                    continue
                best_candidate = ((combined_low, combined_high), end_idx)
                break
            if best_candidate is not None:
                break
        return best_candidate

    def _fallback_ob_base(
        direction: str, choch_idx: int, choch_ts: int, eq_price: float
    ) -> tuple[tuple[float, float], int] | None:
        if not candles:
            return None
        max_age = max(1, int(cfg.base_fallback_max_age))
        distance_limit = float(cfg.base_fallback_max_distance_atr)
        atr_ref = _atr_value(choch_idx)
        for zone in reversed(base_ob_zones):
            if _direction_from_zone(zone.get("type")) != direction:
                continue
            created_at = int(zone.get("created_at", 0))
            if created_at >= choch_ts:
                continue
            if window_start_ms is not None and created_at < window_start_ms:
                continue
            if window_end_ms is not None and created_at > window_end_ms:
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
            if base_idx is None or base_idx > choch_idx:
                continue
            if choch_idx - base_idx > max_age:
                continue
            atr_val = atr_ref if math.isfinite(atr_ref) and atr_ref > 0.0 else _atr_value(base_idx)
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
            return base_range, base_idx
        return None

    for eq_entry in eq_candidates:
        pb_metrics_block["attempts"] = pb_metrics_block.get("attempts", 0) + 1
        metrics_by_tf["attempts"] = metrics_by_tf.get("attempts", 0) + 1
        pb_record: Dict[str, Any] = {
            "tf": timeframe,
            "eq_type": eq_entry.get("type"),
            "eq_idx": int(eq_entry.get("second_idx", -1)),
            "stage": "eq",
            "reason_code": None,
            "bars": {},
        }
        block_confidence = 1.0

        def _push_pb_trace(reason: str, stage: str) -> None:
            pb_record["reason_code"] = reason
            pb_record["stage"] = stage
            entry: Dict[str, Any] = {
                "tf": pb_record.get("tf"),
                "eq_type": pb_record.get("eq_type"),
                "eq_idx": pb_record.get("eq_idx"),
                "stage": stage,
                "reason_code": reason,
                "bars": dict(pb_record.get("bars", {})),
            }
            flags = pb_record.get("flags")
            if isinstance(flags, list) and flags:
                entry["flags"] = list(flags)
            confidence_value = pb_record.get("confidence")
            if confidence_value is not None:
                entry["confidence"] = confidence_value
            pb_trace_list.append(entry)
        flow_counters["eq_found"] = flow_counters.get("eq_found", 0) + 1
        trend_direction = "up" if eq_entry["type"] == "eql" else "down"
        block_direction = "demand" if trend_direction == "up" else "supply"
        eq_price = float(eq_entry["price"])
        rb_debug["last_eq_type"] = eq_entry.get("type")
        rb_debug["last_eq_first_idx"] = int(eq_entry.get("first_idx", -1))
        rb_debug["last_eq_second_idx"] = int(eq_entry.get("second_idx", -1))

        sweep_idx: int | None = None
        recovery_idx: int | None = None
        start_idx = int(eq_entry["second_idx"]) + 1
        stage_counts["sweep"]["in"] += 1
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
            _push_pb_trace("no_sweep", "sweep")
            continue

        flow_counters["sweep"] = flow_counters.get("sweep", 0) + 1
        rb_debug["last_sweep_idx"] = sweep_idx
        rb_debug["last_recovery_idx"] = recovery_idx
        stage_counts["sweep"]["out"] += 1
        sweep_success += 1
        pb_record["stage"] = "sweep"
        pb_record.setdefault("bars", {})["sweep_idx"] = int(sweep_idx)
        pb_record.setdefault("bars", {})["recovery_idx"] = int(recovery_idx)

        min_choch_idx = sweep_idx + 1
        choch_event: Mapping[str, object] | None = None
        choch_idx: int | None = None
        structure_kind = "choch"
        stage_counts["structure"]["in"] += 1
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

        if (choch_event is None or choch_idx is None) and cfg.allow_bos_substitute:
            for event in bos_events_window:
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
                structure_kind = "bos"
                block_confidence *= 0.85
                pb_record.setdefault("flags", []).append("bos_substitute")
                break

        if choch_event is None or choch_idx is None:
            diagnostics["rb_reject_no_choch"] += 1
            reject_counters["no_choch"] = reject_counters.get("no_choch", 0) + 1
            _push_pb_trace("no_structure", "structure")
            continue

        if window_start_ms is not None:
            assert window_start_ms <= int(choch_event.get("t", 0))
        if window_end_ms is not None:
            assert int(choch_event.get("t", 0)) <= window_end_ms

        flow_counters["choch"] = flow_counters.get("choch", 0) + 1
        rb_debug["last_choch_idx"] = choch_idx
        rb_debug["structure_kind"] = structure_kind
        distance_bars = max(1, choch_idx - sweep_idx)
        rb_debug["sweep_to_choch_bars"] = distance_bars
        pb_record.setdefault("bars", {})["sweep_to_structure"] = distance_bars
        sweep_hist_local.append({"tf": timeframe, "bars": distance_bars, "limit": sweep_limit_extended, "kind": structure_kind})
        if distance_bars > sweep_limit_extended:
            if cfg.feature_extended_windows:
                adjustment = max(1.0, distance_bars / max(sweep_limit_extended, 1))
                block_confidence *= max(0.2, 1.0 / adjustment)
                pb_record.setdefault("flags", []).append("structure_extended")
            else:
                diagnostics["rb_reject_no_choch"] += 1
                reject_counters["no_choch"] = reject_counters.get("no_choch", 0) + 1
                _push_pb_trace("sweep_choch_window", "structure")
                continue
        stage_counts["structure"]["out"] += 1
        structure_success += 1
        pb_record["stage"] = "structure"
        pb_record["structure_kind"] = structure_kind

        base_candidate = _range_block_base(sweep_idx, choch_idx, block_direction)
        fallback_used = False
        choch_ts = int(choch_event.get("t", 0))
        stage_counts["base"]["in"] += 1
        if base_candidate is None and cfg.allow_base_fallback:
            base_candidate = _fallback_ob_base(
                block_direction, choch_idx, choch_ts, eq_price
            )
            if base_candidate is not None:
                fallback_used = True
                block_confidence *= 0.9
                pb_record.setdefault("flags", []).append("base_fallback")

        if base_candidate is None:
            diagnostics["rb_reject_no_base"] += 1
            reject_counters["no_base"] = reject_counters.get("no_base", 0) + 1
            _push_pb_trace("no_base", "base")
            continue

        flow_counters["base"] = flow_counters.get("base", 0) + 1
        if fallback_used:
            diagnostics["base_fallback_used"] = True
        stage_counts["base"]["out"] += 1
        base_success += 1
        pb_record["stage"] = "base"

        base_range, base_idx = base_candidate
        base_span = float(base_range[1] - base_range[0])
        base_mean = (float(base_range[0]) + float(base_range[1])) / 2.0
        rb_debug["last_base_idx"] = base_idx
        rb_debug["last_base_range"] = [float(base_range[0]), float(base_range[1])]
        if tick_size and abs(base_range[1] - base_range[0]) < float(tick_size) - 1e-12:
            diagnostics["base_tick_collapse"] = int(diagnostics.get("base_tick_collapse", 0)) + 1

        impulse_candidates = 0
        directional_candidates: List[Dict[str, float]] = []
        cover_max = 0.0
        for offset in range(1, 4):
            idx = choch_idx + offset
            if idx >= len(candles):
                break
            ts_idx = int(candles[idx]["t"])
            if window_end_ms is not None and ts_idx > window_end_ms:
                break
            impulse_candidates += 1
            open_price = float(candles[idx]["o"])
            close_price = float(candles[idx]["c"])
            high_price = float(candles[idx]["h"])
            low_price = float(candles[idx]["l"])
            body_val = abs(close_price - open_price)
            range_val = high_price - low_price
            atr_val = _atr_value(idx)
            sigma_val = _sigma_value(idx)
            cover_ratio = 0.0
            directional = True
            if block_direction == "demand":
                if close_price <= open_price or close_price < base_mean:
                    directional = False
                elif base_span > 0.0:
                    cover_ratio = (close_price - open_price) / max(base_span, 1e-12)
            else:
                if close_price >= open_price or close_price > base_mean:
                    directional = False
                elif base_span > 0.0:
                    cover_ratio = (open_price - close_price) / max(base_span, 1e-12)
            if directional:
                cover_max = max(cover_max, cover_ratio)
                directional_candidates.append(
                    {
                        "idx": float(idx),
                        "body": body_val,
                        "range": range_val,
                        "cover": cover_ratio,
                        "atr": float(atr_val) if math.isfinite(atr_val) else math.nan,
                        "sigma": float(sigma_val) if math.isfinite(sigma_val) else math.nan,
                    }
                )

        impulse_idx: int | None = None
        impulse_atr = math.nan
        impulse_sigma = math.nan
        body_dir_max = 0.0
        range_dir_max = 0.0
        for candidate in directional_candidates:
            body_dir_max = max(body_dir_max, candidate["body"])
            range_dir_max = max(range_dir_max, candidate["range"])
            atr_val = candidate["atr"]
            sigma_val = candidate["sigma"]
            k_body = 1.0
            k_range = 1.5
            if math.isfinite(atr_val) and atr_val > 0.0:
                if math.isfinite(sigma_val) and sigma_val < 0.5 * atr_val:
                    k_body = max(0.8, 0.7)
                    k_range = max(1.2, 1.1)
            cond1 = bool(math.isfinite(atr_val) and atr_val > 0.0 and candidate["body"] >= k_body * atr_val)
            cond2 = bool(math.isfinite(atr_val) and atr_val > 0.0 and candidate["range"] >= k_range * atr_val)
            cond3 = candidate["cover"] >= cfg.impulse_min_cover
            if cond1 or cond2 or cond3:
                impulse_idx = int(candidate["idx"])
                if math.isfinite(atr_val) and atr_val > 0.0:
                    impulse_atr = atr_val
                if math.isfinite(sigma_val):
                    impulse_sigma = sigma_val
                break

        diagnostics["rb_impulse_diag"] = {
            "impulse_candidates": impulse_candidates,
            "impulse_body_max": body_dir_max,
            "impulse_range_max": range_dir_max,
            "cover_pct_vs_base": cover_max,
            "atr": impulse_atr if math.isfinite(impulse_atr) else None,
            "sigma20": impulse_sigma if math.isfinite(impulse_sigma) else None,
        }

        stage_counts["impulse"]["in"] += 1
        if impulse_idx is None:
            diagnostics["rb_reject_no_impulse"] += 1
            reject_counters["no_impulse"] = reject_counters.get("no_impulse", 0) + 1
            pb_record["stage"] = "impulse"
            pb_record["reason_code"] = "no_impulse"
            pb_trace_list.append(dict(pb_record))
            continue

        diagnostics["rb_impulse_ok"] = True
        flow_counters["impulse"] = flow_counters.get("impulse", 0) + 1
        impulse_ts = int(candles[impulse_idx].get("t", choch_ts))
        rb_debug["last_impulse_idx"] = impulse_idx
        stage_counts["impulse"]["out"] += 1
        impulse_success += 1
        pb_record["stage"] = "impulse"

        base_low = float(base_range[0])
        base_high = float(base_range[1])
        rb_direction = "up" if trend_direction == "up" else "down"
        tick_value = float(tick_size) if tick_size else None
        bot_round = _round_to_tick(base_low, tick_value)
        top_round = _round_to_tick(base_high, tick_value)
        use_raw_bounds = bool(tick_value and top_round <= bot_round)
        rb_bot = base_low if use_raw_bounds else bot_round
        rb_top = base_high if use_raw_bounds else top_round
        rb_mid = _round_to_tick((base_low + base_high) / 2.0, tick_value)
        atr_for_min_range = impulse_atr
        if not (math.isfinite(atr_for_min_range) and atr_for_min_range > 0.0):
            atr_for_min_range = _atr_value(impulse_idx)
        if not (math.isfinite(atr_for_min_range) and atr_for_min_range > 0.0):
            atr_for_min_range = _atr_value(base_idx)
        min_rb_range = 0.0
        if tick_value and tick_value > 0.0:
            min_rb_range = max(min_rb_range, 2.0 * tick_value)
        if math.isfinite(atr_for_min_range) and atr_for_min_range > 0.0:
            min_rb_range = max(min_rb_range, 0.1 * atr_for_min_range)

        rb_debug["emit_attempt"] = True
        rb_debug["outside_window"] = False
        rb_debug["rb_bounds_raw"] = [base_low, base_high]
        rb_debug["rb_bounds_rounded"] = [bot_round, top_round]
        rb_debug["min_rb_range"] = min_rb_range
        rb_debug["overlap_with"] = []

        span_below_min = bool(min_rb_range > 0.0 and base_span < min_rb_range - 1e-12)
        allow_small_span = cover_max >= cfg.impulse_min_cover - 1e-12
        if span_below_min and not allow_small_span:
            rb_debug["emit_reason"] = "min_range"
            diagnostics["reason"] = "min_range"
            pb_record.setdefault("flags", []).append("min_range")
            _push_pb_trace("range_collapse", "impulse")
            continue

        overlap_entries: List[Dict[str, object]] = []
        block_confidence = max(0.05, min(block_confidence, 1.0))
        appended, reject_reason, _ = _append_block(
            kind="rb",
            price_range=(base_low, base_high),
            created_idx=impulse_idx,
            direction=block_direction,
            created_at=impulse_ts,
            trace=overlap_entries,
            extra={
                "top": float(rb_top),
                "bot": float(rb_bot),
                "mid": float(rb_mid),
                "direction": rb_direction,
                "confidence": block_confidence,
                "structure_kind": structure_kind,
                "sweep_to_choch_bars": distance_bars,
                "range_mode": "unrounded_range" if use_raw_bounds else "rounded",
            },
        )
        rb_debug["overlap_with"] = overlap_entries
        if appended:
            diagnostics["rb_raw_count"] += 1
            rb_debug["emit_reason"] = "emitted"
            diagnostics.pop("reason", None)
            pb_record["confidence"] = round(block_confidence, 3)
            _push_pb_trace("built", "complete")
            pb_metrics_block["built"] = pb_metrics_block.get("built", 0) + 1
            metrics_by_tf["built"] = metrics_by_tf.get("built", 0) + 1
        else:
            emit_reason = reject_reason or "dedup_priority_loss"
            rb_debug["emit_reason"] = emit_reason
            diagnostics["reason"] = emit_reason
            pb_record.setdefault("flags", []).append("dedup")
            _push_pb_trace(emit_reason, "dedup")
            continue

    has_rb = any(block.get("kind") == "rb" for block in blocks)
    has_bb = any(block.get("kind") == "bb" for block in blocks)

    stage_counts["sweep"]["out"] = sweep_success
    stage_counts["structure"]["out"] = structure_success
    stage_counts["base"]["out"] = base_success
    stage_counts["impulse"]["out"] = impulse_success
    diagnostics.setdefault("sweep_to_choch_bars", []).extend(sweep_hist_local)

    reason_lookup = {
        "eq": ("rb_reject_no_eq", "no_eq"),
        "sweep": ("rb_reject_no_sweep", "no_sweep"),
        "structure": ("rb_reject_no_choch", "no_choch"),
        "base": ("rb_reject_no_base", "no_base"),
        "impulse": ("rb_reject_no_impulse", "no_impulse"),
    }
    trace_entries = diagnostics.setdefault("trace", [])
    for stage_key, metrics in stage_counts.items():
        reject_field, reason_code = reason_lookup.get(stage_key, (None, "ok"))
        top_reason = "ok"
        if reject_field and diagnostics.get(reject_field, 0):
            top_reason = reason_code
        trace_entries.append(
            {
                "tf": timeframe,
                "stage": stage_key,
                "in": int(metrics.get("in", 0)),
                "out": int(metrics.get("out", 0)),
                "top_reason": top_reason,
            }
        )

    if not has_rb:
        reason: str | None = None
        if diagnostics.get("reason"):
            reason = str(diagnostics["reason"])
        elif rb_debug.get("emit_attempt") and rb_debug.get("emit_reason"):
            reason = str(rb_debug.get("emit_reason"))
        elif diagnostics.get("rb_reject_no_impulse"):
            reason = "no_impulse"
        elif diagnostics.get("rb_reject_no_base"):
            reason = "no_base"
        elif diagnostics.get("rb_reject_no_choch"):
            reason = "no_choch"
        elif diagnostics.get("rb_reject_no_sweep"):
            reason = "no_sweep"
        elif diagnostics.get("rb_reject_no_eq"):
            reason = "no_eq"
        diagnostics.setdefault("reason", reason or "no_reversal_blocks")
    if not has_bb:
        bb_reason: str | None = None
        if bb_flow.get("ob_found") == 0:
            bb_reason = "no_ob"
        elif bb_flow.get("invalidated") == 0:
            bb_reason = "no_invalidated_ob"
        elif bb_flow.get("opposite_bos") == 0:
            bb_reason = "no_opposite_bos"
        diagnostics.setdefault("bb_reason", bb_reason)

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
