"""Smart Money Concepts block detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class SMCConfig:
    """Configuration parameters for SMC block detection."""

    min_block_size: float = 0.0
    displacement_factor: float = 1.5
    displacement_lookback: int = 5
    ttl_bars: int = 50


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
) -> List[MutableMapping[str, object]]:
    """Detect breaker, mitigation and reversal blocks on the supplied candles."""

    if not candles:
        return []

    cfg = config or SMCConfig()
    structure_events = _normalise_events(structure_flags)
    base_ob_zones = _normalise_ob_zones(ob_zones)
    liquidity = _liquidity_levels(liquidity_levels)

    blocks: List[MutableMapping[str, object]] = []

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
    sweep_events: List[tuple[int, str]] = []
    for idx, candle in enumerate(candles):
        body_low, body_high = _candle_body_range(candle)
        low, high = _candle_range(candle)
        close_price = float(candle.get("c", body_high))
        for liquidity_key, price in liquidity:
            if price <= 0:
                continue
            if liquidity_key in {"eqh", "pdh"}:
                if high >= price and close_price < price:
                    sweep_events.append((idx, "down"))
            elif liquidity_key in {"eql", "pdl"}:
                if low <= price and close_price > price:
                    sweep_events.append((idx, "up"))
    sweep_events.sort()

    for event in structure_events:
        if event["kind"] != "choch":
            continue
        raw_direction = event.get("direction")
        direction_label = _normalise_direction_label(raw_direction)
        if direction_label not in {"up", "down", "demand", "supply"}:
            continue
        trend_direction = "up" if direction_label in {"up", "demand"} else "down"
        event_idx = _find_candle_index_by_ts(candles, int(event["t"]))
        if event_idx is None:
            continue
        sweep_match = None
        for idx, sweep_dir in reversed(sweep_events):
            if idx > event_idx:
                continue
            if (trend_direction == "up" and sweep_dir == "up") or (
                trend_direction == "down" and sweep_dir == "down"
            ):
                sweep_match = (idx, sweep_dir)
                break
        if sweep_match is None:
            continue
        impulse_idx = None
        for idx in range(event_idx + 1, len(candles)):
            candle = candles[idx]
            body_low, body_high = _candle_body_range(candle)
            body_span = body_high - body_low
            avg_body = _average_body(candles, idx, cfg.displacement_lookback)
            if avg_body <= 0.0:
                continue
            if body_span < cfg.displacement_factor * avg_body:
                continue
            close_price = float(candle.get("c", body_high))
            open_price = float(candle.get("o", body_low))
            if trend_direction == "up" and close_price <= open_price:
                continue
            if trend_direction == "down" and close_price >= open_price:
                continue
            impulse_idx = idx
            break
        if impulse_idx is None or impulse_idx == 0:
            continue
        base_idx = impulse_idx - 1
        base_candle = candles[base_idx]
        base_range = _candle_body_range(base_candle)
        if _range_size(base_range) < cfg.min_block_size - 1e-12:
            continue
        created_at = int(candles[impulse_idx].get("t", event["t"]))
        block_direction = "demand" if trend_direction == "up" else "supply"
        _append_block(
            kind="rb",
            price_range=base_range,
            created_idx=base_idx,
            direction=block_direction,
            created_at=created_at,
        )

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

    return blocks
