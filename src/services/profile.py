"""Volume profile and TPO helpers for inspection payloads."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(slots=True)
class VolumeProfile:
    """Aggregated volume distribution for a given set of candles."""

    prices: List[float]
    volumes: List[float]
    poc: float
    vah: float
    val: float
    diagnostics: Dict[str, str]


def _ensure_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _in_session(moment: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= moment < end
    return moment >= start or moment < end


def _extract_timestamp(candle: Mapping[str, Any]) -> int | None:
    for key in ("t", "time", "openTime"):
        raw = candle.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def split_by_sessions(
    candles: Sequence[Mapping[str, Any]],
    sessions: Iterable[Tuple[str, dtime, dtime]],
    tz: timezone = timezone.utc,
) -> Dict[Tuple[date, str], List[Mapping[str, Any]]]:
    """Group candles by (date, session) buckets similar to VWAP sessions."""

    buckets: Dict[Tuple[date, str], List[Mapping[str, Any]]] = {}
    session_list = list(sessions)
    if not candles or not session_list:
        return buckets

    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        ts = _extract_timestamp(candle)
        if ts is None:
            continue
        dt = datetime.fromtimestamp(ts / 1000.0, tz=tz)
        moment = dt.time()
        for name, start, end in session_list:
            if not _in_session(moment, start, end):
                continue
            session_date = dt.date()
            if start > end and moment < end:
                session_date = session_date - timedelta(days=1)
            bucket_key = (session_date, name)
            bucket = buckets.setdefault(bucket_key, [])
            bucket.append(dict(candle))
    return buckets


def _compute_true_ranges(candles: Sequence[Mapping[str, Any]]) -> List[float]:
    true_ranges: List[float] = []
    prev_close: float | None = None
    for candle in candles:
        high = _ensure_float(candle.get("h") or candle.get("high"))
        low = _ensure_float(candle.get("l") or candle.get("low"))
        close = _ensure_float(candle.get("c") or candle.get("close"))
        if not math.isfinite(high) or not math.isfinite(low):
            continue
        high = float(high)
        low = float(low)
        current_close = close
        tr_components = [high - low]
        if prev_close is not None:
            tr_components.append(abs(high - prev_close))
            tr_components.append(abs(low - prev_close))
        true_range = max(tr_components) if tr_components else 0.0
        if true_range >= 0:
            true_ranges.append(true_range)
        prev_close = current_close if math.isfinite(current_close) else prev_close
    return true_ranges


def _select_bin_size(
    candles: Sequence[Mapping[str, Any]],
    *,
    tick_size: float | None,
    adaptive_bins: bool,
    min_bins: int,
) -> float | None:
    if tick_size is not None:
        try:
            size = float(tick_size)
        except (TypeError, ValueError):
            size = 0.0
        if size > 0:
            return size

    if adaptive_bins:
        true_ranges = _compute_true_ranges(candles)
        if true_ranges:
            avg_tr = sum(true_ranges) / len(true_ranges)
            bin_size = max(avg_tr * 0.5, 0.0)
            if bin_size > 0:
                return bin_size

    n = len(candles)
    if n <= 0:
        return None
    bins = max(min_bins, math.ceil(math.log2(n)) + 1 if n > 1 else min_bins)
    highs = [_ensure_float(c.get("h") or c.get("high")) for c in candles]
    lows = [_ensure_float(c.get("l") or c.get("low")) for c in candles]
    highs = [value for value in highs if math.isfinite(value)]
    lows = [value for value in lows if math.isfinite(value)]
    if not highs or not lows:
        return None
    price_range = max(highs) - min(lows)
    if price_range <= 0:
        return None
    return price_range / bins


def build_volume_profile(
    candles: Sequence[Mapping[str, Any]],
    *,
    tick_size: float | None,
    adaptive_bins: bool,
    value_area_pct: float = 0.70,
    min_bins: int = 10,
) -> VolumeProfile:
    """Construct a volume profile from a sequence of OHLCV candles."""

    entries = [dict(candle) for candle in candles if isinstance(candle, Mapping)]
    diagnostics: Dict[str, str] = {}

    if len(entries) < 30:
        diagnostics["warn"] = "too few bars"

    bin_size = _select_bin_size(entries, tick_size=tick_size, adaptive_bins=adaptive_bins, min_bins=min_bins)
    if not bin_size or bin_size <= 0:
        prices: List[float] = []
        volumes: List[float] = []
        diagnostics.setdefault("warn", "degenerate profile")
        return VolumeProfile(prices=prices, volumes=volumes, poc=math.nan, vah=math.nan, val=math.nan, diagnostics=diagnostics)

    lows = [_ensure_float(item.get("l") or item.get("low")) for item in entries]
    highs = [_ensure_float(item.get("h") or item.get("high")) for item in entries]
    lows = [value for value in lows if math.isfinite(value)]
    highs = [value for value in highs if math.isfinite(value)]
    if not lows or not highs:
        diagnostics.setdefault("warn", "degenerate profile")
        return VolumeProfile(prices=[], volumes=[], poc=math.nan, vah=math.nan, val=math.nan, diagnostics=diagnostics)

    min_price = min(lows)
    max_price = max(highs)
    if max_price <= min_price:
        diagnostics.setdefault("warn", "degenerate profile")
        return VolumeProfile(prices=[], volumes=[], poc=math.nan, vah=math.nan, val=math.nan, diagnostics=diagnostics)

    start_edge = math.floor(min_price / bin_size) * bin_size
    if start_edge > min_price:
        start_edge -= bin_size
    span = max_price - start_edge
    steps = math.ceil(span / bin_size)
    if steps <= 0:
        steps = 1
    edges = [start_edge + idx * bin_size for idx in range(steps + 1)]
    prices = [edge + bin_size / 2 for edge in edges[:-1]]
    volumes = [0.0 for _ in prices]

    total_volume = 0.0

    for candle in entries:
        volume = _ensure_float(candle.get("v") or candle.get("volume"))
        if volume <= 0:
            continue
        footprint_prices = candle.get("prices")
        footprint_volumes = candle.get("volumes")
        if isinstance(footprint_prices, Sequence) and isinstance(footprint_volumes, Sequence):
            paired = list(zip(footprint_prices, footprint_volumes))
            if paired:
                for price_value, volume_value in paired:
                    price_float = _ensure_float(price_value, default=math.nan)
                    vol_float = _ensure_float(volume_value, default=0.0)
                    if not math.isfinite(price_float) or vol_float <= 0:
                        continue
                    index = int(math.floor((price_float - start_edge) / bin_size))
                    if index < 0:
                        index = 0
                    elif index >= len(volumes):
                        index = len(volumes) - 1
                    volumes[index] += vol_float
                    total_volume += vol_float
                continue

        close_price = _ensure_float(candle.get("c") or candle.get("close") or candle.get("o") or candle.get("open"))
        if not math.isfinite(close_price):
            continue
        index = int(math.floor((close_price - start_edge) / bin_size))
        if index < 0:
            index = 0
        elif index >= len(volumes):
            index = len(volumes) - 1
        volumes[index] += volume
        total_volume += volume

    if total_volume <= 0:
        diagnostics.setdefault("warn", "degenerate profile")
        return VolumeProfile(prices=prices, volumes=volumes, poc=math.nan, vah=math.nan, val=math.nan, diagnostics=diagnostics)

    max_volume = max(volumes)
    poc_indices = [idx for idx, value in enumerate(volumes) if math.isclose(value, max_volume)]
    poc_index = min(poc_indices, key=lambda idx: prices[idx]) if poc_indices else 0
    poc_price = prices[poc_index] if prices else math.nan

    threshold = max(0.0, min(1.0, value_area_pct)) * total_volume
    order = sorted(range(len(volumes)), key=lambda idx: (-volumes[idx], prices[idx]))
    covered = 0.0
    selected: set[int] = set()
    for idx in order:
        selected.add(idx)
        covered += volumes[idx]
        if covered >= threshold:
            break
    if not selected:
        selected = {poc_index}

    selected_prices = [prices[idx] for idx in selected]
    vah_price = max(selected_prices) if selected_prices else math.nan
    val_price = min(selected_prices) if selected_prices else math.nan

    return VolumeProfile(
        prices=prices,
        volumes=volumes,
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        diagnostics=diagnostics,
    )


def _sort_session_keys(
    session_map: Mapping[Tuple[date, str], Sequence[Mapping[str, Any]]],
    session_order: Mapping[str, int],
) -> List[Tuple[date, str]]:
    return sorted(
        session_map.keys(),
        key=lambda key: (key[0], session_order.get(key[1], 0)),
    )


def compute_session_profiles(
    candles_1m: Sequence[Mapping[str, Any]],
    *,
    sessions: Iterable[Tuple[str, dtime, dtime]] = (),
    last_n: int = 5,
    tick_size: float | None = None,
    adaptive_bins: bool = False,
    value_area_pct: float = 0.70,
) -> List[Dict[str, object]]:
    """Return volume profile summaries for the latest trading sessions."""

    session_list = list(sessions)
    if not session_list:
        return []

    session_map = split_by_sessions(candles_1m, session_list)
    if not session_map:
        return []

    order_map = {name: index for index, (name, _, _) in enumerate(session_list)}
    ordered_keys = _sort_session_keys(session_map, order_map)
    if last_n > 0:
        ordered_keys = ordered_keys[-last_n:]

    summaries: List[Dict[str, object]] = []
    for key in ordered_keys:
        session_candles = session_map.get(key, [])
        profile = build_volume_profile(
            session_candles,
            tick_size=tick_size,
            adaptive_bins=adaptive_bins,
            value_area_pct=value_area_pct,
        )
        payload: Dict[str, object] = {
            "date": key[0].isoformat(),
            "session": key[1],
        }
        if profile.diagnostics:
            payload["DIAGNOSTICS"] = dict(profile.diagnostics)
        if profile.prices and profile.volumes and math.isfinite(profile.poc):
            payload["POC"] = profile.poc
            if math.isfinite(profile.vah):
                payload["VAH"] = profile.vah
            if math.isfinite(profile.val):
                payload["VAL"] = profile.val
        summaries.append(payload)
    return summaries


def flatten_profile(profile: VolumeProfile) -> List[Dict[str, float]]:
    """Flatten a volume profile into a list of dictionaries for JSON output."""

    return [
        {"price": float(price), "volume": float(volume)}
        for price, volume in zip(profile.prices, profile.volumes)
    ]

