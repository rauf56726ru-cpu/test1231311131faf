"""Volume profile and TPO helpers for inspection payloads."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
import math
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


_PROFILE_CACHE: Dict[Tuple[Any, ...], "VolumeProfile"] = {}


@dataclass(slots=True)
class VolumeProfile:
    """Aggregated volume distribution for a given set of candles."""

    prices: List[float]
    volumes: List[float]
    poc: float | None
    vah: float | None
    val: float | None
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
    atr_multiplier: float,
    target_bins: int,
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
            multiplier = atr_multiplier if atr_multiplier > 0 else 0.5
            bin_size = max(avg_tr * multiplier, 0.0)
            if bin_size > 0:
                return bin_size

    n = len(candles)
    if n <= 0:
        return None
    suggested = math.ceil(math.log2(n)) + 1 if n > 1 else min_bins
    bins = max(min_bins, target_bins if target_bins > 0 else min_bins, suggested)
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
    atr_multiplier: float = 0.5,
    target_bins: int = 80,
    cache_scope: Tuple[Any, ...] | None = None,
    min_value_area_bars: int = 60,
) -> VolumeProfile:
    """Construct a volume profile from a sequence of OHLCV candles."""

    entries = [dict(candle) for candle in candles if isinstance(candle, Mapping)]
    diagnostics: Dict[str, str] = {}

    allow_value_area = len(entries) >= max(1, int(min_value_area_bars))
    if not allow_value_area:
        diagnostics["warn"] = "too_few_bars"

    if value_area_pct <= 0:
        value_area_pct = 0.0
    elif value_area_pct > 1:
        value_area_pct = 1.0

    target_bins = max(min_bins, int(target_bins) if target_bins else min_bins)

    bin_size = _select_bin_size(
        entries,
        tick_size=tick_size,
        adaptive_bins=adaptive_bins,
        min_bins=min_bins,
        atr_multiplier=atr_multiplier,
        target_bins=target_bins,
    )
    if tick_size and tick_size > 0 and bin_size:
        step_multiple = max(1, round(bin_size / tick_size))
        bin_size = tick_size * step_multiple

    if not bin_size or bin_size <= 0:
        prices: List[float] = []
        volumes: List[float] = []
        diagnostics.setdefault("warn", diagnostics.get("warn", "degenerate profile"))
        return VolumeProfile(prices=prices, volumes=volumes, poc=None, vah=None, val=None, diagnostics=diagnostics)

    lows = [_ensure_float(item.get("l") or item.get("low")) for item in entries]
    highs = [_ensure_float(item.get("h") or item.get("high")) for item in entries]
    lows = [value for value in lows if math.isfinite(value)]
    highs = [value for value in highs if math.isfinite(value)]
    if not lows or not highs:
        diagnostics.setdefault("warn", diagnostics.get("warn", "degenerate profile"))
        return VolumeProfile(prices=[], volumes=[], poc=None, vah=None, val=None, diagnostics=diagnostics)

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
        diagnostics.setdefault("warn", diagnostics.get("warn", "degenerate profile"))
        return VolumeProfile(prices=prices, volumes=volumes, poc=None, vah=None, val=None, diagnostics=diagnostics)

    poc_price: float | None = None
    vah_price: float | None = None
    val_price: float | None = None

    if allow_value_area and total_volume > 0 and prices and volumes:
        weighted_price = sum(price * volume for price, volume in zip(prices, volumes)) / total_volume
        max_volume = max(volumes)
        poc_candidates = [idx for idx, value in enumerate(volumes) if math.isclose(value, max_volume)]
        poc_index = min(
            poc_candidates,
            key=lambda idx: (abs(prices[idx] - weighted_price), prices[idx]),
        ) if poc_candidates else 0
        poc_price = prices[poc_index]

        threshold = max(0.0, min(1.0, value_area_pct)) * total_volume
        order = sorted(
            range(len(volumes)),
            key=lambda idx: (-volumes[idx], abs(prices[idx] - weighted_price), prices[idx]),
        )
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
        vah_price = max(selected_prices) if selected_prices else None
        val_price = min(selected_prices) if selected_prices else None

    profile = VolumeProfile(
        prices=prices,
        volumes=volumes,
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        diagnostics=diagnostics,
    )

    if cache_scope is not None:
        _PROFILE_CACHE[cache_scope] = profile

    return profile


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
    atr_multiplier: float = 0.5,
    target_bins: int = 80,
    cache_token: Any | None = None,
) -> List[Dict[str, object]]:
    """Return volume profile summaries grouped by calendar day."""

    session_list = list(sessions)
    session_map = (
        split_by_sessions(candles_1m, session_list) if session_list else {}
    )

    daily_buckets: DefaultDict[date, List[Mapping[str, Any]]] = defaultdict(list)
    for candle in candles_1m:
        if not isinstance(candle, Mapping):
            continue
        ts = _extract_timestamp(candle)
        if ts is None:
            continue
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        daily_buckets[dt.date()].append(dict(candle))


    daily_buckets: DefaultDict[date, List[Mapping[str, Any]]] = defaultdict(list)
    for candle in candles_1m:
        if not isinstance(candle, Mapping):
            continue
        ts = _extract_timestamp(candle)
        if ts is None:
            continue
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        daily_buckets[dt.date()].append(dict(candle))

    if not daily_buckets:
        return []

    ordered_dates = sorted(daily_buckets.keys())
    if last_n > 0:
        ordered_dates = ordered_dates[-last_n:]

    def resolve_profile(
        candles_seq: Sequence[Mapping[str, Any]],
        cache_key: Tuple[date, str],
    ) -> VolumeProfile:
        cache_scope = None
        profile: VolumeProfile | None = None
        if cache_token is not None:
            cache_scope = (
                cache_token,
                cache_key,
                round(float(tick_size or 0.0), 12),
                bool(adaptive_bins),
                round(float(value_area_pct), 4),
                round(float(atr_multiplier), 4),
                int(target_bins),
            )
            profile = _PROFILE_CACHE.get(cache_scope)
        if profile is None:
            profile = build_volume_profile(
                candles_seq,
                tick_size=tick_size,
                adaptive_bins=adaptive_bins,
                value_area_pct=value_area_pct,
                atr_multiplier=atr_multiplier,
                target_bins=target_bins,
                cache_scope=cache_scope,
                min_value_area_bars=60,
            )
        return profile

    def make_payload(
        day_key: date, session_label: str, profile: VolumeProfile
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "date": day_key.isoformat(),
            "session": session_label,
        }
        warn_code = profile.diagnostics.get("warn") if profile.diagnostics else None
        if profile.diagnostics:
            payload["DIAGNOSTICS"] = dict(profile.diagnostics)
        include_value_area = warn_code != "too_few_bars"
        if (
            include_value_area
            and profile.prices
            and profile.volumes
            and profile.poc is not None
            and math.isfinite(float(profile.poc))
        ):
            payload["POC"] = float(profile.poc)
            if profile.vah is not None and math.isfinite(float(profile.vah)):
                payload["VAH"] = float(profile.vah)
            if profile.val is not None and math.isfinite(float(profile.val)):
                payload["VAL"] = float(profile.val)
        if profile.prices and profile.volumes and math.isfinite(profile.poc):
            payload["POC"] = profile.poc
            if math.isfinite(profile.vah):
                payload["VAH"] = profile.vah
            if math.isfinite(profile.val):
                payload["VAL"] = profile.val
        return payload

    summaries: List[Dict[str, object]] = []
    for day_key in ordered_dates:
        day_candles = daily_buckets.get(day_key, [])
        if not day_candles:
            continue

        daily_profile = resolve_profile(day_candles, (day_key, "daily"))
        day_payload = make_payload(day_key, "daily", daily_profile)

        session_payloads: List[Dict[str, object]] = []
        for session_name, _, _ in session_list:
            session_candles = session_map.get((day_key, session_name))
            if not session_candles:
                continue
            session_profile = resolve_profile(session_candles, (day_key, session_name))
            session_payload = make_payload(day_key, session_name, session_profile)
            session_payloads.append(session_payload)

        if session_payloads:
            day_payload["sessions"] = [dict(item) for item in session_payloads]

        summaries.append(day_payload)
        summaries.extend(session_payloads)

    return summaries


def flatten_profile(
    profile: VolumeProfile,
    *,
    clip_threshold: float | None = None,
    smooth_window: int = 1,
) -> List[Dict[str, float]]:
    """Flatten a volume profile into a list of dictionaries for JSON output."""

    prices = list(profile.prices)
    volumes = list(profile.volumes)

    if smooth_window and smooth_window > 1 and len(volumes) > 1:
        window = max(1, int(smooth_window))
        window = min(window, len(volumes))
        smoothed: List[float] = []
        for idx in range(len(volumes)):
            half = window // 2
            start = max(0, idx - half)
            end = start + window
            if end > len(volumes):
                end = len(volumes)
                start = max(0, end - window)
            slice_vols = volumes[start:end]
            smoothed.append(sum(slice_vols) / len(slice_vols) if slice_vols else volumes[idx])
        volumes = smoothed

    if clip_threshold and clip_threshold > 0 and len(volumes) > 1:
        total = sum(volumes)
        if total > 0:
            left = 0
            right = len(volumes) - 1
            threshold = float(clip_threshold)
            while left <= right:
                changed = False
                share_left = volumes[left] / total if total else 0.0
                if share_left <= threshold and left < right:
                    total -= volumes[left]
                    left += 1
                    changed = True
                share_right = volumes[right] / total if total else 0.0
                if share_right <= threshold and right >= left:
                    total -= volumes[right]
                    right -= 1
                    changed = True
                if not changed:
                    break
            prices = prices[left : right + 1]
            volumes = volumes[left : right + 1]

    return [
        {"price": float(price), "volume": float(volume)}
        for price, volume in zip(prices, volumes)
    ]


def build_profile_package(
    candles: Sequence[Mapping[str, Any]],
    *,
    sessions: Iterable[Tuple[str, dtime, dtime]],
    last_n: int,
    tick_size: float | None,
    adaptive_bins: bool,
    value_area_pct: float,
    atr_multiplier: float,
    target_bins: int,
    clip_threshold: float = 0.0,
    smooth_window: int = 1,
    cache_token: Any | None = None,
    tf_key: str = "1m",
) -> Tuple[List[Dict[str, object]], List[Dict[str, float]], List[Dict[str, Any]]]:
    """Compute TPO summaries, flattened profile, and derived zones."""

    session_list = list(sessions)
    if not candles or not session_list:
        return [], [], []

    token = cache_token
    tpo_entries = compute_session_profiles(
        candles,
        sessions=session_list,
        last_n=last_n,
        tick_size=tick_size,
        adaptive_bins=adaptive_bins,
        value_area_pct=value_area_pct,
        atr_multiplier=atr_multiplier,
        target_bins=target_bins,
        cache_token=token,
    )

    flattened: List[Dict[str, float]] = []
    zones: List[Dict[str, Any]] = []

    if not tpo_entries:
        return tpo_entries, flattened, zones

    session_map = split_by_sessions(candles, session_list)
    latest = tpo_entries[-1] if tpo_entries else None
    latest_date = latest.get("date") if isinstance(latest, Mapping) else None
    latest_session = latest.get("session") if isinstance(latest, Mapping) else None

    if latest_date and latest_session and session_map:
        for (session_date, session_name), session_candles in session_map.items():
            if session_date.isoformat() != latest_date or session_name != latest_session:
                continue
            cache_scope = None
            if token is not None:
                cache_scope = (
                    token,
                    (session_date, session_name),
                    round(float(tick_size or 0.0), 12),
                    bool(adaptive_bins),
                    round(float(value_area_pct), 4),
                    round(float(atr_multiplier), 4),
                    int(target_bins),
                )
            last_profile = build_volume_profile(
                session_candles,
                tick_size=tick_size,
                adaptive_bins=adaptive_bins,
                value_area_pct=value_area_pct,
                atr_multiplier=atr_multiplier,
                target_bins=target_bins,
                cache_scope=cache_scope,
                min_value_area_bars=60,
            )
            if last_profile.prices:
                flattened = flatten_profile(
                    last_profile,
                    clip_threshold=clip_threshold,
                    smooth_window=smooth_window,
                )
            break

    seen_zone_keys: set[Tuple[str | None, str | None, str | None]] = set()

    for entry in tpo_entries:
        if not isinstance(entry, Mapping):
            continue
        date_value = entry.get("date")
        session_value = entry.get("session")
        diagnostics = entry.get("DIAGNOSTICS") if isinstance(entry.get("DIAGNOSTICS"), Mapping) else None
        warn_code = diagnostics.get("warn") if isinstance(diagnostics, Mapping) else None
        meta_payload = {
            "value_area_pct": value_area_pct,
            "source": "tpo",
        }
        price_fields = {
            "tpo_poc": entry.get("POC"),
            "tpo_vah": entry.get("VAH"),
            "tpo_val": entry.get("VAL"),
        }
        include_value_area = warn_code != "too_few_bars"
        for zone_type, price_value in price_fields.items():
            if price_value is None:
                continue
            if not include_value_area:
                continue
            try:
                price_float = float(price_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price_float):
                continue
            key = (zone_type, str(date_value) if date_value is not None else None, str(session_value) if session_value is not None else None)
            if key in seen_zone_keys:
                continue
            seen_zone_keys.add(key)
            zones.append(
                {
                    "type": zone_type,
                    "price": price_float,
                    "date": date_value,
                    "session": session_value,
                    "tf": tf_key,
                    "meta": meta_payload,
                }
            )

    return tpo_entries, flattened, zones

