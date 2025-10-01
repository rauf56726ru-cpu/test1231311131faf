"""Utilities to enrich inspection payloads with SMC/ICT metrics."""
from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import httpx
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_DELTA_BARS = 50; MAX_FOOTPRINT_ROWS = 10


def _parse_ts(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            if "T" in value:
                return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
            return int(float(value))
        except (TypeError, ValueError):
            return None
    return None


def _normalise_row(row: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    ts = _parse_ts(row.get("t") or row.get("time") or row.get("openTime"))
    if ts is None:
        return None
    try:
        o = float(row.get("o") or row.get("open"))
        h = float(row.get("h") or row.get("high"))
        l = float(row.get("l") or row.get("low"))
        c = float(row.get("c") or row.get("close"))
        v = float(row.get("v") or row.get("volume") or 0.0)
    except (TypeError, ValueError):
        return None
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": max(v, 0.0)}


def _map_minutes(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Dict[str, float]]:
    minutes: Dict[int, Dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candle = _normalise_row(row)
        if candle is not None:
            minutes[candle["t"]] = candle
    return minutes


async def _fetch_recent(symbol: str, interval: str, limit: int) -> List[Dict[str, float]]:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": str(min(max(limit, 1), 1000))}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        response = await client.get(BINANCE_KLINES_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    candles: List[Dict[str, float]] = []
    if isinstance(payload, Sequence):
        for row in payload:
            try:
                candles.append({
                    "t": int(row[0]),
                    "o": float(row[1]),
                    "h": float(row[2]),
                    "l": float(row[3]),
                    "c": float(row[4]),
                    "v": max(float(row[5]), 0.0),
                })
            except (IndexError, TypeError, ValueError):
                continue
    return candles


def _zone_state(series: Sequence[Mapping[str, float]], idx: int, direction: str) -> str:
    base = series[idx]
    body_low = min(base["o"], base["c"])
    body_high = max(base["o"], base["c"])
    tapped = False
    for candle in series[idx + 1 :]:
        low = candle.get("l", body_low)
        high = candle.get("h", body_high)
        if direction == "demand":
            if low < body_low:
                return "invalidated"
            if low <= body_high <= high:
                tapped = True
        else:
            if high > body_high:
                return "invalidated"
            if high >= body_low >= low:
                tapped = True
    return "tapped" if tapped else "fresh"


def _supply_demand(candles: Sequence[Mapping[str, float]]) -> Dict[str, List[Dict[str, Any]]]:
    avg = statistics.fmean([float(c.get("v", 0.0)) for c in candles]) if candles else 0.0
    result = {"demand": [], "supply": []}
    for idx, candle in enumerate(candles):
        if avg and candle.get("v", 0.0) < avg:
            continue
        direction = "demand" if candle.get("c", 0.0) >= candle.get("o", 0.0) else "supply"
        zone = {
            "open": float(min(candle.get("o", 0.0), candle.get("c", 0.0))),
            "close": float(max(candle.get("o", 0.0), candle.get("c", 0.0))),
            "mean": float((candle.get("o", 0.0) + candle.get("h", 0.0) + candle.get("l", 0.0) + candle.get("c", 0.0)) / 4),
            "origin_utc": datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": _zone_state(candles, idx, direction),
        }
        result[direction].append(zone)
    return result


def _delta_series(candles: Iterable[Mapping[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candle in list(candles)[-MAX_DELTA_BARS:]:
        spread = max(float(candle.get("h", 0.0)) - float(candle.get("l", 0.0)), 1e-9)
        change = float(candle.get("c", 0.0)) - float(candle.get("o", 0.0))
        volume = float(candle.get("v", 0.0))
        rows.append({"t": int(candle.get("t", 0)), "delta": volume * (change / spread)})
    return rows


def _cvd(delta_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    running = 0.0
    series: List[Dict[str, Any]] = []
    for row in delta_rows:
        running += float(row.get("delta", 0.0))
        series.append({"t": row.get("t"), "cvd": running})
    return series


def _footprint_summary(footprint: Sequence[Mapping[str, Any]], minute_map: Mapping[int, Mapping[str, float]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, float]] = defaultdict(lambda: {"buy": 0.0, "sell": 0.0})
    for row in footprint:
        ts = _parse_ts(row.get("t"))
        if ts is None:
            continue
        grouped[ts]["buy"] += max(float(row.get("ask", 0.0)), 0.0)
        grouped[ts]["sell"] += max(float(row.get("bid", 0.0)), 0.0)
    summaries: List[Dict[str, Any]] = []
    for ts in sorted(grouped)[-MAX_FOOTPRINT_ROWS:]:
        totals = grouped[ts]
        total_vol = totals["buy"] + totals["sell"]
        candle = minute_map.get(ts, {})
        price_span = max(float(candle.get("h", 0.0)) - float(candle.get("l", 0.0)), 1e-9)
        price_move = float(candle.get("c", 0.0)) - float(candle.get("o", 0.0))
        buy_share = (totals["buy"] / total_vol * 100.0) if total_vol else 0.0
        sell_share = (totals["sell"] / total_vol * 100.0) if total_vol else 0.0
        imbalance = abs(totals["buy"] - totals["sell"])
        summaries.append({
            "t": ts,
            "buy_imbalance": round(buy_share, 3),
            "sell_imbalance": round(sell_share, 3),
            "absorption": bool(total_vol and imbalance > 0.55 * total_vol and abs(price_move) < 0.25 * price_span),
        })
    return summaries


def _block_candidates(candles: Sequence[Mapping[str, float]], tf: str, *, min_ratio: float) -> List[Dict[str, Any]]:
    if len(candles) < 5:
        return []
    avg = statistics.fmean([float(c.get("v", 0.0)) for c in candles]) if candles else 0.0
    zones: List[Dict[str, Any]] = []
    for idx, candle in enumerate(candles[-20:]):
        body = abs(float(candle.get("c", 0.0)) - float(candle.get("o", 0.0)))
        span = max(float(candle.get("h", 0.0)) - float(candle.get("l", 0.0)), 1e-9)
        if not span or body / span < min_ratio:
            continue
        if avg and candle.get("v", 0.0) < 1.2 * avg:
            continue
        direction = "demand" if candle.get("c", 0.0) >= candle.get("o", 0.0) else "supply"
        zones.append({
            "tf": tf,
            "type": direction,
            "open": float(min(candle.get("o", 0.0), candle.get("c", 0.0))),
            "close": float(max(candle.get("o", 0.0), candle.get("c", 0.0))),
            "mean": float((candle.get("o", 0.0) + candle.get("h", 0.0) + candle.get("l", 0.0) + candle.get("c", 0.0)) / 4),
            "origin_utc": datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": _zone_state(candles, max(0, len(candles) - 20) + idx, direction),
        })
    return zones


def _breaker_blocks(candles: Sequence[Mapping[str, float]], tf: str) -> List[Dict[str, Any]]:
    breakers: List[Dict[str, Any]] = []
    for idx in range(3, len(candles)):
        window = candles[idx - 3 : idx]
        prev_high = max(float(c.get("h", 0.0)) for c in window)
        prev_low = min(float(c.get("l", 0.0)) for c in window)
        open_price = float(candles[idx].get("o", 0.0))
        close_price = float(candles[idx].get("c", 0.0))
        direction = None
        if close_price > prev_high and open_price < prev_high:
            direction = "demand"
        elif close_price < prev_low and open_price > prev_low:
            direction = "supply"
        if direction:
            breakers.append({
                "tf": tf,
                "type": direction,
                "open": float(min(open_price, close_price)),
                "close": float(max(open_price, close_price)),
                "mean": float((candles[idx].get("o", 0.0) + candles[idx].get("h", 0.0) + candles[idx].get("l", 0.0) + candles[idx].get("c", 0.0)) / 4),
                "origin_utc": datetime.fromtimestamp(candles[idx]["t"] / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": _zone_state(candles, idx, direction),
            })
    return breakers


def _swing_levels(candles: Sequence[Mapping[str, float]]) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []
    for idx in range(2, len(candles) - 2):
        cur = candles[idx]
        high = cur.get("h", 0.0)
        low = cur.get("l", 0.0)
        if all(high >= candles[j].get("h", 0.0) for j in range(idx - 2, idx + 3)):
            levels.append({"type": "resistance", "price": float(high), "ts": int(cur.get("t", 0)), "valid": True})
        if all(low <= candles[j].get("l", 0.0) for j in range(idx - 2, idx + 3)):
            levels.append({"type": "support", "price": float(low), "ts": int(cur.get("t", 0)), "valid": True})
    return levels


def _equal_extremes(candles: Sequence[Mapping[str, float]], field: str, tolerance: float) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []
    for prev, cur in zip(candles, candles[1:]):
        prev_val = float(prev.get(field, 0.0))
        cur_val = float(cur.get(field, 0.0))
        if not prev_val:
            continue
        if abs(cur_val - prev_val) / prev_val <= tolerance:
            levels.append({"price": round((cur_val + prev_val) / 2, 6), "ts": int(cur.get("t", 0))})
    return levels


def _profile_summary(tpo_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sessions = {"asia": [], "london": [], "ny": []}
    daily: Dict[str, Dict[str, List[float]]] = {}
    for entry in tpo_entries:
        if not isinstance(entry, Mapping):
            continue
        date = entry.get("date")
        session = str(entry.get("session") or "").lower() or None
        poc, vah, val = entry.get("POC"), entry.get("VAH"), entry.get("VAL")
        if date is not None:
            bucket = daily.setdefault(str(date), {"poc": [], "vah": [], "val": []})
            for key, value in (("poc", poc), ("vah", vah), ("val", val)):
                try:
                    bucket[key].append(float(value))
                except (TypeError, ValueError):
                    continue
        if session in sessions:
            try:
                sessions[session].append({"date": date, "poc": float(poc), "vah": float(vah), "val": float(val)})
            except (TypeError, ValueError):
                continue
    daily_levels = []
    for date, payload in sorted(daily.items())[-3:]:
        daily_levels.append({
            "date": date,
            "poc": float(statistics.fmean(payload["poc"])) if payload["poc"] else None,
            "vah": float(statistics.fmean(payload["vah"])) if payload["vah"] else None,
            "val": float(statistics.fmean(payload["val"])) if payload["val"] else None,
        })
    return {"daily": daily_levels, "sessions": sessions}


def _context(daily_candles: Sequence[Mapping[str, float]], zones: Mapping[str, Any]) -> Dict[str, Any]:
    bias = "neutral"
    narrative = "No data"
    if daily_candles:
        last = daily_candles[-1]
        open_price = last.get("o", 0.0)
        close_price = last.get("c", 0.0)
        if close_price > open_price:
            bias = "bull"
        elif close_price < open_price:
            bias = "bear"
        narrative = f"Daily candle closed {'higher' if close_price >= open_price else 'lower'} ({close_price:.2f})"
    fresh_demand = any(zone.get("status") == "fresh" for zone in zones.get("demand", [])) if isinstance(zones, Mapping) else False
    fresh_supply = any(zone.get("status") == "fresh" for zone in zones.get("supply", [])) if isinstance(zones, Mapping) else False
    return {"globalBias": bias, "narrative": narrative, "openOppositeZones": fresh_demand and fresh_supply}


async def enrich_inspection_snapshot(
    snapshot: Mapping[str, Any],
    *,
    cache: MutableMapping[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _ = cache
    symbol = str(snapshot.get("symbol") or "BTCUSDT").upper()
    minute_rows = snapshot.get("ohlcv", {}).get("1m", {}).get("candles") if isinstance(snapshot.get("ohlcv"), Mapping) else []
    if not minute_rows:
        minute_rows = snapshot.get("candles", [])
    minute_map = _map_minutes(minute_rows or [])

    # Pull compact higher-timeframe history directly from Binance.
    one_day, four_hour, one_hour = await asyncio.gather(
        _fetch_recent(symbol, "1d", 3),
        _fetch_recent(symbol, "4h", 6),
        _fetch_recent(symbol, "1h", 72),
    )

    # Build OHLCV blocks augmented with basic supply/demand heuristics.
    ohlcv_additions = {
        "1d": {"symbol": symbol, "tf": "1d", "candles": one_day, **_supply_demand(one_day)},
        "4h": {"symbol": symbol, "tf": "4h", "candles": four_hour, **_supply_demand(four_hour)},
        "1h": {"symbol": symbol, "tf": "1h", "candles": one_hour[-24:], **_supply_demand(one_hour)},
    }

    # Derive simplified order-flow metrics from available 1m candles and footprint rows.
    delta_rows = _delta_series(minute_map.values())
    footprint_raw = []
    orderflow_snapshot = snapshot.get("orderflow") if isinstance(snapshot.get("orderflow"), Mapping) else {}
    if isinstance(orderflow_snapshot, Mapping):
        footprint_raw = [row for row in orderflow_snapshot.get("footprint", []) if isinstance(row, Mapping)]
    orderflow_block = {
        "delta": delta_rows,
        "cvd": _cvd(delta_rows),
        "footprint": _footprint_summary(footprint_raw, minute_map),
        "raw": footprint_raw,
    }

    enriched_zones = {
        "mb": _block_candidates(one_hour, "1h", min_ratio=0.6) + _block_candidates(four_hour, "4h", min_ratio=0.6),
        "bb": _breaker_blocks(one_hour, "1h"),
        "rb": [], "pb": [],
        "sr": _swing_levels(one_hour) + _swing_levels(four_hour),
    }

    liquidity_levels = {"eqh": _equal_extremes(one_hour, "h", 0.001), "eql": _equal_extremes(one_hour, "l", 0.001)}

    data_section = snapshot.get("DATA") if isinstance(snapshot.get("DATA"), Mapping) else {}
    tpo_entries = []
    if isinstance(data_section, Mapping):
        tpo_payload = data_section.get("tpo") if isinstance(data_section.get("tpo"), Mapping) else {}
        tpo_entries = [entry for entry in tpo_payload.get("sessions", []) if isinstance(entry, Mapping)]
    profile_levels = _profile_summary(tpo_entries)

    enrichment = {
        "status": "ok",
        "missing_fields": [],
        "ohlcv": ohlcv_additions,
        "orderflow": orderflow_block,
        "zones": enriched_zones,
        "liquidity": liquidity_levels,
        "profile_levels": profile_levels,
        "risk_prefs": {"rr_min": 2.5, "risk_per_trade_pct": 1.0},
        "context": _context(one_day, ohlcv_additions.get("1h", {})),
    }
    return enrichment


def apply_enrichment_to_payload(payload: Dict[str, Any], enrichment: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or not isinstance(enrichment, Mapping):
        return
    data = payload.setdefault("DATA", {})
    if not isinstance(data, dict):
        return
    ohlcv_block = data.setdefault("ohlcv", {})
    if isinstance(ohlcv_block, dict):
        for tf, block in enrichment.get("ohlcv", {}).items():
            if isinstance(block, Mapping):
                merged = dict(ohlcv_block.get(tf, {}))
                merged.update(block)
                ohlcv_block[tf] = merged
    orderflow_block = data.setdefault("orderflow", {})
    if isinstance(orderflow_block, dict):
        if "cvd" in orderflow_block:
            orderflow_block.setdefault("cvd_legacy", orderflow_block["cvd"])
        enriched_flow = enrichment.get("orderflow")
        if isinstance(enriched_flow, Mapping):
            for key, value in enriched_flow.items():
                if key == "raw":
                    orderflow_block.setdefault("footprint_raw", value)
                else:
                    orderflow_block[key] = value
    zones_block = data.setdefault("zones", {})
    if isinstance(zones_block, dict):
        container = zones_block.setdefault("zones", {}) if isinstance(zones_block.get("zones"), Mapping) else zones_block
        extra = enrichment.get("zones")
        if isinstance(container, dict) and isinstance(extra, Mapping):
            for key, value in extra.items():
                container[key] = value
    data["liquidity"] = enrichment.get("liquidity")
    data["profile_levels"] = enrichment.get("profile_levels")
    data["risk_prefs"] = enrichment.get("risk_prefs")
    data["context"] = enrichment.get("context")
    data["enrichment_status"] = {
        "status": enrichment.get("status", "ok"),
        "missing_fields": enrichment.get("missing_fields", []),
    }
