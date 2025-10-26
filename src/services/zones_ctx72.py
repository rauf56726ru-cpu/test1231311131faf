"""Builder for 72-hour SMC zone context payloads."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.common.ts import ensure_epoch_ms
from src.services.zones_context import ZoneDetector, ZoneRecord, TouchRecord

MINUTE_MS = 60_000
EXPECTED_BARS = 72 * 60


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _compute_atr(frame: pd.DataFrame, n: int = 14) -> pd.Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)

    hl = (high - low).abs()
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()

    stacked = np.vstack([hl.to_numpy(), hc.to_numpy(), lc.to_numpy()])
    tr = np.nanmax(stacked, axis=0)
    atr = pd.Series(tr, index=frame.index).rolling(n, min_periods=1).mean()
    return atr


def _prepare_frame(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.dropna(subset=["ts_open", "open", "high", "low", "close", "volume"])
    frame["ts_open"] = frame["ts_open"].astype("int64")
    frame = frame[(frame["ts_open"] >= start_ms) & (frame["ts_open"] < end_ms)]
    frame.sort_values("ts_open", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _to_zone_contract(zone: ZoneRecord, touches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "id": zone.id,
        "type": zone.type,
        "side": zone.side,
        "price_hi": _safe_float(zone.high),
        "price_lo": _safe_float(zone.low),
        "formed_ms": int(zone.created_ts),
        "strength": _safe_float(zone.strength),
        "status": zone.status,
        "touches": touches,
    }
    if zone.last_seen_ts:
        payload["last_seen_ms"] = int(zone.last_seen_ts)
    return payload


def _touches_by_zone(touches: Sequence[TouchRecord], zone_ids: set[str]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {zone_id: [] for zone_id in zone_ids}
    for touch in touches:
        if touch.zone_id not in grouped:
            continue
        grouped[touch.zone_id].append(
            {
                "ts_ms": int(touch.at_ts),
                "kind": touch.touch_kind,
                "depth": _safe_float(touch.depth),
            }
        )
    for items in grouped.values():
        items.sort(key=lambda item: item["ts_ms"])
    return grouped


def build_smc_72h_ctx_v1(
    symbol: str,
    df_m1_72h: pd.DataFrame,
    *,
    session_split_ts: int,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Build the SMC_72h_ctx_v1 payload from 72h minute data."""

    build_started = time.perf_counter()

    if df_m1_72h is None or df_m1_72h.empty:
        raise ValueError("df_m1_72h must contain data")

    session_split_ms = ensure_epoch_ms(session_split_ts)
    start_ms = ensure_epoch_ms(int(df_m1_72h["ts_open"].min()))
    last_bar_ms = ensure_epoch_ms(int(df_m1_72h["ts_open"].max()))
    end_ms = last_bar_ms + MINUTE_MS

    frame = _prepare_frame(df_m1_72h, start_ms, end_ms)
    if frame.empty:
        raise ValueError("72h frame empty after sanitisation")

    frame["atr14"] = _compute_atr(frame)
    volume_median = _safe_float(frame["volume"].median()) or 1.0
    taker_buy = frame["taker_buy_vol"] if "taker_buy_vol" in frame else frame["volume"] / 2.0
    frame["delta"] = (taker_buy * 2.0) - frame["volume"]
    frame["rvol"] = frame["volume"] / max(volume_median, 1e-9)

    detector = ZoneDetector(frame, session_start_ms=session_split_ms)
    zones = detector.detect()
    filtered = [zone for zone in zones if zone.status != "filled"]
    filtered.sort(key=lambda zone: (-float(zone.strength), -int(zone.created_ts)))
    zones_top = filtered[: max(1, int(top_n))]

    session_end_ms = min(end_ms - MINUTE_MS, last_bar_ms)
    touches = detector.detect_session_touches(zones_top, session_end_ms=session_end_ms)
    zone_ids = {zone.id for zone in zones_top}
    touches_map = _touches_by_zone(touches, zone_ids)

    atr14_mean = _safe_float(frame["atr14"].mean())
    rvol_mean = _safe_float(frame["rvol"].mean())

    bars = int(len(frame))
    expected_bars = EXPECTED_BARS
    coverage = bars / expected_bars if expected_bars else 1.0
    ensure_meta = df_m1_72h.attrs.get("ensure")
    if ensure_meta is not None:
        coverage = getattr(ensure_meta, "coverage", coverage)
        source_seq = list(getattr(ensure_meta, "source_sequence", []))
    else:
        source_seq = []

    if coverage < 0.99:
        raise RuntimeError("insufficient_coverage")

    zones_payload = [
        _to_zone_contract(zone, touches_map.get(zone.id, []))
        for zone in zones_top
    ]

    counts = {
        "fvg": sum(1 for zone in zones_top if zone.type.upper() == "FVG"),
        "ob": sum(1 for zone in zones_top if zone.type.upper() == "OB"),
    }
    counts["others"] = max(0, len(zones_top) - counts["fvg"] - counts["ob"])

    latency_ms = int((time.perf_counter() - build_started) * 1000)

    return {
        "symbol": symbol.upper(),
        "window": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "bars": EXPECTED_BARS,
            "coverage_pct": round(coverage * 100.0, 4),
            "source_seq": source_seq,
        },
        "aggregates": {
            "atr14_mean": atr14_mean,
            "rvol_mean": rvol_mean,
        },
        "zones_top": zones_payload,
        "counts": counts,
        "latency_ms": latency_ms,
    }


__all__ = ["build_smc_72h_ctx_v1"]
