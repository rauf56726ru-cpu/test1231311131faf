"""Builder for 72-hour SMC zone context payloads."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.analysis.bias import compute_bias
from src.common.ts import ensure_epoch_ms
from src.services.zones_context import ZoneDetector, ZoneRecord, TouchRecord, summarise_touch_records

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


def _build_vwap_context(frame: pd.DataFrame) -> Dict[str, Any]:
    from src.analysis.vwap_context import build_vwap_context as _build

    return _build(frame)


def _build_tpo_context(frame: pd.DataFrame) -> Dict[str, Any]:
    from src.analysis.tpo_context import build_tpo_context as _build

    return _build(frame)


def _build_session_ib(frame: pd.DataFrame, session_split_ms: int) -> Dict[str, Any]:
    from src.analysis.session_ib import compute_session_ib

    session_end_ms = session_split_ms
    session_start_ms = session_end_ms - (24 * 60 * 60 * 1000)
    if session_start_ms < 0:
        session_start_ms = 0
    payload = compute_session_ib(
        frame,
        session_start_ms=session_start_ms,
        session_end_ms=session_end_ms,
        ib_minutes=60,
    )
    return payload


def _to_zone_contract(zone: ZoneRecord, touch_bundle: Dict[str, Any] | None) -> Dict[str, Any]:
    events = list(touch_bundle.get("events") or []) if touch_bundle else []
    stats = dict(touch_bundle.get("stats") or {}) if touch_bundle else {}
    if not stats:
        _, stats = summarise_touch_records([])
    payload = {
        "id": zone.id,
        "type": zone.type,
        "side": zone.side,
        "price_hi": _safe_float(zone.high),
        "price_lo": _safe_float(zone.low),
        "formed_ms": int(zone.created_ts),
        "strength": _safe_float(zone.strength),
        "status": zone.status,
    }
    payload["touch_count"] = int(zone.touches)
    payload["touches"] = events
    payload["touch_stats"] = stats
    if zone.last_seen_ts:
        payload["last_seen_ms"] = int(zone.last_seen_ts)
    return payload


def _touches_by_zone(
    touches: Sequence[TouchRecord],
    zone_ids: set[str],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[TouchRecord]] = {zone_id: [] for zone_id in zone_ids}
    for touch in touches:
        if touch.zone_id not in grouped:
            continue
        grouped[touch.zone_id].append(touch)
    compressed: Dict[str, Dict[str, Any]] = {}
    for zone_id, records in grouped.items():
        events, stats = summarise_touch_records(records)
        compressed[zone_id] = {"events": events, "stats": stats}
    return compressed


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

    frame_for_metrics = frame.assign(ts_open=frame["ts_open"].astype("int64"))
    bias_block = compute_bias(
        frame_for_metrics,
        timeframes=("1h", "4h", "1d"),
        neutral_pct=0.1,
    )

    zones_payload = [
        _to_zone_contract(zone, touches_map.get(zone.id))
        for zone in zones_top
    ]

    counts = {
        "fvg": sum(1 for zone in zones_top if zone.type.upper() == "FVG"),
        "ob": sum(1 for zone in zones_top if zone.type.upper() == "OB"),
    }
    counts["others"] = max(0, len(zones_top) - counts["fvg"] - counts["ob"])

    latency_ms = int((time.perf_counter() - build_started) * 1000)

    aggregates = {
        "atr14_mean": atr14_mean,
        "rvol_mean": rvol_mean,
    }
    if bias_block:
        aggregates["bias"] = bias_block
    vwap_context = _build_vwap_context(frame_for_metrics[["ts_open", "open", "high", "low", "close", "volume"]])
    if vwap_context:
        aggregates["vwap_context"] = vwap_context
    tpo_context = _build_tpo_context(frame_for_metrics[["ts_open", "open", "high", "low", "close", "volume"]])
    if tpo_context:
        aggregates["tpo_context"] = tpo_context
    session_ib = _build_session_ib(frame_for_metrics[["ts_open", "open", "high", "low", "close"]], session_split_ms)
    if session_ib:
        aggregates["sessions"] = {"last_closed": session_ib}

    return {
        "symbol": symbol.upper(),
        "window": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "bars": EXPECTED_BARS,
            "coverage_pct": round(coverage * 100.0, 4),
            "source_seq": source_seq,
        },
        "aggregates": aggregates,
        "zones_top": zones_payload,
        "counts": counts,
        "latency_ms": latency_ms,
    }


__all__ = ["build_smc_72h_ctx_v1"]
