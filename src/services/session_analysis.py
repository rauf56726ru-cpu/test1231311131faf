"""Builder for structured SMC session analytics."""

from __future__ import annotations

import math
import time
from datetime import date
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.config import AppConfig
from src.common.ts import ensure_epoch_ms
from src.storage.parquet import ParquetStorage

MINUTE_MS = 60_000
ATR_PERIOD = 14
RVOL_BASELINE_SESSIONS = 5


@lru_cache(maxsize=1)
def _get_storage() -> ParquetStorage:
    cfg = AppConfig.load()
    return ParquetStorage(root=cfg.data_dir, market=cfg.market, index_path=cfg.duckdb_path)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _compute_atr(frame: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = (frame["high"] - frame["low"]).abs()
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.bfill().ffill()


def _session_expected_minutes(start_ms: int, end_ms: int) -> int:
    return max(0, (end_ms - start_ms) // MINUTE_MS)


def _load_previous_session_frame(symbol: str, session_start_ms: int, duration_ms: int) -> pd.DataFrame:
    storage = _get_storage()
    start_prev = session_start_ms - duration_ms
    if start_prev < 0:
        return pd.DataFrame()
    end_prev_exclusive = session_start_ms
    load_end = max(start_prev, end_prev_exclusive - MINUTE_MS)
    frame = storage.load_window(symbol, "1m", start_prev, load_end)
    if frame.empty:
        return frame
    mask = (frame["ts_open"] >= start_prev) & (frame["ts_open"] < end_prev_exclusive)
    return frame.loc[mask].copy()


def _compute_rvol(symbol: str, session_start_ms: int, duration_ms: int, session_volume: float) -> float | None:
    if duration_ms <= 0 or session_volume <= 0:
        return None
    volumes: List[float] = []
    for idx in range(1, RVOL_BASELINE_SESSIONS + 1):
        offset_start = session_start_ms - idx * duration_ms
        if offset_start < 0:
            break
        frame = _load_previous_session_frame(symbol, session_start_ms - (idx - 1) * duration_ms, duration_ms)
        if frame.empty:
            continue
        expected = duration_ms // MINUTE_MS
        if expected <= 0:
            continue
        coverage = len(frame) / expected
        if coverage < 0.9:
            continue
        volumes.append(float(frame["volume"].sum()))
    if not volumes:
        return None
    baseline = sum(volumes) / len(volumes)
    if baseline <= 0:
        return None
    return session_volume / baseline


def _compute_micro_metrics(frame: pd.DataFrame) -> Dict[str, float | None]:
    metrics = {
        "spread_p50": None,
        "spread_p95": None,
        "l1_imbalance_p50": None,
    }
    if "spread_bp_p50" in frame:
        metrics["spread_p50"] = _safe_float(frame["spread_bp_p50"].median())
    if "spread_bp_p95" in frame:
        metrics["spread_p95"] = _safe_float(frame["spread_bp_p95"].quantile(0.95))
    if "l1_imbalance_p50" in frame:
        metrics["l1_imbalance_p50"] = _safe_float(frame["l1_imbalance_p50"].median())
    return metrics


def _compute_orderflow(frame: pd.DataFrame) -> Dict[str, Any]:
    delta_sum = None
    cvd_last = None
    impulses: List[Dict[str, Any]] = []

    if "delta" in frame:
        delta_sum = _safe_float(frame["delta"].sum())
        ranked = frame.assign(abs_delta=frame["delta"].abs())
        top_rows = ranked.nlargest(10, "abs_delta")
        impulses = []
        for row in top_rows.itertuples():
            delta_value = _safe_float(row.delta)
            volume_value = _safe_float(row.volume)
            if delta_value is None:
                continue
            impulses.append(
                {
                    "ts": int(row.ts_open),
                    "delta": delta_value,
                    "volume": volume_value,
                }
            )

    if "cvd" in frame:
        cvd_last = _safe_float(frame["cvd"].iloc[-1])

    return {
        "delta_sum": delta_sum,
        "cvd_last": cvd_last,
        "impulses": impulses,
    }


def _detect_session_events(
    frame: pd.DataFrame,
    *,
    pdh: float | None,
    pdl: float | None,
    vwap: float | None,
) -> List[str]:
    events: List[str] = []
    closes = frame["close"]
    highs = frame["high"]
    lows = frame["low"]

    if pdh is not None and (highs > pdh).any():
        events.append("breakout_pdh")
        if (closes < pdh).any():
            events.append("false_break_pdh")
    if pdl is not None and (lows < pdl).any():
        events.append("breakout_pdl")
        if (closes > pdl).any():
            events.append("false_break_pdl")

    if vwap is not None:
        above = closes > vwap
        below = closes < vwap
        if above.any() and below.any():
            events.append("retest_vwap")

    return events[:10]


def build_smc_session_v1(
    symbol: str,
    df_m1: pd.DataFrame,
    meta: Dict[str, Any],
    *,
    session_start_ms: int,
    session_end_ms: int,
) -> Dict[str, Any]:
    """Build metrics for the SMC_session_v1 contract from minute-level data."""

    build_started = time.perf_counter()

    session_start_ms = ensure_epoch_ms(session_start_ms)
    session_end_ms = ensure_epoch_ms(session_end_ms)
    if session_end_ms <= session_start_ms:
        raise ValueError("session_end_ms must be greater than session_start_ms")

    if df_m1 is None or df_m1.empty:
        raise ValueError("df_m1 must contain session data")

    frame = df_m1.copy()
    if "ts_open" not in frame:
        raise ValueError("df_m1 must include 'ts_open'")

    frame = frame.dropna(subset=["ts_open", "open", "high", "low", "close", "volume"])
    if frame.empty:
        raise ValueError("session frame empty after sanitisation")

    frame["ts_open"] = frame["ts_open"].astype("int64")
    frame.sort_values("ts_open", inplace=True)
    mask = (frame["ts_open"] >= session_start_ms) & (frame["ts_open"] < session_end_ms)
    frame = frame.loc[mask].reset_index(drop=True)
    if frame.empty:
        raise ValueError("session frame has no rows within requested window")

    expected_minutes = _session_expected_minutes(session_start_ms, session_end_ms)
    coverage = float(meta.get("coverage", len(frame) / max(expected_minutes, 1)))
    if expected_minutes > 0 and coverage < 0.99:
        raise RuntimeError("insufficient_coverage")

    source_seq = list(meta.get("source_sequence", []))

    price_high = _safe_float(frame["high"].max()) or 0.0
    price_low = _safe_float(frame["low"].min()) or 0.0
    price_range = price_high - price_low
    atr_series = _compute_atr(frame)
    frame["atr14"] = atr_series
    atr_value = _safe_float(atr_series.iloc[-1]) if not atr_series.empty else None

    volume_total = _safe_float(frame["volume"].sum()) or 0.0
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    numerator = _safe_float((typical * frame["volume"]).sum()) or 0.0
    denominator = _safe_float(frame["volume"].sum()) or 0.0
    last_close = _safe_float(frame["close"].iloc[-1]) or 0.0
    vwap = numerator / denominator if denominator > 0 else last_close

    duration_ms = expected_minutes * MINUTE_MS
    prev_frame = _load_previous_session_frame(symbol, session_start_ms, duration_ms)
    pdh = _safe_float(prev_frame["high"].max()) if not prev_frame.empty else None
    pdl = _safe_float(prev_frame["low"].min()) if not prev_frame.empty else None
    pdc = _safe_float(prev_frame["close"].iloc[-1]) if not prev_frame.empty else None

    rvol = _compute_rvol(symbol, session_start_ms, duration_ms, volume_total) if volume_total > 0 else None

    micro = _compute_micro_metrics(frame)
    orderflow = _compute_orderflow(frame)

    events = _detect_session_events(frame, pdh=pdh, pdl=pdl, vwap=vwap)

    basis_p50 = None
    if "basis_bp" in frame:
        basis_p50 = _safe_float(frame["basis_bp"].median())

    funding_hint = None
    if "funding_rate" in frame:
        funding_hint = _safe_float(frame["funding_rate"].iloc[-1])

    window = {
        "start_ms": session_start_ms,
        "end_ms": session_end_ms,
        "bars": int(len(frame)),
        "coverage_pct": round(coverage * 100.0, 4),
        "source_seq": source_seq,
    }

    session_block = {
        "pdc": pdc,
        "pdh": pdh,
        "pdl": pdl,
    }

    ib_frame = frame.head(60)
    if not ib_frame.empty:
        session_block["ib_high"] = _safe_float(ib_frame["high"].max())
        session_block["ib_low"] = _safe_float(ib_frame["low"].min())

    orderflow_block = {
        "delta_sum": orderflow["delta_sum"],
        "cvd_last": orderflow["cvd_last"],
        "impulses": orderflow["impulses"],
    }

    delta_series: List[Dict[str, Any]] = []
    delta_values = None
    if "delta" in frame:
        delta_values = frame["delta"]
    elif "taker_buy_vol" in frame and "volume" in frame:
        delta_values = (frame["taker_buy_vol"] * 2.0) - frame["volume"]

    if delta_values is not None:
        delta_list = delta_values.tolist()
        for row, delta_raw in zip(frame.itertuples(), delta_list):
            delta_value = _safe_float(delta_raw)
            if delta_value is None:
                continue
            delta_series.append(
                {
                    "ts_ms": int(row.ts_open),
                    "delta": delta_value,
                    "volume": _safe_float(getattr(row, "volume", None)),
                }
            )

    cvd_value = orderflow_block["cvd_last"]
    if cvd_value is None and delta_series:
        cvd_value = float(sum(item["delta"] for item in delta_series if item["delta"] is not None))
    if orderflow_block["delta_sum"] is None and delta_series:
        orderflow_block["delta_sum"] = float(
            sum(item["delta"] for item in delta_series if item["delta"] is not None)
        )

    micro_metrics = dict(micro)
    micro_metrics["basis_p50"] = basis_p50
    micro_metrics["funding_hint"] = funding_hint

    latency_ms = int((time.perf_counter() - build_started) * 1000)

    metrics = {
        "ATR": atr_value,
        "VWAP": vwap,
        "RVOL": rvol,
        "range": price_range,
        "levels": session_block,
        "delta": {
            "cvd": cvd_value,
            "sum": orderflow_block["delta_sum"],
            "per_bar": delta_series,
            "impulses": orderflow_block["impulses"],
        },
        "microstructure": micro_metrics,
        "events": events,
    }

    return {
        "symbol": symbol.upper(),
        "window": window,
        "metrics": metrics,
        "latency_ms": latency_ms,
    }


class ZoneDetectionConfig:
    """Compatibility stub used by legacy pipelines."""

    def __init__(self) -> None:
        self.top_n_zones = 20


async def build_session_snapshot(
    symbol: str,
    *,
    session_date: Optional[date] = None,
    cfg: Any | None = None,
) -> Dict[str, Any] | None:
    """Placeholder legacy hook; returns ``None`` to signal absence."""

    return None


async def build_72h_context(
    symbol: str,
    *,
    hours: int = 72,
    cfg: Any | None = None,
) -> Dict[str, Any] | None:
    """Placeholder legacy hook; returns ``None`` to signal absence."""

    return None


__all__ = ["build_smc_session_v1", "ZoneDetectionConfig", "build_session_snapshot", "build_72h_context"]
