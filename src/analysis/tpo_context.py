"""TPO context helper utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pandas as pd

from src.services.tpo import calculate_tpo, TPOCalculationError


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(numeric):
        return None
    return float(numeric)


def _prepare_candles(frame: pd.DataFrame) -> List[Dict[str, float]]:
    required = {"ts_open", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"build_tpo_context requires columns: {sorted(missing)}")

    candles: List[Dict[str, float]] = []
    has_volume = "volume" in frame.columns
    for row in frame.itertuples():
        try:
            record = {
                "t": int(getattr(row, "ts_open")),
                "o": float(getattr(row, "open")),
                "h": float(getattr(row, "high")),
                "l": float(getattr(row, "low")),
                "c": float(getattr(row, "close")),
            }
        except (TypeError, ValueError):
            continue
        record["v"] = float(getattr(row, "volume")) if has_volume else 0.0
        candles.append(record)
    return candles


def build_tpo_context(frame: pd.DataFrame, *, last_n: int = 3) -> Dict[str, Any]:
    """Return compact TPO context (POC/VAH/VAL) for recent daily sessions."""

    if frame is None or frame.empty:
        return {}
    if last_n <= 0:
        return {}

    candles = _prepare_candles(frame.dropna(subset=["ts_open", "open", "high", "low", "close"]))
    if not candles:
        return {}

    try:
        payload = calculate_tpo(candles)
    except (TPOCalculationError, ValueError):
        return {}

    days = payload.get("days")
    if not isinstance(days, list) or not days:
        return {}

    recent = days[-last_n:]
    entries: List[Dict[str, Any]] = []
    for item in recent:
        if not isinstance(item, Mapping):
            continue
        entries.append(
            {
                "date": item.get("date"),
                "vwap": _safe_float(item.get("vwap")),
                "poc": _safe_float(item.get("POC")),
                "vah": _safe_float(item.get("VAH")),
                "val": _safe_float(item.get("VAL")),
                "sd1_plus": _safe_float(item.get("sd1_plus")),
                "sd1_minus": _safe_float(item.get("sd1_minus")),
                "sd2_plus": _safe_float(item.get("sd2_plus")),
                "sd2_minus": _safe_float(item.get("sd2_minus")),
                "ib_high": _safe_float(item.get("IBH")),
                "ib_low": _safe_float(item.get("IBL")),
            }
        )

    if not entries:
        return {}

    return {
        "days": entries,
        "latest": entries[-1],
    }


__all__ = ["build_tpo_context"]
