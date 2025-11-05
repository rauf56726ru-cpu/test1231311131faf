"""VWAP context builders for daily and session windows."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

import pandas as pd

from src.services.tpo import calculate_session_tpo, calculate_tpo, TPOCalculationError

LOGGER = logging.getLogger(__name__)

SESSION_NAMES = ("asia", "london", "ny")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(numeric):
        return None
    return float(numeric)


def _prepare_candles(frame: pd.DataFrame) -> list[Dict[str, float]]:
    required = {"ts_open", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"build_vwap_context requires columns: {sorted(missing)}")

    candles: list[Dict[str, float]] = []
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


def _build_daily_block(candles: list[Dict[str, float]]) -> Dict[str, Any]:
    try:
        tpo_payload = calculate_tpo(candles)
    except (TPOCalculationError, ValueError) as exc:
        LOGGER.debug("vwap_context.daily.failed", exc_info=exc)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("vwap_context.daily.unexpected", exc_info=exc)
        return {}

    days = tpo_payload.get("days")
    if not isinstance(days, list) or not days:
        return {}
    latest = days[-1]
    if not isinstance(latest, Mapping):
        return {}
    return {
        "date": latest.get("date"),
        "vwap": _safe_float(latest.get("vwap")),
        "poc": _safe_float(latest.get("POC")),
        "vah": _safe_float(latest.get("VAH")),
        "val": _safe_float(latest.get("VAL")),
        "sd1_plus": _safe_float(latest.get("sd1_plus")),
        "sd1_minus": _safe_float(latest.get("sd1_minus")),
        "sd2_plus": _safe_float(latest.get("sd2_plus")),
        "sd2_minus": _safe_float(latest.get("sd2_minus")),
    }


def _build_session_block(candles: list[Dict[str, float]], session: str) -> Dict[str, Any]:
    try:
        payload = calculate_session_tpo(candles, session)
    except (TPOCalculationError, ValueError) as exc:
        LOGGER.debug("vwap_context.session.failed", extra={"session": session}, exc_info=exc)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("vwap_context.session.unexpected", extra={"session": session}, exc_info=exc)
        return {}

    if not isinstance(payload, Mapping):
        return {}
    return {
        "vwap": _safe_float(payload.get("vwap")),
        "poc": _safe_float(payload.get("POC")),
        "vah": _safe_float(payload.get("VAH")),
        "val": _safe_float(payload.get("VAL")),
        "high": _safe_float(payload.get("High")),
        "low": _safe_float(payload.get("Low")),
        "ib_high": _safe_float(payload.get("IBH")),
        "ib_low": _safe_float(payload.get("IBL")),
    }


def build_vwap_context(frame: pd.DataFrame) -> Dict[str, Any]:
    """Calculate daily and session VWAP context for the supplied minute frame."""

    if frame is None or frame.empty:
        return {}

    candles = _prepare_candles(frame.dropna(subset=["ts_open", "open", "high", "low", "close"]))
    if not candles:
        return {}

    daily = _build_daily_block(candles)
    sessions = {name: _build_session_block(candles, name) for name in SESSION_NAMES}

    result = {"daily": daily, "sessions": sessions}
    if not any(daily.values()) and not any(any(block.values()) for block in sessions.values()):
        return {}
    return result


__all__ = ["build_vwap_context"]
