"""Bias calculation helpers for multiple timeframes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class BiasResult:
    direction: str
    change_pct: float
    confidence: float
    close: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "direction": self.direction,
            "change_pct": round(self.change_pct, 4),
            "confidence": round(self.confidence, 4),
            "close": round(self.close, 8),
        }


def _resolve_direction(change_pct: float, *, neutral_threshold: float) -> str:
    if change_pct >= neutral_threshold:
        return "bull"
    if change_pct <= -neutral_threshold:
        return "bear"
    return "neutral"


def _confidence(body: float, high: float, low: float) -> float:
    price_range = max(high - low, 0.0)
    if price_range <= 0.0:
        return 0.0
    value = abs(body) / price_range
    return float(max(0.0, min(1.0, value)))


def _resample_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    return agg.dropna(subset=["open", "high", "low", "close"])


def compute_bias(
    frame: pd.DataFrame,
    *,
    timeframes: Iterable[str] = ("1h", "4h", "1d"),
    neutral_pct: float = 0.1,
) -> Dict[str, Mapping[str, float | str]]:
    """
    Compute directional bias for the supplied timeframes.

    Parameters
    ----------
    frame:
        Minute-level candles with columns ``ts_open``, ``open``, ``high``, ``low``, ``close``.
    timeframes:
        Iterable of pandas-compatible resample rules (e.g. ``1h``).
    neutral_pct:
        Threshold in percent that separates bull/bear from neutral.

    Returns
    -------
    dict
        Mapping timeframe label -> bias payload.
    """

    if frame is None or frame.empty:
        return {}

    required = {"ts_open", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise KeyError(f"bias calculation requires columns: {missing}")

    working = frame[["ts_open", "open", "high", "low", "close"]].copy()
    working["ts_open"] = pd.to_datetime(working["ts_open"], unit="ms", utc=True)
    working.set_index("ts_open", inplace=True)
    working.sort_index(inplace=True)

    neutral_threshold = abs(neutral_pct)

    results: Dict[str, Mapping[str, float | str]] = {}
    for tf in timeframes:
        try:
            resampled = _resample_frame(working, tf)
        except ValueError:
            continue
        if resampled.empty:
            continue
        latest = resampled.iloc[-1]
        open_price = float(latest["open"])
        close_price = float(latest["close"])
        high_price = float(latest["high"])
        low_price = float(latest["low"])
        if open_price == 0:
            change_pct = 0.0
        else:
            change_pct = ((close_price - open_price) / abs(open_price)) * 100.0
        direction = _resolve_direction(change_pct, neutral_threshold=neutral_threshold)
        body = close_price - open_price
        confidence = _confidence(body, high_price, low_price)
        label = tf.upper()
        results[label] = BiasResult(
            direction=direction,
            change_pct=change_pct,
            confidence=confidence,
            close=close_price,
        ).to_dict()

    return results


__all__ = ["compute_bias"]
