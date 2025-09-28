"""Integration tests for automatic OHLCV completion."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.ohlc import ensure_complete_ohlcv, TIMEFRAME_TO_MS


MINUTE_MS = TIMEFRAME_TO_MS["1m"]
TARGET_TIMEFRAMES: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")


@pytest.fixture(scope="module")
def synthetic_minutes() -> List[Dict[str, float]]:
    """Generate 1m candles covering >3 days to satisfy all aggregations."""

    total_minutes = (3 * 24 * 60) + (4 * 60)
    start_ts = 0
    candles: List[Dict[str, float]] = []
    price = 100.0
    for index in range(total_minutes):
        ts = start_ts + index * MINUTE_MS
        open_price = price
        high_price = price + 0.75
        low_price = price - 0.5
        close_price = price + 0.25
        volume = 10.0 + (index % 5)
        candles.append({
            "t": ts,
            "o": open_price,
            "h": high_price,
            "l": low_price,
            "c": close_price,
            "v": volume,
        })
        price += 0.1
    return candles


@pytest.fixture(scope="module")
def completed_ohlcv(
    synthetic_minutes: List[Dict[str, float]]
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Any]]]:
    bundle, diagnostics = ensure_complete_ohlcv(synthetic_minutes)
    return bundle, diagnostics


def test_ohlcv_presence(
    completed_ohlcv: Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Any]]]
) -> None:
    bundle, diagnostics = completed_ohlcv
    for tf in TARGET_TIMEFRAMES:
        assert tf in bundle, f"missing timeframe {tf}"
        series = bundle[tf]
        assert series, f"empty series for {tf}"
        diag = diagnostics.get(tf, {})
        assert diag.get("bars", 0) == len(series)


def test_ohlcv_alignment(
    completed_ohlcv: Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Any]]]
) -> None:
    bundle, _ = completed_ohlcv
    for tf, series in bundle.items():
        interval = TIMEFRAME_TO_MS[tf]
        timestamps = [candle["t"] for candle in series]
        for left, right in zip(timestamps, timestamps[1:]):
            assert right - left == interval
        assert all(ts % interval == 0 for ts in timestamps)


def test_ohlcv_consistency(
    synthetic_minutes: List[Dict[str, float]],
    completed_ohlcv: Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Any]]],
) -> None:
    bundle, _ = completed_ohlcv
    minute_index = {candle["t"]: candle for candle in synthetic_minutes}
    for tf, series in bundle.items():
        interval = TIMEFRAME_TO_MS[tf]
        first = series[0]
        start = first["t"]
        end = start + interval - MINUTE_MS
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[float] = []
        cursor = start
        while cursor <= end:
            candle = minute_index[cursor]
            opens.append(candle["o"])
            highs.append(candle["h"])
            lows.append(candle["l"])
            closes.append(candle["c"])
            volumes.append(candle["v"])
            cursor += MINUTE_MS
        assert math.isclose(first["o"], opens[0])
        assert math.isclose(first["h"], max(highs))
        assert math.isclose(first["l"], min(lows))
        assert math.isclose(first["c"], closes[-1])
        assert math.isclose(first["v"], sum(volumes))


def test_ohlcv_range(
    completed_ohlcv: Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Any]]]
) -> None:
    bundle, _ = completed_ohlcv
    expected_min_bars = {
        "1m": 4 * 60,
        "3m": 4 * 60 // 3,
        "5m": 4 * 60 // 5,
        "15m": (3 * 24 * 60) // 15,
        "1h": (3 * 24 * 60) // 60,
        "4h": (3 * 24 * 60) // 240,
        "1d": 3,
    }
    for tf, minimum in expected_min_bars.items():
        assert len(bundle[tf]) >= minimum, f"{tf} shorter than expected range"
