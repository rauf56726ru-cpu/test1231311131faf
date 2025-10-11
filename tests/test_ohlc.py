"""Tests for OHLC normalization utilities."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services import ohlc


def test_candles_are_sorted_and_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        [60_000, "2", "3", "1", "2.5", "10"],
        [0, "1", "2", "0.5", "1.5", "8"],
        [180_000, "4", "5", "3", "4.5", "12"],
        [120_000, "3", "4", "2", "3.5", "9"],
    ]
    payload = ohlc.normalise_ohlcv("BTCUSDT", "1m", rows, include_diagnostics=True)
    candles = payload["candles"]
    times = [candle["t"] for candle in candles]
    assert times == [0, 60_000, 120_000, 180_000]
    assert all((t % 60_000) == 0 for t in times)
    diagnostics = payload["diagnostics"]
    assert diagnostics["duplicates"] == []
    assert diagnostics["missing_bars"] == []


def test_duplicate_rows_are_deduplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        [0, "1", "2", "0.5", "1.5", "8"],
        [60_000, "2", "3", "1", "2.5", "10"],
        [60_000, "2.1", "3.1", "1.1", "2.6", "11"],
        [120_000, "3", "4", "2", "3.5", "9"],
    ]
    payload = ohlc.normalise_ohlcv("BTCUSDT", "1m", rows, include_diagnostics=True)
    diagnostics = payload["diagnostics"]
    assert diagnostics["duplicates"] == [60_000]
    # Ensure only one candle per timestamp is exported.
    times = [candle["t"] for candle in payload["candles"]]
    assert times == [0, 60_000, 120_000]


def test_missing_rows_are_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        [0, "10", "15", "5", "12", "100"],
        [120_000, "13", "16", "9", "14", "90"],
    ]
    payload = ohlc.normalise_ohlcv("BTCUSDT", "1m", rows, include_diagnostics=True)
    candles = payload["candles"]
    assert [candle["t"] for candle in candles] == [0, 60_000, 120_000]
    # The missing bar should use the previous close (12) for OHLC values and zero volume.
    missing_candle = candles[1]
    assert missing_candle == {"t": 60_000, "o": 12.0, "h": 12.0, "l": 12.0, "c": 12.0, "v": 0.0}
    diagnostics = payload["diagnostics"]
    assert diagnostics["missing_bars"] == [{"t": 60_000, "filled_with": 12.0}]
    # Diagnostics series must expose the missing flag while export data does not.
    series_missing_flags = [item.get("missing") for item in diagnostics["series"]]
    assert series_missing_flags == [False, True, False]
