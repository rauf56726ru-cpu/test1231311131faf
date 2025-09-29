from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Dict

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.check_all_datas import VALUE_AREA_PCT, _build_volume_profile_stats
from src.services.inspection import compute_session_vwaps

UTC = timezone.utc


def make_candle(
    when: datetime,
    *,
    high: float,
    low: float,
    close: float | None = None,
    volume: float = 1.0,
) -> Dict[str, float]:
    close_price = close if close is not None else (high + low) / 2.0
    timestamp_ms = int(when.timestamp() * 1000)
    return {
        "t": timestamp_ms,
        "o": close_price,
        "h": high,
        "l": low,
        "c": close_price,
        "v": volume,
    }


def find_session(payload: Dict[str, object], session: str) -> Dict[str, object]:
    for entry in payload.get("vwap", []):
        if entry.get("session") == session:
            return entry
    raise AssertionError(f"Session {session} not found in payload")


def test_session_extrema_match_high_low() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    candles = [
        make_candle(base.replace(hour=0, minute=10), high=101.5, low=98.5, volume=2.0),
        make_candle(base.replace(hour=6, minute=45), high=105.0, low=97.0, volume=3.0),
        make_candle(base.replace(hour=7, minute=15), high=112.0, low=108.0, volume=1.5),
        make_candle(base.replace(hour=12, minute=45), high=114.0, low=109.5, volume=1.0),
        make_candle(base.replace(hour=13, minute=45), high=125.0, low=115.0, volume=2.5),
    ]

    result = compute_session_vwaps("btcusdt", candles)

    asia_entry = find_session(result, "asia")
    london_entry = find_session(result, "london")
    ny_entry = find_session(result, "ny")

    assert asia_entry["session_high"] == pytest.approx(105.0)
    assert asia_entry["session_low"] == pytest.approx(97.0)
    assert london_entry["session_high"] == pytest.approx(114.0)
    assert london_entry["session_low"] == pytest.approx(108.0)
    assert ny_entry["session_high"] == pytest.approx(125.0)
    assert ny_entry["session_low"] == pytest.approx(115.0)


def test_session_boundary_bars_assigned_to_start_only() -> None:
    base = datetime(2024, 3, 2, tzinfo=UTC)
    candles = [
        make_candle(base.replace(hour=6, minute=59), high=90.0, low=85.0),
        make_candle(base.replace(hour=7, minute=0), high=110.0, low=105.0),
        make_candle(base.replace(hour=13, minute=30), high=150.0, low=140.0),
        make_candle(base.replace(hour=21, minute=0), high=200.0, low=195.0),
    ]

    result = compute_session_vwaps("ethusdt", candles)

    asia_entry = find_session(result, "asia")
    london_entry = find_session(result, "london")
    ny_entry = find_session(result, "ny")

    assert asia_entry["session_high"] == pytest.approx(90.0)
    assert asia_entry["session_low"] == pytest.approx(85.0)
    assert london_entry["session_high"] == pytest.approx(110.0)
    assert london_entry["session_low"] == pytest.approx(105.0)
    assert ny_entry["session_high"] == pytest.approx(150.0)
    assert ny_entry["session_low"] == pytest.approx(140.0)


def test_no_extrema_when_session_empty() -> None:
    start_ms = int(datetime(2024, 4, 1, 13, 30, tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime(2024, 4, 1, 21, 0, tzinfo=UTC).timestamp() * 1000)

    stats = _build_volume_profile_stats(
        [],
        start_ms=start_ms,
        end_ms=end_ms,
        tick_size=0.5,
        value_area_pct=VALUE_AREA_PCT,
    )

    assert "session_high" not in stats
    assert "session_low" not in stats
