from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.meta import Meta
from src.services.inspection import compute_session_vwaps


def make_candle(dt: datetime, price: float, volume: float = 1.0) -> dict[str, object]:
    ms = int(dt.timestamp() * 1000)
    return {"t": ms, "o": price, "h": price, "l": price, "c": price, "v": volume}


def extract_map(payload: dict[str, object]) -> dict[tuple[str, str], float]:
    return {
        (entry["date"], entry["session"]): entry["value"]
        for entry in payload["vwap"]
    }


def test_constant_price_vwap() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        make_candle(base + timedelta(hours=2), 100.0, 2.0),
        make_candle(base + timedelta(hours=9), 100.0, 1.5),
        make_candle(base + timedelta(hours=14), 100.0, 1.0),
        make_candle(base + timedelta(days=1, hours=2), 120.0, 2.0),
        make_candle(base + timedelta(days=1, hours=9), 120.0, 1.5),
        make_candle(base + timedelta(days=1, hours=14), 120.0, 1.0),
    ]

    result = compute_session_vwaps("btcusdt", candles)
    mapping = extract_map(result)

    assert mapping[("2024-01-01", "daily")] == pytest.approx(100.0)
    assert mapping[("2024-01-01", "asia")] == pytest.approx(100.0)
    assert mapping[("2024-01-01", "london")] == pytest.approx(100.0)
    assert mapping[("2024-01-01", "ny")] == pytest.approx(100.0)
    assert mapping[("2024-01-02", "daily")] == pytest.approx(120.0)


def test_session_boundaries_inclusive_start_exclusive_end() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        make_candle(base.replace(hour=8, minute=0), 110.0, 1.0),
        make_candle(base.replace(hour=16, minute=0), 150.0, 1.0),
    ]

    result = compute_session_vwaps("ethusdt", candles)
    mapping = extract_map(result)

    assert ("2024-01-01", "london") in mapping
    assert ("2024-01-01", "ny") in mapping
    # london entry should correspond to price at 08:00, ny to price at 16:00
    assert mapping[("2024-01-01", "london")] == pytest.approx(110.0)
    assert mapping[("2024-01-01", "ny")] == pytest.approx(150.0)


def test_zero_volume_skips_sessions() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [make_candle(base, 100.0, 0.0)]

    result = compute_session_vwaps("xrpusdt", candles)
    assert result["vwap"] == []


def test_lookback_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Meta, "VWAP_LOOKBACK_DAYS", 3)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for offset in range(6):
        day = base + timedelta(days=offset)
        candles.append(make_candle(day + timedelta(hours=2), 100 + offset, 1.0))

    result = compute_session_vwaps("adausdt", candles)
    dates = {entry["date"] for entry in result["vwap"]}
    assert "2024-01-01" not in dates
    assert min(dates) >= "2024-01-04"
