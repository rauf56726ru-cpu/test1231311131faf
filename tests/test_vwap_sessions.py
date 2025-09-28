from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

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


def extract_sigma_map(payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    sigma_entries = payload.get("vwap_sigma", [])
    return {
        (entry["date"], entry["session"]): entry
        for entry in sigma_entries
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
        make_candle(base.replace(hour=12, minute=0), 150.0, 1.0),
        make_candle(base.replace(hour=20, minute=0), 190.0, 1.0),
    ]

    result = compute_session_vwaps("ethusdt", candles)
    mapping = extract_map(result)

    assert ("2024-01-01", "london") in mapping
    assert ("2024-01-01", "ny") in mapping
    # london entry should correspond to price at 08:00, ny to price at 12:00 (20:00 excluded)
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


def test_sigma_channels_structure_and_values() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        make_candle(base.replace(hour=1), 100.0, 2.0),
        make_candle(base.replace(hour=2), 110.0, 1.0),
        make_candle(base.replace(hour=9), 120.0, 3.0),
        make_candle(base.replace(hour=13), 130.0, 4.0),
    ]

    result = compute_session_vwaps("ltcusdt", candles)
    vwap_map = extract_map(result)
    sigma_map = extract_sigma_map(result)

    assert sigma_map
    for key, sigma_entry in sigma_map.items():
        assert key in vwap_map
        sigma_levels = sigma_entry["sigma"]
        assert isinstance(sigma_levels, list) and len(sigma_levels) == 2
        center = vwap_map[key]
        for level in sigma_levels:
            minus = level["price_minus"]
            plus = level["price_plus"]
            assert center - minus == pytest.approx(plus - center)
        gap1 = abs(center - sigma_levels[0]["price_minus"])
        gap2 = abs(center - sigma_levels[1]["price_minus"])
        if gap1 > 0:
            assert gap2 > gap1
        else:
            assert gap2 == pytest.approx(0.0)
        if sigma_entry["session"] == "daily":
            assert sigma_entry["basis"] == "daily"
        else:
            assert sigma_entry["basis"] == "session"
        if gap1 == pytest.approx(0.0):
            assert gap2 == pytest.approx(0.0)

    date_key = "2024-01-01"
    asia_entry = sigma_map[(date_key, "asia")]
    asia_center = vwap_map[(date_key, "asia")]
    asia_sigma = asia_center - asia_entry["sigma"][0]["price_minus"]
    asia_prices = [100.0, 110.0]
    asia_volumes = [2.0, 1.0]
    asia_mean = sum(p * v for p, v in zip(asia_prices, asia_volumes)) / sum(asia_volumes)
    expected_asia_sigma = math.sqrt(
        sum(v * (p - asia_mean) ** 2 for p, v in zip(asia_prices, asia_volumes)) / sum(asia_volumes)
    )
    assert asia_sigma == pytest.approx(expected_asia_sigma)
    assert asia_entry["sigma"][1]["price_plus"] - asia_center == pytest.approx(asia_sigma * 2)
