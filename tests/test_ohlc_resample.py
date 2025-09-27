from datetime import datetime, timedelta, timezone

import pytest

from src.services.ohlc import TIMEFRAME_TO_MS, resample_ohlcv

UTC = timezone.utc


def _minute_block(start: datetime, o: float, h: float, l: float, c: float, *, minutes: int = 15):
    candles = []
    for offset in range(minutes):
        ts = int((start + timedelta(minutes=offset)).timestamp() * 1000)
        open_price = o if offset == 0 else c
        close_price = c
        high_price = h if offset == 0 else max(c, o)
        low_price = l if offset == 0 else min(c, o)
        candles.append({
            "t": ts,
            "o": open_price,
            "h": high_price,
            "l": low_price,
            "c": close_price,
            "v": 1.0,
        })
    return candles


def test_resample_builds_expected_15m_and_1h_buckets():
    base = datetime(2024, 1, 1, tzinfo=UTC)
    minute_candles = []
    specs = [
        (100.0, 105.0, 99.5, 102.0),
        (102.0, 106.0, 101.0, 104.0),
        (104.0, 110.0, 103.0, 108.0),
        (108.0, 112.0, 107.0, 111.0),
    ]
    for index, spec in enumerate(specs):
        start = base + timedelta(minutes=15 * index)
        minute_candles.extend(_minute_block(start, *spec))

    resampled_15m = resample_ohlcv(minute_candles, TIMEFRAME_TO_MS["15m"])
    assert len(resampled_15m) == len(specs)
    first = resampled_15m[0]
    assert first["t"] == int(base.timestamp() * 1000)
    assert pytest.approx(first["o"]) == 100.0
    assert pytest.approx(first["h"]) == 105.0
    assert pytest.approx(first["l"]) == 99.5
    assert pytest.approx(first["c"]) == 102.0
    assert pytest.approx(first["v"]) == 15.0

    resampled_1h = resample_ohlcv(minute_candles, TIMEFRAME_TO_MS["1h"])
    assert len(resampled_1h) == 1
    hourly = resampled_1h[0]
    assert hourly["t"] == int(base.timestamp() * 1000)
    assert pytest.approx(hourly["o"]) == 100.0
    assert pytest.approx(hourly["h"]) == 112.0
    assert pytest.approx(hourly["l"]) == 99.5
    assert pytest.approx(hourly["c"]) == 111.0
    assert pytest.approx(hourly["v"]) == 60.0
