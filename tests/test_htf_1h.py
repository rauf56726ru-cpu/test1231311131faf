from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.services.ohlc import aggregate_1m_to_1h


UTC = timezone.utc


def _minute(ts: datetime, price: float, volume: float) -> dict[str, float | int]:
    ts_ms = int(ts.timestamp() * 1000)
    return {
        "t": ts_ms,
        "o": price,
        "h": price + 0.5,
        "l": price - 0.5,
        "c": price + 0.25,
        "v": volume,
    }


def test_aggregate_1m_to_1h_produces_hourly_candles() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    candles = []
    for index in range(120):
        minute_ts = base + timedelta(minutes=index)
        candles.append(_minute(minute_ts, price=100.0 + index, volume=1.0 + index * 0.1))

    hourly = aggregate_1m_to_1h(candles)

    assert len(hourly) == 2
    first_open = candles[0]
    first_close = candles[59]
    second_open = candles[60]
    second_close = candles[-1]

    assert hourly[0]["t"] == int(base.timestamp() * 1000)
    assert hourly[0]["o"] == pytest.approx(first_open["o"])
    assert hourly[0]["c"] == pytest.approx(first_close["c"])
    assert hourly[0]["h"] == pytest.approx(max(c["h"] for c in candles[:60]))
    assert hourly[0]["l"] == pytest.approx(min(c["l"] for c in candles[:60]))
    assert hourly[0]["v"] == pytest.approx(sum(c["v"] for c in candles[:60]))

    second_hour_start = int((base + timedelta(hours=1)).timestamp() * 1000)
    assert hourly[1]["t"] == second_hour_start
    assert hourly[1]["o"] == pytest.approx(second_open["o"])
    assert hourly[1]["c"] == pytest.approx(second_close["c"])

    assert all(candle["t"] % 3_600_000 == 0 for candle in hourly)


def test_aggregate_1m_to_1h_handles_missing_minutes() -> None:
    base = datetime(2024, 2, 1, tzinfo=UTC)
    candles = []
    skip_offsets = {5, 17, 42}
    for index in range(60):
        if index in skip_offsets:
            continue
        minute_ts = base + timedelta(minutes=index)
        candles.append(_minute(minute_ts, price=200.0 + index * 0.5, volume=2.0 + index * 0.2))

    hourly = aggregate_1m_to_1h(candles)

    assert len(hourly) == 1
    assert hourly[0]["t"] == int(base.timestamp() * 1000)
    assert hourly[0]["o"] == pytest.approx(candles[0]["o"])
    assert hourly[0]["c"] == pytest.approx(candles[-1]["c"])
    assert hourly[0]["h"] == pytest.approx(max(c["h"] for c in candles))
    assert hourly[0]["l"] == pytest.approx(min(c["l"] for c in candles))
    assert hourly[0]["v"] == pytest.approx(sum(c["v"] for c in candles))
