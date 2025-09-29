from datetime import datetime, timedelta, timezone

import pytest

from src.services.ohlc import (
    TIMEFRAME_TO_MS,
    build_multi_timeframe_ohlcv,
    resample_ohlcv,
)

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


def test_build_multi_timeframe_ohlcv_respects_complete_windows():
    base = datetime(2024, 5, 1, tzinfo=UTC)
    total_minutes = (24 * 60) + 30
    candles = []
    for index in range(total_minutes):
        ts = int((base + timedelta(minutes=index)).timestamp() * 1000)
        open_price = 100.0 + index * 0.5
        close_price = open_price + 0.2
        high_price = close_price + 0.1
        low_price = open_price - 0.1
        volume = float(1 + (index % 5))
        candles.append(
            {
                "t": ts,
                "o": open_price,
                "h": high_price,
                "l": low_price,
                "c": close_price,
                "v": volume,
            }
        )

    block = build_multi_timeframe_ohlcv(candles)

    assert set(block) == set(TIMEFRAME_TO_MS)
    assert len(block["1m"]["candles"]) == total_minutes

    for tf, payload in block.items():
        series = payload.get("candles", [])
        times = [candle["t"] for candle in series]
        assert times == sorted(times)
        interval_ms = TIMEFRAME_TO_MS[tf]
        if times:
            assert all((ts % interval_ms) == 0 for ts in times)

    candles_3m = block["3m"]["candles"]
    assert len(candles_3m) == total_minutes // 3
    first_3m = candles_3m[0]
    assert first_3m["t"] == int(base.timestamp() * 1000)
    assert pytest.approx(first_3m["o"]) == candles[0]["o"]
    assert pytest.approx(first_3m["c"]) == candles[2]["c"]
    assert pytest.approx(first_3m["h"]) == candles[2]["h"]
    assert pytest.approx(first_3m["l"]) == candles[0]["l"]
    assert pytest.approx(first_3m["v"]) == sum(c["v"] for c in candles[:3])

    candles_5m = block["5m"]["candles"]
    assert len(candles_5m) == total_minutes // 5
    assert candles_5m[0]["t"] == int(base.timestamp() * 1000)

    hourly = block["1h"]["candles"]
    assert len(hourly) == 24
    assert hourly[-1]["t"] == int((base + timedelta(hours=23)).timestamp() * 1000)
    assert pytest.approx(hourly[-1]["c"]) == candles[(24 * 60) - 1]["c"]
    assert all((bar["t"] % TIMEFRAME_TO_MS["1h"]) == 0 for bar in hourly)

    four_hour = block["4h"]["candles"]
    assert len(four_hour) == 6
    assert four_hour[-1]["t"] == int((base + timedelta(hours=20)).timestamp() * 1000)

    daily = block["1d"]["candles"]
    assert len(daily) == 1
    assert daily[0]["t"] == int(base.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    assert pytest.approx(daily[0]["c"]) == candles[1440 - 1]["c"]
    assert pytest.approx(daily[0]["v"]) == sum(c["v"] for c in candles[:1440])


def test_build_multi_timeframe_ohlcv_fetches_missing():
    calls: list[tuple[str, str]] = []

    base_ts = 1_700_000_000_000

    def fake_fetch(symbol: str, timeframe: str, hours: int | None = None):
        calls.append((symbol, timeframe))
        interval = TIMEFRAME_TO_MS.get(timeframe, 60_000)
        aligned = base_ts - (base_ts % interval)
        candles = []
        for idx in range(2):
            open_ts = aligned + idx * interval
            candles.append(
                {
                    "t": open_ts,
                    "o": 100.0 + idx,
                    "h": 101.0 + idx,
                    "l": 99.0 + idx,
                    "c": 100.5 + idx,
                    "v": 5.0 + idx,
                }
            )
        return {"symbol": symbol, "tf": timeframe, "candles": candles, "last_ts": candles[-1]["t"]}

    block = build_multi_timeframe_ohlcv([], symbol="BTCUSDT", fetcher=fake_fetch)

    assert calls and calls[0] == ("BTCUSDT", "1m")
    for tf, payload in block.items():
        assert payload["candles"], f"Expected fetched candles for {tf}"
