import pytest

import src.services.ohlcv as ohlcv


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_fetch_ohlcv_uses_shared_cache(monkeypatch):
    calls = 0

    async def fake_collect(symbol: str, lookback_days: int):
        nonlocal calls
        calls += 1
        return [
            ohlcv.Candle(ts=0, open=1.0, high=1.5, low=0.5, close=1.2, volume=10.0),
            ohlcv.Candle(ts=60_000, open=1.2, high=1.7, low=0.8, close=1.4, volume=12.0),
        ]

    monkeypatch.setattr(ohlcv, "_collect_1m_candles", fake_collect)

    cache: dict[str, dict[str, object]] = {}
    first = await ohlcv.fetch_ohlcv("BTCUSDT", "1m", 1, cache=cache)
    second = await ohlcv.fetch_ohlcv("BTCUSDT", "1m", 1, cache=cache)

    assert calls == 1
    assert first == second
