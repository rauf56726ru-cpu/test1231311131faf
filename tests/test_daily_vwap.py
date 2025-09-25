from __future__ import annotations
import asyncio

from datetime import datetime, timezone
from typing import List

import pytest

from src.services.vwap import fetch_daily_vwap


class DummyResponse:
    def __init__(self, data: List[list]):
        self._data = data

    def json(self) -> List[list]:
        return self._data

    def raise_for_status(self) -> None:
        return None


class DummyClient:
    def __init__(self, batches: List[List[list]]):
        self._batches = batches

    async def __aenter__(self) -> "DummyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, params=None, timeout=None) -> DummyResponse:
        if self._batches:
            batch = self._batches.pop(0)
        else:
            batch = []
        return DummyResponse(batch)


def make_kline(open_ms: int, close_ms: int, high: float, low: float, close: float, volume: float) -> list:
    return [
        open_ms,
        "0",
        str(high),
        str(low),
        str(close),
        str(volume),
        close_ms,
        "0",
        0,
        "0",
        "0",
        "0",
    ]



def test_daily_vwap_constant_prices() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    open_ms = int(base.timestamp() * 1000)
    k1 = make_kline(open_ms, open_ms + 60_000 - 1, 101, 99, 100, 5)
    k2 = make_kline(open_ms + 60_000, open_ms + 120_000 - 1, 102, 100, 101, 10)
    now_ms = open_ms + 120_000

    result = asyncio.run(fetch_daily_vwap(
        "BTCUSDT",
        now_ms=now_ms,
        client_factory=lambda: DummyClient([[k1, k2]]),
    ))

    assert result["candles_used"] == 2
    assert result["cum_volume"] == pytest.approx(15.0)
    assert result["vwap_at_last_closed"] == pytest.approx(100.6666666667)
    assert result["last_closed_candle_time"].startswith("2024-01-01T00:01:59")



def test_future_candle_excluded() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    open_ms = int(base.timestamp() * 1000)
    k1 = make_kline(open_ms, open_ms + 60_000 - 1, 101, 99, 100, 5)
    future = make_kline(open_ms + 60_000, open_ms + 120_000 - 1, 102, 100, 101, 10)
    now_ms = open_ms + 60_000  # close time of first candle

    result = asyncio.run(fetch_daily_vwap(
        "ETHUSDT",
        now_ms=now_ms,
        client_factory=lambda: DummyClient([[k1, future]]),
    ))

    assert result["candles_used"] == 1
    assert result["vwap_at_last_closed"] == pytest.approx(100.0)



def test_zero_volume_filtered() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    open_ms = int(base.timestamp() * 1000)
    zero = make_kline(open_ms, open_ms + 60_000 - 1, 101, 99, 100, 0.0)
    now_ms = open_ms + 60_000

    with pytest.raises(ValueError, match="no data"):
        asyncio.run(fetch_daily_vwap(
            "BNBUSDT",
            now_ms=now_ms,
            client_factory=lambda: DummyClient([[zero]]),
        ))



def test_multipage_fetch() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    open_ms = int(base.timestamp() * 1000)
    filler = [
        make_kline(open_ms + i * 60_000, open_ms + (i + 1) * 60_000 - 1, 100, 100, 100, 0.0)
        for i in range(999)
    ]
    first = make_kline(open_ms + 999 * 60_000, open_ms + 1_000 * 60_000 - 1, 101, 99, 100, 5)
    second = make_kline(open_ms + 1_000 * 60_000, open_ms + 1_001 * 60_000 - 1, 103, 101, 102, 5)
    batches = [
        filler + [first],
        [second],
        [],
    ]
    now_ms = open_ms + 1_001 * 60_000

    result = asyncio.run(fetch_daily_vwap(
        "SOLUSDT",
        now_ms=now_ms,
        client_factory=lambda: DummyClient([list(batch) for batch in batches]),
    ))

    assert result["candles_used"] == 2
    assert result["vwap_at_last_closed"] == pytest.approx(101.0)


