import asyncio
from datetime import datetime, timezone
from typing import List, Sequence

import pytest

import src.services.ohlc as ohlc


class StubFetcher:
    def __init__(self, *, interval_ms: int, start: datetime, count: int) -> None:
        self.interval_ms = interval_ms
        self.calls: List[tuple[str, str, int | None, int | None, int | None]] = []
        self.rows: List[List[float | int]] = []
        self._seed(start, count)

    def _seed(self, start: datetime, count: int) -> None:
        base_ms = int(start.timestamp() * 1000)
        for idx in range(count):
            ts = base_ms + idx * self.interval_ms
            price = 100.0 + idx * 0.1
            self.rows.append([ts, price, price + 1, price - 1, price + 0.5, 1.0])

    def extend(self, count: int) -> None:
        if not self.rows:
            raise AssertionError("No existing data to extend")
        last_ts = int(self.rows[-1][0])
        start_dt = datetime.fromtimestamp((last_ts + self.interval_ms) / 1000, tz=timezone.utc)
        self._seed(start_dt, count)

    async def __call__(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int | None,
        end_ms: int | None,
        limit: int | None,
    ) -> Sequence[Sequence[object]]:
        self.calls.append((symbol, timeframe, start_ms, end_ms, limit))
        data = []
        for row in self.rows:
            ts = int(row[0])
            if start_ms is not None and ts < start_ms:
                continue
            if end_ms is not None and ts >= end_ms:
                continue
            data.append(row)
        if limit is not None:
            data = data[:limit]
        return data


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    ohlc._CANDLE_CACHE.clear()
    yield
    ohlc._CANDLE_CACHE.clear()


def test_fetch_ohlcv_reuses_cached_window() -> None:
    interval_ms = ohlc.TIMEFRAME_TO_MS["1m"]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    fetcher = StubFetcher(interval_ms=interval_ms, start=start, count=120)

    payload = asyncio.run(ohlc.fetch_ohlcv("BTCUSDT", "1m", hours=2, fetcher=fetcher))
    assert fetcher.calls and fetcher.calls[0][2] is None
    candles = payload["candles"]
    assert len(candles) == 120
    assert candles[-1]["t"] == fetcher.rows[119][0]

    fetcher.extend(5)
    payload_fresh = asyncio.run(ohlc.fetch_ohlcv("BTCUSDT", "1m", hours=2, fetcher=fetcher))
    assert len(fetcher.calls) == 2
    tail_call = fetcher.calls[-1]
    assert tail_call[2] == candles[-1]["t"] + interval_ms
    fresh_candles = payload_fresh["candles"]
    assert len(fresh_candles) == 120
    assert fresh_candles[-1]["t"] == fetcher.rows[-1][0]
    assert fresh_candles[0]["t"] == fresh_candles[-1]["t"] - 119 * interval_ms
