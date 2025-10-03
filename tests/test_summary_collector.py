import pytest
from datetime import datetime, timezone

from src.services.ohlc import TIMEFRAME_TO_MS
from src.services.summary_collector import collect_recent_summary
import src.services.summary_collector as summary_collector

import src.services.candles_repository as candles_repository


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_collect_recent_summary_fills_only_missing(monkeypatch):
    symbol = "BTCUSDT"
    interval = "1m"
    repo = candles_repository.get_repository()
    interval_ms = TIMEFRAME_TO_MS[interval]
    start_ms = 1_700_000_000_000
    start_ms = (start_ms // interval_ms) * interval_ms
    end_ms = start_ms + interval_ms * 4

    seed = []
    for index in range(5):
        if index in {2, 3}:
            continue
        open_ms = start_ms + index * interval_ms
        seed.append(
            {"t": open_ms, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1.0}
        )
    repo.upsert_candles(symbol, interval, seed, stage="test.seed")
    fetched_before = repo.fetch_open_times(symbol, interval, start_ms, end_ms)
    assert fetched_before == [start_ms, start_ms + interval_ms, start_ms + interval_ms * 4]

    calls = []

    async def fake_request(client, *, symbol, interval, start_ms, end_ms, limit, bucket):
        calls.append((start_ms, end_ms, limit))
        cursor = start_ms
        rows = []
        while cursor <= end_ms - interval_ms:
            rows.append([cursor, 100.0, 101.0, 99.0, 100.5, 1.0])
            cursor += interval_ms
        return rows

    monkeypatch.setattr(summary_collector, "_request_klines", fake_request)

    result = await collect_recent_summary(
        symbol,
        days=1,
        end_ms=end_ms,
        intervals=[interval],
        start_ms=start_ms,
    )

    assert calls, "Expected at least one REST call for missing gaps"
    # ChartGapViewer mirrors Binance page sizing with a minimum of 50 candles,
    # verify the collector does the same when only a handful of bars are missing.
    assert calls[0][2] == 50
    fetched_opens = repo.fetch_open_times(symbol, interval, start_ms, end_ms)
    expected = [start_ms + i * interval_ms for i in range(5)]
    for ts in expected:
        assert ts in fetched_opens
    interval_summary = result.intervals[interval]
    assert interval_summary.gaps_total == 1
    assert interval_summary.gaps_filled == 1
    assert interval_summary.remaining_gaps == []


async def test_collect_recent_summary_skips_when_complete(monkeypatch):
    symbol = "ETHUSDT"
    interval = "3m"
    repo = candles_repository.get_repository()
    interval_ms = TIMEFRAME_TO_MS[interval]
    start_ms = 1_700_100_000_000
    start_ms = (start_ms // interval_ms) * interval_ms
    end_ms = start_ms + interval_ms * 4

    seed = [
        {"t": start_ms + i * interval_ms, "o": 200.0, "h": 201.0, "l": 199.0, "c": 200.5, "v": 2.0}
        for i in range(5)
    ]
    repo.upsert_candles(symbol, interval, seed, stage="test.seed")
    fetched_before = repo.fetch_open_times(symbol, interval, start_ms, end_ms)
    assert fetched_before == [start_ms + i * interval_ms for i in range(5)]

    async def fail_request(*args, **kwargs):
        raise AssertionError("_request_klines should not be called when coverage is complete")

    monkeypatch.setattr(summary_collector, "_request_klines", fail_request)

    result = await collect_recent_summary(
        symbol,
        days=1,
        end_ms=end_ms,
        intervals=[interval],
        start_ms=start_ms,
    )

    interval_summary = result.intervals[interval]
    assert interval_summary.gaps_total == 0
    assert interval_summary.requests == 0
    assert interval_summary.remaining_gaps == []


async def test_collect_recent_summary_ignores_open_tail(monkeypatch):
    symbol = "XRPUSDT"
    interval = "1m"
    repo = candles_repository.get_repository()
    interval_ms = TIMEFRAME_TO_MS[interval]
    start_ms = 1_701_000_000_000
    start_ms = (start_ms // interval_ms) * interval_ms

    seed = [
        {"t": start_ms + i * interval_ms, "o": 50.0, "h": 51.0, "l": 49.5, "c": 50.5, "v": 0.5}
        for i in range(5)
    ]
    repo.upsert_candles(symbol, interval, seed, stage="test.seed")

    now_ms = start_ms + interval_ms * 5 + interval_ms // 2

    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            base = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
            if tz is None:
                return base.replace(tzinfo=None)
            return base.astimezone(tz)

    async def fail_request(*args, **kwargs):
        raise AssertionError("_request_klines should not be invoked for an open tail")

    monkeypatch.setattr(summary_collector, "datetime", FrozenDateTime)
    monkeypatch.setattr(summary_collector, "_request_klines", fail_request)

    result = await collect_recent_summary(
        symbol,
        days=1,
        intervals=[interval],
        start_ms=start_ms,
    )

    interval_summary = result.intervals[interval]
    assert interval_summary.gaps_total == 0
    assert interval_summary.requests == 0
    assert interval_summary.remaining_gaps == []


async def test_collect_recent_summary_reports_progress(monkeypatch):
    symbol = "ADAUSDT"
    interval = "1m"
    repo = candles_repository.get_repository()
    interval_ms = TIMEFRAME_TO_MS[interval]
    start_ms = 1_702_000_000_000
    start_ms = (start_ms // interval_ms) * interval_ms
    end_ms = start_ms + interval_ms * 2

    events: list[str] = []

    async def fake_request(client, *, symbol, interval, start_ms, end_ms, limit, bucket):
        rows = []
        cursor = start_ms
        while cursor <= end_ms - interval_ms:
            rows.append([cursor, 100.0, 101.0, 99.0, 100.5, 1.0])
            cursor += interval_ms
        return rows

    async def reporter(event: str, payload):
        events.append(event)

    monkeypatch.setattr(summary_collector, "_request_klines", fake_request)

    await collect_recent_summary(
        symbol,
        days=1,
        end_ms=end_ms,
        intervals=[interval],
        start_ms=start_ms,
        progress=reporter,
    )

    assert "summary_collector:start" in events
    assert any(event.startswith("summary_collector:gap") for event in events)
    assert "summary_collector:finished" in events
