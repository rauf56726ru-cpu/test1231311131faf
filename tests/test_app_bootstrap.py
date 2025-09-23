import asyncio
from datetime import datetime, timezone

from src.live.candles_store import CandleStore
from src.live.market import MarketDataProvider


def test_market_provider_reuses_persisted_history(tmp_path, monkeypatch):
    async def runner() -> None:
        store_path = tmp_path / "candles.sqlite"
        candles_store = CandleStore(store_path, max_bars=100)
        symbol = "BTCUSDT"
        interval = "1m"

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        interval_ms = 60_000
        raw_rows = []
        for idx in range(8):
            open_time = now_ms - interval_ms * (7 - idx)
            raw_rows.append(
                [
                    open_time,
                    100.0 + idx,
                    101.0 + idx,
                    99.0 + idx,
                    100.5 + idx,
                    10.0 + idx,
                    open_time + interval_ms - 1,
                ]
            )

        call_counter = {"count": 0}

        async def fake_fetch(symbol_arg, interval_arg, limit=1000, start_time=None, end_time=None):
            assert symbol_arg == symbol
            assert interval_arg == interval
            call_counter["count"] += 1
            filtered = [
                row
                for row in raw_rows
                if (start_time is None or row[0] >= start_time) and (end_time is None or row[0] <= end_time)
            ]
            return filtered

        monkeypatch.setattr("src.live.market.fetch_klines", fake_fetch)

        provider_a = MarketDataProvider(store=candles_store)
        preload_counts = await provider_a.preload_cache(symbol, [interval], candles=len(raw_rows))
        assert preload_counts[interval] == len(raw_rows)
        assert call_counter["count"] >= 1

        async def fail_fetch(*args, **kwargs):  # pragma: no cover - sanity guard
            raise AssertionError("unexpected network fetch")

        monkeypatch.setattr("src.live.market.fetch_klines", fail_fetch)

        provider_b = MarketDataProvider(store=candles_store)
        cached_counts = await provider_b.preload_cache(symbol, [interval], candles=len(raw_rows))
        assert cached_counts[interval] == len(raw_rows)

        history = await provider_b.history(
            symbol,
            interval,
            limit=5,
            date_from_ms=raw_rows[0][0],
        )
        assert len(history) == 5
        assert history[-1]["ts_ms_utc"] == raw_rows[-1][0]
        assert all(isinstance(entry["timestamp"], datetime) for entry in history)

    asyncio.run(runner())


def test_aggregated_history_uses_cached_minute(monkeypatch):
    async def runner() -> None:
        provider = MarketDataProvider()
        symbol = "BTCUSDT"
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        base_rows = []
        for idx in range(10):
            open_time = now_ms - (9 - idx) * 60_000
            base_rows.append(
                [
                    open_time,
                    100.0 + idx,
                    101.0 + idx,
                    99.0 + idx,
                    100.5 + idx,
                    10.0 + idx,
                    open_time + 59_000,
                ]
            )

        calls = []

        async def fake_fetch(symbol_arg, interval_arg, limit=1000, start_time=None, end_time=None):
            calls.append(interval_arg)
            filtered = [
                row
                for row in base_rows
                if (start_time is None or row[0] >= start_time) and (end_time is None or row[0] <= end_time)
            ]
            return filtered

        monkeypatch.setattr("src.live.market.fetch_klines", fake_fetch)
        await provider._preload_interval(symbol, "1m", candles=len(base_rows))
        calls.clear()

        candles = await provider.history(symbol, "5m", limit=2)
        assert calls == []
        assert len(candles) == 2

    asyncio.run(runner())

def test_aggregated_gap_persists(tmp_path, monkeypatch):
    async def runner() -> None:
        store = CandleStore(tmp_path / "candles.sqlite", max_bars=200)
        provider = MarketDataProvider(store=store)
        symbol = "BTCUSDT"
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        interval_ms = 60_000
        base_rows = []
        for idx in range(12):
            open_time = now_ms - (11 - idx) * interval_ms
            base_rows.append(
                [
                    open_time,
                    100.0 + idx,
                    101.0 + idx,
                    99.0 + idx,
                    100.5 + idx,
                    10.0 + idx,
                    open_time + interval_ms - 1,
                ]
            )

        async def fake_fetch(symbol_arg, interval_arg, limit=1000, start_time=None, end_time=None):
            assert interval_arg in {"1m", "5m"}
            filtered = [
                row
                for row in base_rows
                if (start_time is None or row[0] >= start_time) and (end_time is None or row[0] <= end_time)
            ]
            return filtered

        monkeypatch.setattr("src.live.market.fetch_klines", fake_fetch)

        start = base_rows[0][0]
        end = base_rows[-1][0]
        candles = await provider.fetch_gap(symbol, "5m", start, end)
        assert candles

        stored = await store.read(symbol, "5m")
        assert stored
        assert stored[-1]["ts_ms_utc"] == candles[-1]["ts_ms_utc"]

    asyncio.run(runner())

