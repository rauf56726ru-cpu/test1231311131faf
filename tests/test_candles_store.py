import asyncio
from datetime import datetime, timedelta, timezone

from src.live.candles_store import CandleStore


def test_candle_store_roundtrip(tmp_path):
    async def runner() -> None:
        store = CandleStore(tmp_path / "candles.sqlite", max_bars=3)
        base = datetime.now(timezone.utc).replace(microsecond=0)
        bars = []
        for idx in range(5):
            ts = base + timedelta(minutes=idx)
            bars.append(
                {
                    "ts_ms_utc": int(ts.timestamp() * 1000),
                    "open": 100.0 + idx,
                    "high": 101.0 + idx,
                    "low": 99.0 + idx,
                    "close": 100.5 + idx,
                    "volume": 10.0 + idx,
                    "close_time_ms_utc": int(ts.timestamp() * 1000) + 59_000,
                    "timestamp": ts,
                }
            )
        await store.write("BTCUSDT", "1m", bars)

        loaded = await store.read("BTCUSDT", "1m")
        assert len(loaded) == 3
        assert [entry["ts_ms_utc"] for entry in loaded] == [bar["ts_ms_utc"] for bar in bars[-3:]]
        assert all(isinstance(entry["timestamp"], datetime) and entry["timestamp"].tzinfo for entry in loaded)

        # Writing overlapping data should keep the most recent bars without duplication.
        await store.write("BTCUSDT", "1m", bars[-2:])
        reread = await store.read("BTCUSDT", "1m")
        assert len(reread) == 3
        assert reread[-1]["close"] == bars[-1]["close"]

        # Clearing removes all cached candles.
        await store.write("BTCUSDT", "1m", [])
        assert await store.read("BTCUSDT", "1m") == []

    asyncio.run(runner())
