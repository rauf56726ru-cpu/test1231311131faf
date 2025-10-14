from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest

import httpx

from src.services import vision_store as vision_store_module
from src.services.vision_store import VisionStore
from src.services.ohlc import fetch_ohlcv
from src.services.orderflow import fetch_footprint, _fetch_trades
from src.services.derivatives import fetch_derivatives


@pytest.fixture()
def vision_store(tmp_path, monkeypatch):
    store_path = tmp_path / "vision.sqlite"
    store = VisionStore(store_path)
    monkeypatch.setattr(vision_store_module, "_DEFAULT_STORE", store, raising=False)
    return store


@pytest.mark.asyncio()
async def test_fetch_ohlcv_prefers_vision_store(vision_store, monkeypatch):
    base_open = int(datetime(2024, 5, 10, 12, tzinfo=timezone.utc).timestamp() * 1000)
    records = []
    for index in range(3):
        open_time = base_open + index * 60_000
        records.append(
            {
                "ts": open_time,
                "open_time": open_time,
                "close_time": open_time + 59_000,
                "o": 100.0 + index,
                "h": 101.0 + index,
                "l": 99.0 + index,
                "c": 100.5 + index,
                "v": 10.0 + index,
                "quote_volume": 20.0 + index,
                "trades": 50 + index,
            }
        )
    vision_store.upsert_klines("BTCUSDT", "1m", "2024-05-10", records)

    async_client_get = AsyncMock(side_effect=AssertionError("network access should not occur"))
    monkeypatch.setattr(httpx.AsyncClient, "get", async_client_get)

    payload = await fetch_ohlcv("BTCUSDT", "1m", hours=1)
    assert payload["candles"], "Expected candles sourced from vision store"
    assert len(payload["candles"]) == 3
    assert payload["candles"][-1]["c"] == pytest.approx(102.5)


@pytest.mark.asyncio()
async def test_fetch_footprint_uses_vision_store(vision_store, monkeypatch):
    base_ts = int(datetime(2024, 5, 10, 12, tzinfo=timezone.utc).timestamp() * 1000)
    trades = []
    for idx in range(5):
        trades.append(
            {
                "agg_id": idx + 1,
                "ts": base_ts + idx * 1_000,
                "price": 100.0 + idx * 0.5,
                "qty": 1.0 + idx * 0.1,
                "side": "buy" if idx % 2 == 0 else "sell",
                "buyer_maker": 0 if idx % 2 == 0 else 1,
            }
        )
    vision_store.insert_agg_trades("BTCUSDT", "2024-05-10", trades)

    monkeypatch.setattr(
        "src.services.orderflow._fetch_trades",
        AsyncMock(side_effect=AssertionError("network fallback not expected")),
    )

    snapshot = await fetch_footprint("BTCUSDT", 1)
    assert snapshot["per_bar"], "Expected per-bar orderflow rows"
    assert snapshot["aggregates"]["15m"], "Expected aggregates for 15m timeframe"


@pytest.mark.asyncio()
async def test_fetch_derivatives_from_vision_store(vision_store):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=2), now - timedelta(hours=1)]
    funding_records = []
    oi_records = []
    liq_records = []
    for bucket in buckets:
        ts = int(bucket.timestamp() * 1000)
        funding_records.append(
            {
                "ts": ts,
                "funding_rate": 0.0001,
                "mark_price": 100.0,
            }
        )
        oi_records.append(
            {
                "ts": ts,
                "open_interest": 1_000_000 + ts % 1_000,
                "notional": 500_000 + ts % 500,
            }
        )
        liq_records.append(
            {
                "ts": ts + 10_000,
                "price": 100.0,
                "qty": 50.0,
                "side": "sell",
                "notional": 5_000.0,
            }
        )
    vision_store.upsert_funding_rates("BTCUSDT", "2024-05-10", funding_records)
    vision_store.upsert_open_interest("BTCUSDT", "2024-05-10", oi_records)
    vision_store.upsert_liquidations("BTCUSDT", "2024-05-10", liq_records)

    rows = await fetch_derivatives("BTCUSDT", 4)
    assert rows, "Derivatives rows should be produced from store datasets"
    assert all("oi" in row for row in rows)
    assert any(row["liq_long"] > 0 for row in rows)
