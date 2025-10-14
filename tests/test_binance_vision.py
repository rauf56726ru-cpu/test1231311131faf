from __future__ import annotations

import io
import json
from datetime import date
from zipfile import ZipFile

import pytest

from src.services import binance_vision as vision
from src.services.settings import BinanceVisionSettings


def _zip_bytes(name: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio()
async def test_fetch_agg_trades_parses_rows(monkeypatch):
    payload = _zip_bytes(
        "BTCUSDT-aggTrades-2024-01-01.csv",
        "1,100.0,2.0,10,11,1700000000000,true\n",
    )

    async def fake_download(url, *, client=None, timeout=0, trace=None):
        return payload

    monkeypatch.setattr(vision, "_download_bytes", fake_download)

    batch = await vision.fetch_dataset(
        vision.DATASET_AGG_TRADES,
        symbol="BTCUSDT",
        day=date(2024, 1, 1),
        settings=BinanceVisionSettings(),
    )

    assert batch is not None
    assert batch.dataset == vision.DATASET_AGG_TRADES
    assert batch.count == 1
    row = batch.records[0]
    assert row["agg_id"] == 1
    assert row["side"] == "sell"
    assert row["ts"] == 1_700_000_000_000


@pytest.mark.anyio()
async def test_fetch_klines_parses_rows(monkeypatch):
    payload = _zip_bytes(
        "BTCUSDT-1m-2024-01-01.csv",
        "1700000000000,100,110,90,105,15,1700000059999,20,45\n",
    )

    async def fake_download(url, *, client=None, timeout=0, trace=None):
        return payload

    monkeypatch.setattr(vision, "_download_bytes", fake_download)

    settings = BinanceVisionSettings()
    batch = await vision.fetch_dataset(
        vision.DATASET_KLINES,
        symbol="BTCUSDT",
        day=date(2024, 1, 1),
        interval="1m",
        settings=settings,
    )

    assert batch is not None
    assert batch.count == 1
    candle = batch.records[0]
    assert candle["interval"] == "1m"
    assert candle["o"] == 100.0
    assert candle["ts"] == 1_700_000_059_999
    assert candle["trades"] == 45


@pytest.mark.anyio()
async def test_fetch_exchange_info(monkeypatch):
    payload = json.dumps({"symbols": [{"symbol": "BTCUSDT"}]}).encode("utf-8")

    async def fake_download(url, *, client=None, timeout=0, trace=None):
        return payload

    monkeypatch.setattr(vision, "_download_bytes", fake_download)

    batch = await vision.fetch_dataset(
        vision.DATASET_EXCHANGE_INFO,
        settings=BinanceVisionSettings(),
    )

    assert batch is not None
    assert batch.records[0]["symbols"][0]["symbol"] == "BTCUSDT"


@pytest.mark.anyio()
async def test_fetch_book_depth_from_json_zip(monkeypatch):
    depth_payload = json.dumps(
        {
            "timestamp": 1_700_000_000_000,
            "bids": [[100.0, 1.0]],
            "asks": [[100.5, 2.0]],
        }
    )
    payload = _zip_bytes("snapshot.json", depth_payload)

    async def fake_download(url, *, client=None, timeout=0, trace=None):
        return payload

    monkeypatch.setattr(vision, "_download_bytes", fake_download)

    batch = await vision.fetch_dataset(
        vision.DATASET_BOOK_DEPTH,
        symbol="BTCUSDT",
        day=date(2024, 1, 1),
        settings=BinanceVisionSettings(),
    )

    assert batch is not None
    assert batch.count == 1
    snap = batch.records[0]
    assert snap["bids"] == [[100.0, 1.0]]
    assert snap["asks"][0][0] == 100.5
