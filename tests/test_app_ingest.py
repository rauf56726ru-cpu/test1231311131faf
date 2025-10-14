from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.app import app
from src.services.binance_vision import DATASET_EXCHANGE_INFO


def test_ingest_binance_vision_endpoint(monkeypatch):
    client = TestClient(app)
    captured = {}

    async def fake_ingest(
        *,
        symbol,
        start_ms,
        end_ms,
        datasets,
        klines_intervals,
        include_exchange_info,
        trace,
        settings=None,
    ):
        captured.update(
            {
                "symbol": symbol,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "datasets": datasets,
                "klines_intervals": klines_intervals,
                "include_exchange_info": include_exchange_info,
            }
        )
        return {"status": "ok", "ingested": {"aggTrades": {"count": 1}}}

    monkeypatch.setattr("src.api.app.ingest_binance_vision", fake_ingest)

    payload = {
        "symbol": "BTCUSDT",
        "startUtc": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
        "endUtc": datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat(),
        "datasets": ["aggTrades", "klines"],
        "includeExchangeInfo": True,
        "klinesIntervals": ["1m", "1h"],
    }

    response = client.post("/ingest/binance-vision", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert captured["symbol"] == "BTCUSDT"
    assert DATASET_EXCHANGE_INFO in captured["datasets"]
