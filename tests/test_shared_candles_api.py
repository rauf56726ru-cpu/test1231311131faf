from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app
from src.services import shared_candles_store


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _merge(client: TestClient, payload: Dict[str, object]) -> Dict[str, object]:
    response = client.post("/shared-candles", json=payload)
    assert response.status_code == 200
    return response.json()


def _fetch(client: TestClient, symbol: str, interval: str) -> Dict[str, object]:
    response = client.get(
        "/shared-candles",
        params={"symbol": symbol, "interval": interval},
    )
    assert response.status_code == 200
    return response.json()


def test_shared_candles_returns_empty_payload(client: TestClient) -> None:
    payload = _fetch(client, "BTCUSDT", "1m")
    assert payload["symbol"] == "BTCUSDT"
    assert payload["interval"] == "1m"
    assert payload["candles"] == []
    assert payload["intervalMs"] is None
    assert payload["lastUpdateMs"] is None


def test_shared_candles_merge_and_fetch_roundtrip(client: TestClient) -> None:
    merge_payload = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "candles": [
            {"time": 1_700_000_000, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1},
            {"time": 1_700_000_060, "open": 1.1, "high": 1.4, "low": 1.0, "close": 1.3},
        ],
        "intervalMs": 60_000,
        "lastUpdateMs": 1_700_000_120_000,
        "reset": True,
    }

    created = _merge(client, merge_payload)
    assert created["status"] == "ok"
    assert created["written"] is True
    assert created["lastUpdateMs"] == 1_700_000_120_000

    fetched = _fetch(client, "btcusdt", "1M")
    assert fetched["symbol"] == "BTCUSDT"
    assert fetched["interval"] == "1m"
    assert [bar["time"] for bar in fetched["candles"]] == [1_700_000_000, 1_700_000_060]

    incremental = _merge(
        client,
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "candles": [
                {"time": 1_700_000_120, "open": 1.3, "high": 1.6, "low": 1.2, "close": 1.5},
            ],
            "intervalMs": 60_000,
            "lastUpdateMs": 1_700_000_180_000,
        },
    )
    assert incremental["status"] == "ok"
    assert incremental["written"] is True
    assert incremental["lastUpdateMs"] == 1_700_000_180_000

    fetched_after = _fetch(client, "BTCUSDT", "1m")
    assert [bar["time"] for bar in fetched_after["candles"]] == [
        1_700_000_000,
        1_700_000_060,
        1_700_000_120,
    ]
    assert fetched_after["lastUpdateMs"] == 1_700_000_180_000


def test_shared_candles_idempotent_updates(client: TestClient) -> None:
    base_payload = {
        "symbol": "ETHUSDT",
        "interval": "5m",
        "candles": [
            {"time": 1_700_000_000, "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0},
        ],
        "lastUpdateMs": 1_700_000_060_000,
        "intervalMs": 300_000,
        "reset": True,
    }
    first = _merge(client, base_payload)
    assert first["status"] == "ok"

    duplicate = _merge(client, base_payload)
    assert duplicate["status"] == "noop"
    assert duplicate["written"] is False
    assert duplicate["lastUpdateMs"] == 1_700_000_060_000


def test_shared_candles_rate_limit(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setattr(shared_candles_store, "_WRITE_RATE_LIMIT_SECONDS", 10.0)

    first = _merge(
        client,
        {
            "symbol": "XRPUSDT",
            "interval": "1m",
            "candles": [
                {"time": 1_700_001_000, "open": 0.5, "high": 0.6, "low": 0.4, "close": 0.55},
            ],
            "lastUpdateMs": 1_700_001_000_000,
            "intervalMs": 60_000,
            "reset": True,
        },
    )
    assert first["status"] == "ok"

    throttled = _merge(
        client,
        {
            "symbol": "XRPUSDT",
            "interval": "1m",
            "candles": [
                {"time": 1_700_001_060, "open": 0.55, "high": 0.7, "low": 0.5, "close": 0.65},
            ],
            "lastUpdateMs": 1_700_001_060_000,
            "intervalMs": 60_000,
        },
    )
    assert throttled["status"] == "rate_limited"
    assert throttled["written"] is False
    assert "retryAfterMs" in throttled


def test_shared_candles_requires_valid_arguments(client: TestClient) -> None:
    missing_symbol = client.post("/shared-candles", json={"interval": "1m", "candles": []})
    assert missing_symbol.status_code == 400

    invalid_candles = client.post(
        "/shared-candles",
        json={"symbol": "BTCUSDT", "interval": "1m", "candles": "oops"},
    )
    assert invalid_candles.status_code == 400
