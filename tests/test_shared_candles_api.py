from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_shared_candles_returns_empty_payload(client: TestClient) -> None:
    response = client.get("/shared-candles", params={"symbol": "BTCUSDT", "interval": "1m"})
    assert response.status_code == 200
    payload = response.json()
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

    first_response = client.post("/shared-candles", json=merge_payload)
    assert first_response.status_code == 200
    created = first_response.json()
    assert [bar["time"] for bar in created["candles"]] == [1_700_000_000, 1_700_000_060]
    assert created["intervalMs"] == 60_000
    assert created["lastUpdateMs"] == 1_700_000_120_000

    fetch_response = client.get("/shared-candles", params={"symbol": "btcusdt", "interval": "1M"})
    assert fetch_response.status_code == 200
    fetched = fetch_response.json()
    assert fetched["symbol"] == "BTCUSDT"
    assert fetched["interval"] == "1m"
    assert [bar["time"] for bar in fetched["candles"]] == [1_700_000_000, 1_700_000_060]

    incremental_response = client.post(
        "/shared-candles",
        json={
            "symbol": "BTCUSDT",
            "interval": "1m",
            "candles": [
                {"time": 1_700_000_120, "open": 1.3, "high": 1.6, "low": 1.2, "close": 1.5},
            ],
            "intervalMs": 60_000,
            "lastUpdateMs": 1_700_000_180_000,
        },
    )
    assert incremental_response.status_code == 200
    merged = incremental_response.json()
    assert [bar["time"] for bar in merged["candles"]] == [
        1_700_000_000,
        1_700_000_060,
        1_700_000_120,
    ]
    assert merged["lastUpdateMs"] == 1_700_000_180_000


def test_shared_candles_requires_valid_arguments(client: TestClient) -> None:
    missing_symbol = client.post("/shared-candles", json={"interval": "1m", "candles": []})
    assert missing_symbol.status_code == 400

    invalid_candles = client.post(
        "/shared-candles",
        json={"symbol": "BTCUSDT", "interval": "1m", "candles": "oops"},
    )
    assert invalid_candles.status_code == 400
