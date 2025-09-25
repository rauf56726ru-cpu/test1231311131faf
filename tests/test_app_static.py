"""Integration smoke-tests for serving the chart UI from FastAPI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_index_served_with_inspection_button(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "<div id=\"chart\"" in body
    assert "id=\"show-inspection\"" in body


def test_static_assets_are_available(client: TestClient) -> None:
    response = client.get("/public/app.js")
    assert response.status_code == 200
    assert "window.LightweightCharts" in response.text


def test_inspection_snapshot_roundtrip(client: TestClient) -> None:
    snapshot_payload = {
        "symbol": "BTCUSDT",
        "tf": "1m",
        "candles": [
            {"t": 0, "o": 1.0, "h": 1.5, "l": 0.8, "c": 1.2, "v": 10.0},
            {"t": 60_000, "o": 1.2, "h": 1.7, "l": 1.0, "c": 1.4, "v": 12.0},
        ],
    }

    create_response = client.post("/inspection/snapshot", json=snapshot_payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    fetch_response = client.get(
        f"/inspection?snapshot={snapshot_id}", headers={"Accept": "application/json"}
    )
    assert fetch_response.status_code == 200
    body = fetch_response.json()
    assert body["DATA"]["symbol"] == "BTCUSDT"
    assert "frames" in body["DATA"]
    candles = body["DATA"]["frames"]["1m"]["candles"]
    assert [candle["t"] for candle in candles] == [0, 60_000]

    list_response = client.get("/inspection/snapshots")
    assert list_response.status_code == 200
    entries = list_response.json()
    assert any(item["id"] == snapshot_id for item in entries)
