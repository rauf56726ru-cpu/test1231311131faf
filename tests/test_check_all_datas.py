from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.services.inspection as inspection


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def snapshot_storage(tmp_path, monkeypatch):
    storage_dir = tmp_path / "snapshots"
    monkeypatch.setattr(inspection, "SNAPSHOT_STORAGE_DIR", storage_dir)
    inspection._SNAPSHOT_STORE.clear()
    inspection._ensure_storage_dir()
    inspection._load_existing_snapshots()
    yield storage_dir
    inspection._SNAPSHOT_STORE.clear()


def _build_snapshot_payload(base: datetime, count: int = 12) -> dict:
    candles = []
    for index in range(count):
        moment = base + timedelta(minutes=index)
        timestamp_ms = int(moment.timestamp() * 1000)
        open_price = 100.0 + index
        close_price = open_price + 0.5
        high_price = close_price + 0.25
        low_price = open_price - 0.25
        volume = 5.0 + index * 0.1
        candles.append(
            {
                "t": timestamp_ms,
                "o": round(open_price, 2),
                "h": round(high_price, 2),
                "l": round(low_price, 2),
                "c": round(close_price, 2),
                "v": round(volume, 3),
            }
        )

    return {"symbol": "BTCUSDT", "tf": "1m", "candles": candles}


def test_historical_snapshot_still_populates_window(client: TestClient) -> None:
    base = (
        datetime.now(timezone.utc)
        .replace(hour=12, minute=0, second=0, microsecond=0)
        - timedelta(days=5)
    )
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get(f"/inspection/check-all?snapshot={snapshot_id}")
    assert response.status_code == 200
    body = response.json()

    expected_last = base + timedelta(minutes=len(payload["candles"]) - 1)
    expected_last_iso = expected_last.isoformat()

    assert body["snapshot_id"] == snapshot_id
    assert body["asof_utc"].startswith(expected_last_iso)
    assert body["latest_candle_utc"].startswith(expected_last_iso)
    assert body["last_candle"]["t"] == int(expected_last.timestamp() * 1000)
    assert body["current_day"]["summary"]["count"] == len(payload["candles"])
    assert body["last_hours"]["summary"]["count"] == len(payload["candles"])


def test_snapshot_persisted_locally(client: TestClient) -> None:
    base = datetime.now(timezone.utc) - timedelta(days=2)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    stored_path = inspection._snapshot_path(snapshot_id)
    assert stored_path.exists()

    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored.get("id") == snapshot_id
    assert stored.get("frames", {}).get("1m", {}).get("candles")
    first_stored = stored["frames"]["1m"]["candles"][0]
    assert first_stored["t"] == payload["candles"][0]["t"]

    inspection._SNAPSHOT_STORE.clear()
    inspection._load_existing_snapshots()

    reloaded = inspection.get_snapshot(snapshot_id)
    assert reloaded is not None
    assert reloaded["frames"]["1m"]["candles"][-1]["t"] == payload["candles"][-1]["t"]


def test_check_all_after_reload_returns_data(client: TestClient) -> None:
    base = (
        datetime.now(timezone.utc)
        .replace(hour=10, minute=0, second=0, microsecond=0)
        - timedelta(days=3)
    )
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    inspection._SNAPSHOT_STORE.clear()
    inspection._load_existing_snapshots()

    response = client.get(f"/inspection/check-all?snapshot={snapshot_id}")
    assert response.status_code == 200
    body = response.json()

    expected_last = base + timedelta(minutes=len(payload["candles"]) - 1)
    assert body["snapshot_id"] == snapshot_id
    assert body["last_candle"]["t"] == int(expected_last.timestamp() * 1000)
    assert body["last_hours"]["summary"]["count"] == len(payload["candles"])
