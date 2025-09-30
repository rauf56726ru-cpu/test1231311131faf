from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Mapping

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.app import app
import src.services.check_all_datas as check_all_datas
import src.services.inspection as inspection
from src.services import presets

UTC = timezone.utc


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def preset_storage(tmp_path, monkeypatch):
    storage_path = tmp_path / "presets.json"
    monkeypatch.setattr(presets, "_PRESET_STORAGE_PATH", storage_path)
    presets._PRESET_CACHE.clear()
    presets._STORAGE_LOADED = False
    yield
    presets._PRESET_CACHE.clear()
    presets._STORAGE_LOADED = False


@pytest.fixture(autouse=True)
def snapshot_storage(tmp_path, monkeypatch):
    storage_dir = tmp_path / "snapshots"
    monkeypatch.setattr(inspection, "SNAPSHOT_STORAGE_DIR", storage_dir)
    inspection._SNAPSHOT_STORE.clear()
    inspection._ensure_storage_dir()
    inspection._load_existing_snapshots()
    yield storage_dir
    inspection._SNAPSHOT_STORE.clear()


@pytest.fixture(autouse=True)
def stub_missing_minutes(monkeypatch):
    def filler(symbol: str, start_ms: int, end_ms: int, gaps):
        candles = []
        for gap in gaps:
            cursor = int(gap["from"])
            limit = int(gap["to"])
            while cursor <= limit:
                candles.append(
                    {
                        "t": cursor,
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1.0,
                    }
                )
                cursor += 60_000
        return candles

    monkeypatch.setattr(check_all_datas, "_download_missing_minutes", filler)

    def filler_htf(symbol: str, gaps, *, fetcher, target):
        inserted = 0
        for gap in gaps:
            cursor = int(gap.get("from", 0))
            limit = int(gap.get("to", cursor))
            while cursor <= limit:
                candle = {
                    "t": cursor,
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1.0,
                }
                if cursor not in target:
                    inserted += 1
                target[cursor] = candle
                cursor += 60_000
        return inserted

    monkeypatch.setattr(inspection, "_download_missing_minutes", filler_htf)
    yield


def _build_snapshot_payload() -> Mapping[str, object]:
    base = datetime(2024, 5, 1, 0, 0, tzinfo=UTC)
    candles = []
    # Cover all sessions with distinct extremes
    schedule = [
        (0, 0, 6, 59, 101.0, 95.0),  # asia extremes
        (7, 0, 12, 59, 115.0, 108.0),  # london extremes
        (13, 30, 20, 59, 140.0, 130.0),  # ny extremes
    ]
    for entry in schedule:
        start_hour, start_minute, end_hour, end_minute, high, low = entry
        start_dt = base.replace(hour=start_hour, minute=start_minute)
        end_dt = base.replace(hour=end_hour, minute=end_minute)
        cursor = start_dt
        while cursor <= end_dt:
            timestamp_ms = int(cursor.timestamp() * 1000)
            offset = (cursor - start_dt).seconds // 60
            price = high - offset * 0.1
            candles.append(
                {
                    "t": timestamp_ms,
                    "o": price,
                    "h": high,
                    "l": low,
                    "c": price,
                    "v": 5.0 + offset * 0.1,
                }
            )
            cursor += timedelta(minutes=1)
    return {"symbol": "BTCUSDT", "tf": "1m", "candles": candles}


def _create_snapshot(client: TestClient) -> str:
    payload = _build_snapshot_payload()
    response = client.post("/inspection/snapshot", json=payload)
    assert response.status_code == 200
    return response.json()["snapshot_id"]


def test_check_all_sessions_include_extrema(client: TestClient) -> None:
    snapshot_id = _create_snapshot(client)

    response = client.get(
        "/inspection/check-all",
        params={"snapshot": snapshot_id, "hours": 8},
    )
    assert response.status_code == 200
    body = response.json()

    vwap_sessions = body["vwap_tpo"]["sessions"]
    for session_name, session_payload in vwap_sessions.items():
        assert session_payload["sessionHigh"] is not None
        assert session_payload["sessionLow"] is not None

    composite_day = body["tpo"]["composite_day"]
    assert set(composite_day.keys()) == {"poc", "vah", "val"}


def test_profile_sessions_include_extrema(client: TestClient) -> None:
    snapshot_id = _create_snapshot(client)

    response = client.get(
        "/profile",
        params={"snapshot": snapshot_id, "tf": "1m", "last_n": 1},
    )
    assert response.status_code == 200
    body = response.json()

    sessions = [
        entry for entry in body["tpo"]["sessions"] if entry.get("session") != "daily"
    ]
    assert sessions, "expected session entries in profile payload"
    for entry in sessions:
        assert "session_high" in entry
        assert "session_low" in entry
