from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.services.inspection as inspection
from src.services import presets


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

    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]} if candles else None
    payload = {"symbol": "BTCUSDT", "tf": "1m", "candles": candles}
    if selection:
        payload["selection"] = selection
    return payload


def _build_timeframe_candles(
    base: datetime,
    *,
    count: int,
    interval: timedelta,
    open_increment: float = 1.0,
    symbol: str = "BTCUSDT",
    tf: str = "4h",
) -> dict:
    candles = []
    for index in range(count):
        moment = base + index * interval
        ts_ms = int(moment.timestamp() * 1000)
        open_price = 1000.0 + index * open_increment
        close_price = open_price + 10.0
        high_price = close_price + 5.0
        low_price = open_price - 5.0
        volume = 50.0 + index
        candles.append(
            {
                "t": ts_ms,
                "o": round(open_price, 2),
                "h": round(high_price, 2),
                "l": round(low_price, 2),
                "c": round(close_price, 2),
                "v": round(volume, 3),
            }
        )

    selection = None
    if candles:
        selection = {
            "start": candles[0]["t"],
            "end": candles[-1]["t"] + int(interval.total_seconds() * 1000),
        }

    payload = {"symbol": symbol, "tf": tf, "candles": candles}
    if selection:
        payload["selection"] = selection
    return payload


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

    params = {
        "snapshot": snapshot_id,
        "selection_start": payload["candles"][0]["t"],
        "selection_end": payload["candles"][-1]["t"],
        "hours": 2,
    }
    response = client.get("/inspection/check-all", params=params)
    assert response.status_code == 200
    body = response.json()

    expected_last = base + timedelta(minutes=len(payload["candles"]) - 1)
    expected_reference = expected_last + timedelta(minutes=1)
    expected_last_iso = expected_reference.isoformat()

    assert body["snapshot_id"] == snapshot_id
    assert body["asof_utc"].startswith(expected_last_iso)
    assert body["latest_candle_utc"].startswith(expected_last.isoformat())
    assert body["latest_candle"]["t"] == int(expected_last.timestamp() * 1000)
    assert body["datas_for_last_N_hours"]["hours"] == 2
    assert (
        body["datas_for_last_N_hours"]["frames"]["1m"]["summary"]["count"]
        == len(payload["candles"])
    )
    assert (
        body["datas_for_last_N_hours"]["frames"]["1m"]["candles"][-1]["t"]
        == payload["candles"][-1]["t"]
    )
    detailed_start_ms = (payload["candles"][-1]["t"] + 60_000) - 2 * 3_600_000
    expected_detailed_start = datetime.fromtimestamp(
        detailed_start_ms / 1000, tz=timezone.utc
    ).isoformat()
    assert body["datas_for_last_N_hours"]["range"]["start_utc"].startswith(
        expected_detailed_start
    )
    movement_key = next(
        key for key in body.keys() if isinstance(key, str) and key.startswith("movement_datas_for_")
    )
    assert body[movement_key]["days"] == 0
    expected_movement_end_ms = max(payload["candles"][0]["t"], detailed_start_ms)
    expected_movement_end = datetime.fromtimestamp(
        expected_movement_end_ms / 1000, tz=timezone.utc
    ).isoformat()
    assert body[movement_key]["range"]["end_utc"].startswith(expected_movement_end)
    assert "tpo" in body and isinstance(body["tpo"], list)
    assert "profile" in body and isinstance(body["profile"], list)
    assert "zones" in body and isinstance(body["zones"], list)
    preset_payload = body.get("profile_preset")
    assert preset_payload is not None
    assert preset_payload["symbol"] == "BTCUSDT"
    assert preset_payload["builtin"] is True
    assert body.get("profile_preset_required") is False
    if body["zones"]:
        zone_types = {zone["type"] for zone in body["zones"]}
        assert {"tpo_poc", "tpo_vah", "tpo_val"}.issubset(zone_types)


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

    params = {
        "snapshot": snapshot_id,
        "selection_start": payload["candles"][0]["t"],
        "selection_end": payload["candles"][-1]["t"],
        "hours": 3,
    }
    response = client.get("/inspection/check-all", params=params)
    assert response.status_code == 200
    body = response.json()

    expected_last = base + timedelta(minutes=len(payload["candles"]) - 1)
    expected_reference = expected_last + timedelta(minutes=1)
    assert body["snapshot_id"] == snapshot_id
    assert body["latest_candle"]["t"] == int(expected_last.timestamp() * 1000)
    assert (
        body["datas_for_last_N_hours"]["frames"]["1m"]["summary"]["count"]
        == len(payload["candles"])
    )
    assert body["datas_for_last_N_hours"]["hours"] == 3
    detailed_start_ms = (payload["candles"][-1]["t"] + 60_000) - 3 * 3_600_000
    expected_detailed_start = datetime.fromtimestamp(
        detailed_start_ms / 1000, tz=timezone.utc
    ).isoformat()
    assert body["datas_for_last_N_hours"]["range"]["start_utc"].startswith(
        expected_detailed_start
    )
    movement_key = next(
        key for key in body.keys() if isinstance(key, str) and key.startswith("movement_datas_for_")
    )
    expected_movement_end_ms = max(payload["candles"][0]["t"], detailed_start_ms)
    expected_movement_end = datetime.fromtimestamp(
        expected_movement_end_ms / 1000, tz=timezone.utc
    ).isoformat()
    assert body[movement_key]["range"]["end_utc"].startswith(expected_movement_end)
    assert "zones" in body


def test_detailed_section_backfills_minute_frame(client: TestClient) -> None:
    base = (
        datetime.now(timezone.utc)
        .replace(hour=8, minute=0, second=0, microsecond=0)
        - timedelta(days=1)
    )
    interval = timedelta(hours=4)
    payload = _build_timeframe_candles(
        base,
        count=3,
        interval=interval,
        symbol="ETHUSDT",
        tf="4h",
    )

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    params = {
        "snapshot": snapshot_id,
        "selection_start": payload["selection"]["start"],
        "selection_end": payload["selection"]["end"],
        "hours": 1,
    }
    response = client.get("/inspection/check-all", params=params)
    assert response.status_code == 200
    body = response.json()

    detailed = body["datas_for_last_N_hours"]
    assert "1m" in detailed["frames"]

    minute_frame = detailed["frames"]["1m"]
    assert minute_frame["candles"], "minute candles should be synthesised from higher timeframe"

    expected_end_ms = payload["selection"]["end"]
    expected_start_ms = expected_end_ms - 3_600_000

    minute_candles = minute_frame["candles"]
    assert minute_candles[0]["t"] >= expected_start_ms
    assert minute_candles[-1]["t"] <= expected_end_ms
    assert minute_frame["summary"]["count"] == len(minute_candles) == 60

    delta_cvd = detailed["indicators"]["delta_cvd"]
    assert "1m" in delta_cvd
    assert delta_cvd["1m"], "minute delta/CVD series should be populated"
    assert "3m" not in delta_cvd
    assert "5m" not in delta_cvd

    assert "3m" not in detailed["frames"]
    assert "5m" not in detailed["frames"]
