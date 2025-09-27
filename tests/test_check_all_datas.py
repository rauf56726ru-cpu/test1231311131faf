from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.services.check_all_datas as check_all_datas
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


@pytest.fixture(autouse=True)
def stub_binance_minutes(monkeypatch):
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
    window_end_ms = payload["candles"][-1]["t"]
    window_start_ms = window_end_ms - 2 * 3_600_000
    total_minutes = ((window_end_ms - window_start_ms) // 60_000) + 1

    assert body["snapshot_id"] == snapshot_id
    assert body["asof_utc"].startswith(expected_last_iso)
    assert body["latest_candle_utc"].startswith(expected_last.isoformat())
    assert body["latest_candle"]["t"] == int(expected_last.timestamp() * 1000)
    assert body["datas_for_last_N_hours"]["hours"] == 2
    assert (
        body["datas_for_last_N_hours"]["frames"]["1m"]["summary"]["count"]
        == total_minutes
    )
    assert (
        body["datas_for_last_N_hours"]["frames"]["1m"]["candles"][-1]["t"]
        == payload["candles"][-1]["t"]
    )
    detailed_start_ms = window_start_ms
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
    assert "tpo" in body and isinstance(body["tpo"], dict)
    assert isinstance(body["tpo"].get("sessions"), list)
    assert isinstance(body["tpo"].get("zones"), list)
    assert "profile" in body and isinstance(body["profile"], list)
    assert "zones" in body and isinstance(body["zones"], dict)
    assert body["zones"].get("zones") is not None
    assert "htf" in body and isinstance(body["htf"], dict)
    assert set(body["htf"].get("candles", {}).keys()).issuperset({"15m", "1h", "4h", "1d"})
    dq_htf = body.get("data_quality_htf")
    assert isinstance(dq_htf, dict)
    assert dq_htf.get("minute_missing_before") >= 0
    assert dq_htf.get("minute_missing_after") == 0
    preset_payload = body.get("profile_preset")
    assert preset_payload is not None
    assert preset_payload["symbol"] == "BTCUSDT"
    assert preset_payload["builtin"] is True
    assert body.get("profile_preset_required") is False
    if body["tpo"]["zones"]:
        zone_types = {zone["type"] for zone in body["tpo"]["zones"]}
        assert {"tpo_poc", "tpo_vah", "tpo_val"}.issubset(zone_types)

    dq = body["data_quality"]
    assert dq["window"]["start_ms"] == window_start_ms
    assert dq["window"]["end_ms"] == window_end_ms
    assert dq["minute_missing_after"] == 0
    assert dq["tf_missing_after"] == 0
    assert dq["minute_missing_before"] == total_minutes - len(payload["candles"])
    assert dq["fetched_1m_count"] == dq["minute_missing_before"]
    assert dq["tf_missing_before"] == dq["minute_missing_before"]


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


def test_missing_15m_bar_rebuilt_from_minutes(client: TestClient, monkeypatch) -> None:
    base = (
        datetime.now(timezone.utc)
        .replace(minute=0, second=0, microsecond=0)
        - timedelta(hours=3)
    )
    interval = timedelta(minutes=15)
    payload = _build_timeframe_candles(base, count=4, interval=interval, tf="15m")
    missing_ts = payload["candles"][1]["t"]
    del payload["candles"][1]
    payload["selection"]["end"] = payload["candles"][-1]["t"] + int(interval.total_seconds() * 1000)

    def forced_resolve(symbol, meta):
        config = presets.resolve_profile_config(symbol, meta)
        config["target_tf_key"] = "15m"
        return config

    monkeypatch.setattr(check_all_datas, "resolve_profile_config", forced_resolve)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 1})
    assert response.status_code == 200
    body = response.json()

    dq = body["data_quality"]
    assert dq["tf_missing_before"] > 0
    assert dq["tf_missing_after"] == 0
    assert dq["minute_missing_after"] == 0

    candles_15m = body["datas_for_last_N_hours"]["frames"]["15m"]["candles"]
    assert any(candle["t"] == missing_ts for candle in candles_15m)


def test_binance_failure_returns_quality_error(client: TestClient, monkeypatch) -> None:
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    def failing_fetch(*args, **kwargs):
        raise check_all_datas.BinanceDownloadError(0, "rate limited")

    monkeypatch.setattr(check_all_datas, "_download_missing_minutes", failing_fetch)

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 1})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["data_quality"]["downloaded"] == 0
    assert detail["data_quality"]["time_gaps"]


def test_incomplete_minute_fill_triggers_quality_error(client: TestClient, monkeypatch) -> None:
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    def partial_fetch(symbol, start_ms, end_ms, gaps):
        # Return only half of the requested minutes to keep a gap open.
        candles = []
        for gap in gaps:
            cursor = int(gap["from"])
            limit = cursor + (gap["count"] // 2) * 60_000
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

    monkeypatch.setattr(check_all_datas, "_download_missing_minutes", partial_fetch)

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 1})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["data_quality"]["minute_missing_after"] > 0
    assert detail["data_quality"]["downloaded"] >= 0

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
    window_end_ms = payload["candles"][-1]["t"]
    window_start_ms = window_end_ms - 3 * 3_600_000
    total_minutes = ((window_end_ms - window_start_ms) // 60_000) + 1
    assert (
        body["datas_for_last_N_hours"]["frames"]["1m"]["summary"]["count"]
        == total_minutes
    )
    assert body["datas_for_last_N_hours"]["hours"] == 3
    detailed_start_ms = window_start_ms
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
    assert minute_candles[0]["t"] <= expected_start_ms
    assert minute_candles[-1]["t"] <= expected_end_ms
    expected_minutes = ((expected_end_ms - expected_start_ms) // 60_000) + 1
    assert minute_frame["summary"]["count"] == len(minute_candles) == expected_minutes

    delta_cvd = detailed["indicators"]["delta_cvd"]
    assert "1m" in delta_cvd
    assert delta_cvd["1m"], "minute delta/CVD series should be populated"
    assert "3m" not in delta_cvd
    assert "5m" not in delta_cvd

    assert "3m" not in detailed["frames"]
    assert "5m" not in detailed["frames"]
