from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import math

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


def test_check_all_includes_vwap_profiles(client: TestClient) -> None:
    base = datetime(2024, 1, 1, 6, 55, tzinfo=timezone.utc)
    candles = []
    total_minutes = (
        int(
            (
                datetime(2024, 1, 1, 14, 5, tzinfo=timezone.utc) - base
            ).total_seconds()
            // 60
        )
        + 1
    )
    for index in range(total_minutes):
        moment = base + timedelta(minutes=index)
        timestamp_ms = int(moment.timestamp() * 1000)
        price = 100.0 + (index % 5) * 0.25
        volume = 2.0 + (index % 3) * 0.5
        candles.append(
            {
                "t": timestamp_ms,
                "o": price,
                "h": price + 0.2,
                "l": price - 0.2,
                "c": price + 0.1,
                "v": volume,
            }
        )

    payload = {"symbol": "BTCUSDT", "tf": "1m", "candles": candles}
    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 4})
    assert response.status_code == 200
    body = response.json()

    vwap_block = body.get("vwap")
    assert isinstance(vwap_block, dict)
    assert set(vwap_block.get("sessions", {}).keys()) == {"asia", "london", "ny"}

    daily_window = vwap_block["daily"]["window"]
    assert daily_window["start"].startswith("2024-01-01T00:00:00")
    assert daily_window["end"].startswith("2024-01-01T14:05:00")
    assert vwap_block["daily"]["vwap"] > 0

    asia_window = vwap_block["sessions"]["asia"]["window"]
    assert asia_window["start"].startswith("2024-01-01T00:00:00")
    assert asia_window["end"].startswith("2024-01-01T02:59:00")
    assert vwap_block["sessions"]["asia"]["vwap"] == 0.0
    assert vwap_block["sessions"]["asia"].get("session_high") is None
    assert vwap_block["sessions"]["asia"].get("session_low") is None
    assert vwap_block["sessions"]["asia"].get("high") is None
    assert vwap_block["sessions"]["asia"].get("low") is None
    assert vwap_block["sessions"]["asia"].get("open_utc", "").startswith("2024-01-01T00:00:00")
    assert vwap_block["sessions"]["asia"].get("close_utc", "").startswith("2024-01-01T03:00:00")

    london_window = vwap_block["sessions"]["london"]["window"]
    assert london_window["start"].startswith("2024-01-01T07:00:00")
    assert london_window["end"].startswith("2024-01-01T09:59:00")
    assert vwap_block["sessions"]["london"]["poc"] is not None
    assert vwap_block["sessions"]["london"]["vah"] is not None
    assert vwap_block["sessions"]["london"]["val"] is not None
    assert "session_high" in vwap_block["sessions"]["london"]
    assert "session_low" in vwap_block["sessions"]["london"]
    assert vwap_block["sessions"]["london"].get("open_utc", "").startswith("2024-01-01T07:00:00")
    assert vwap_block["sessions"]["london"].get("close_utc", "").startswith("2024-01-01T10:00:00")
    assert vwap_block["sessions"]["london"].get("ib_high") is not None
    assert vwap_block["sessions"]["london"].get("ib_low") is not None

    ny_window = vwap_block["sessions"]["ny"]["window"]
    assert ny_window["start"].startswith("2024-01-01T13:30:00")
    assert ny_window["end"].startswith("2024-01-01T14:05:00")
    assert vwap_block["sessions"]["ny"]["vwap"] > 0
    assert "session_high" in vwap_block["sessions"]["ny"]
    assert "session_low" in vwap_block["sessions"]["ny"]
    assert vwap_block["sessions"]["ny"].get("open_utc", "").startswith("2024-01-01T13:30:00")
    assert vwap_block["sessions"]["ny"].get("close_utc", "").startswith("2024-01-01T16:30:00")
    assert vwap_block["sessions"]["ny"].get("ib_high") is not None
    assert vwap_block["sessions"]["ny"].get("ib_low") is not None

    vwap_tpo_block = body.get("vwap_tpo")
    assert isinstance(vwap_tpo_block, dict)
    assert vwap_tpo_block["daily"]["open_utc"].startswith("2024-01-01T00:00:00")
    assert vwap_tpo_block["daily"]["sd1"]["plus"] >= vwap_tpo_block["daily"]["sd1"]["minus"]
    session_alias = vwap_tpo_block["sessions"]["ny"]
    assert session_alias["open_utc"].startswith("2024-01-01T13:30:00")
    assert session_alias["close_utc"].startswith("2024-01-01T16:30:00")
    assert session_alias["poc"] == vwap_block["sessions"]["ny"]["poc"]
    assert session_alias["ib_high"] == vwap_block["sessions"]["ny"].get("ib_high")
    assert session_alias["sd2"]["plus"] >= session_alias["sd1"]["plus"]

    composite_day = body["tpo"].get("composite_day")
    assert isinstance(composite_day, dict)
    assert composite_day["poc"] == pytest.approx(vwap_block["daily"]["poc"])
    assert composite_day["vah"] == pytest.approx(vwap_block["daily"]["vah"])
    assert composite_day["val"] == pytest.approx(vwap_block["daily"]["val"])

    tpo_sessions = body["tpo"]["sessions"]
    assert isinstance(tpo_sessions, list)
    ny_sessions = [entry for entry in tpo_sessions if entry.get("session") == "ny"]
    assert ny_sessions, "expected NY session entries in TPO payload"
    latest_ny = ny_sessions[-1]
    assert latest_ny["high"] == latest_ny.get("session_high")
    assert latest_ny["low"] == latest_ny.get("session_low")
    assert latest_ny["open_utc"].startswith("2024-01-01T13:30:00")
    assert latest_ny["close_utc"].startswith("2024-01-01T16:30:00")
    assert "ib_high" in latest_ny and "ib_low" in latest_ny


def test_vwap_profile_tick_size_stability(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    base = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    candles = []
    for index in range(180):
        moment = base + timedelta(minutes=index)
        timestamp_ms = int(moment.timestamp() * 1000)
        price = 50.0 + (index % 4) * 0.05
        volume = 3.0 + (index % 2) * 0.2
        candles.append(
            {
                "t": timestamp_ms,
                "o": price,
                "h": price + 0.03,
                "l": price - 0.03,
                "c": price + 0.01,
                "v": volume,
            }
        )

    payload = {"symbol": "ETHUSDT", "tf": "1m", "candles": candles}
    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    original_resolve = check_all_datas.resolve_profile_config
    tick_holder = {"value": 0.5}

    def stub_resolve(symbol, meta):
        config = original_resolve(symbol, meta)
        config["tick_size"] = tick_holder["value"]
        config["adaptive_bins"] = False
        return config

    monkeypatch.setattr(check_all_datas, "resolve_profile_config", stub_resolve)

    def fetch_with_tick(tick_value: float) -> dict:
        tick_holder["value"] = tick_value
        response = client.get(
            "/inspection/check-all",
            params={"snapshot": snapshot_id, "hours": 4},
        )
        assert response.status_code == 200
        return response.json()["vwap"]

    coarse_tick = 0.5
    fine_tick = 0.05
    coarse = fetch_with_tick(coarse_tick)
    fine = fetch_with_tick(fine_tick)

    tolerance = max(coarse_tick, fine_tick)
    assert abs(coarse["daily"]["poc"] - fine["daily"]["poc"]) <= tolerance
    assert abs(coarse["daily"]["vah"] - fine["daily"]["vah"]) <= tolerance
    assert abs(coarse["daily"]["val"] - fine["daily"]["val"]) <= tolerance


def test_check_all_payload_includes_multi_tf_ohlcv_block(client: TestClient) -> None:
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    payload = _build_snapshot_payload(base, count=8 * 60)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get(
        "/inspection/check-all",
        params={"snapshot": snapshot_id, "hours": 4},
    )
    assert response.status_code == 200
    body = response.json()

    ohlcv_block = body.get("ohlcv")
    assert isinstance(ohlcv_block, dict)
    assert set(ohlcv_block.keys()) == {"1m", "3m", "5m", "15m", "1h", "4h", "1d"}

    minute_series = ohlcv_block["1m"].get("candles", [])
    assert minute_series
    total_minutes = len(minute_series)
    minute_map = {candle["t"]: candle for candle in minute_series}

    minute_interval = 60_000
    for tf in ("3m", "5m", "15m", "1h", "4h", "1d"):
        candles = ohlcv_block[tf].get("candles", [])
        interval_ms = check_all_datas.TIMEFRAME_TO_MS[tf]
        expected = total_minutes // max(1, interval_ms // minute_interval)
        assert len(candles) == expected
        for candle in candles:
            assert candle["t"] % interval_ms == 0

    hourly = ohlcv_block["1h"]["candles"]
    if hourly:
        first_hour = hourly[0]
        step_count = check_all_datas.TIMEFRAME_TO_MS["1h"] // minute_interval
        expected_minutes = [first_hour["t"] + index * minute_interval for index in range(step_count)]
        assert all(ts in minute_map for ts in expected_minutes)
        assert pytest.approx(first_hour["o"]) == minute_map[first_hour["t"]]["o"]
        assert pytest.approx(first_hour["c"]) == minute_map[expected_minutes[-1]]["c"]
        assert pytest.approx(first_hour["h"]) == max(minute_map[ts]["h"] for ts in expected_minutes)
        assert pytest.approx(first_hour["l"]) == min(minute_map[ts]["l"] for ts in expected_minutes)
        assert pytest.approx(first_hour["v"]) == sum(minute_map[ts]["v"] for ts in expected_minutes)

    assert ohlcv_block["1d"]["candles"] == []


def test_check_all_payload_includes_orderflow_block(client: TestClient) -> None:
    base = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
    minute = timedelta(minutes=1)
    candles = []
    for offset in range(3):
        moment = base + offset * minute
        ts = int(moment.timestamp() * 1000)
        if offset == 0:
            candle = {"t": ts, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.4, "v": 10.0}
        elif offset == 1:
            candle = {"t": ts, "o": 100.4, "h": 100.6, "l": 99.0, "c": 99.05, "v": 12.0}
        else:
            candle = {"t": ts, "o": 99.05, "h": 101.0, "l": 98.8, "c": 100.95, "v": 15.0}
        candles.append(candle)

    trades = [
        {"t": candles[0]["t"] + 10_000, "q": 2.0, "side": "buy"},
        {"t": candles[0]["t"] + 20_000, "q": 1.5, "side": "buy"},
        {"t": candles[0]["t"] + 30_000, "q": 0.5, "side": "sell"},
        {"t": candles[1]["t"] + 5_000, "q": 4.0, "side": "buy"},
        {"t": candles[1]["t"] + 10_000, "q": 1.0, "side": "sell"},
        {"t": candles[2]["t"] + 15_000, "q": 1.0, "side": "buy"},
        {"t": candles[2]["t"] + 20_000, "q": 6.0, "side": "sell"},
    ]

    payload = {
        "symbol": "BTCUSDT",
        "tf": "1m",
        "candles": candles,
        "agg_trades": {"symbol": "BTCUSDT", "agg": trades},
    }

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 1})
    assert response.status_code == 200
    body = response.json()

    orderflow_block = body.get("orderflow")
    assert isinstance(orderflow_block, dict)
    assert set(orderflow_block.keys()) == {"1m", "3m", "5m", "15m"}

    minute_series = orderflow_block["1m"].get("per_bar")
    assert isinstance(minute_series, list)
    minute_map = {entry["ts"]: entry for entry in minute_series}

    first_bar = minute_map[candles[0]["t"]]
    assert first_bar["delta"] == pytest.approx(3.0)
    assert first_bar["cvd"] == pytest.approx(3.0)
    assert first_bar["ask_vol"] == pytest.approx(3.5)
    assert first_bar["bid_vol"] == pytest.approx(0.5)
    assert first_bar["imbalance_buy"] is True
    assert first_bar["imbalance_sell"] is False
    assert first_bar["absorption_low"] is False
    assert first_bar["absorption_high"] is False
    assert first_bar["large_trades_count"] == 0

    second_bar = minute_map[candles[1]["t"]]
    assert second_bar["delta"] == pytest.approx(3.0)
    assert second_bar["cvd"] == pytest.approx(6.0)
    assert second_bar["ask_vol"] == pytest.approx(4.0)
    assert second_bar["bid_vol"] == pytest.approx(1.0)
    assert second_bar["imbalance_buy"] is True
    assert second_bar["imbalance_sell"] is False
    assert second_bar["absorption_low"] is True
    assert second_bar["absorption_high"] is False
    assert second_bar["large_trades_count"] == 0

    third_bar = minute_map[candles[2]["t"]]
    assert third_bar["delta"] == pytest.approx(-5.0)
    assert third_bar["cvd"] == pytest.approx(1.0)
    assert third_bar["ask_vol"] == pytest.approx(1.0)
    assert third_bar["bid_vol"] == pytest.approx(6.0)
    assert third_bar["imbalance_buy"] is False
    assert third_bar["imbalance_sell"] is True
    assert third_bar["absorption_low"] is False
    assert third_bar["absorption_high"] is True
    assert third_bar["large_trades_count"] == 1

    three_minute = orderflow_block["3m"].get("per_bar")
    assert isinstance(three_minute, list)
    interval_3m = check_all_datas.TIMEFRAME_TO_MS["3m"]
    bucket_ts = (candles[0]["t"] // interval_3m) * interval_3m
    aggregated = {entry["ts"]: entry for entry in three_minute}.get(bucket_ts)
    assert aggregated is not None
    assert aggregated["delta"] == pytest.approx(1.0)
    assert aggregated["cvd"] == pytest.approx(1.0)
    assert aggregated["ask_vol"] == pytest.approx(8.5)
    assert aggregated["bid_vol"] == pytest.approx(7.5)
    assert aggregated["large_trades_count"] == 1
    assert aggregated["imbalance_buy"] is True
    assert aggregated["imbalance_sell"] is True
    assert aggregated["absorption_low"] is True
    assert aggregated["absorption_high"] is True


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
    assert isinstance(body["tpo"].get("composite_day"), dict)
    assert "profile" in body and isinstance(body["profile"], list)
    assert "zones" in body and isinstance(body["zones"], dict)
    assert body["zones"].get("zones") is not None
    liquidity_section = body.get("liquidity")
    assert isinstance(liquidity_section, dict)
    assert {"eqh", "eql", "pdh", "pdl", "sweeps"}.issubset(liquidity_section)
    assert "htf" in body and isinstance(body["htf"], list)
    hourly_entry = next((block for block in body["htf"] if block.get("tf") == "1h"), None)
    assert hourly_entry is not None
    assert isinstance(hourly_entry["candles"], list)
    htf_details = body.get("htf_details")
    assert isinstance(htf_details, dict)
    assert set(htf_details.get("candles", {}).keys()).issuperset({"15m", "1h", "4h", "1d"})
    dq_htf = body.get("data_quality_htf")
    assert isinstance(dq_htf, dict)
    assert dq_htf.get("minute_missing_before") >= 0
    assert dq_htf.get("minute_missing_after") == 0
    preset_payload = body.get("profile_preset")
    assert preset_payload is not None
    assert preset_payload["symbol"] == "BTCUSDT"
    assert preset_payload["builtin"] is True
    assert "profile_preset_required" not in body
    if body["tpo"]["zones"]:
        zone_types = {zone["type"] for zone in body["tpo"]["zones"]}
        assert {"tpo_poc", "tpo_vah", "tpo_val"}.issubset(zone_types)

    dq = body["data_quality"]
    assert dq["window"]["start_ms"] == window_start_ms
    assert dq["window"]["end_ms"] == window_end_ms
    assert dq["minute_missing_after"] == 0
    assert dq["tf_missing_after"] == 0
    assert dq["minute_missing_before"] == total_minutes - len(payload["candles"])
    assert "fetched_1m_count" not in dq
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
    liquidity_section = body.get("liquidity")
    assert isinstance(liquidity_section, dict)
    detailed_start_ms = window_start_ms
    expected_detailed_start = datetime.fromtimestamp(
        detailed_start_ms / 1000, tz=timezone.utc
    ).isoformat()
    assert body["datas_for_last_N_hours"]["range"]["start_utc"].startswith(
        expected_detailed_start
    )
    vwap_sigma_block = body.get("vwap_sigma")
    assert isinstance(vwap_sigma_block, dict)
    daily_sigma = vwap_sigma_block.get("daily")
    assert isinstance(daily_sigma, dict)
    assert daily_sigma.get("basis") == "daily"
    sigma_entries = daily_sigma.get("sigma")
    assert isinstance(sigma_entries, list)
    assert len(sigma_entries) == 2
    first_level, second_level = sigma_entries
    assert first_level.get("k") == 1
    assert second_level.get("k") == 2
    first_spread = first_level["price_plus"] - first_level["price_minus"]
    second_spread = second_level["price_plus"] - second_level["price_minus"]
    assert first_spread >= 0
    assert second_spread >= first_spread
    vwap_center = body["datas_for_last_N_hours"]["frames"]["1m"]["vwap"]
    assert math.isclose(
        vwap_center,
        first_level["price_minus"] + first_spread / 2,
        rel_tol=1e-6,
    )
    session_sigma = vwap_sigma_block.get("sessions")
    assert isinstance(session_sigma, dict)
    assert session_sigma, "expected per-session sigma levels"
    for session_name, entry in session_sigma.items():
        assert entry["basis"] == "session"
        levels = entry.get("sigma")
        assert isinstance(levels, list)
        assert len(levels) == 2
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
