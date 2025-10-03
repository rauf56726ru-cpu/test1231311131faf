from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.services.check_all_datas as check_all_datas
import src.services.inspection as inspection
from src.services import presets
from src.services.collection_state import reset_state, set_last_collection_time

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


@pytest.fixture
def anyio_backend():
    yield "asyncio"


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
    async def filler(symbol: str, start_ms: int, end_ms: int, gaps):
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

    monkeypatch.setattr(check_all_datas, "_download_missing_minutes_async", filler)

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


@pytest.fixture(autouse=True)
def stub_fetch_ohlcv(monkeypatch):
    import src.services.ohlc as ohlc

    base_epoch = 1_700_000_000_000

    def _aligned_base(interval_ms: int) -> int:
        return base_epoch - (base_epoch % interval_ms)

    def fake_fetch(symbol: str, timeframe: str, hours: int | None = None):
        interval_ms = ohlc.TIMEFRAME_TO_MS.get(timeframe, 60_000)
        start_ts = _aligned_base(interval_ms)
        candles = []
        for index in range(3):
            open_ts = start_ts + index * interval_ms
            candles.append(
                {
                    "t": open_ts,
                    "o": 200.0 + index,
                    "h": 201.0 + index,
                    "l": 199.0 + index,
                    "c": 200.5 + index,
                    "v": 10.0 + index,
                }
            )
        return {"symbol": symbol, "tf": timeframe, "candles": candles, "last_ts": candles[-1]["t"]}

    monkeypatch.setattr(ohlc, "fetch_ohlcv_sync", fake_fetch)
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

    payload = {"symbol": "BTCUSDT", "tf": "1m", "candles": candles}
    if candles:
        payload["selection"] = {"start": candles[0]["t"], "end": candles[-1]["t"]}
    return payload


def test_check_all_returns_structured_payload(client: TestClient) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=60 * 30)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 4})
    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) == {
        "status",
        "meta",
        "data",
        "availability",
        "missing_fields",
        "notes",
    }
    assert body["status"] == "ok"

    meta = body["meta"]
    assert meta["symbol"] == payload["symbol"]
    assert meta["tz"] == "Europe/Berlin"
    assert "last_price" in meta
    assert "last_ts_utc" in meta
    assert "last_tf" in meta
    assert meta["last_price_source"] in {"stream", "ohlcv"}
    assert "stale" in meta
    assert meta["invalid_candles_count"] >= 0
    assert meta["invalid_ts_count"] >= 0
    assert meta["invalid_ohlc_count"] >= 0
    assert meta["invalid_candles_count"] == meta["invalid_ts_count"] + meta["invalid_ohlc_count"]
    assert isinstance(meta["invalid_candle_stages"], dict)
    assert meta.get("sanitized") is True
    assert isinstance(meta.get("sessions_empty"), bool)
    for stage_counts in meta["invalid_candle_stages"].values():
        assert isinstance(stage_counts, dict)

    data_block = body["data"]

    ohlcv = data_block["ohlcv"]
    assert set(ohlcv.keys()) == {"1m", "3m", "5m", "15m", "1h", "4h", "1d"}
    for series in ohlcv.values():
        assert isinstance(series["candles"], list)

    orderflow = data_block["orderflow"]
    assert set(orderflow.keys()) == {"15m", "1h"}
    for block in orderflow.values():
        assert isinstance(block["per_bar"], list)

    assert isinstance(body["notes"], list)
    assert not body["notes"]

    vwap_tpo = data_block["vwap_tpo"]
    assert vwap_tpo["daily"]["open_utc"].startswith("2024-01-02T00:00:00")
    assert set(vwap_tpo["sessions"].keys()) == {"asia", "london", "ny"}
    for session_payload in vwap_tpo["sessions"].values():
        assert set(session_payload.keys()) == {
            "open_utc",
            "close_utc",
            "vwap",
            "sd1",
            "sd2",
            "poc",
            "vah",
            "val",
            "ib_high",
            "ib_low",
            "high",
            "low",
        }

    composite_day = data_block["tpo"]["composite_day"]
    assert set(composite_day.keys()) == {"poc", "vah", "val"}

    prev_day = data_block["prev_day"]
    assert set(prev_day.keys()) == {"pdh", "pdl", "close", "poc", "vah", "val"}
    prev_minutes = payload["candles"][: 60 * 24]
    expected_high = max(candle["h"] for candle in prev_minutes)
    expected_low = min(candle["l"] for candle in prev_minutes)
    expected_close = prev_minutes[-1]["c"]
    assert prev_day["pdh"] == pytest.approx(expected_high)
    assert prev_day["pdl"] == pytest.approx(expected_low)
    assert prev_day["close"] == pytest.approx(expected_close)

    zones = data_block["zones"]
    assert set(zones.keys()) == {"fvg", "fvl", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels"}
    for zone_series in zones.values():
        assert isinstance(zone_series, list)

    liquidity = data_block["liquidity"]
    assert set(liquidity.keys()) == {"eqh", "eql"}
    for levels in liquidity.values():
        assert isinstance(levels, list)

    assert data_block["risk_prefs"] == {
        "rr_min": pytest.approx(2.5),
        "risk_per_trade_pct": pytest.approx(1.0),
    }
    assert data_block["context"] == {
        "globalBias": "neutral",
        "narrative": "",
        "openOppositeZones": False,
    }

    availability = body["availability"]
    assert set(availability.keys()) == {"ohlcv", "vwap_sessions", "zones", "orderflow"}
    assert isinstance(body["missing_fields"], list)


@pytest.mark.anyio("asyncio")
async def test_check_all_reports_invalid_timestamps() -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=120)
    candles = payload["candles"]
    bad_index = len(candles) // 2
    candles[bad_index]["t"] = -1

    snapshot = {
        "id": "invalid-ts",  # stable identifier for caching paths
        "symbol": payload["symbol"],
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": candles}},
        "selection": {"start": candles[0]["t"], "end": candles[-1]["t"]},
    }

    result = await check_all_datas.build_check_all_datas(
        snapshot,
        now_utc=base + timedelta(hours=4),
        hours=4,
    )

    assert result is not None
    assert result["status"] == "ok"
    meta = result["meta"]
    assert meta["invalid_ts_count"] >= 1
    assert meta["invalid_candles_count"] >= meta["invalid_ts_count"]
    assert meta["invalid_candle_stages"]
    assert result["notes"], "expected notes for invalid timestamps"
    assert any("invalid timestamps" in note for note in result["notes"])


def test_topup_limits_window_to_last_collection(client: TestClient) -> None:
    reset_state()
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=60 * 6)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    last_collection_dt = base + timedelta(hours=5, minutes=55)
    set_last_collection_time(last_collection_dt)

    try:
        response = client.get(
            "/inspection/check-all",
            params={"snapshot": snapshot_id, "mode": "topup"},
        )
        assert response.status_code == 200
        body = response.json()
        zones_block = body["data"]["zones"]

        for zone_key in ("fvg", "ob", "mb", "bb", "rb", "pb", "sr"):
            zone_entries = zones_block.get(zone_key, [])
            assert zone_entries, f"expected informational entry for {zone_key}"
            message_entry = zone_entries[0]
            assert "message" in message_entry
            assert "выбранный период" in message_entry["message"].lower()
    finally:
        reset_state()


def test_multi_timeframe_ohlcv_alignment(client: TestClient) -> None:
    base = datetime(2024, 2, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=8 * 60)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 4})
    assert response.status_code == 200
    body = response.json()

    ohlcv_block = body["data"]["ohlcv"]
    assert ohlcv_block["1m"]["candles"]

    minute_map = {candle["t"]: candle for candle in ohlcv_block["1m"]["candles"]}
    first_hour = ohlcv_block["1h"]["candles"][0]
    interval = check_all_datas.TIMEFRAME_TO_MS["1m"]
    hour_interval = check_all_datas.TIMEFRAME_TO_MS["1h"]
    step_count = hour_interval // interval
    expected_minutes = [first_hour["t"] + index * interval for index in range(step_count)]
    assert all(ts in minute_map for ts in expected_minutes)
    assert pytest.approx(first_hour["o"]) == minute_map[first_hour["t"]]["o"]
    assert pytest.approx(first_hour["c"]) == minute_map[expected_minutes[-1]]["c"]
    assert pytest.approx(first_hour["h"]) == max(minute_map[ts]["h"] for ts in expected_minutes)
    assert pytest.approx(first_hour["l"]) == min(minute_map[ts]["l"] for ts in expected_minutes)
    assert pytest.approx(first_hour["v"]) == sum(minute_map[ts]["v"] for ts in expected_minutes)

    assert ohlcv_block["1d"]["candles"]


def test_orderflow_block_matches_spec(client: TestClient) -> None:
    base = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
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

    orderflow_block = body["data"]["orderflow"]
    assert set(orderflow_block.keys()) == {"15m", "1h"}

    fifteen_series = orderflow_block["15m"]["per_bar"]
    assert isinstance(fifteen_series, list)
    assert fifteen_series
    fifteen_entry = fifteen_series[0]
    for key in ("delta", "cvd", "ask_vol", "bid_vol"):
        assert isinstance(fifteen_entry[key], (int, float))
    assert isinstance(fifteen_entry["large_trades_count"], int)
    assert isinstance(fifteen_entry["imbalance_buy"], bool)
    assert isinstance(fifteen_entry["imbalance_sell"], bool)
    assert isinstance(fifteen_entry["absorption_low"], bool)
    assert isinstance(fifteen_entry["absorption_high"], bool)

    hourly_series = orderflow_block["1h"]["per_bar"]
    assert isinstance(hourly_series, list)
    if hourly_series:
        hourly_entry = hourly_series[0]
        for key in ("delta", "cvd", "ask_vol", "bid_vol"):
            assert isinstance(hourly_entry[key], (int, float))
        assert isinstance(hourly_entry["large_trades_count"], int)


def test_vwap_tpo_sessions_include_aliases(client: TestClient) -> None:
    base = datetime(2024, 4, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=12 * 60)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get(
        "/inspection/check-all",
        params={"snapshot": snapshot_id, "hours": 8},
    )
    assert response.status_code == 200
    body = response.json()

    sessions = body["data"]["vwap_tpo"]["sessions"]
    ny_session = sessions["ny"]
    assert ny_session["open_utc"].endswith("13:30:00Z")
    assert ny_session["close_utc"].endswith("16:30:00Z")
    assert ny_session["sd2"]["plus"] >= ny_session["sd1"]["plus"]
    assert "poc" in ny_session
    assert "ib_high" in ny_session
    assert "ib_low" in ny_session

    composite_day = body["data"]["tpo"]["composite_day"]
    assert composite_day["poc"] is not None
    assert composite_day["vah"] is not None
    assert composite_day["val"] is not None


def test_session_detailed_mode_returns_placeholder(client: TestClient) -> None:
    base = datetime(2024, 5, 1, 7, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=120)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get(
        "/inspection/check-all",
        params={"snapshot": snapshot_id, "mode": "session_detailed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "session_detailed.v1"
    assert body["status"] == "insufficient_data"
    assert body["meta"]["symbol"] == payload["symbol"]
    assert body["meta"]["tz"] == "Europe/Berlin"
    assert "session" in body
    assert body["session"]["coverage_pct"] == 0.0
    assert "ohlcv.coverage" in body["missing_fields"]


@pytest.mark.anyio("asyncio")
async def test_async_builder_timeout_returns_insufficient(monkeypatch):
    base = datetime(2024, 5, 1, 0, 0, tzinfo=UTC)
    snapshot = _build_snapshot_payload(base, count=5)

    async def slow_builder(snapshot, **kwargs):
        await asyncio.sleep(0.2)
        return {
            "status": "ok",
            "meta": {"symbol": snapshot.get("symbol", "UNKNOWN")},
            "data": {},
            "availability": {},
            "missing_fields": [],
        }

    monkeypatch.setattr(check_all_datas, "build_check_all_datas", slow_builder)

    result = await check_all_datas.build_check_all_datas_async(snapshot, timeout=0.05)

    assert result is not None
    assert result["status"] == "insufficient_data"
    assert result["meta"]["insufficient_reason"] == "stale_or_unseeded_buffers"

def test_build_inspection_error_payload_sets_reason() -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot = {
        "symbol": "BTCUSDT",
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": []}},
    }
    payload = check_all_datas.build_inspection_error_payload(
        snapshot,
        now_utc=now,
        missing_fields=["ohlcv.1m"],
        reason="invalid_timestamps",
    )
    assert payload["status"] == "insufficient_data"
    assert "ohlcv.1m" in payload["missing_fields"]
    meta = payload["meta"]
    assert meta["insufficient_reason"] == "invalid_timestamps"
    assert meta.get("sanitized") is True

def test_check_all_returns_insufficient_on_internal_error(client: TestClient, monkeypatch) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=60)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    async def boom(*args, **kwargs):  # pragma: no cover - monkeypatch helper
        raise RuntimeError("boom")

    monkeypatch.setattr("src.api.app.build_check_all_datas_async", boom)

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["meta"].get("insufficient_reason") == "invalid_timestamps"
