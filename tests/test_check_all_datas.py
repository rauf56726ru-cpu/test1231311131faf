from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import time

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.api.app as app_module
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
    async def filler(
        symbol: str,
        start_ms: int,
        end_ms: int,
        gaps,
        *,
        budget=None,
        api_diag=None,
        **_,
    ):
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
        if api_diag is not None:
            api_diag["downloaded_bars"] = int(api_diag.get("downloaded_bars", 0)) + len(candles)
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

    async def fake_fetch_async(
        symbol: str,
        timeframe: str,
        *,
        hours: int | None = None,
        fetcher=None,
    ) -> dict:
        return fake_fetch(symbol, timeframe, hours)

    monkeypatch.setattr(ohlc, "fetch_ohlcv_sync", fake_fetch)
    monkeypatch.setattr(ohlc, "fetch_ohlcv", fake_fetch_async)
    monkeypatch.setattr(check_all_datas, "fetch_ohlcv", fake_fetch_async)

    async def fake_klines(
        symbol: str,
        timeframe: str,
        start_ms: int | None,
        end_ms: int | None,
        limit: int | None,
    ):
        interval_ms = ohlc.TIMEFRAME_TO_MS.get(timeframe, 60_000)
        if start_ms is None:
            cursor = _aligned_base(interval_ms)
        else:
            cursor = start_ms - (start_ms % interval_ms)
        end_cursor = end_ms if end_ms is not None else cursor + interval_ms * 3
        max_rows = limit or 500
        rows: list[list[float]] = []
        while cursor < end_cursor and len(rows) < max_rows:
            open_price = 200.0 + (cursor - base_epoch) / max(interval_ms, 1) * 0.01
            high_price = open_price + 1.0
            low_price = open_price - 1.0
            close_price = open_price + 0.5
            rows.append([cursor, open_price, high_price, low_price, close_price, 1.0])
            cursor += interval_ms
        return rows

    monkeypatch.setattr(check_all_datas, "_fetch_binance_klines", fake_klines)

    async def fake_fetch_orderbook(symbol: str, window_minutes: int, *, trace=None):
        from datetime import datetime, timezone
        return {
            "symbol": symbol.upper(),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "window_minutes": window_minutes,
            "top_levels": [
                {"side": "bid", "p": 100.0, "sz": 5.0},
                {"side": "ask", "p": 100.5, "sz": 4.0},
            ],
            "imbalance": 1.0,
            "spoofing_flags": [],
        }

    monkeypatch.setattr(app_module, "fetch_orderbook", fake_fetch_orderbook)
    monkeypatch.setattr(check_all_datas, "fetch_ohlcv_sync", fake_fetch)
    monkeypatch.setattr(app_module, "fetch_ohlcv_enhanced", fake_fetch_async)
    yield


@pytest.fixture(autouse=True)
def stub_vision_ingest(monkeypatch):
    async def fake_ingest(*args, **kwargs):
        return {"status": "cached"}

    monkeypatch.setattr(check_all_datas, "_maybe_ingest_vision_data", fake_ingest)


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
        payload["book"] = {"top_levels": [
            {"bid": candles[-1]["c"], "ask": candles[-1]["c"] + 0.5, "price": candles[-1]["c"], "ts": candles[-1]["t"]}
        ]}
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
        "timing",
    }
    timing_block = body["timing"]
    assert {
        "fetch_ms",
        "db_ms",
        "compute_ms",
        "serialize_ms",
        "size_bytes",
    }.issubset(timing_block.keys())
    assert body["status"] == "ready"

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
    assert meta.get("pipeline_preset") == "summary_72h"
    stage_timing = meta.get("stage_timing")
    assert isinstance(stage_timing, Mapping)
    expected_stages = {"gaps", "rollups", "delta", "vwap_tpo", "zones", "vitrines"}
    assert expected_stages.issubset(stage_timing.keys())
    for stage_name in expected_stages:
        stage_entry = stage_timing[stage_name]
        assert isinstance(stage_entry, Mapping)
        assert stage_entry.get("status") in {"complete", "error"}
    for stage_counts in meta["invalid_candle_stages"].values():
        assert isinstance(stage_counts, dict)
    readiness = meta.get("readiness", {})
    assert readiness.get("ready") is True
    assert readiness.get("missing") == []
    readiness_requirements = readiness.get("requirements", {})
    assert isinstance(readiness_requirements, Mapping)

    data_block = body["data"]
    assert data_block["symbol"] == payload["symbol"]
    timeframes = data_block["timeframes"]
    expected_tfs = {"1m", "5m", "15m", "1h", "4h", "1d"}
    assert set(timeframes.keys()) == expected_tfs
    showcases = data_block["showcases"]
    assert set(showcases.keys()) == {"A", "B"}
    showcase_a = showcases["A"]
    showcase_b = showcases["B"]
    for showcase in (showcase_a, showcase_b):
        assert {"items", "completeness", "sources"}.issubset(showcase.keys())
        assert isinstance(showcase["items"], list)
        assert isinstance(showcase["completeness"], Mapping)
        assert isinstance(showcase["sources"], Mapping)
    completeness_a = showcase_a["completeness"]
    assert {"window_hours", "counts", "open_entries", "missing_modules", "valid", "validation_errors"}.issubset(
        completeness_a.keys()
    )
    assert isinstance(completeness_a["missing_modules"], list)
    assert isinstance(completeness_a["validation_errors"], list)
    completeness_b = showcase_b["completeness"]
    assert {"sessions", "missing_modules", "valid", "validation_errors"}.issubset(completeness_b.keys())
    assert isinstance(completeness_b["missing_modules"], list)
    assert isinstance(completeness_b["validation_errors"], list)
    for tf, summary in timeframes.items():
        assert set(summary.keys()) == {"zones", "sweeps", "atr", "vwap_sessions"}
        zones_summary = summary["zones"]
        assert set(zones_summary.keys()) == {"eqh", "eql", "fvg", "ob"}
        assert isinstance(summary["sweeps"], int)
        vwap_flags = summary["vwap_sessions"]
        assert set(vwap_flags.keys()) == {"asia", "london", "ny"}
        for flag in vwap_flags.values():
            assert isinstance(flag, bool)
        atr_value = summary["atr"]
        if atr_value is not None:
            assert isinstance(atr_value, float)

    availability = body["availability"]
    assert set(availability.keys()) == {"timeframes"}
    for tf, tf_block in availability["timeframes"].items():
        assert set(tf_block.keys()) == {"zones", "sweeps", "atr", "vwap_sessions"}
        assert set(tf_block["zones"].keys()) == {"eqh", "eql", "fvg", "ob"}

    assert isinstance(body["notes"], list)

    diagnostics = meta.get("diagnostics", {})
    assert {"liquidity", "zones", "vwap_tpo", "orderflow", "coverage", "api", "sessions", "readiness"}.issubset(
        diagnostics.keys()
    )
    showcases_diag = diagnostics.get("showcases", {})
    assert isinstance(showcases_diag, Mapping)
    assert set(showcases_diag.keys()) == {"A", "B"}
    api_diag = diagnostics.get("api", {})
    assert {"requests", "retries", "rate_limit_hits", "backoffs"}.issubset(api_diag.keys())
    coverage_diag = diagnostics.get("coverage", {})
    assert "1m" in coverage_diag
    vwap_tpo = diagnostics.get("vwap_tpo", {})
    sessions_diag = vwap_tpo.get("sessions", {}) if isinstance(vwap_tpo, Mapping) else {}
    assert set(sessions_diag.keys()) == {"asia", "london", "ny"}

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
    assert result["status"] == "ready"
    meta = result["meta"]
    assert meta["invalid_ts_count"] >= 1
    assert meta["invalid_candles_count"] >= meta["invalid_ts_count"]
    assert meta["invalid_candle_stages"]
    assert result["notes"], "expected notes for invalid timestamps"
    assert any("invalid timestamps" in note for note in result["notes"])


@pytest.mark.anyio("asyncio")
async def test_readiness_requires_ohlcv(monkeypatch) -> None:
    base = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=120)
    snapshot = {
        "id": "readiness-ohlcv",
        "symbol": payload["symbol"],
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": payload["candles"]}},
        "selection": {"start": payload["candles"][0]["t"], "end": payload["candles"][-1]["t"]},
    }

    original_build_multi = check_all_datas.build_multi_tf_ohlcv

    async def fake_build_multi_tf(*args, **kwargs):
        result = await original_build_multi(*args, **kwargs)
        timeframes = kwargs.get("timeframes")
        if not isinstance(timeframes, Sequence):
            if len(args) >= 3:
                timeframes = args[2]
        if not isinstance(timeframes, Sequence):
            timeframes = ()
        for tf in ("15m", "1h"):
            frame = result.get(tf)
            if isinstance(frame, Mapping):
                frame["candles"] = []
            else:
                result[tf] = {"candles": []}
        return result

    monkeypatch.setattr(check_all_datas, "build_multi_tf_ohlcv", fake_build_multi_tf)

    result = await check_all_datas.build_check_all_datas(
        snapshot,
        now_utc=base + timedelta(hours=4),
        hours=4,
    )

    assert result is not None
    assert result["status"] == "insufficient_data"
    assert "readiness.ohlcv" in result["missing_fields"]
    readiness = result["meta"].get("readiness", {})
    assert readiness.get("ready") is False
    assert "readiness.ohlcv" in readiness.get("missing", [])


@pytest.mark.anyio("asyncio")
async def test_readiness_requires_orderflow(monkeypatch) -> None:
    base = datetime(2024, 1, 3, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=120)
    snapshot = {
        "id": "readiness-orderflow",
        "symbol": payload["symbol"],
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": payload["candles"]}},
        "selection": {"start": payload["candles"][0]["t"], "end": payload["candles"][-1]["t"]},
    }

    original_orderflow = check_all_datas._build_orderflow_block

    async def fake_orderflow_block(*args, **kwargs):
        block, diag = await original_orderflow(*args, **kwargs)
        for tf in ("15m", "1h"):
            frame = block.get(tf)
            if isinstance(frame, Mapping):
                frame["per_bar"] = []
            else:
                block[tf] = {"per_bar": [], "summary": {}}
        return block, diag

    monkeypatch.setattr(check_all_datas, "_build_orderflow_block", fake_orderflow_block)

    result = await check_all_datas.build_check_all_datas(
        snapshot,
        now_utc=base + timedelta(hours=4),
        hours=4,
    )

    assert result is not None
    assert result["status"] == "insufficient_data"
    assert "readiness.orderflow" in result["missing_fields"]
    readiness = result["meta"].get("readiness", {})
    assert readiness.get("ready") is False
    assert "readiness.orderflow" in readiness.get("missing", [])


@pytest.mark.anyio("asyncio")
async def test_readiness_requires_session_completeness(monkeypatch) -> None:
    base = datetime(2024, 1, 4, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=120)
    snapshot = {
        "id": "readiness-sessions",
        "symbol": payload["symbol"],
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": payload["candles"]}},
        "selection": {"start": payload["candles"][0]["t"], "end": payload["candles"][-1]["t"]},
    }

    original_compute = check_all_datas._compute_session_completeness

    def fake_compute_session_completeness(*args, **kwargs):
        result = original_compute(*args, **kwargs)
        overriden = dict(result)
        overriden["status"] = "empty"
        return overriden

    monkeypatch.setattr(check_all_datas, "_compute_session_completeness", fake_compute_session_completeness)

    result = await check_all_datas.build_check_all_datas(
        snapshot,
        now_utc=base + timedelta(hours=4),
        hours=4,
    )

    assert result is not None
    assert result["status"] == "insufficient_data"
    assert "readiness.sessions" in result["missing_fields"]
    readiness = result["meta"].get("readiness", {})
    assert readiness.get("ready") is False
    assert "readiness.sessions" in readiness.get("missing", [])


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
        zones_diag_block = body["meta"]["diagnostics"].get("zones")
        assert isinstance(zones_diag_block, dict)
        gating_diag = zones_diag_block.get("gating")
        if gating_diag is not None:
            assert isinstance(gating_diag, dict)
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

    diagnostics = body["meta"].get("diagnostics", {})
    coverage = diagnostics.get("coverage", {})
    assert {"1m", "5m", "15m", "1h", "4h", "1d"}.issubset(coverage.keys())
    hour_state = coverage["1h"]
    assert hour_state["interval_ms"] == check_all_datas.TIMEFRAME_TO_MS["1h"]
    assert hour_state["gap_count"] == 0
    assert hour_state["present_bars"] == hour_state["expected_bars"]
    daily_state = coverage["1d"]
    assert daily_state["interval_ms"] == check_all_datas.TIMEFRAME_TO_MS["1d"]
    daily_diag = diagnostics.get("daily", {})
    assert daily_diag.get("available") is True


def test_timeframe_series_do_not_embed_minute_data(client: TestClient) -> None:
    base = datetime(2024, 5, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=6 * 60)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get("/inspection/check-all", params={"snapshot": snapshot_id, "hours": 4})
    assert response.status_code == 200
    body = response.json()

    diagnostics = body["meta"].get("diagnostics", {})
    coverage = diagnostics.get("coverage", {})
    assert set(coverage.keys()) >= {"3m", "5m", "15m", "1h"}
    for tf in ("3m", "5m", "15m", "1h"):
        state = coverage[tf]
        assert state["interval_ms"] == check_all_datas.TIMEFRAME_TO_MS[tf]
        assert state["expected_bars"] >= state["present_bars"]
    trace_entries = diagnostics.get("coverage_trace", [])
    counts = {tf: 0 for tf in ("3m", "5m", "15m", "1h")}
    for entry in trace_entries:
        tf = entry.get("tf")
        if tf in counts:
            counts[tf] += 1
    for tf in counts:
        assert counts[tf] >= 1, f"expected at least one coverage trace entry for {tf}"


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

    orderflow_block = body["meta"]["diagnostics"]["orderflow"]
    expected_tfs = {"1m", "3m", "5m", "15m", "1h"}
    assert expected_tfs.issubset(orderflow_block.keys())
    diag_block = orderflow_block.get("diag")
    assert isinstance(diag_block, dict)

    minute_block = orderflow_block["1m"]
    minute_series = minute_block["per_bar"]
    assert isinstance(minute_series, list)
    assert minute_series
    assert len(minute_series) == check_all_datas.ORDERFLOW_REQUIRED_HOURS * 60
    minute_entry = minute_series[-1]
    assert "delta" in minute_entry and "cvd" in minute_entry
    summary_1m = minute_block.get("summary", {})
    assert summary_1m.get("minutes") == check_all_datas.ORDERFLOW_REQUIRED_HOURS * 60
    assert summary_1m.get("minutes_with_trades") >= 0

    for tf in ("3m", "5m", "15m", "1h"):
        block = orderflow_block[tf]
        series = block["per_bar"]
        assert isinstance(series, list)
        if series:
            entry = series[0]
            for key in ("delta_sum", "cvd_close", "vol_sum"):
                assert isinstance(entry[key], (int, float))
            if "bars" in entry:
                assert isinstance(entry["bars"], int)
        summary = block.get("summary", {})
        assert "delta_sum" in summary and "volume_sum" in summary
        if tf in {"15m", "1h"}:
            assert series, f"Expected aggregated series for {tf} to be non-empty"

    orderflow_data = body["data"]["orderflow"]
    agg_trades_block = orderflow_data.get("agg_trades")
    assert isinstance(agg_trades_block, dict)
    assert agg_trades_block.get("symbol") == "BTCUSDT"
    window_block = agg_trades_block.get("range")
    assert isinstance(window_block, dict)
    assert window_block["end_ms"] >= window_block["start_ms"]
    summary = agg_trades_block.get("summary", {})
    assert isinstance(summary.get("count"), int)
    assert isinstance(summary.get("buy"), int)
    assert isinstance(summary.get("sell"), int)
    assert isinstance(summary.get("volume"), (int, float))
    counts_block = agg_trades_block.get("counts")
    assert isinstance(counts_block, dict)
    assert counts_block.get("resolved") == summary.get("count")
    assert counts_block.get("snapshot") >= 0
    assert counts_block.get("downloaded") >= 0
    exported_trades = agg_trades_block.get("trades")
    assert isinstance(exported_trades, list)
    if summary.get("count"):
        assert exported_trades
    else:
        assert exported_trades == []


@pytest.mark.anyio
async def test_download_agg_trades_paginates_full_window(monkeypatch) -> None:
    class _DummyStore:
        def fetch_agg_trades(self, *_, **__):
            return []

    monkeypatch.setattr(check_all_datas, "get_store", lambda: _DummyStore())

    base_ts = 1_700_000_000_000
    step_ms = 60_000
    total = check_all_datas._AGG_TRADES_LIMIT * 2 + 500
    trades = [
        {
            "t": base_ts + index * step_ms,
            "p": 100.0 + index * 0.01,
            "q": 1.0 + (index % 5) * 0.25,
            "m": bool(index % 2),
        }
        for index in range(total)
    ]

    monkeypatch.setattr(check_all_datas, "_fetch_binance_agg_trades", None)

    class _DummyResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return self._payload

    async def fake_http_request(method, url, *, params=None, **_):
        assert method == "GET"
        assert params is not None
        start_ms = int(params["startTime"])
        end_ms = int(params["endTime"])
        limit = int(params["limit"])
        window = [row for row in trades if start_ms <= row["t"] < end_ms]
        payload = [dict(row) for row in window[:limit]]
        return _DummyResponse(payload)

    monkeypatch.setattr(check_all_datas, "http_request", fake_http_request)

    records, diag = await check_all_datas._download_agg_trades_async(
        "BTCUSDT",
        base_ts,
        trades[-1]["t"] + step_ms,
        allow_network=True,
        trace_ctx=None,
        budget=check_all_datas._TimeBudget(None),
        page_span_ms=12 * 60 * 60 * 1000,
    )

    assert len(records) == total
    assert records[0]["t"] == trades[0]["t"]
    assert records[-1]["t"] == trades[-1]["t"]
    assert diag["downloaded"] == total
    assert diag["batches"] >= 3
    assert diag["status"] == 200


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

    sessions = body["meta"]["diagnostics"]["vwap_tpo"]["sessions"]
    ny_session = sessions["ny"]
    assert ny_session["open_utc"].endswith("11:30:00Z")
    assert ny_session["close_utc"].endswith("14:30:00Z")
    assert ny_session["sd2"]["plus"] >= ny_session["sd1"]["plus"]
    assert "poc" in ny_session
    assert "ib_high" in ny_session
    assert "ib_low" in ny_session
    completeness_block = ny_session.get("completeness", {})
    assert completeness_block
    assert isinstance(completeness_block, dict)
    assert completeness_block.get("status") in {"complete", "partial", "empty", "na"}
    assert completeness_block.get("tf") in {"1m", "3m"}

    daily_block = body["meta"]["diagnostics"]["vwap_tpo"]["daily"]
    assert daily_block["vwap"] is not None


def test_recent_zone_focus_filters_window(client: TestClient, monkeypatch) -> None:
    base = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    payload = _build_snapshot_payload(base, count=12 * 60)

    recent_iso = (base + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    stale_iso = (base - timedelta(hours=80)).isoformat().replace("+00:00", "Z")

    def fake_detect_zones(*args, **kwargs):
        return {
            "zones": {
                "fvg": [
                    {
                        "tf": "15m",
                        "direction": "up",
                        "top": 105.0,
                        "bot": 103.0,
                        "mid": 104.0,
                        "created_utc": recent_iso,
                        "status": "open",
                    },
                    {
                        "tf": "1h",
                        "direction": "down",
                        "top": 140.0,
                        "bot": 138.0,
                        "mid": 139.0,
                        "created_utc": stale_iso,
                        "status": "tapped",
                    },
                ],
                "fvl": [],
                "ob": [
                    {
                        "tf": "1h",
                        "type": "supply",
                        "open": 120.0,
                        "close": 121.2,
                        "mean": 120.6,
                        "origin_utc": recent_iso,
                        "status": "fresh",
                        "fill_ratio": 0.7,
                    },
                    {
                        "tf": "15m",
                        "type": "demand",
                        "open": 90.0,
                        "close": 91.0,
                        "mean": 90.5,
                        "origin_utc": stale_iso,
                        "status": "tapped",
                    },
                ],
                "mb": [],
                "bb": [],
                "rb": [],
                "pb": [],
                "sr": [],
                "profile_levels": [],
            },
            "meta": {},
        }

    monkeypatch.setattr(check_all_datas, "detect_zones", fake_detect_zones)

    captured_configs: list[Mapping[str, Any]] = []
    sweep_ts = int((base + timedelta(hours=2)).timestamp() * 1000)

    def fake_build_liquidity_snapshot(
        frames,
        *,
        symbol,
        tick_size,
        tick_source_hint=None,
        meta=None,
        selection=None,
        config=None,
    ):
        captured_configs.append(config or {})
        sweeps = [
            {
                "type": "sweep_top",
                "tf": "15m",
                "level_type": "eqh",
                "level_price": 104.0,
                "t": sweep_ts,
                "retest_t": sweep_ts,
                "atr_tolerance": 0.1,
                "min_move": 0.2,
                "level_source": "eq",
            },
            {
                "type": "sweep_top",
                "tf": "1h",
                "level_type": "eqh",
                "level_price": 104.0,
                "t": sweep_ts,
                "retest_t": sweep_ts,
                "atr_tolerance": 0.1,
                "min_move": 0.2,
                "level_source": "eq",
            },
        ]
        diagnostics = {
            "config": dict(config or {}),
            "metrics": {},
            "summary": {},
            "tick_size": {"value": tick_size},
        }
        return {
            "eqh": [],
            "eql": [],
            "pdh": [],
            "pdl": [],
            "sweeps": sweeps,
            "candidates": {"eqh": [], "eql": []},
            "diagnostics": diagnostics,
        }

    monkeypatch.setattr(check_all_datas, "build_liquidity_snapshot", fake_build_liquidity_snapshot)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    response = client.get(
        "/inspection/check-all",
        params={"snapshot": snapshot_id, "hours": 4},
    )
    assert response.status_code == 200
    body = response.json()

    zones_recent = body["data"]["zones"]["recent"]
    assert zones_recent["window_hours"] == 72
    assert zones_recent["meta"]["has_recent"] is True

    fvg_recent = zones_recent["fvg"]
    assert len(fvg_recent) == 1
    assert fvg_recent[0]["status"] == "open"
    assert fvg_recent[0]["mid"] == pytest.approx(104.0)
    assert "sweeps" in fvg_recent[0]
    assert fvg_recent[0]["sweeps"], "Expected sweep linkage on recent zone"

    ob_recent = zones_recent["ob"]
    assert len(ob_recent) == 1
    assert ob_recent[0]["status"] == "mitigated"
    assert ob_recent[0]["type"] == "supply"

    reasons_fvg = zones_recent["meta"]["reasons"]["fvg"]
    reasons_ob = zones_recent["meta"]["reasons"]["ob"]

    sweeps_public = body["data"]["liquidity"].get("sweeps", [])
    assert len(sweeps_public) == 1
    sweep_entry = sweeps_public[0]
    assert sweep_entry.get("zone_links"), "Expected sweep to include linked zones"

    assert captured_configs, "Expected liquidity config to be captured"
    dynamic_config = captured_configs[0]
    assert isinstance(dynamic_config, Mapping)


def test_session_window_converts_berlin_to_utc() -> None:
    ny_start, ny_end = next(
        (start, end)
        for name, start, end in check_all_datas.VWAP_TPO_SESSIONS
        if name == "ny"
    )
    anchor_pre = datetime(2024, 3, 29, 20, 0, tzinfo=UTC)
    pre_start, pre_end, pre_close = check_all_datas._session_window(
        int(anchor_pre.timestamp() * 1000), ny_start, ny_end
    )
    expected_pre_start = datetime(2024, 3, 29, 12, 30, tzinfo=UTC)
    expected_pre_close = datetime(2024, 3, 29, 15, 30, tzinfo=UTC)
    assert pre_start == int(expected_pre_start.timestamp() * 1000)
    assert pre_close == int(expected_pre_close.timestamp() * 1000)
    assert pre_end == pre_close - check_all_datas.MINUTE_INTERVAL_MS

    anchor_post = datetime(2024, 4, 2, 20, 0, tzinfo=UTC)
    post_start, post_end, post_close = check_all_datas._session_window(
        int(anchor_post.timestamp() * 1000), ny_start, ny_end
    )
    expected_post_start = datetime(2024, 4, 2, 11, 30, tzinfo=UTC)
    expected_post_close = datetime(2024, 4, 2, 14, 30, tzinfo=UTC)
    assert post_start == int(expected_post_start.timestamp() * 1000)
    assert post_close == int(expected_post_close.timestamp() * 1000)
    assert post_end == post_close - check_all_datas.MINUTE_INTERVAL_MS
    assert post_close - post_start == 3 * 60 * 60 * 1000


def test_session_completeness_partial_detection() -> None:
    start_dt = datetime(2024, 4, 1, 11, 30, tzinfo=UTC)
    start_ms = int(start_dt.timestamp() * 1000)
    close_ms = start_ms + 3 * 60 * 60 * 1000
    end_ms = start_ms + (30 - 1) * check_all_datas.MINUTE_INTERVAL_MS
    candles = [
        {"t": start_ms + index * check_all_datas.MINUTE_INTERVAL_MS, "o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "v": 1.0}
        for index in range(30)
    ]

    completeness = check_all_datas._compute_session_completeness(
        candles,
        start_ms=start_ms,
        end_ms=end_ms,
        close_ms=close_ms,
        interval_ms=check_all_datas.MINUTE_INTERVAL_MS,
        timeframe="1m",
    )

    assert completeness["bars_observed"] == 30
    assert completeness["bars_expected"] == 180
    assert completeness["status"] == "partial"
    assert completeness["coverage_ratio"] == pytest.approx(30 / 180)


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
    assert result["status"] == "ok"
    notes = result.get("notes", [])
    assert any(
        "extended fallback" in str(note) for note in notes
    ), "Expected fallback note in response notes"

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


@pytest.mark.anyio("asyncio")
async def test_check_all_with_fixture_snapshot_file() -> None:
    fixture_path = Path(__file__).with_name("data") / "fake_snapshot.json"
    snapshot = json.loads(fixture_path.read_text(encoding="utf-8"))
    candles = list(snapshot["candles"])
    start_ts = candles[0]["t"]
    end_ts = candles[-1]["t"]
    enriched_snapshot = dict(snapshot)
    enriched_snapshot["selection"] = {"start": start_ts, "end": end_ts}

    result = await check_all_datas.build_check_all_datas(
        enriched_snapshot,
        now_utc=datetime.fromtimestamp(end_ts / 1000, tz=UTC) + timedelta(minutes=5),
        hours=3,
        trace=None,
    )

    assert result is not None
    assert result["status"] == "ready"
    meta = result["meta"]
    assert meta["symbol"] == snapshot["symbol"]
    assert meta["invalid_candles_count"] == 0
    assert meta["invalid_ts_count"] == 0
    assert isinstance(meta["stale"], bool)
    assert isinstance(result["notes"], list)
    if result["notes"]:
        assert any("Minute coverage" in note for note in result["notes"])

    diagnostics = meta.get("diagnostics", {})
    coverage = diagnostics.get("coverage", {})
    minute_state = coverage.get("1m", {})
    assert minute_state.get("present_bars", 0) > 0
    first_ts = minute_state.get("start_ms")
    assert first_ts is None or abs(first_ts - start_ts) <= 5 * 60_000

    availability = result["availability"].get("timeframes", {})
    if isinstance(availability, Mapping) and "1m" in availability:
        minute_availability = availability["1m"]
        assert set(minute_availability.keys()) == {"zones", "sweeps", "atr", "vwap_sessions"}
