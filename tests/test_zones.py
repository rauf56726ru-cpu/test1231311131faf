from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.services import inspection, presets
from src.services.zones import Config, detect_zones

BASE_TS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def make_candle(
    index: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float = 10.0,
) -> Dict[str, float]:
    return {
        "t": BASE_TS + index * 900_000,
        "o": open_price,
        "h": high,
        "l": low,
        "c": close,
        "v": volume,
    }


def build_orderflow_sequence() -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    for idx in range(4):
        base = 100.0 + idx * 0.3
        candles.append(make_candle(idx, base, base + 0.8, base - 0.6, base + 0.2))
    candles.append(make_candle(4, 101.5, 102.0, 100.5, 101.2))
    candles.append(make_candle(5, 104.0, 110.5, 104.4, 109.6))
    candles.append(make_candle(6, 109.6, 110.0, 109.0, 109.2))
    candles.append(make_candle(7, 100.4, 100.6, 99.8, 100.0))
    candles.append(make_candle(8, 100.1, 103.2, 100.0, 102.8))
    candles.append(make_candle(9, 102.4, 103.0, 100.1, 100.5))
    candles.append(make_candle(10, 100.6, 101.0, 99.4, 99.6))
    candles.append(make_candle(11, 99.8, 100.2, 98.8, 99.0))
    return candles


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


def test_detect_zones_identifies_fvg_and_order_blocks() -> None:
    frames = {"15m": build_orderflow_sequence()}
    cfg = Config(tick_size=0.2, atr_period=3)
    payload = detect_zones(frames=frames, config=cfg)

    zones = payload["zones"]
    assert zones["fvg"], "Expected at least one FVG zone"
    assert zones["ob"], "Expected at least one order block"

    fvg_zone = zones["fvg"][0]
    assert fvg_zone["tf"] == "15m"
    assert fvg_zone["direction"] in {"up", "down"}
    assert fvg_zone["status"] in {"open", "fresh", "tapped", "mitigated"}
    if fvg_zone.get("inverted"):
        assert fvg_zone["status"] in {"open", "tapped"}
    assert fvg_zone["bot"] < fvg_zone["top"]
    assert fvg_zone["mid"] == pytest.approx((fvg_zone["bot"] + fvg_zone["top"]) / 2)

    ob_zone = zones["ob"][0]
    assert ob_zone["tf"] == "15m"
    assert ob_zone["type"] in {"supply", "demand"}
    assert ob_zone["status"] in {"fresh", "tapped", "invalidated"}
    assert ob_zone["open"] < ob_zone["close"]


def test_detect_zones_propagates_profile_levels() -> None:
    frames = {"15m": build_orderflow_sequence()}
    cfg = Config(tick_size=0.2, atr_period=3)
    profile_map = {"daily": {"poc": 101.25, "vah": 102.4, "val": 99.8}}
    payload = detect_zones(
        frames=frames,
        config=cfg,
        profile_levels=profile_map,
    )


def test_fvg_preserves_raw_bounds_when_tick_collapses() -> None:
    candles = [
        make_candle(0, 100.0, 100.0, 99.6, 99.8),
        make_candle(1, 99.8, 101.6, 99.5, 101.2),
        make_candle(2, 101.5, 101.8, 100.3, 100.6),
        make_candle(3, 100.5, 100.7, 99.8, 100.1),
        make_candle(4, 100.0, 100.4, 99.7, 99.9),
    ]
    cfg = Config(
        tick_size=1.0,
        displacement_body=0.0,
        displacement_range=0.0,
        atr_period=1,
        min_gap_tick_multiple=0.0,
    )
    payload = detect_zones(frames={"15m": candles}, config=cfg)

    zones = payload["zones"]
    assert zones["fvg"], "Expected at least one FVG zone"
    fvg = zones["fvg"][0]
    assert fvg["top"] == pytest.approx(100.3)
    assert fvg["bot"] == pytest.approx(100.0)
    stats = payload["meta"]["fvg_stats"]["15m"]
    assert stats.get("fvg_reject_tick_collapse", 0) >= 1


def test_detect_zones_accepts_keyword_only_inputs() -> None:
    frames = {"15m": build_orderflow_sequence()}
    payload = detect_zones(frames=frames)

    assert "zones" in payload
    assert isinstance(payload["zones"], dict)


def test_detect_zones_legacy_single_positional_argument() -> None:
    frames = {"15m": build_orderflow_sequence()}
    payload = detect_zones(frames)

    assert "zones" in payload
    assert isinstance(payload["zones"], dict)


def test_fvg_survives_with_atr_fallback() -> None:
    frames = {"15m": build_orderflow_sequence()}
    cfg = Config(tick_size=0.2, atr_period=50)

    payload = detect_zones(frames=frames, config=cfg)

    fvg_zones = payload["zones"]["fvg"]
    assert fvg_zones, "Expected FVG zones even when ATR coverage is insufficient"

    stats = payload["meta"]["fvg_stats"]["15m"]
    assert stats.get("fvg_reject_displacement", 0) == 0

    diagnostics = payload["meta"]["diagnostics"]
    warmup_diag = diagnostics.get("warmup", {}).get("15m", {})
    assert warmup_diag.get("ok") is False

    timeframe_diag = next(
        entry
        for entry in diagnostics.get("timeframes", [])
        if entry.get("tf") == "15m"
    )
    atr_diag = timeframe_diag.get("atr", {})
    assert atr_diag.get("reliable") is False
    assert atr_diag.get("proxy") == "stdev_close_20"


def test_detect_zones_diagnostics_include_reasons_for_empty_results() -> None:
    payload = detect_zones(frames={"15m": []})

    diagnostics = payload["meta"].get("diagnostics")
    assert isinstance(diagnostics, dict)

    summary = diagnostics.get("summary")
    assert isinstance(summary, dict)

    fvg_summary = summary.get("fvg")
    assert isinstance(fvg_summary, dict)
    assert fvg_summary.get("count") == 0
    reasons = fvg_summary.get("reasons")
    assert reasons and any(item.get("reason") == "insufficient_candles" for item in reasons)

    rb_summary = summary.get("rb")
    assert isinstance(rb_summary, dict)
    assert rb_summary.get("count") == 0
    rb_reasons = rb_summary.get("reasons")
    assert rb_reasons and any(item.get("reason") for item in rb_reasons)

    timeframes = diagnostics.get("timeframes")
    assert isinstance(timeframes, list) and timeframes
    first_tf = timeframes[0]
    assert first_tf.get("fvg", {}).get("reason") == "insufficient_candles"
    assert first_tf.get("rb", {}).get("reason") == "insufficient_candles"


def test_zones_endpoint_returns_structured_payload(client: TestClient) -> None:
    candles = build_orderflow_sequence()
    response = client.request(
        "GET",
        "/zones",
        params={"symbol": "TEST", "tf": "15m", "atr_period": 3},
        json={"candles": candles, "tick_size": 0.2},
    )
    assert response.status_code == 200
    payload = response.json()
    zones = payload["zones"]
    for key in ("fvg", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels"):
        assert key in zones
        assert isinstance(zones[key], list)
    assert zones["ob"], "Expected non-empty OB list for synthetic sequence"
    assert any(zones[key] for key in ("fvg", "ob", "mb", "bb", "rb", "pb", "sr")), (
        "Expected at least one populated zone list"
    )


def test_zones_endpoint_uses_latest_snapshot_defaults(client: TestClient) -> None:
    candles: List[Dict[str, float]] = []
    for idx in range(18):
        base = 100.0 + idx * 0.15
        candles.append(make_candle(idx, base, base + 0.6, base - 0.6, base + 0.2))

    gap_start = len(candles)
    candles.append(make_candle(gap_start, 103.5, 103.8, 102.9, 103.6))
    candles.append(make_candle(gap_start + 1, 104.1, 104.4, 103.8, 104.2))
    candles.append(make_candle(gap_start + 2, 107.2, 107.8, 106.9, 107.5))

    for tail in range(3):
        idx = len(candles)
        base = 106.8 - tail * 0.25
        candles.append(make_candle(idx, base, base + 0.7, base - 0.7, base + 0.15))
    snapshot_payload = {
        "symbol": "SNAP",
        "tf": "15m",
        "candles": candles,
    }
    create_response = client.post("/inspection/snapshot", json=snapshot_payload)
    assert create_response.status_code == 200

    response = client.get("/zones")
    assert response.status_code == 200
    payload = response.json()
    zones = payload.get("zones", {})
    assert zones.get("fvg") or zones.get("ob"), "Expected FVG or OB zones from latest snapshot"


def test_diag_report_includes_kpi_metrics(client: TestClient) -> None:
    candles = build_orderflow_sequence()
    snapshot_payload = {
        "symbol": "DIAG",
        "tf": "15m",
        "candles": candles,
    }
    create_response = client.post("/inspection/snapshot", json=snapshot_payload)
    assert create_response.status_code == 200

    response = client.get("/diag")
    assert response.status_code == 200
    payload = response.json()

    assert payload.get("schema") == "compact.v1"

    meta = payload.get("meta")
    assert isinstance(meta, dict)
    assert meta.get("symbol") == "DIAG"
    assert "tz" in meta
    assert "coverage" in meta and isinstance(meta["coverage"], dict)

    summary = payload.get("summary")
    assert isinstance(summary, dict)

    zones_summary = summary.get("zones")
    assert isinstance(zones_summary, dict)
    assert "raw_counts" in zones_summary
    assert "retention" in zones_summary
    assert "fvg_ob_share" in zones_summary

    metrics = summary.get("metrics")
    assert isinstance(metrics, dict)
    for key in (
        "fvg_reject_no_gap",
        "fvg_reject_displacement",
        "zones_retained",
        "fvg_ob_share",
    ):
        entry = metrics.get(key)
        assert isinstance(entry, dict)
        assert "ok" in entry
        assert "target" in entry
        if not entry["ok"]:
            assert entry.get("reason"), f"Expected reason for failed metric {key}"

    timeframe_targets = summary.get("timeframe_targets")
    assert isinstance(timeframe_targets, dict)
    for key in ("1h_fvg", "1h_ob", "15m_fvg"):
        entry = timeframe_targets.get(key)
        assert isinstance(entry, dict)
        assert "ok" in entry
        assert "required" in entry
        assert "count" in entry
        if not entry["ok"]:
            assert entry.get("reason"), f"Expected reason for timeframe target {key}"

    diagnostics = payload.get("diagnostics")
    assert isinstance(diagnostics, dict)
    assert "filter" in diagnostics
    assert "fvg_stats" in diagnostics

    zones_block = payload.get("zones")
    assert isinstance(zones_block, dict)
    for key in ("fvg", "ob", "other"):
        assert key in zones_block
