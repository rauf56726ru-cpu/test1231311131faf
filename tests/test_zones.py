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
    payload = detect_zones(frames, symbol="TEST", cfg=cfg)

    zones = payload["zones"]
    assert zones["fvg"], "Expected at least one FVG zone"
    assert zones["ob"], "Expected at least one order block"

    fvg_zone = zones["fvg"][0]
    assert fvg_zone["tf"] == "15m"
    assert fvg_zone["direction"] in {"up", "down"}
    assert fvg_zone["status"] in {"open", "fulfilled", "inverted"}
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
    payload = detect_zones(frames, symbol="TEST", cfg=cfg, profile_levels=profile_map)

    levels = payload["zones"]["profile_levels"]
    assert {level["type"] for level in levels} == {"poc", "vah", "val"}
    assert {level["session"] for level in levels} == {"daily"}


def test_zones_endpoint_returns_structured_payload(client: TestClient) -> None:
    candles = build_orderflow_sequence()
    response = client.request(
        "GET",
        "/zones",
        params={"symbol": "TEST", "tf": "15m"},
        json={"candles": candles, "tick_size": 0.2},
    )
    assert response.status_code == 200
    payload = response.json()
    zones = payload["zones"]
    for key in ("fvg", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels"):
        assert key in zones
        assert isinstance(zones[key], list)
