from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.services.inspection as inspection
from src.services import presets
from src.services.zones import (
    Config,
    compute_swings,
    detect_cisd,
    detect_fvg,
    detect_inducement,
    detect_ob,
    detect_zones,
)


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
        "t": BASE_TS + index * 60_000,
        "o": open_price,
        "h": high,
        "l": low,
        "c": close,
        "v": volume,
    }


def build_series_with_fvg() -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    for index in range(60):
        base = 100.0 + 0.05 * index
        candles.append(
            make_candle(
                index,
                base,
                base + 0.8,
                base - 0.8,
                base + 0.1,
            )
        )

    start = len(candles)
    candles.append(make_candle(start, 104.0, 104.6, 103.6, 104.2))
    candles.append(make_candle(start + 1, 104.3, 104.9, 103.9, 104.5))
    candles.append(make_candle(start + 2, 107.0, 108.2, 107.1, 107.6))
    candles.append(make_candle(start + 3, 102.5, 103.2, 101.8, 102.0))
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


def test_detect_fvg_up_down_with_filters() -> None:
    candles = [
        make_candle(0, 100.0, 100.0, 98.0, 99.5),
        make_candle(1, 99.8, 100.8, 98.4, 99.9),
        make_candle(2, 102.0, 104.0, 103.0, 103.4),
        make_candle(3, 96.8, 97.9, 94.8, 95.0),
        make_candle(4, 95.0, 96.0, 94.0, 94.5),
    ]

    cfg = Config(min_gap_pct=0.05, tick_size=0.5)
    zones = detect_fvg(candles, cfg, "1m")

    assert len(zones) == 2

    up_zone = next(zone for zone in zones if zone["dir"] == "up")
    down_zone = next(zone for zone in zones if zone["dir"] == "down")

    assert up_zone["bot"] == pytest.approx(100.0)
    assert up_zone["top"] == pytest.approx(103.0)
    assert up_zone["fvl"] == pytest.approx(101.5)
    assert up_zone["created_at"] == candles[1]["t"]
    assert up_zone["status"] == "open"

    assert down_zone["bot"] == pytest.approx(96.0)
    assert down_zone["top"] == pytest.approx(103.0)
    assert down_zone["status"] == "open"


def test_detect_fvg_retains_small_gap_with_tick_inference() -> None:
    candles = [
        make_candle(0, 100000.0, 100000.0, 99999.5, 100000.00),
        make_candle(1, 100000.1, 100000.2, 99999.9, 100000.01),
        make_candle(2, 100000.6, 100001.0, 100000.5, 100000.51),
    ]

    cfg_explicit = Config(min_gap_pct=0.0001, tick_size=0.01)
    zones_explicit = detect_fvg(candles, cfg_explicit, "1m")
    assert len(zones_explicit) == 1
    zone = zones_explicit[0]
    assert zone["bot"] == pytest.approx(100000.0)
    assert zone["top"] == pytest.approx(100000.5)
    assert zone["status"] == "open"

    cfg_inferred = Config(min_gap_pct=0.0001, tick_size=None)
    zones_inferred = detect_fvg(candles, cfg_inferred, "1m")
    assert len(zones_inferred) == 1
    inferred_zone = zones_inferred[0]
    assert inferred_zone["bot"] == pytest.approx(zone["bot"])
    assert inferred_zone["top"] == pytest.approx(zone["top"])
    assert inferred_zone["status"] == zone["status"]


def test_detect_fvg_merges_overlapping_and_marks_closed() -> None:
    candles = [
        make_candle(0, 100.0, 100.5, 99.2, 99.6),
        make_candle(1, 101.0, 102.0, 100.2, 101.6),
        make_candle(2, 103.0, 105.0, 103.0, 104.4),
        make_candle(3, 104.8, 107.0, 104.2, 106.8),
        make_candle(4, 101.0, 105.0, 99.0, 101.5),
    ]

    cfg = Config(min_gap_pct=0.01, tick_size=0.5)
    zones = detect_fvg(candles, cfg, "1m")

    assert len(zones) == 1
    zone = zones[0]
    assert zone["bot"] == pytest.approx(100.5)
    assert zone["top"] == pytest.approx(104.0)
    assert zone["created_at"] == candles[1]["t"]
    assert zone["status"] == "closed"


def test_detect_ob_supply_status_transitions() -> None:
    candles = [
        make_candle(0, 100.0, 101.0, 99.0, 100.5),
        make_candle(1, 100.5, 101.5, 100.0, 101.0),
        make_candle(2, 101.0, 102.0, 100.5, 101.4),
        make_candle(3, 100.8, 102.6, 100.2, 102.3),
        make_candle(4, 99.0, 99.5, 92.0, 93.0),
        make_candle(5, 93.5, 101.5, 93.0, 95.0),
        make_candle(6, 101.0, 105.0, 99.5, 103.5),
    ]

    cfg = Config(atr_period=3, k_impulse=1.1, tick_size=0.5)

    open_zone = detect_ob(candles[:5], cfg, "1m")[0]
    tapped_zone = detect_ob(candles[:6], cfg, "1m")[0]
    inverted_zone = detect_ob(candles, cfg, "1m")[0]

    assert open_zone["status"] == "open"
    assert tapped_zone["status"] == "tapped"
    assert inverted_zone["status"] == "inverted"
    assert inverted_zone["range"] == pytest.approx([100.0, 102.5])


def test_compute_swings_identifies_local_extrema() -> None:
    candles = [
        make_candle(0, 100.0, 101.0, 99.0, 100.5),
        make_candle(1, 101.0, 103.0, 100.5, 102.5),
        make_candle(2, 102.0, 102.5, 98.5, 99.0),
        make_candle(3, 99.0, 104.0, 98.8, 103.2),
        make_candle(4, 103.0, 102.0, 99.5, 100.0),
    ]

    swings = compute_swings(candles, w=1)
    assert [s["type"] for s in swings] == ["high", "low", "high"]
    assert swings[0]["price"] == pytest.approx(103.0)
    assert swings[1]["price"] == pytest.approx(98.5)


def test_detect_inducement_classifies_before_inside_after() -> None:
    candles = [
        make_candle(0, 100.0, 100.5, 99.0, 99.8),
        make_candle(1, 100.4, 101.0, 99.5, 100.6),
        make_candle(2, 102.0, 104.0, 103.7, 103.8),
        make_candle(3, 102.2, 104.2, 100.2, 100.3),
        make_candle(4, 101.0, 102.5, 100.5, 101.5),
        make_candle(5, 102.0, 104.1, 101.0, 102.4),
        make_candle(6, 100.5, 102.0, 99.5, 100.0),
        make_candle(7, 101.5, 103.8, 99.8, 100.2),
        make_candle(8, 100.0, 101.0, 99.0, 99.5),
    ]

    cfg = Config(
        tick_size=0.5,
        atr_period=3,
        w_swing=1,
        r_zone_pct=0.7,
        m_wick_atr=2.0,
    )

    fvg_zones = detect_fvg(candles, cfg, "1m")
    ob_zones: Sequence[Mapping[str, Any]] = []
    inducements = detect_inducement(candles, fvg_zones, ob_zones, cfg, "1m")

    assert [item["type"] for item in inducements] == ["before", "inside", "after"]


def test_detect_cisd_identifies_bull_scenario() -> None:
    candles = [
        make_candle(0, 104.5, 105.2, 104.0, 104.5),
        make_candle(1, 105.2, 106.5, 105.0, 105.2),
        make_candle(2, 103.5, 104.0, 100.8, 103.0),
        make_candle(3, 102.5, 104.4, 101.5, 102.0),
        make_candle(4, 101.5, 101.8, 99.2, 100.0),
        make_candle(5, 100.5, 103.0, 98.5, 99.0),
        make_candle(6, 99.0, 100.5, 96.5, 97.5),
        make_candle(7, 98.0, 105.5, 97.0, 105.2),
    ]

    cfg = Config(tick_size=0.1, w_swing=1)
    fvg_zones = detect_fvg(candles, cfg, "1m")
    swings = compute_swings(candles, w=1)
    cisd = detect_cisd(candles, fvg_zones, swings, cfg, "1m")

    assert {item["type"] for item in cisd} == {"bull"}
    assert cisd[0]["delivery_candle"] == candles[7]["t"]


def test_zones_endpoint_returns_structured_payload(client: TestClient) -> None:
    candles = [
        make_candle(0, 100.0, 100.0, 98.0, 99.5),
        make_candle(1, 99.8, 100.8, 98.4, 99.9),
        make_candle(2, 102.0, 104.0, 103.0, 103.4),
        make_candle(3, 96.8, 97.9, 94.8, 95.0),
        make_candle(4, 95.0, 96.0, 94.0, 94.5),
    ]

    response = client.request(
        "GET",
        "/zones",
        params={"symbol": "TEST", "tf": "1m", "tick_size": 0.5, "atr_period": 3, "w_swing": 1},
        json={"candles": candles},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "TEST"
    assert "fvg" in payload["zones"]
    assert payload["zones"]["fvg"], "FVG zones should be detected"


def test_profile_and_inspection_include_structured_zones(client: TestClient) -> None:
    candles = build_series_with_fvg()

    snapshot_payload = {
        "id": "snap-test",
        "symbol": "TEST",
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": candles}},
    }

    snapshot_id = inspection.register_snapshot(snapshot_payload)

    profile_response = client.get("/profile", params={"snapshot": snapshot_id, "tf": "1m"})
    assert profile_response.status_code == 200
    profile_payload = profile_response.json()
    assert "zones" in profile_payload
    assert profile_payload["zones"]["zones"]["fvg"]

    snapshot = inspection.get_snapshot(snapshot_id)
    inspection_payload = inspection.build_inspection_payload(snapshot)
    zones_structured = inspection_payload["DATA"]["zones"]
    assert zones_structured["zones"]["fvg"]
