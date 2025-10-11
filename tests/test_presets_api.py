from __future__ import annotations

from typing import Iterable

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.services import presets


@pytest.fixture()
def client() -> Iterable[TestClient]:
    with TestClient(app) as http_client:
        yield http_client


@pytest.fixture(autouse=True)
def preset_storage(tmp_path, monkeypatch):
    storage_path = tmp_path / "presets.json"
    monkeypatch.setattr(presets, "_PRESET_STORAGE_PATH", storage_path)
    presets._PRESET_CACHE.clear()
    presets._STORAGE_LOADED = False
    yield
    presets._PRESET_CACHE.clear()
    presets._STORAGE_LOADED = False


def test_list_presets_includes_builtins(client: TestClient) -> None:
    response = client.get("/presets")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    symbols = {item["symbol"] for item in payload["presets"]}
    assert {"BTCUSDT", "ETHUSDT", "SOLUSDT"}.issubset(symbols)
    builtin_flags = {item["symbol"]: item["builtin"] for item in payload["presets"]}
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert builtin_flags[symbol] is True


def test_get_preset_returns_builtin_when_available(client: TestClient) -> None:
    response = client.get("/presets/BTCUSDT")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    preset = payload["preset"]
    assert preset["symbol"] == "BTCUSDT"
    assert preset["builtin"] is True


def test_get_preset_returns_null_when_missing(client: TestClient) -> None:
    response = client.get("/presets/ABCUSDT")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["preset"] is None


def test_create_update_delete_preset(client: TestClient) -> None:
    create_payload = {
        "symbol": "ABCUSDT",
        "tf": "1m",
        "last_n": 5,
        "value_area_pct": 0.9,
        "binning": {
            "mode": "tick",
            "tick_size": -0.01,
            "atr_multiplier": 5.0,
            "target_bins": 500,
        },
        "extras": {
            "clip_low_volume_tail": 0.02,
            "smooth_window": 4,
        },
    }
    create_response = client.post("/presets", json=create_payload)
    assert create_response.status_code == 200
    created = create_response.json()["preset"]
    assert created["symbol"] == "ABCUSDT"
    assert created["builtin"] is False
    assert created["value_area_pct"] == pytest.approx(0.9)
    assert created["binning"]["mode"] == "adaptive"
    assert created["binning"]["tick_size"] is None
    assert created["binning"]["atr_multiplier"] == pytest.approx(2.0)
    assert created["binning"]["target_bins"] == 200
    assert created["extras"]["clip_low_volume_tail"] == pytest.approx(0.02)
    assert created["extras"]["smooth_window"] == 4

    list_response = client.get("/presets")
    symbols = {item["symbol"]: item for item in list_response.json()["presets"]}
    assert "ABCUSDT" in symbols
    assert symbols["ABCUSDT"]["builtin"] is False

    update_payload = {
        "value_area_pct": 0.65,
        "binning": {"mode": "tick", "tick_size": 0.5},
    }
    update_response = client.put("/presets/ABCUSDT", json=update_payload)
    assert update_response.status_code == 200
    updated = update_response.json()["preset"]
    assert updated["value_area_pct"] == pytest.approx(0.65)
    assert updated["binning"]["mode"] == "tick"
    assert updated["binning"]["tick_size"] == pytest.approx(0.5)
    assert updated["binning"]["atr_multiplier"] == pytest.approx(0.5)
    assert updated["binning"]["target_bins"] == 80

    delete_response = client.delete("/presets/ABCUSDT")
    assert delete_response.status_code == 200
    assert delete_response.json()["ok"] is True

    post_delete = client.get("/presets/ABCUSDT")
    assert post_delete.status_code == 200
    assert post_delete.json()["preset"] is None
