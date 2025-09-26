from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Mapping

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.services.inspection as inspection
from src.meta import Meta
from src.services import presets
from src.services.profile import (
    build_volume_profile,
    compute_session_profiles,
    flatten_profile,
    split_by_sessions,
)


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


def make_unimodal(
    *,
    center: float = 100.0,
    spread: float = 0.5,
    n: int = 200,
    seed: int = 42,
    base: datetime | None = None,
) -> List[Mapping[str, float]]:
    rng = random.Random(seed)
    start = base or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for index in range(n):
        moment = start + timedelta(minutes=index)
        timestamp = int(moment.timestamp() * 1000)
        price = rng.gauss(center, spread)
        high = price + abs(rng.gauss(spread * 0.9, spread * 0.4))
        low = price - abs(rng.gauss(spread * 0.9, spread * 0.4))
        open_price = price + rng.uniform(-spread * 0.4, spread * 0.4)
        close_price = price
        volume = abs(rng.gauss(12.0, 2.0)) + 1.0
        candles.append(
            {
                "t": timestamp,
                "o": round(open_price, 4),
                "h": round(high, 4),
                "l": round(low, 4),
                "c": round(close_price, 4),
                "v": round(volume, 4),
            }
        )
    return candles


def make_bimodal(
    *,
    left: float = 100.0,
    right: float = 102.0,
    n: int = 240,
    seed: int = 7,
    base: datetime | None = None,
) -> List[Mapping[str, float]]:
    rng = random.Random(seed)
    start = base or datetime(2024, 1, 2, tzinfo=timezone.utc)
    candles = []
    for index in range(n):
        moment = start + timedelta(minutes=index)
        timestamp = int(moment.timestamp() * 1000)
        cluster_center = right if index % 2 else left
        spread = 0.35 if index % 2 else 0.55
        price = rng.gauss(cluster_center, spread)
        high = price + abs(rng.gauss(spread * 1.2, spread * 0.5))
        low = price - abs(rng.gauss(spread * 1.2, spread * 0.5))
        open_price = price + rng.uniform(-spread * 0.4, spread * 0.4)
        close_price = price
        volume = (18.0 if index % 2 else 10.0) + abs(rng.gauss(0.0, 1.5))
        candles.append(
            {
                "t": timestamp,
                "o": round(open_price, 4),
                "h": round(high, 4),
                "l": round(low, 4),
                "c": round(close_price, 4),
                "v": round(volume, 4),
            }
        )
    return candles


def test_volume_profile_unimodal_focus() -> None:
    candles = make_unimodal(center=250.0, spread=0.8, n=360)
    profile = build_volume_profile(candles, tick_size=0.1, adaptive_bins=False, value_area_pct=0.7)

    assert profile.prices, "Profile should contain bins"
    assert math.isfinite(profile.poc)
    assert abs(profile.poc - 250.0) <= 0.6
    flattened = flatten_profile(profile)
    total_volume = sum(entry["volume"] for entry in flattened)
    assert total_volume > 0
    within_value_area = [
        entry for entry in flattened if profile.val <= entry["price"] <= profile.vah
    ]
    value_volume = sum(entry["volume"] for entry in within_value_area)
    assert value_volume / total_volume >= 0.65
    assert "warn" not in profile.diagnostics


def test_volume_profile_bimodal_prefers_dominant_peak() -> None:
    candles = make_bimodal(left=200.0, right=205.0, n=300)
    profile = build_volume_profile(candles, tick_size=0.25, adaptive_bins=False, value_area_pct=0.7)

    assert math.isfinite(profile.poc)
    assert abs(profile.poc - 205.0) <= 0.75
    assert profile.vah >= profile.val
    flattened = flatten_profile(profile)
    total_volume = sum(entry["volume"] for entry in flattened)
    assert total_volume > 0
    covered = sum(entry["volume"] for entry in flattened if profile.val <= entry["price"] <= profile.vah)
    assert covered / total_volume >= 0.65


def test_volume_profile_warns_on_short_series() -> None:
    candles = make_unimodal(n=10)
    profile = build_volume_profile(candles, tick_size=0.1, adaptive_bins=False, value_area_pct=0.7)
    assert profile.diagnostics.get("warn") == "too few bars"


def test_split_by_sessions_groups_candles() -> None:
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    candles = make_unimodal(n=600, base=base)
    sessions = list(Meta.iter_vwap_sessions())
    buckets = split_by_sessions(candles, sessions)
    assert buckets, "Expected session buckets"
    asia_key = (base.date(), sessions[0][0])
    london_key = (base.date(), sessions[1][0])
    assert asia_key in buckets
    assert london_key in buckets
    assert len(buckets[asia_key]) > 0
    assert len(buckets[london_key]) > 0


def _snapshot_payload(candles: Iterable[Mapping[str, float]]) -> Mapping[str, object]:
    return {
        "symbol": "BTCUSDT",
        "tf": "1m",
        "frames": {
            "1m": {
                "tf": "1m",
                "candles": list(candles),
            }
        },
    }


def test_compute_session_profiles_respects_last_n() -> None:
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    candles = make_unimodal(n=60 * 24 * 4, base=base)
    sessions = list(Meta.iter_vwap_sessions())
    summaries = compute_session_profiles(
        candles,
        sessions=sessions,
        last_n=2,
        tick_size=0.1,
        adaptive_bins=False,
        value_area_pct=0.7,
    )
    assert summaries, "Expected at least one TPO summary"

    daily_entries = [entry for entry in summaries if entry.get("session") == "daily"]
    assert daily_entries, "Expected daily TPO entries"
    assert len(daily_entries) <= 2

    allowed_dates = {entry["date"] for entry in daily_entries}
    assert allowed_dates
    assert allowed_dates == {entry["date"] for entry in summaries}
    expected_dates = {
        (base.date() + timedelta(days=offset)).isoformat() for offset in range(2, 4)
    }
    assert allowed_dates == expected_dates

    ordered_dates = [entry["date"] for entry in summaries]
    assert ordered_dates == sorted(ordered_dates)

    for day_entry in daily_entries:
        nested = day_entry.get("sessions", [])
        assert isinstance(nested, list)
        for nested_entry in nested:
            assert nested_entry.get("date") == day_entry["date"]
    if any(day_entry.get("sessions") for day_entry in daily_entries):
        assert summaries[-1]["session"] != "daily"


def test_profile_endpoint_returns_payload(client: TestClient) -> None:
    candles = make_unimodal(n=720, base=datetime(2024, 6, 1, tzinfo=timezone.utc))
    response = client.post("/inspection/snapshot", json=_snapshot_payload(candles))
    assert response.status_code == 200
    snapshot_id = response.json()["snapshot_id"]

    params = {
        "snapshot": snapshot_id,
        "last_n": 2,
        "value_area_pct": 0.7,
    }
    profile_response = client.get("/profile", params=params)
    assert profile_response.status_code == 200
    body = profile_response.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["tf"] == "1m"
    assert "tpo" in body and isinstance(body["tpo"], list)
    assert "profile" in body and isinstance(body["profile"], list)
    assert "zones" in body and isinstance(body["zones"], list)
    preset = body.get("preset")
    assert preset is not None
    assert preset["symbol"] == "BTCUSDT"
    assert preset["builtin"] is True
    assert body.get("preset_required") is False
    if body["zones"]:
        zone_types = {item["type"] for item in body["zones"]}
        assert {"tpo_poc", "tpo_vah", "tpo_val"}.issubset(zone_types)
    if body["tpo"]:
        latest = body["tpo"][-1]
        assert "session" in latest
        assert "date" in latest


def test_profile_endpoint_includes_all_snapshot_days(client: TestClient) -> None:
    base = datetime(2025, 9, 21, tzinfo=timezone.utc)
    candles = make_unimodal(n=60 * 24 * 5, base=base)

    response = client.post("/inspection/snapshot", json=_snapshot_payload(candles))
    assert response.status_code == 200
    snapshot_id = response.json()["snapshot_id"]

    profile_response = client.get(
        "/profile",
        params={
            "snapshot": snapshot_id,
            "last_n": 5,
            "value_area_pct": 0.7,
        },
    )
    assert profile_response.status_code == 200
    body = profile_response.json()

    tpo_entries = body.get("tpo", [])
    assert tpo_entries, "Expected TPO entries to be present"
    daily_entries = [entry for entry in tpo_entries if entry.get("session") == "daily"]
    assert len(daily_entries) == 5

    expected_dates = {
        (base.date() + timedelta(days=offset)).isoformat() for offset in range(5)
    }
    observed_dates = {entry.get("date") for entry in daily_entries}
    assert observed_dates == expected_dates

    zone_dates = {zone.get("date") for zone in body.get("zones", [])}
    assert expected_dates.issubset(zone_dates)


def test_profile_endpoint_is_deterministic(client: TestClient) -> None:
    candles = make_bimodal(n=360, base=datetime(2024, 7, 1, tzinfo=timezone.utc))
    response = client.post("/inspection/snapshot", json=_snapshot_payload(candles))
    assert response.status_code == 200
    snapshot_id = response.json()["snapshot_id"]

    params = {"snapshot": snapshot_id, "adaptive_bins": True, "value_area_pct": 0.75}
    first = client.get("/profile", params=params)
    second = client.get("/profile", params=params)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_profile_endpoint_marks_missing_preset(client: TestClient) -> None:
    candles = make_unimodal(
        n=360,
        base=datetime(2024, 8, 1, tzinfo=timezone.utc),
    )
    payload = dict(_snapshot_payload(candles))
    payload["symbol"] = "ABCUSDT"
    response = client.post("/inspection/snapshot", json=payload)
    assert response.status_code == 200
    snapshot_id = response.json()["snapshot_id"]

    profile_response = client.get("/profile", params={"snapshot": snapshot_id})
    assert profile_response.status_code == 200
    body = profile_response.json()
    assert body["symbol"] == "ABCUSDT"
    assert body.get("preset") is None
    assert body.get("preset_required") is True
