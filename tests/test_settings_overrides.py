from __future__ import annotations

import json

import pytest


def _write_settings(tmp_path, payload):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload))
    return path


def test_equal_high_low_settings_override(tmp_path, monkeypatch):
    payload = {
        "eql_settings": {
            "tolerance_bps_by_tf": {"15m": 12.0},
            "min_separation_bars_by_tf": {"15m": 8},
            "pivot_radius_by_tf": {"15m": 5},
        }
    }
    settings_path = _write_settings(tmp_path, payload)
    monkeypatch.setenv("APP_SETTINGS_PATH", str(settings_path))

    from src.services import settings as settings_module
    settings_module.get_settings.cache_clear()

    import src.services.check_all_datas as check_all_datas

    try:
        assert check_all_datas._relative_tolerance_for_equal_levels("15m") == pytest.approx(0.0012)
        assert check_all_datas._minimum_separation_for_equal_levels("15m") == 8
        assert check_all_datas._pivot_radius_for_equal_levels("15m") == 5
    finally:
        settings_module.get_settings.cache_clear()


def test_zone_threshold_overrides_apply(tmp_path, monkeypatch):
    payload = {
        "zone_thresholds": {
            "tolerance_bps": 3.0,
            "max_fill_percent": 75.0,
            "min_displacement": 0.5,
            "min_separation_bars": 4,
        }
    }
    settings_path = _write_settings(tmp_path, payload)
    monkeypatch.setenv("APP_SETTINGS_PATH", str(settings_path))

    from src.services import settings as settings_module
    settings_module.get_settings.cache_clear()

    import src.services.check_all_datas as check_all_datas

    try:
        cfg = check_all_datas._apply_zone_threshold_overrides(check_all_datas.ZonesConfig())
        assert cfg.sr_merge_pct == pytest.approx(0.0003)
        assert cfg.mitigation_fill_ratio == pytest.approx(0.75)
        assert cfg.ob_distance_atr == pytest.approx(0.5)
        assert cfg.base_min_bars == 4
    finally:
        settings_module.get_settings.cache_clear()
