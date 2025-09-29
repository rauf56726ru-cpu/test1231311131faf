from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from src.services import inspection


UTC = timezone.utc


def _make_candle(ts: int, o: float, h: float, l: float, c: float) -> Dict[str, Any]:
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": 1.0}


def _sample_candles(base: datetime) -> List[Dict[str, Any]]:
    specs = [
        (100.0, 101.0, 99.0, 100.5),
        (100.5, 102.0, 99.5, 101.2),
        (101.2, 110.0, 100.0, 109.5),
        (109.5, 108.0, 101.0, 102.0),
        (102.0, 110.0, 101.5, 108.5),
        (108.5, 107.0, 100.5, 101.0),
        (101.0, 104.0, 100.0, 103.0),
    ]
    candles: List[Dict[str, Any]] = []
    for index, (o, h, l, c) in enumerate(specs):
        ts = int((base + timedelta(minutes=15 * index)).timestamp() * 1000)
        candles.append(_make_candle(ts, o, h, l, c))
    return candles


def _install_common_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candles_15m: List[Dict[str, Any]],
    candles_1d: List[Dict[str, Any]],
) -> None:
    monkeypatch.setattr(
        inspection,
        "build_htf_section",
        lambda *args, **kwargs: (
            {"candles": {"15m": candles_15m, "1d": candles_1d}},
            {"timeframes": {}},
        ),
    )
    monkeypatch.setattr(inspection, "build_profile_package", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(
        inspection,
        "detect_zones",
        lambda *args, **kwargs: {
            "zones": {
                "fvg": [],
                "ob": [],
                "mb": [],
                "bb": [],
                "rb": [],
                "pb": [],
                "sr": [],
                "profile_levels": [],
            },
            "meta": {},
        },
    )
    monkeypatch.setattr(
        inspection,
        "resolve_profile_config",
        lambda sym, meta: {
            "preset": None,
            "preset_payload": None,
            "preset_required": False,
            "target_tf_key": "15m",
            "tick_size": None,
        },
    )


@pytest.mark.parametrize(
    "symbol, expected_tick",
    [
        ("BTCUSDT", 0.1),
        ("btcusdt", 0.1),
        ("BTCUSDTPERP", 0.1),
        ("BTCUSDT_PERP", 0.1),
        ("BTCUSDT:BINANCE", 0.1),
        ("BINANCE:BTCUSDT", 0.1),
        ("ETHUSDT", 0.01),
        ("SOLUSDT", 0.001),
    ],
)
def test_inspection_liquidity_uses_normalised_tick_size(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    expected_tick: float,
) -> None:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    candles_15m = _sample_candles(base)
    candles_1d = [
        _make_candle(int((base - timedelta(days=1)).timestamp() * 1000), 100.0, 105.0, 95.0, 101.0),
        _make_candle(int(base.timestamp() * 1000), 101.0, 106.0, 96.0, 104.0),
    ]

    _install_common_stubs(monkeypatch, candles_15m=candles_15m, candles_1d=candles_1d)

    selection = {"start": candles_15m[0]["t"], "end": candles_15m[-1]["t"]}
    snapshot = {
        "id": "snap-hft",
        "symbol": symbol,
        "frames": {"15m": {"candles": candles_15m}, "1d": {"candles": candles_1d}},
        "selection": selection,
        "agg_trades": {"status": "unavailable"},
        "meta": {
            "liquidity": {"r_ticks": 5, "swing_window": 1, "lookback": 20},
        },
    }

    payload = inspection.build_inspection_payload(snapshot)
    liquidity = payload["DATA"]["liquidity"]
    eqh_levels = liquidity["eqh"]
    assert eqh_levels, "Expected EQH levels to be detected for fallback tick size"

    tolerance_per_tick = [level["tolerance"] / 5 for level in eqh_levels if level.get("tolerance")]
    assert tolerance_per_tick, "Liquidity levels should report tolerance values"
    assert any(math.isclose(value, expected_tick, rel_tol=1e-9, abs_tol=1e-9) for value in tolerance_per_tick)

    diagnostics = payload["DIAGNOSTICS"].get("liquidity")
    assert isinstance(diagnostics, dict)
    summary = diagnostics.get("summary")
    assert isinstance(summary, dict)
    assert summary.get("eqh") >= 0
    tick_diag = diagnostics.get("tick_size")
    assert isinstance(tick_diag, dict)
    symbol_map = {0.1: "BTCUSDT", 0.01: "ETHUSDT", 0.001: "SOLUSDT"}
    assert tick_diag.get("normalized_symbol") == symbol_map[expected_tick]
    assert math.isclose(tick_diag.get("value", 0.0), expected_tick, rel_tol=1e-9, abs_tol=1e-9)
    assert tick_diag.get("source") == "hardcoded"
    levels_diag = diagnostics.get("levels")
    assert isinstance(levels_diag, dict)
    tf_diag = levels_diag.get("15m")
    assert isinstance(tf_diag, dict)
    eqh_diag = tf_diag.get("eqh")
    assert isinstance(eqh_diag, dict)
    assert "pairs_within_tol_before_cluster" in eqh_diag
    assert "sample_pairs_top10" in eqh_diag


def test_inspection_liquidity_prefers_hardcoded_over_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    candles_15m = _sample_candles(base)
    candles_1d = [
        _make_candle(int((base - timedelta(days=1)).timestamp() * 1000), 100.0, 105.0, 95.0, 101.0),
        _make_candle(int(base.timestamp() * 1000), 101.0, 106.0, 96.0, 104.0),
    ]

    _install_common_stubs(monkeypatch, candles_15m=candles_15m, candles_1d=candles_1d)

    snapshot = {
        "id": "snap-exchange",
        "symbol": "ETHUSDT",
        "frames": {"15m": {"candles": candles_15m}, "1d": {"candles": candles_1d}},
        "selection": {"start": candles_15m[0]["t"], "end": candles_15m[-1]["t"]},
        "agg_trades": {"status": "unavailable"},
        "meta": {
            "liquidity": {"r_ticks": 5, "swing_window": 1, "lookback": 20},
            "exchange_info": {
                "symbols": [
                    {
                        "symbol": "ETHUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.05"},
                        ],
                    }
                ]
            },
        },
    }

    payload = inspection.build_inspection_payload(snapshot)
    diagnostics = payload["DIAGNOSTICS"]["liquidity"]
    assert diagnostics["config"]["tick_size"] == pytest.approx(0.01)
    tick_diag = diagnostics.get("tick_size")
    assert isinstance(tick_diag, dict)
    assert tick_diag.get("normalized_symbol") == "ETHUSDT"
    assert tick_diag.get("source") == "hardcoded"
    assert tick_diag.get("value") == pytest.approx(0.01)
    eqh_levels = payload["DATA"]["liquidity"]["eqh"]
    assert eqh_levels
    tolerances = {level["tolerance"] for level in eqh_levels if level.get("tolerance")}
    assert tolerances, "Expected tolerances to be reported"
    assert all(math.isclose(tol, max(5 * 0.01, 0.01)) for tol in tolerances)
