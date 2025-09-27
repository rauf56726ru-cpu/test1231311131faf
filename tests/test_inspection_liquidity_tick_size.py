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


@pytest.mark.parametrize(
    "symbol, expected_tick",
    [
        ("BTCUSDT", 0.1),
        ("ETHUSDT", 0.01),
        ("SOLUSDT", 0.001),
    ],
)
def test_inspection_liquidity_uses_hardcoded_tick_size(monkeypatch: pytest.MonkeyPatch, symbol: str, expected_tick: float) -> None:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    candles_15m = _sample_candles(base)
    candles_1d = [
        _make_candle(int((base - timedelta(days=1)).timestamp() * 1000), 100.0, 105.0, 95.0, 101.0),
        _make_candle(int(base.timestamp() * 1000), 101.0, 106.0, 96.0, 104.0),
    ]

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
        lambda *args, **kwargs: {"symbol": symbol, "zones": {"fvg": [], "ob": [], "inducement": [], "cisd": []}},
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
