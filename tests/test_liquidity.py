from __future__ import annotations

from datetime import datetime, timezone, timedelta

from src.services.liquidity import build_liquidity_snapshot


UTC = timezone.utc


def _make_candle(ts: int, o: float, h: float, l: float, c: float) -> dict[str, float]:
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": 1.0}


def test_liquidity_detects_equal_levels_and_sweeps() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    candles_15m = [
        _make_candle(int((base + timedelta(minutes=15 * idx)).timestamp() * 1000), *values)
        for idx, values in enumerate(
            [
                (100.0, 100.5, 97.0, 99.0),
                (99.0, 102.0, 98.0, 101.0),
                (101.0, 105.0, 100.0, 104.0),  # swing high
                (104.0, 104.2, 95.0, 96.0),   # swing low
                (96.0, 104.95, 95.3, 103.0),  # swing high close to previous
                (103.0, 103.5, 95.1, 96.5),   # swing low close to previous
                (96.5, 99.0, 95.8, 98.0),
                (98.0, 99.5, 96.7, 99.0),
                (99.0, 105.3, 99.0, 104.2),   # sweep top candle
                (104.2, 105.0, 94.8, 95.6),   # sweep bottom candle
            ]
        )
    ]

    day_base = datetime(2023, 12, 31, tzinfo=UTC)
    candles_1d = [
        _make_candle(int((day_base + timedelta(days=offset)).timestamp() * 1000), *values)
        for offset, values in enumerate(
            [
                (100.0, 105.0, 95.0, 102.0),
                (102.0, 111.0, 91.0, 109.0),  # previous day
                (109.0, 112.0, 100.0, 105.0),
            ]
        )
    ]

    selection_end = int((base + timedelta(days=2, hours=12)).timestamp() * 1000)
    frames = {"15m": {"candles": candles_15m}, "1d": {"candles": candles_1d}}

    liquidity = build_liquidity_snapshot(
        frames,
        tick_size=0.1,
        selection={"end": selection_end},
        config={
            "r_ticks": 2,
            "lookback": 10,
            "swing_window": 1,
            "atr_period": 3,
            "sweep_atr_multiplier": 1.5,
        },
    )

    expected_day = candles_1d[2]
    assert liquidity["pdh"] == {
        "t": expected_day["t"],
        "price": expected_day["h"],
    }
    assert liquidity["pdl"] == {
        "t": expected_day["t"],
        "price": expected_day["l"],
    }

    eqh_levels = liquidity["eqh"]
    eql_levels = liquidity["eql"]
    assert len(eqh_levels) == 1
    assert len(eql_levels) == 1

    eqh = eqh_levels[0]
    eql = eql_levels[0]
    assert eqh["tf"] == "15m"
    assert eql["tf"] == "15m"
    eqh_swings = set(eqh["swings"])
    eql_swings = set(eql["swings"])
    assert {candles_15m[2]["t"], candles_15m[4]["t"]}.issubset(eqh_swings)
    assert {candles_15m[3]["t"], candles_15m[5]["t"]}.issubset(eql_swings)
    assert abs(eqh["price"] - 105.0) < 0.2
    assert abs(eql["price"] - 95.05) < 0.2
    assert eqh["tolerance"] == 0.2
    assert eql["tolerance"] == 0.2

    sweep_types = {event["type"] for event in liquidity["sweeps"]}
    level_types = {event["level_type"] for event in liquidity["sweeps"]}
    assert sweep_types == {"sweep_top", "sweep_bottom"}
    assert level_types == {"eqh", "eql"}
    for event in liquidity["sweeps"]:
        assert event["atr_tolerance"] >= 0


def test_liquidity_respects_atr_tolerance() -> None:
    base = datetime(2024, 5, 1, tzinfo=UTC)
    highs = [100.2, 100.4, 100.6, 100.3, 100.62, 100.35, 101.1]
    lows = [99.8, 100.0, 100.2, 100.1, 100.3, 100.2, 99.5]
    closes = [100.0, 100.2, 100.5, 100.2, 100.58, 100.25, 100.0]
    candles = []
    for idx in range(len(highs)):
        ts = int((base + timedelta(minutes=15 * idx)).timestamp() * 1000)
        candles.append(_make_candle(ts, closes[idx], highs[idx], lows[idx], closes[idx]))

    frames = {"15m": {"candles": candles}}
    liquidity = build_liquidity_snapshot(
        frames,
        tick_size=0.01,
        config={
            "r_ticks": 2,
            "lookback": 10,
            "swing_window": 1,
            "atr_period": 3,
            "sweep_atr_multiplier": 0.1,
        },
    )

    # Equal highs should still be detected, but sweep should be filtered out.
    assert liquidity["eqh"]
    assert liquidity["sweeps"] == []

