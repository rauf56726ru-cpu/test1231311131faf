from __future__ import annotations

from datetime import datetime, timezone, timedelta

from src.services.check_all_datas import build_equal_liquidity_levels
from src.services.liquidity import build_liquidity_snapshot


UTC = timezone.utc


def _make_candle(ts: int, o: float, h: float, l: float, c: float) -> dict[str, float]:
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": 1.0}


def _expand_block_to_minutes(start: datetime, o: float, h: float, l: float, c: float) -> list[dict[str, float]]:
    candles: list[dict[str, float]] = []
    for minute in range(15):
        moment = start + timedelta(minutes=minute)
        ts = int(moment.timestamp() * 1000)
        open_price = o if minute == 0 else c
        close_price = c
        high_price = h if minute == 0 else max(c, o)
        low_price = l if minute == 0 else min(c, o)
        candles.append({
            "t": ts,
            "o": open_price,
            "h": high_price,
            "l": low_price,
            "c": close_price,
            "v": 1.0,
        })
    return candles


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
                (99.0, 105.6, 99.0, 104.2),   # sweep top candle
                (104.2, 105.0, 94.4, 95.6),   # sweep bottom candle
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
        symbol="BTCUSDT",
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

    diagnostics = liquidity.get("diagnostics")
    assert isinstance(diagnostics, dict)
    summary = diagnostics.get("summary")
    assert isinstance(summary, dict)
    assert summary.get("eqh") == len(eqh_levels)
    assert summary.get("eqh_filtered") == len(eqh_levels)
    assert summary.get("eqh_raw") >= len(eqh_levels)
    assert isinstance(summary.get("eqh_reasons"), list)
    assert summary.get("has_degraded") is False
    metrics_block = diagnostics.get("metrics")
    assert isinstance(metrics_block, dict)
    assert metrics_block.get("raw_candidates_eqh") >= len(eqh_levels)
    assert metrics_block.get("tick_size_unresolved") == 0
    levels_diag = diagnostics.get("levels")
    assert isinstance(levels_diag, dict)
    tf_diag = levels_diag.get("15m")
    assert isinstance(tf_diag, dict)
    eqh_diag = tf_diag.get("eqh")
    eql_diag = tf_diag.get("eql")
    assert isinstance(eqh_diag, dict)
    assert isinstance(eql_diag, dict)
    assert eqh_diag.get("cluster_count") == len(eqh_levels)
    assert eql_diag.get("cluster_count") == len(eql_levels)
    assert eqh_diag.get("pairs_within_tol_before_cluster") == eqh_diag.get("pairs_within_tol")
    assert eql_diag.get("pairs_within_tol_before_cluster") == eql_diag.get("pairs_within_tol")
    assert isinstance(eqh_diag.get("sample_pairs_top10"), list)
    assert isinstance(eql_diag.get("sample_pairs_top10"), list)
    assert eqh_diag.get("raw_count") >= eqh_diag.get("filtered_count")
    assert eql_diag.get("raw_count") >= eql_diag.get("filtered_count")
    assert isinstance(eqh_diag.get("reason_histogram"), dict)
    assert isinstance(eql_diag.get("reason_histogram"), dict)
    candidates = liquidity.get("candidates")
    assert isinstance(candidates, dict)
    eqh_candidates = candidates.get("eqh")
    assert isinstance(eqh_candidates, list)
    assert any(not entry.get("rejected") for entry in eqh_candidates)
    eql_candidates = candidates.get("eql")
    assert isinstance(eql_candidates, list)
    assert any(not entry.get("rejected") for entry in eql_candidates)
    tick_diag = diagnostics.get("tick_size")
    assert isinstance(tick_diag, dict)
    assert tick_diag.get("normalized_symbol") == "BTCUSDT"
    assert tick_diag.get("source") == "param"
    assert tick_diag.get("value") == 0.1


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
        symbol="ETHUSDT",
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

    diagnostics = liquidity.get("diagnostics")
    assert isinstance(diagnostics, dict)
    summary = diagnostics.get("summary")
    assert isinstance(summary, dict)
    assert summary.get("sweeps") == 0
    assert summary.get("eqh_raw") >= summary.get("eqh_filtered")
    metrics_block = diagnostics.get("metrics")
    assert isinstance(metrics_block, dict)
    assert metrics_block.get("raw_candidates_eqh") >= len(liquidity["eqh"])
    candidates = liquidity.get("candidates")
    assert isinstance(candidates, dict)
    assert isinstance(candidates.get("eqh"), list)



def test_liquidity_aggregates_minute_seed() -> None:
    base = datetime(2024, 2, 1, tzinfo=UTC)
    minute_candles: list[dict[str, float]] = []
    fifteen_specs = [
        (100.0, 100.5, 97.0, 99.0),
        (99.0, 102.0, 98.0, 101.0),
        (101.0, 105.0, 100.0, 104.0),
        (104.0, 104.2, 95.0, 96.0),
        (96.0, 104.95, 95.3, 103.0),
        (103.0, 103.5, 95.1, 96.5),
        (96.5, 99.0, 95.8, 98.0),
        (98.0, 99.5, 96.7, 99.0),
        (99.0, 105.6, 99.0, 104.2),
        (104.2, 105.0, 94.4, 95.6),
    ]
    for idx, spec in enumerate(fifteen_specs):
        block_start = base + timedelta(minutes=15 * idx)
        minute_candles.extend(_expand_block_to_minutes(block_start, *spec))

    day_base = datetime(2024, 1, 31, tzinfo=UTC)
    candles_1d = [
        _make_candle(int((day_base + timedelta(days=offset)).timestamp() * 1000), *values)
        for offset, values in enumerate(
            [
                (100.0, 105.0, 95.0, 102.0),
                (102.0, 111.0, 91.0, 109.0),
                (109.0, 112.0, 100.0, 105.0),
            ]
        )
    ]

    selection_end = int((base + timedelta(days=2)).timestamp() * 1000)
    frames = {"1m": {"candles": minute_candles}, "1d": {"candles": candles_1d}}

    liquidity = build_liquidity_snapshot(
        frames,
        symbol="BTCUSDT",
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

    assert liquidity["eqh"]
    assert liquidity["eql"]
    sweep_types = {event["type"] for event in liquidity["sweeps"]}
    assert "sweep_top" in sweep_types
    assert "sweep_bottom" in sweep_types

    diagnostics = liquidity.get("diagnostics")
    assert isinstance(diagnostics, dict)
    summary = diagnostics.get("summary")
    assert isinstance(summary, dict)
    assert summary.get("eqh") == len(liquidity["eqh"])
    assert isinstance(summary.get("degraded_timeframes"), list)


def test_liquidity_detects_pdh_pdl_sweeps_from_minute_seed() -> None:
    base = datetime(2024, 3, 2, tzinfo=UTC)
    minute_candles: list[dict[str, float]] = []
    blocks = [
        (200.0, 205.0, 198.0, 203.0),
        (203.0, 206.5, 200.0, 204.5),
        (204.5, 221.5, 203.5, 219.0),  # sweep top above PDH
        (219.0, 220.0, 179.0, 181.0),  # sweep bottom below PDL
        (181.0, 190.0, 180.0, 188.0),
    ]
    for idx, spec in enumerate(blocks):
        block_start = base + timedelta(minutes=15 * idx)
        minute_candles.extend(_expand_block_to_minutes(block_start, *spec))

    previous_day = datetime(2024, 3, 1, tzinfo=UTC)
    candles_1d = [
        _make_candle(int(previous_day.timestamp() * 1000), 195.0, 220.0, 180.0, 210.0),
        _make_candle(int((previous_day + timedelta(days=1)).timestamp() * 1000), 210.0, 212.0, 190.0, 200.0),
    ]

    selection_end = int((base + timedelta(hours=6)).timestamp() * 1000)
    frames = {"1m": {"candles": minute_candles}, "1d": {"candles": candles_1d}}

    liquidity = build_liquidity_snapshot(
        frames,
        symbol="BTCUSDT",
        tick_size=1.0,
        selection={"end": selection_end},
        config={
            "r_ticks": 2,
            "lookback": 10,
            "swing_window": 1,
            "atr_period": 3,
            "sweep_atr_multiplier": 2.0,
        },
    )

    sweeps = liquidity["sweeps"]
    assert sweeps
    level_types = {event["level_type"] for event in sweeps}
    assert level_types == {"pdh", "pdl"}
    types = {event["type"] for event in sweeps}
    assert types == {"sweep_top", "sweep_bottom"}


def test_build_equal_liquidity_levels_detects_second_touch() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    highs = [95.0, 98.0, 101.0, 100.0, 99.0, 97.0, 96.0, 97.0, 101.02, 98.0, 97.0, 96.0, 95.0]
    lows = [93.0, 92.0, 91.0, 90.0, 88.0, 89.0, 90.0, 91.0, 92.0, 91.0, 88.02, 89.0, 90.0]
    candles: list[dict[str, float]] = []
    for idx, (high, low) in enumerate(zip(highs, lows)):
        moment = base + timedelta(minutes=15 * idx)
        ts = int(moment.timestamp() * 1000)
        mid = (high + low) / 2.0
        candles.append({"t": ts, "o": mid, "h": high, "l": low, "c": mid, "v": 1.0})

    levels = build_equal_liquidity_levels({"15m": candles})
    eqh_levels = levels["eqh"]
    eql_levels = levels["eql"]

    assert len(eqh_levels) == 1
    assert len(eql_levels) == 1

    eqh_entry = eqh_levels[0]
    eql_entry = eql_levels[0]

    expected_high_ts = (base + timedelta(minutes=15 * 8)).isoformat().replace("+00:00", "Z")
    expected_low_ts = (base + timedelta(minutes=15 * 10)).isoformat().replace("+00:00", "Z")
    assert eqh_entry["ts"] == expected_high_ts
    assert eql_entry["ts"] == expected_low_ts

    expected_high_price = (highs[2] + highs[8]) / 2.0
    expected_low_price = (lows[4] + lows[10]) / 2.0
    assert abs(eqh_entry["price"] - expected_high_price) < 1e-6
    assert abs(eql_entry["price"] - expected_low_price) < 1e-6
