from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.services.smc import SMCConfig, detect_smc_blocks

BASE_TS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def make_hour_candle(idx: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> dict[str, float]:
    return {
        "t": BASE_TS + idx * 3_600_000,
        "o": float(o),
        "h": float(h),
        "l": float(l),
        "c": float(c),
        "v": float(v),
    }


def test_breaker_block_detected_after_bos_retest() -> None:
    candles = [
        make_hour_candle(0, 100.0, 102.0, 99.0, 99.0),
        make_hour_candle(1, 99.0, 100.0, 97.0, 98.0),
        make_hour_candle(2, 98.0, 104.0, 97.0, 103.0),
        make_hour_candle(3, 103.0, 104.0, 98.8, 99.2),
        make_hour_candle(4, 99.1, 99.8, 98.9, 99.4),
    ]

    ob_zones = [
        {
            "type": "supply",
            "range": [99.0, 101.0],
            "created_at": candles[0]["t"],
            "touches": 0,
            "tf": "1h",
        }
    ]
    structure_flags = [
        {"kind": "bos", "direction": "up", "t": candles[2]["t"], "close": 103.0}
    ]

    config = SMCConfig(min_block_size=0.5, ttl_bars=10)
    blocks = detect_smc_blocks(
        candles,
        structure_flags=structure_flags,
        ob_zones=ob_zones,
        liquidity_levels={},
        config=config,
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["kind"] == "bb"
    assert block["type"] == "demand"
    assert block["status"] == "tapped"
    assert block["range"][0] == pytest.approx(99.0)
    assert block["range"][1] == pytest.approx(100.0)


def test_mitigation_block_uses_unfilled_body() -> None:
    candles = [
        make_hour_candle(0, 100.0, 102.0, 99.5, 102.0),
        make_hour_candle(1, 103.0, 104.0, 100.5, 101.2),
        make_hour_candle(2, 101.0, 106.0, 99.0, 104.0),
        make_hour_candle(3, 104.0, 105.0, 99.4, 100.5),
        make_hour_candle(4, 100.3, 101.0, 99.8, 100.9),
    ]

    ob_zones = [
        {
            "type": "demand",
            "range": [100.0, 102.0],
            "created_at": candles[0]["t"],
            "touches": 0,
            "tf": "1h",
        }
    ]

    config = SMCConfig(min_block_size=0.5, ttl_bars=10)
    blocks = detect_smc_blocks(
        candles,
        structure_flags=[],
        ob_zones=ob_zones,
        liquidity_levels={},
        config=config,
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["kind"] == "mb"
    assert block["type"] == "demand"
    assert block["status"] == "tapped"
    assert block["range"][0] == pytest.approx(100.0)
    assert block["range"][1] == pytest.approx(101.2)


def test_reversal_block_after_liquidity_grab() -> None:
    candles = [
        make_hour_candle(0, 100.0, 102.0, 99.0, 101.0),
        make_hour_candle(1, 101.0, 102.0, 99.5, 100.0),
        make_hour_candle(2, 100.0, 101.5, 94.5, 96.0),
        make_hour_candle(3, 96.0, 100.0, 95.5, 99.0),
        make_hour_candle(4, 99.0, 100.5, 98.5, 100.1),
        make_hour_candle(5, 100.2, 110.0, 99.5, 108.0),
        make_hour_candle(6, 103.0, 105.0, 98.8, 99.6),
    ]

    structure_flags = [
        {"kind": "choch", "direction": "up", "t": candles[3]["t"]}
    ]
    liquidity = {"pdl": {"price": 95.0}}

    config = SMCConfig(min_block_size=0.5, displacement_factor=1.5, displacement_lookback=3, ttl_bars=10)
    blocks = detect_smc_blocks(
        candles,
        structure_flags=structure_flags,
        ob_zones=[],
        liquidity_levels=liquidity,
        config=config,
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["kind"] == "rb"
    assert block["type"] == "demand"
    assert block["status"] == "tapped"
    assert block["range"][0] == pytest.approx(99.0, rel=1e-3)
    assert block["range"][1] == pytest.approx(100.1, rel=1e-3)
    assert block["created_at"] == candles[5]["t"]
