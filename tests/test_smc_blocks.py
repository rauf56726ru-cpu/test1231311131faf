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


def make_15m_candle(idx: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> dict[str, float]:
    return {
        "t": BASE_TS + idx * 900_000,
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
    blocks, stats = detect_smc_blocks(
        candles,
        structure_flags=structure_flags,
        ob_zones=ob_zones,
        liquidity_levels={},
        config=config,
        atr=[5.0] * len(candles),
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["kind"] == "bb"
    assert block["block_type"] == "breaker block"
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
    blocks, stats = detect_smc_blocks(
        candles,
        structure_flags=[],
        ob_zones=ob_zones,
        liquidity_levels={},
        config=config,
        atr=[5.0] * len(candles),
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["kind"] == "mb"
    assert block["block_type"] == "mitigation block"
    assert block["type"] == "demand"
    assert block["status"] == "tapped"
    assert block["range"][0] == pytest.approx(100.0)
    assert block["range"][1] == pytest.approx(101.2)


def test_reversal_block_after_liquidity_grab() -> None:
    candles = [
        make_15m_candle(0, 103.0, 104.0, 102.0, 103.2),
        make_15m_candle(1, 103.1, 103.4, 101.2, 102.4),
        make_15m_candle(2, 101.8, 102.0, 99.0, 99.8),
        make_15m_candle(3, 100.2, 101.9, 100.0, 101.4),
        make_15m_candle(4, 101.6, 103.2, 100.8, 102.6),
        make_15m_candle(5, 102.7, 104.6, 101.6, 103.5),
        make_15m_candle(6, 103.4, 104.0, 102.2, 103.2),
        make_15m_candle(7, 102.6, 103.0, 101.2, 102.0),
        make_15m_candle(8, 101.4, 102.4, 99.0, 100.5),
        make_15m_candle(9, 100.6, 101.4, 99.6, 100.8),
        make_15m_candle(10, 100.8, 101.2, 99.3, 100.4),
        make_15m_candle(11, 100.4, 100.8, 98.8, 99.3),
        make_15m_candle(12, 99.2, 99.9, 99.1, 99.8),
        make_15m_candle(13, 99.3, 100.0, 99.25, 99.7),
        make_15m_candle(14, 99.9, 101.8, 99.8, 101.6),
        make_15m_candle(15, 101.5, 102.2, 100.9, 101.2),
    ]

    structure_flags = [
        {"kind": "choch", "direction": "up", "t": candles[13]["t"], "tf": "15m"}
    ]

    atr_values = [1.0] * len(candles)
    sigma_values = [0.1] * len(candles)
    config = SMCConfig(min_block_size=0.2, ttl_bars=20)
    blocks, stats = detect_smc_blocks(
        candles,
        timeframe="15m",
        structure_flags=structure_flags,
        ob_zones=[],
        liquidity_levels={},
        config=config,
        atr=atr_values,
        returns_sigma=sigma_values,
        tick_size=0.1,
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["kind"] == "rb"
    assert block["block_type"] == "reversal block"
    assert block["type"] == "demand"
    assert block["status"] == "fresh"
    assert stats["rb_raw_count"] == 1
    flow = stats.get("rb_flow", {})
    assert flow.get("eq_found", 0) >= 1
    assert flow.get("impulse", 0) >= 1
    assert not stats.get("base_fallback_used", False)
