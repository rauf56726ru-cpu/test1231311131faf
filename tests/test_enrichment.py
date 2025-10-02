"""Tests for inspection enrichment helpers."""

from __future__ import annotations

import pytest

from src.services import enrichment


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_enrichment_adds_daily_and_orderflow(monkeypatch: pytest.MonkeyPatch) -> None:
    base_ts = 1_700_000_000_000
    minute_candles = []
    for idx in range(60):
        open_price = 100.0 + idx * 0.1
        close_price = open_price + (0.2 if idx % 2 == 0 else -0.15)
        high_price = max(open_price, close_price) + 0.1
        low_price = min(open_price, close_price) - 0.1
        minute_candles.append(
            {
                "t": base_ts + idx * 60_000,
                "o": open_price,
                "h": high_price,
                "l": low_price,
                "c": close_price,
                "v": 20.0 + idx,
            }
        )

    async def _fake_fetch_recent(symbol: str, interval: str, limit: int):  # type: ignore[override]
        if interval == "1d":
            return [
                {
                    "t": base_ts - (3 - idx) * 86_400_000,
                    "o": 100.0 + idx,
                    "h": 101.0 + idx,
                    "l": 99.0 + idx,
                    "c": 100.5 + idx,
                    "v": 1000.0 + idx * 10,
                }
                for idx in range(3)
            ]
        if interval == "4h":
            return [
                {
                    "t": base_ts - (6 - idx) * 14_400_000,
                    "o": 100.0 + idx * 0.3,
                    "h": 100.6 + idx * 0.3,
                    "l": 99.4 + idx * 0.3,
                    "c": 100.2 + idx * 0.3,
                    "v": 400.0 + idx * 5,
                }
                for idx in range(6)
            ]
        if interval == "1h":
            return [
                {
                    "t": base_ts - (limit - idx) * 3_600_000,
                    "o": 100.0 + idx * 0.2,
                    "h": 100.4 + idx * 0.2,
                    "l": 99.6 + idx * 0.2,
                    "c": 100.1 + idx * 0.2,
                    "v": 200.0 + idx * 3,
                }
                for idx in range(limit)
            ]
        raise AssertionError(f"unexpected interval {interval}")

    monkeypatch.setattr(enrichment, "_fetch_recent", _fake_fetch_recent)

    snapshot = {
        "symbol": "BTCUSDT",
        "tf": "1m",
        "ohlcv": {"1m": {"candles": minute_candles}},
    }

    payload = await enrichment.enrich_inspection_snapshot(snapshot)

    ohlcv_block = payload["ohlcv"]["1d"]
    assert len(ohlcv_block["candles"]) == 3
    assert ohlcv_block["demand"] or ohlcv_block["supply"]

    delta_series = payload["orderflow"]["delta"]
    assert 0 < len(delta_series) <= enrichment.MAX_DELTA_BARS

    cvd_series = payload["orderflow"]["cvd"]
    assert len(cvd_series) == len(delta_series)

    footprint = payload["orderflow"]["footprint"]
    assert footprint, "synthetic footprint should be populated"

    assert payload["status"] == "ok"
