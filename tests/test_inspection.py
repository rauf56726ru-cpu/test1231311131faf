"""Tests for inspection snapshot normalisation behaviour."""
from __future__ import annotations

from datetime import datetime, timezone

from src.services.inspection import build_inspection_payload


def test_inspection_respects_selection_window() -> None:
    interval_ms = 60_000
    base_ts = 1_700_000_000_000 - (1_700_000_000_000 % interval_ms)
    days = 3
    total_candles = days * 24 * 60

    candles = []
    for idx in range(total_candles):
        ts = base_ts + idx * interval_ms
        price = 100.0 + idx * 0.1
        candles.append(
            {
                "t": ts,
                "o": price,
                "h": price + 1.0,
                "l": price - 1.0,
                "c": price + 0.5,
                "v": 1.0,
            }
        )

    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]}
    snapshot = {
        "id": "test",
        "symbol": "BTCUSDT",
        "tf": "1m",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "frames": {
            "1m": {
                "tf": "1m",
                "candles": candles,
            }
        },
        "selection": selection,
    }

    payload = build_inspection_payload(snapshot)
    exported = payload["DATA"]["frames"]["1m"]["candles"]

    assert len(exported) == total_candles
    assert exported[0]["t"] == selection["start"]
    assert exported[-1]["t"] == selection["end"]
