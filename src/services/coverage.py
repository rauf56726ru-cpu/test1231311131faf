"""Coverage metrics helpers for minute time series."""
from __future__ import annotations

from typing import Mapping, Sequence, Tuple

EXPECTED_MINUTES = 72 * 60


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_coverage(minutes: Sequence[Mapping[str, object]]) -> Tuple[dict[str, float], list[int]]:
    expected = EXPECTED_MINUTES if EXPECTED_MINUTES > 0 else max(len(minutes), 1)
    if not minutes:
        return (
            {
                "klines_pct": 0.0,
                "aggtrades_pct": 0.0,
                "recon_mismatch_pct": 0.0,
                "vwap_oob_pct": 0.0,
            },
            [],
        )

    valid_ohlcv = 0
    agg_present = 0
    recon_flags = 0
    vwap_oob_flags = 0
    silent_minutes: list[int] = []

    for entry in minutes:
        ts = int(entry.get("ts_min", 0))
        volume_value = entry.get("volume")
        gap_mask = bool(entry.get("gap_mask"))
        if not gap_mask and volume_value is not None:
            valid_ohlcv += 1

        vol = _as_float(entry.get("vol"))
        trades_cnt = entry.get("trades_cnt")
        if vol > 0.0 or (trades_cnt is not None and int(trades_cnt) > 0):
            agg_present += 1
        else:
            silent_minutes.append(ts)

        if entry.get("recon_flag"):
            recon_flags += 1
        if entry.get("vwap_oob"):
            vwap_oob_flags += 1

    metrics = {
        "klines_pct": round((valid_ohlcv / expected) * 100.0, 3),
        "aggtrades_pct": round((agg_present / expected) * 100.0, 3),
        "recon_mismatch_pct": round((recon_flags / expected) * 100.0, 3),
        "vwap_oob_pct": round((vwap_oob_flags / expected) * 100.0, 3),
    }

    return metrics, silent_minutes


__all__ = ["compute_coverage"]
