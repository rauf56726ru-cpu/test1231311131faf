"""Synthetic derivatives analytics built from OHLCV context."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, MutableMapping

from .ohlcv import fetch_ohlcv


@dataclass(slots=True)
class DerivativeRow:
    """Represent a single derivatives sample."""

    timestamp: datetime
    open_interest: float
    funding_rate: float
    liq_long: float
    liq_short: float
    basis_bps: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "t": self.timestamp.isoformat().replace("+00:00", "Z"),
            "oi": round(self.open_interest, 2),
            "funding": round(self.funding_rate, 6),
            "liq_long": round(self.liq_long, 2),
            "liq_short": round(self.liq_short, 2),
            "basis_bps": round(self.basis_bps, 3),
        }


async def fetch_derivatives(
    symbol: str,
    window_hours: int,
    *,
    cache: MutableMapping[str, Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """Generate derivatives statistics using OHLCV series."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    lookback_days = max(1, (window_hours + 23) // 24)
    ohlcv_payload = await fetch_ohlcv(symbol_clean, "1m", lookback_days, cache=cache)
    candles = ohlcv_payload.get("candles", [])
    if not isinstance(candles, list) or not candles:
        raise ValueError("OHLCV candles not available for derivatives computation")

    hourly_rows: Dict[int, List[Mapping[str, object]]] = {}
    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        ts_text = candle.get("t")
        try:
            ts = datetime.fromisoformat(str(ts_text).replace("Z", "+00:00"))
        except ValueError:
            continue
        bucket = int(ts.timestamp() // 3600)
        hourly_rows.setdefault(bucket, []).append(candle)

    buckets = sorted(bucket for bucket in hourly_rows.keys())
    if not buckets:
        return []

    rows: List[DerivativeRow] = []
    for bucket in buckets[-window_hours:]:
        candles_in_bucket = hourly_rows[bucket]
        mid_ts = datetime.fromtimestamp(bucket * 3600, tz=timezone.utc)
        closes = [float(candle.get("c", 0.0)) for candle in candles_in_bucket]
        volumes = [float(candle.get("v", 0.0)) for candle in candles_in_bucket]
        if not closes or not volumes:
            continue
        avg_close = sum(closes) / len(closes)
        avg_volume = sum(volumes) / len(volumes)
        open_interest = max(avg_close * avg_volume * 10, 1e6)
        funding_rate = (avg_close - min(closes)) / avg_close / 24
        basis_bps = ((max(closes) - avg_close) / avg_close) * 10_000
        liq_long = 0.0
        liq_short = 0.0
        for candle in candles_in_bucket:
            high = float(candle.get("h", 0.0))
            low = float(candle.get("l", 0.0))
            close = float(candle.get("c", 0.0))
            liq_long += max(0.0, high - close)
            liq_short += max(0.0, close - low)
        rows.append(
            DerivativeRow(
                timestamp=mid_ts,
                open_interest=open_interest,
                funding_rate=funding_rate,
                liq_long=liq_long,
                liq_short=liq_short,
                basis_bps=basis_bps,
            )
        )

    return [row.as_dict() for row in rows]
