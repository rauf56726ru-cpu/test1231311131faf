"""Compute metrics for the last closed UTC trading session."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import pandas as pd

from .zones_72h import Zone

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionMetrics:
    symbol: str
    session_start: int
    session_end: int
    o: float
    h: float
    l: float
    c: float
    range: float
    body_ratio: float
    volume: float
    vwap: float
    taker_buy_delta: float
    atr: float
    zones_total: int
    zone_touches: int
    zones_mitigated: List[str]
    zone_ids: List[str]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "open": self.o,
            "high": self.h,
            "low": self.l,
            "close": self.c,
            "range": self.range,
            "body_ratio": self.body_ratio,
            "volume": self.volume,
            "vwap": self.vwap,
            "taker_buy_delta": self.taker_buy_delta,
            "atr": self.atr,
            "zones_total": self.zones_total,
            "zone_touches": self.zone_touches,
            "zones_mitigated": list(self.zones_mitigated),
            "zone_ids": list(self.zone_ids),
        }


def last_closed_session_bounds(reference_ms: int | None = None) -> tuple[int, int]:
    now = datetime.fromtimestamp((reference_ms or int(time.time() * 1000)) / 1000, tz=UTC)
    start = (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1))
    end = start + timedelta(days=1) - timedelta(milliseconds=1 * 60_000)
    start_ms = int(start.timestamp() * 1000)
    end_ms = start_ms + (24 * 60 * 60_000) - 60_000
    return start_ms, end_ms


def compute_session_metrics(
    frame: pd.DataFrame,
    *,
    symbol: str,
    session_start_ms: int,
    session_end_ms: int,
    zones: Sequence[Zone] | None = None,
) -> SessionMetrics | None:
    if frame.empty:
        return None

    scoped = frame[(frame["ts_open"] >= session_start_ms) & (frame["ts_open"] <= session_end_ms)].copy()
    if scoped.empty:
        return None

    scoped.sort_values("ts_open", inplace=True)
    first = scoped.iloc[0]
    last = scoped.iloc[-1]
    high = float(scoped["high"].max())
    low = float(scoped["low"].min())
    volume = float(scoped["volume"].sum())
    taker_buy_delta = float(scoped.get("taker_buy_vol", pd.Series([0.0])).sum())

    price_range = high - low
    body = abs(float(last["close"]) - float(first["open"]))
    body_ratio = (body / price_range) if price_range > 0 else 0.0

    vwap_value = _compute_vwap(scoped)
    atr_value = _compute_atr(scoped)

    zone_ids: list[str] = []
    mitigated: list[str] = []
    touches = 0
    zones_total = 0
    if zones:
        for zone in zones:
            if zone.symbol != symbol:
                continue
            zones_total += 1
            zone_ids.append(zone.id)
            if zone.filled_pct > 0:
                mitigated.append(zone.id)
            touches += int(zone.touches)

    return SessionMetrics(
        symbol=symbol,
        session_start=session_start_ms,
        session_end=session_end_ms,
        o=float(first["open"]),
        h=high,
        l=low,
        c=float(last["close"]),
        range=price_range,
        body_ratio=body_ratio,
        volume=volume,
        vwap=vwap_value,
        taker_buy_delta=taker_buy_delta,
        atr=atr_value,
        zones_total=zones_total,
        zone_touches=touches,
        zones_mitigated=mitigated,
        zone_ids=zone_ids,
    )


def export_sessions_jsonl(metrics: Iterable[SessionMetrics], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in metrics:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    LOGGER.info("session_last.export", extra={"path": str(output_path), "rows": count})
    return count


def _compute_vwap(scoped: pd.DataFrame) -> float:
    typical = (scoped["high"] + scoped["low"] + scoped["close"]) / 3.0
    volume = scoped["volume"]
    numerator = (typical * volume).sum()
    denominator = volume.sum()
    if denominator <= 0:
        return float(scoped["close"].iloc[-1])
    return float(numerator / denominator)


def _compute_atr(scoped: pd.DataFrame, period: int = 14) -> float:
    high = scoped["high"]
    low = scoped["low"]
    close = scoped["close"]
    previous_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=min(3, period)).mean()
    last = atr.iloc[-1]
    if math.isfinite(last):
        return float(last)
    valid = atr.dropna()
    return float(valid.iloc[-1]) if not valid.empty else float(tr.mean())


__all__ = [
    "SessionMetrics",
    "last_closed_session_bounds",
    "compute_session_metrics",
    "export_sessions_jsonl",
]
