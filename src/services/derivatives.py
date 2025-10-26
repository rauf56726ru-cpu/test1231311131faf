"""Derivatives metrics sourced from Binance Vision archives."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timezone, timedelta, date
from typing import Any, Dict, List, MutableMapping, Sequence, Mapping

from .binance_vision import DATASET_METRICS, fetch_dataset
from .vision_store import get_store

MS_IN_HOUR = 3_600_000
LOGGER = logging.getLogger(__name__)


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


def _bucket_hour(ts_ms: int) -> int:
    return (ts_ms // MS_IN_HOUR) * MS_IN_HOUR


def _ensure_range(start_ms: int, end_ms: int) -> Sequence[int]:
    if end_ms <= start_ms:
        return []
    buckets = []
    cursor = _bucket_hour(start_ms)
    end_bucket = _bucket_hour(end_ms)
    while cursor <= end_bucket:
        buckets.append(cursor)
        cursor += MS_IN_HOUR
    return buckets


async def fetch_derivatives(
    symbol: str,
    window_hours: int,
    *,
    cache: MutableMapping[str, Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """Aggregate derivatives metrics using Binance Vision datasets."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - max(1, window_hours) * MS_IN_HOUR
    store = get_store()

    funding_rows, oi_rows, liq_rows = await asyncio.gather(
        asyncio.to_thread(store.fetch_funding_rates, symbol_clean, start_ms, now_ms),
        asyncio.to_thread(store.fetch_open_interest, symbol_clean, start_ms, now_ms),
        asyncio.to_thread(store.fetch_liquidations, symbol_clean, start_ms, now_ms),
    )

    if not funding_rows or not liq_rows or not oi_rows:
        metrics_rows = await asyncio.to_thread(store.fetch_metrics, symbol_clean, start_ms, now_ms)
        if not metrics_rows:
            try:
                metrics_rows = await _fetch_metrics(symbol_clean, start_ms, now_ms, store)
            except Exception as exc:
                LOGGER.warning("Metrics fallback failed for %s: %s", symbol_clean, exc)
                metrics_rows = []

        if metrics_rows:
            if not oi_rows:
                oi_rows = metrics_rows
            if not funding_rows:
                funding_rows = _convert_metrics_to_funding(metrics_rows)
            if not liq_rows:
                liq_rows = _convert_metrics_to_liquidations(metrics_rows)

    if not funding_rows and not oi_rows and not liq_rows:
        return []

    buckets = {bucket: {"funding": [], "mark": [], "oi": None, "liq_long": 0.0, "liq_short": 0.0} for bucket in _ensure_range(start_ms, now_ms)}

    for row in funding_rows:
        ts = int(row["ts"])
        bucket = _bucket_hour(ts)
        if bucket not in buckets:
            continue
        buckets[bucket]["funding"].append(float(row["funding_rate"]))
        mark_price = row.get("mark_price")
        if mark_price is not None:
            buckets[bucket]["mark"].append(float(mark_price))

    for row in oi_rows:
        ts = int(row["ts"])
        bucket = _bucket_hour(ts)
        if bucket not in buckets:
            continue
        buckets[bucket]["oi"] = float(row["open_interest"])

    for row in liq_rows:
        ts = int(row["ts"])
        bucket = _bucket_hour(ts)
        if bucket not in buckets:
            continue
        value = row.get("notional")
        if value is None or not value:
            value = row.get("qty") or 0.0
        amount = float(value)
        side = (row.get("side") or "").lower()
        if side == "buy":
            buckets[bucket]["liq_short"] += amount
        elif side == "sell":
            buckets[bucket]["liq_long"] += amount
        else:
            buckets[bucket]["liq_long"] += amount / 2.0
            buckets[bucket]["liq_short"] += amount / 2.0

    rows: List[DerivativeRow] = []
    for bucket in sorted(buckets.keys())[-window_hours:]:
        entry = buckets[bucket]
        funding_series = entry["funding"]
        if funding_series:
            funding_rate = sum(funding_series) / len(funding_series)
        else:
            funding_rate = 0.0
        open_interest = entry["oi"] if entry["oi"] is not None else 0.0
        mark_series = entry["mark"]
        if mark_series:
            mid_price = sum(mark_series) / len(mark_series)
            if mid_price > 0:
                basis_bps = ((max(mark_series) - min(mark_series)) / mid_price) * 10_000
            else:
                basis_bps = 0.0
        else:
            basis_bps = funding_rate * 24 * 10_000
        rows.append(
            DerivativeRow(
                timestamp=datetime.fromtimestamp(bucket / 1000, tz=timezone.utc),
                open_interest=open_interest,
                funding_rate=funding_rate,
                liq_long=entry["liq_long"],
                liq_short=entry["liq_short"],
                basis_bps=basis_bps,
            )
        )

    return [row.as_dict() for row in rows if any((row.open_interest, row.funding_rate, row.liq_long, row.liq_short))]


def _iter_metric_days(start_ms: int, end_ms: int) -> List[date]:
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    start_day = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date()
    end_day = datetime.fromtimestamp(end_ms / 1000, tz=UTC).date()
    today = datetime.now(UTC).date()
    if end_day >= today:
        end_day = today - timedelta(days=1)
    if end_day < start_day:
        end_day = start_day
    days: List[date] = []
    cursor = start_day
    while cursor <= end_day:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


async def _fetch_metrics(symbol: str, start_ms: int, end_ms: int, store) -> List[Dict[str, Any]]:
    days = _iter_metric_days(start_ms, end_ms)
    if not days:
        return []

    records: List[Dict[str, Any]] = []
    for day in days:
        batch = await fetch_dataset(DATASET_METRICS, symbol=symbol, day=day)
        if batch is None or not batch.records:
            continue
        records.extend(batch.records)
        await asyncio.to_thread(store.upsert_metrics, symbol, day.isoformat(), batch.records)
    return records


def _convert_metrics_to_funding(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    funding: List[Dict[str, Any]] = []
    for row in rows:
        ts = row.get("ts")
        if ts is None:
            continue
        count = _to_float(row.get("top_trader_count"))
        total = _to_float(row.get("top_trader_sum"))
        funding_rate = 0.0
        if count and total is not None and count > 0:
            funding_rate = (total / count) - 1.0
        funding.append({"ts": int(ts), "funding_rate": funding_rate, "mark_price": None})
    return funding


def _convert_metrics_to_liquidations(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    liqs: List[Dict[str, Any]] = []
    for row in rows:
        ts = row.get("ts")
        if ts is None:
            continue
        ratio = _to_float(row.get("taker_vol_ratio")) or 0.0
        if ratio >= 0:
            liq_long = ratio
            liq_short = 0.0
        else:
            liq_long = 0.0
            liq_short = -ratio
        liqs.append(
            {
                "ts": int(ts),
                "price": None,
                "qty": abs(ratio),
                "side": "sell" if ratio >= 0 else "buy",
                "notional": abs(ratio),
                "liq_long": liq_long,
                "liq_short": liq_short,
            }
        )
    return liqs


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_rows_by_day(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        ts = row.get("ts")
        if ts is None:
            continue
        try:
            day_label = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            continue
        buckets.setdefault(day_label, []).append(dict(row))
    return buckets


async def _persist_funding(store, symbol: str, rows: Sequence[Mapping[str, Any]]) -> None:
    day_map = _group_rows_by_day(rows)
    if not day_map:
        return
    await asyncio.gather(
        *[
            asyncio.to_thread(store.upsert_funding_rates, symbol, day, payload)
            for day, payload in day_map.items()
        ]
    )


async def _persist_liquidations(store, symbol: str, rows: Sequence[Mapping[str, Any]]) -> None:
    day_map = _group_rows_by_day(rows)
    if not day_map:
        return
    await asyncio.gather(
        *[
            asyncio.to_thread(store.upsert_liquidations, symbol, day, payload)
            for day, payload in day_map.items()
        ]
    )
