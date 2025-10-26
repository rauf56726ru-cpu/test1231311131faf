"""Orchestration for ingesting Binance Vision archives into the local store."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import httpx

from .binance_vision import (
    DATASET_AGG_TRADES,
    DATASET_BOOK_DEPTH,
    DATASET_EXCHANGE_INFO,
    DATASET_FUNDING_RATE,
    DATASET_KLINES,
    DATASET_LIQ_ORDERS,
    DATASET_OPEN_INTEREST,
    DATASET_METRICS,
    VisionBatch,
    fetch_dataset,
)
from . import candles_repository
from .timeutils import ensure_ms_epoch
from .settings import BinanceVisionSettings, get_settings
from .tracing import TraceContext
from .vision_store import IngestionStats, get_store


DEFAULT_DATASETS: tuple[str, ...] = (
    DATASET_AGG_TRADES,
    DATASET_BOOK_DEPTH,
    DATASET_KLINES,
    DATASET_METRICS,
    DATASET_OPEN_INTEREST,
    DATASET_FUNDING_RATE,
    DATASET_LIQ_ORDERS,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionTask:
    dataset: str
    day: date
    interval: str | None = None


def _iter_days(start_ms: int, end_ms: int) -> List[date]:
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


def _make_tasks(
    days: Iterable[date],
    datasets: Sequence[str],
    intervals: Sequence[str],
) -> List[IngestionTask]:
    tasks: List[IngestionTask] = []
    for day in days:
        for dataset in datasets:
            if dataset == DATASET_KLINES:
                for interval in intervals:
                    tasks.append(IngestionTask(dataset=dataset, day=day, interval=interval))
            else:
                tasks.append(IngestionTask(dataset=dataset, day=day))
    return tasks


def _empty_summary() -> Dict[str, MutableMapping[str, object]]:
    return {
        DATASET_AGG_TRADES: {"count": 0, "inserted": 0, "days": {}},
        DATASET_KLINES: {"count": 0, "inserted": 0, "intervals": {}, "days": {}},
        DATASET_FUNDING_RATE: {"count": 0, "inserted": 0, "days": {}},
        DATASET_OPEN_INTEREST: {"count": 0, "inserted": 0, "days": {}},
        DATASET_LIQ_ORDERS: {"count": 0, "inserted": 0, "days": {}},
        DATASET_BOOK_DEPTH: {"count": 0, "inserted": 0, "days": {}},
        DATASET_METRICS: {"count": 0, "inserted": 0, "days": {}},
        DATASET_EXCHANGE_INFO: {"count": 0, "status": "skipped"},
    }


async def _store_batch(
    symbol: str,
    batch: VisionBatch,
    *,
    day_label: str,
) -> IngestionStats | None:
    store = get_store()
    if batch.dataset == DATASET_AGG_TRADES:
        return store.insert_agg_trades(symbol, day_label, batch.records, bytes_downloaded=batch.bytes_downloaded)
    if batch.dataset == DATASET_KLINES and batch.interval:
        stats = store.upsert_klines(
            symbol,
            batch.interval,
            day_label,
            batch.records,
            bytes_downloaded=batch.bytes_downloaded,
        )
        if batch.interval == "1m" and batch.records:
            repo = candles_repository.get_repository()
            repo_payload: List[Dict[str, object]] = []
            for record in batch.records:
                ts_source = (
                    record.get("open_time")
                    or record.get("openTime")
                    or record.get("t")
                    or record.get("ts")
                )
                timestamp_ms = ensure_ms_epoch(ts_source)
                if timestamp_ms is None:
                    continue
                try:
                    open_price = float(record.get("o"))
                    high_price = float(record.get("h"))
                    low_price = float(record.get("l"))
                    close_price = float(record.get("c"))
                    volume_value = record.get("v", 0.0)
                    volume = float(volume_value) if volume_value is not None else 0.0
                except (TypeError, ValueError):
                    continue
                repo_payload.append(
                    {
                        "t": timestamp_ms,
                        "o": open_price,
                        "h": high_price,
                        "l": low_price,
                        "c": close_price,
                        "v": volume,
                    }
                )
            if repo_payload:
                repo.upsert_candles(
                    symbol,
                    "1m",
                    repo_payload,
                    stage="vision.klines.seed",
                )
        return stats
    if batch.dataset == DATASET_FUNDING_RATE:
        return store.upsert_funding_rates(symbol, day_label, batch.records, bytes_downloaded=batch.bytes_downloaded)
    if batch.dataset == DATASET_OPEN_INTEREST:
        return store.upsert_open_interest(symbol, day_label, batch.records, bytes_downloaded=batch.bytes_downloaded)
    if batch.dataset == DATASET_METRICS:
        return store.upsert_metrics(symbol, day_label, batch.records, bytes_downloaded=batch.bytes_downloaded)
    if batch.dataset == DATASET_LIQ_ORDERS:
        return store.upsert_liquidations(symbol, day_label, batch.records, bytes_downloaded=batch.bytes_downloaded)
    if batch.dataset == DATASET_BOOK_DEPTH:
        return store.upsert_depth_snapshots(symbol, day_label, batch.records, bytes_downloaded=batch.bytes_downloaded)
    return None


async def ingest_binance_vision(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    datasets: Sequence[str] | None = None,
    klines_intervals: Sequence[str] | None = None,
    settings: BinanceVisionSettings | None = None,
    trace: TraceContext | None = None,
    include_exchange_info: bool = True,
) -> Dict[str, object]:
    """Download and persist the requested Binance Vision datasets."""

    app_settings = settings or get_settings().binance_vision
    dataset_order: List[str]
    exchange_flag = include_exchange_info
    if datasets:
        dataset_order = [ds for ds in datasets if ds != DATASET_EXCHANGE_INFO]
        if DATASET_EXCHANGE_INFO in datasets:
            exchange_flag = True
    else:
        dataset_order = list(DEFAULT_DATASETS)

    intervals = tuple(klines_intervals or app_settings.default_intervals)
    days = _iter_days(start_ms, end_ms)
    tasks = _make_tasks(days, dataset_order, intervals)
    semaphore = asyncio.Semaphore(max(1, int(app_settings.max_parallel_downloads)))

    LOGGER.info(
        "vision.ingest.start | symbol=%s | start_ms=%s | end_ms=%s | datasets=%s | intervals=%s | days=%s",
        symbol,
        start_ms,
        end_ms,
        dataset_order,
        intervals,
        [day.isoformat() for day in days],
    )

    summary = {
        "status": "ok",
        "symbol": symbol,
        "range": {"start": start_ms, "end": end_ms},
        "ingested": _empty_summary(),
        "errors": [],
        "missing": [],
    }

    async def run_task(task: IngestionTask, client: httpx.AsyncClient) -> None:
        day_label = task.day.isoformat()
        LOGGER.info(
            "vision.ingest.task.start | symbol=%s | dataset=%s | day=%s | interval=%s",
            symbol,
            task.dataset,
            day_label,
            task.interval,
        )
        try:
            async with semaphore:
                batch = await fetch_dataset(
                    task.dataset,
                    symbol=symbol,
                    day=task.day,
                    interval=task.interval,
                    client=client,
                    settings=app_settings,
                    trace=trace,
                )
        except Exception as exc:  # pragma: no cover - network failure path
            summary["status"] = "partial"
            summary["errors"].append(
                {
                    "dataset": task.dataset,
                    "interval": task.interval,
                    "day": day_label,
                    "error": str(exc),
                }
            )
            LOGGER.warning(
                "vision.ingest.task.error | symbol=%s | dataset=%s | day=%s | interval=%s | error=%s",
                symbol,
                task.dataset,
                day_label,
                task.interval,
                exc,
            )
            return

        if batch is None or not batch.records:
            summary["status"] = "partial"
            summary["missing"].append(
                {"dataset": task.dataset, "day": day_label, "interval": task.interval}
            )
            dataset_entry = summary["ingested"][task.dataset]
            dataset_entry.setdefault("days", {})[day_label] = {"status": "missing"}
            if task.dataset == DATASET_KLINES and task.interval:
                dataset_entry.setdefault("intervals", {}).setdefault(task.interval, {}).setdefault(
                    "days", {}
                )[day_label] = {"status": "missing"}
            LOGGER.warning(
                "vision.ingest.task.missing | symbol=%s | dataset=%s | day=%s | interval=%s",
                symbol,
                task.dataset,
                day_label,
                task.interval,
            )
            return

        stats = await _store_batch(symbol, batch, day_label=day_label)
        dataset_entry = summary["ingested"][task.dataset]
        dataset_entry["count"] = int(dataset_entry.get("count", 0)) + batch.count
        inserted = stats.inserted if stats else batch.count
        dataset_entry["inserted"] = int(dataset_entry.get("inserted", 0)) + inserted
        dataset_entry.setdefault("days", {})[day_label] = {
            "count": batch.count,
            "inserted": inserted,
            "status": "ok" if inserted else "skipped",
        }
        if task.dataset == DATASET_KLINES and task.interval:
            interval_entry = dataset_entry.setdefault("intervals", {}).setdefault(
                task.interval, {"count": 0, "inserted": 0, "days": {}}
            )
            interval_entry["count"] += batch.count
            interval_entry["inserted"] += inserted
            interval_entry["days"][day_label] = {
                "count": batch.count,
                "inserted": inserted,
                "status": "ok" if inserted else "skipped",
            }
        LOGGER.info(
            "vision.ingest.task.complete | symbol=%s | dataset=%s | day=%s | interval=%s | records=%s | inserted=%s",
            symbol,
            task.dataset,
            day_label,
            task.interval,
            batch.count,
            stats.inserted if stats else batch.count,
        )

    async with httpx.AsyncClient(timeout=app_settings.request_timeout_seconds or 30.0) as client:
        if exchange_flag:
            LOGGER.info("vision.ingest.exchange_info.start | symbol=%s", symbol)
            exchange_batch = await fetch_dataset(
                DATASET_EXCHANGE_INFO,
                client=client,
                settings=app_settings,
                trace=trace,
            )
            if exchange_batch and exchange_batch.records:
                store = get_store()
                store.store_exchange_info(exchange_batch.records[0])
                entry = summary["ingested"][DATASET_EXCHANGE_INFO]
                entry["count"] = len(exchange_batch.records)
                entry["status"] = "ok"
                LOGGER.info(
                    "vision.ingest.exchange_info.complete | symbol=%s | records=%s",
                    symbol,
                    len(exchange_batch.records),
                )
            else:
                entry = summary["ingested"][DATASET_EXCHANGE_INFO]
                entry["status"] = "missing"
                summary["status"] = "partial"
                LOGGER.warning("vision.ingest.exchange_info.missing | symbol=%s", symbol)
        for task in tasks:
            await run_task(task, client)

    metrics_entry = summary["ingested"].get(DATASET_METRICS, {})
    if metrics_entry.get("inserted"):
        open_interest_entry = summary["ingested"].setdefault(
            DATASET_OPEN_INTEREST,
            {"count": 0, "inserted": 0, "days": {}},
        )
        open_interest_entry["count"] = max(
            open_interest_entry.get("count", 0), metrics_entry.get("count", 0)
        )
        open_interest_entry["inserted"] = max(
            open_interest_entry.get("inserted", 0), metrics_entry.get("inserted", 0)
        )
        open_interest_entry["source"] = "metrics"
        open_days = open_interest_entry.setdefault("days", {})
        for day_label, info in metrics_entry.get("days", {}).items():
            existing = open_days.setdefault(day_label, {})
            existing.update(info)
            existing.setdefault("status", info.get("status", "ok"))
        summary["missing"] = [
            item for item in summary["missing"] if item.get("dataset") != DATASET_OPEN_INTEREST
        ]

    if summary["status"] == "ok" and not summary["ingested"][DATASET_KLINES]["count"]:
        summary["status"] = "partial"
    LOGGER.info(
        "vision.ingest.complete | symbol=%s | status=%s | errors=%s | missing=%s",
        symbol,
        summary["status"],
        len(summary["errors"]),
        len(summary["missing"]),
    )
    return summary


__all__ = ["ingest_binance_vision"]
