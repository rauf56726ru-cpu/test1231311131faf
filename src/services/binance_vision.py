"""Download helpers for Binance Vision archive datasets."""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from zipfile import ZipFile, BadZipFile

import httpx

from .http_client import (
    RATE_LIMIT_STATUSES,
    TRANSIENT_STATUSES,
    request as http_request,
)
from .settings import BinanceVisionSettings, get_settings
from .tracing import TraceContext

LOGGER = logging.getLogger(__name__)


class BinanceVisionError(RuntimeError):
    """Raised when archive downloads fail irrecoverably."""


@dataclass(slots=True)
class VisionBatch:
    """Container with normalised archive records for a single dataset/day."""

    dataset: str
    symbol: str
    day: date
    records: List[Dict[str, Any]]
    url: str
    interval: Optional[str] = None
    bytes_downloaded: int = 0

    @property
    def count(self) -> int:
        return len(self.records)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "symbol": self.symbol,
            "day": self.day.isoformat(),
            "interval": self.interval,
            "count": self.count,
            "bytes": self.bytes_downloaded,
            "url": self.url,
        }


DatasetName = str

DATASET_AGG_TRADES: DatasetName = "aggTrades"
DATASET_KLINES: DatasetName = "klines"
DATASET_FUNDING_RATE: DatasetName = "fundingRate"
DATASET_OPEN_INTEREST: DatasetName = "openInterest"
DATASET_LIQ_ORDERS: DatasetName = "liquidationOrders"
DATASET_BOOK_DEPTH: DatasetName = "bookDepth"
DATASET_METRICS: DatasetName = "metrics"
DATASET_EXCHANGE_INFO: DatasetName = "exchangeInfo"

_TRUTHY = {"1", "true", "True", "TRUE"}


def _normalise_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"buy", "long"}:
        return "buy"
    if text in {"sell", "short"}:
        return "sell"
    return text


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_symbol(symbol: str) -> str:
    clean = (symbol or "").strip().upper()
    if not clean:
        raise ValueError("symbol is required")
    return clean


def _make_url(
    settings: BinanceVisionSettings,
    dataset: DatasetName,
    symbol: str | None,
    day: date | None,
    interval: str | None,
) -> str:
    base = settings.base_url.rstrip("/")
    paths = settings.datasets

    if dataset == DATASET_EXCHANGE_INFO:
        return f"{base}/{paths.exchange_info}"

    if day is None:
        raise ValueError(f"{dataset} requires a date")

    day_text = day.strftime("%Y-%m-%d")
    symbol_clean = _normalise_symbol(symbol or "")

    if dataset == DATASET_AGG_TRADES:
        path = f"{paths.agg_trades}/{symbol_clean}/{symbol_clean}-aggTrades-{day_text}.zip"
    elif dataset == DATASET_KLINES:
        if not interval:
            raise ValueError("klines dataset requires interval")
        interval_clean = interval.strip()
        path = f"{paths.klines}/{symbol_clean}/{interval_clean}/{symbol_clean}-{interval_clean}-{day_text}.zip"
    elif dataset == DATASET_FUNDING_RATE:
        path = f"{paths.funding_rate}/{symbol_clean}/{symbol_clean}-fundingRate-{day_text}.zip"
    elif dataset == DATASET_OPEN_INTEREST:
        path = f"{paths.open_interest}/{symbol_clean}/{symbol_clean}-openInterest-{day_text}.zip"
    elif dataset == DATASET_METRICS:
        path = f"{paths.metrics}/{symbol_clean}/{symbol_clean}-metrics-{day_text}.zip"
    elif dataset == DATASET_LIQ_ORDERS:
        path = f"{paths.liquidation_orders}/{symbol_clean}/{symbol_clean}-liquidationOrders-{day_text}.zip"
    elif dataset == DATASET_BOOK_DEPTH:
        path = f"{paths.depth_snapshots}/{symbol_clean}/{symbol_clean}-bookDepth-{day_text}.zip"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return f"{base}/{path}"


async def _download_bytes(
    url: str,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
    trace: TraceContext | None,
) -> Optional[bytes]:
    try:
        response = await http_request(
            "GET",
            url,
            scope="vision.download",
            trace=trace,
            timeout=timeout,
            client=client,
            max_retries=2,
            retry_statuses=TRANSIENT_STATUSES,
            rate_limit_statuses=RATE_LIMIT_STATUSES,
            retry_on_request_error=True,
        )
    except httpx.RequestError as exc:
        raise BinanceVisionError(f"network error fetching {url}") from exc

    if response.status_code == 404:
        LOGGER.debug("Archive not found: %s", url)
        return None
    if response.status_code != 200:
        raise BinanceVisionError(f"unexpected status {response.status_code} for {url}")
    return response.content


def _unzip_members(payload: bytes) -> List[tuple[str, bytes]]:
    try:
        archive = ZipFile(io.BytesIO(payload))
    except BadZipFile as exc:
        raise BinanceVisionError("invalid zip payload") from exc

    members: List[tuple[str, bytes]] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        with archive.open(info) as handle:
            members.append((info.filename, handle.read()))
    return members


def _iter_csv_rows(content: bytes) -> Iterable[List[str]]:
    text_stream = io.TextIOWrapper(io.BytesIO(content), encoding="utf-8")
    reader = csv.reader(text_stream)
    for row in reader:
        if not row:
            continue
        yield [field.strip() for field in row]


def _parse_agg_trades(symbol: str, rows: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if len(row) < 7:
            continue
        try:
            agg_id = int(row[0])
            price = float(row[1])
            qty = float(row[2])
            first_id = int(row[3])
            last_id = int(row[4])
            ts = int(row[5])
            maker_flag = str(row[6]).strip()
        except (ValueError, TypeError):
            continue
        is_buyer_maker = maker_flag in _TRUTHY
        parsed.append(
            {
                "symbol": symbol,
                "agg_id": agg_id,
                "price": price,
                "qty": qty,
                "first_id": first_id,
                "last_id": last_id,
                "ts": ts,
                "buyer_maker": is_buyer_maker,
                "side": "sell" if is_buyer_maker else "buy",
            }
        )
    return parsed


def _parse_klines(symbol: str, interval: str, rows: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if len(row) < 6:
            continue
        try:
            open_time = int(row[0])
            open_price = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            volume = float(row[5])
            close_time = int(row[6]) if len(row) > 6 else open_time
            quote_volume = float(row[7]) if len(row) > 7 else 0.0
            trades = int(row[8]) if len(row) > 8 else 0
        except (ValueError, TypeError):
            continue
        parsed.append(
            {
                "symbol": symbol,
                "interval": interval,
                "open_time": open_time,
                "close_time": close_time,
                "ts": close_time,
                "o": open_price,
                "h": high,
                "l": low,
                "c": close,
                "v": volume,
                "quote_volume": quote_volume,
                "trades": trades,
            }
        )
    return parsed


def _parse_funding_rate(symbol: str, rows: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            funding_time = int(row[1])
            rate = float(row[2])
            mark_price = float(row[3]) if len(row) > 3 else None
        except (ValueError, TypeError):
            continue
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "ts": funding_time,
            "funding_rate": rate,
        }
        if mark_price is not None:
            payload["mark_price"] = mark_price
        parsed.append(payload)
    return parsed


def _parse_open_interest(symbol: str, rows: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            open_interest = float(row[1])
            ts = int(row[2 if len(row) == 3 else 3])
            notional = float(row[2]) if len(row) > 3 else None
        except (ValueError, TypeError, IndexError):
            continue
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "ts": ts,
            "open_interest": open_interest,
        }
        if notional is not None:
            payload["notional"] = notional
        parsed.append(payload)
    return parsed


def _parse_metrics(symbol: str, rows: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if len(row) < 4:
            continue
        if row[0].lower().startswith("create_time"):
            continue
        try:
            dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
            open_interest = float(row[2])
        except (ValueError, TypeError):
            continue
        open_interest_value = None
        raw_value = row[3]
        if raw_value not in (None, ""):
            try:
                open_interest_value = float(raw_value)
            except (ValueError, TypeError):
                open_interest_value = None
        top_count = _safe_float(row[4])
        top_sum = _safe_float(row[5])
        trader_count = _safe_float(row[6])
        taker_ratio = _safe_float(row[7])
        parsed.append(
            {
                "symbol": symbol,
                "ts": ts,
                "open_interest": open_interest,
                "open_interest_value": open_interest_value,
                "top_trader_count": top_count,
                "top_trader_sum": top_sum,
                "trader_count": trader_count,
                "taker_vol_ratio": taker_ratio,
            }
        )
    return parsed


def _parse_liquidations(symbol: str, rows: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    header_detected = False
    for row in rows:
        if len(row) < 4:
            continue
        if not header_detected:
            lowered = [cell.lower() for cell in row]
            if any(cell in {"time", "timestamp"} for cell in lowered):
                header_detected = True
                continue
        ts_value: Optional[int] = None
        ts_index = 0
        for idx, cell in enumerate(row):
            try:
                candidate = int(float(cell))
            except (TypeError, ValueError):
                continue
            if candidate > 1_000_000_000:  # allow seconds or ms
                ts_value = candidate if candidate > 10_000_000_000 else candidate * 1000
                ts_index = idx
                break
        if ts_value is None:
            continue
        numeric_values: List[float] = []
        side_value: Optional[str] = None
        for idx, cell in enumerate(row):
            if idx == ts_index:
                continue
            text = str(cell).strip()
            text_lower = text.lower()
            if text_lower in {"buy", "sell", "long", "short"}:
                side_value = _normalise_side(text_lower)
                continue
            try:
                numeric = float(text)
            except ValueError:
                continue
            numeric_values.append(numeric)
        if not numeric_values:
            continue
        price = numeric_values[0]
        qty = numeric_values[1] if len(numeric_values) > 1 else None
        notional = numeric_values[2] if len(numeric_values) > 2 else None
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "ts": ts_value,
            "price": price,
            "qty": qty,
            "side": side_value,
        }
        if notional is not None:
            payload["notional"] = notional
        parsed.append(payload)
    return parsed


def _parse_json_sequence(content: bytes) -> List[Dict[str, Any]]:
    text = content.decode("utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
    if isinstance(data, list):
        return [row for row in data if isinstance(row, Mapping)]
    if isinstance(data, Mapping):
        return [dict(data)]
    return []


def _parse_depth(symbol: str, members: Sequence[tuple[str, bytes]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for _, content in members:
        entries = _parse_json_sequence(content)
        if entries:
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                ts = entry.get("timestamp") or entry.get("T") or entry.get("time")
                bids = entry.get("bids") or entry.get("bid")
                asks = entry.get("asks") or entry.get("ask")
                try:
                    ts_value = int(ts)
                except (TypeError, ValueError):
                    continue
                parsed.append(
                    {
                        "symbol": symbol,
                        "ts": ts_value,
                        "bids": list(bids) if isinstance(bids, Sequence) else [],
                        "asks": list(asks) if isinstance(asks, Sequence) else [],
                    }
                )
            continue

        # Fallback to CSV structure (timestamp, percentage, depth, notional)
        buckets: Dict[int, Dict[str, Any]] = {}
        for row in _iter_csv_rows(content):
            if not row or row[0].lower().startswith("timestamp"):
                continue
            if len(row) < 4:
                continue
            try:
                dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                ts_value = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                percentage = float(row[1])
                depth_value = float(row[2])
            except (ValueError, TypeError):
                continue
            notional_value = None
            notional_raw = row[3]
            if notional_raw not in (None, ""):
                try:
                    notional_value = float(notional_raw)
                except (ValueError, TypeError):
                    notional_value = None

            bucket = buckets.setdefault(
                ts_value,
                {
                    "symbol": symbol,
                    "ts": ts_value,
                    "bids": [],
                    "asks": [],
                },
            )

            level_payload = {
                "p": percentage,
                "sz": depth_value,
            }
            if notional_value is not None:
                level_payload["notional"] = notional_value

            target = bucket["bids"] if percentage <= 0 else bucket["asks"]
            target.append(level_payload)

        parsed.extend(buckets.values())
    return parsed


def _parse_exchange_info(payload: bytes) -> Dict[str, Any]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinanceVisionError("invalid exchangeInfo payload") from exc
    if not isinstance(data, Mapping):
        raise BinanceVisionError("exchangeInfo payload must be a mapping")
    return dict(data)


def _parse_liquidation_members(symbol: str, members: Sequence[tuple[str, bytes]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for name, content in members:
        if name.endswith(".json") or name.endswith(".jsonl"):
            entries = _parse_json_sequence(content)
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                ts = entry.get("time") or entry.get("T") or entry.get("timestamp")
                try:
                    ts_value = int(ts)
                except (TypeError, ValueError):
                    continue
                price = entry.get("price")
                qty = entry.get("qty") or entry.get("quantity")
                side = _normalise_side(entry.get("side") or entry.get("S"))
                notional = entry.get("notional") or entry.get("quoteQty")
                payload: Dict[str, Any] = {
                    "symbol": symbol,
                    "ts": ts_value,
                    "price": float(price) if price is not None else None,
                    "qty": float(qty) if qty is not None else None,
                    "side": side,
                }
                if notional is not None:
                    try:
                        payload["notional"] = float(notional)
                    except (TypeError, ValueError):
                        pass
                parsed.append(payload)
        else:
            rows = _iter_csv_rows(content)
            parsed.extend(_parse_liquidations(symbol, rows))
    return parsed


def _parse_liquidation_csv(symbol: str, members: Sequence[tuple[str, bytes]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for _, content in members:
        rows = _iter_csv_rows(content)
        parsed.extend(_parse_liquidations(symbol, rows))
    return parsed


PARSER_REGISTRY: Dict[str, str] = {
    DATASET_AGG_TRADES: "csv",
    DATASET_KLINES: "csv",
    DATASET_FUNDING_RATE: "csv",
    DATASET_OPEN_INTEREST: "csv",
    DATASET_METRICS: "csv",
    DATASET_LIQ_ORDERS: "mixed",
    DATASET_BOOK_DEPTH: "json",
}


async def fetch_dataset(
    dataset: DatasetName,
    *,
    symbol: str | None = None,
    day: date | None = None,
    interval: str | None = None,
    client: httpx.AsyncClient | None = None,
    settings: BinanceVisionSettings | None = None,
    trace: TraceContext | None = None,
) -> Optional[VisionBatch]:
    """Download and normalise a single Binance Vision dataset."""

    settings = settings or get_settings().binance_vision
    url = _make_url(settings, dataset, symbol, day, interval)
    timeout = settings.request_timeout_seconds or 30.0
    payload = await _download_bytes(url, client=client, timeout=timeout, trace=trace)
    if payload is None:
        return None

    symbol_clean = _normalise_symbol(symbol or "") if dataset != DATASET_EXCHANGE_INFO else symbol or ""

    if dataset == DATASET_EXCHANGE_INFO:
        parsed = _parse_exchange_info(payload)
        return VisionBatch(
            dataset=dataset,
            symbol="*",
            day=day or date.today(),
            interval=None,
            records=[parsed],
            url=url,
            bytes_downloaded=len(payload),
        )

    parser_kind = PARSER_REGISTRY.get(dataset, "csv")
    members = _unzip_members(payload)

    if parser_kind == "csv":
        rows: List[Sequence[str]] = []
        for _, content in members:
            rows.extend(list(_iter_csv_rows(content)))
        if dataset == DATASET_AGG_TRADES:
            records = _parse_agg_trades(symbol_clean, rows)
        elif dataset == DATASET_KLINES:
            if interval is None:
                raise ValueError("Interval required for klines dataset")
            records = _parse_klines(symbol_clean, interval, rows)
        elif dataset == DATASET_FUNDING_RATE:
            records = _parse_funding_rate(symbol_clean, rows)
        elif dataset == DATASET_OPEN_INTEREST:
            records = _parse_open_interest(symbol_clean, rows)
        elif dataset == DATASET_METRICS:
            records = _parse_metrics(symbol_clean, rows)
        else:
            records = _parse_liquidations(symbol_clean, rows)
    elif dataset == DATASET_BOOK_DEPTH:
        records = _parse_depth(symbol_clean, members)
    elif dataset == DATASET_LIQ_ORDERS:
        records = _parse_liquidation_members(symbol_clean, members)
    else:
        records = []

    return VisionBatch(
        dataset=dataset,
        symbol=symbol_clean,
        day=day or date.today(),
        interval=interval,
        records=records,
        url=url,
        bytes_downloaded=len(payload),
    )


__all__ = [
    "BinanceVisionError",
    "DATASET_AGG_TRADES",
    "DATASET_BOOK_DEPTH",
    "DATASET_EXCHANGE_INFO",
    "DATASET_FUNDING_RATE",
    "DATASET_KLINES",
    "DATASET_LIQ_ORDERS",
    "DATASET_OPEN_INTEREST",
    "DATASET_METRICS",
    "VisionBatch",
    "fetch_dataset",
]
