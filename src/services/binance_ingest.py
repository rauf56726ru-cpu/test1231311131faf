"""Binance Futures REST ingestion helpers for minute candles and agg trades."""
from __future__ import annotations

import asyncio
import csv
import io
import random
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from zipfile import ZipFile

import httpx

from .binance import BINANCE_FAPI_BASE_URL
from .http_client import RATE_LIMIT_STATUSES, TRANSIENT_STATUSES, request as http_request
from .cache import FileCache
from .settings import get_settings
from .tracing import TraceContext
from .timeutils import ensure_ms_epoch

LOGGER = logging.getLogger(__name__)

MINUTE_MS = 60_000
KLINES_LIMIT = 1_000
AGG_TRADES_LIMIT = 1_000
KLINES_ENDPOINT = f"{BINANCE_FAPI_BASE_URL}/klines"
AGG_TRADES_ENDPOINT = f"{BINANCE_FAPI_BASE_URL}/aggTrades"
AGG_TRADES_WINDOW_MS = 30 * MINUTE_MS
AGG_TRADES_MIN_WINDOW_MS = 10 * MINUTE_MS
MAX_RETRY_ATTEMPTS = 7
BACKOFF_BASE_SECONDS = 0.25
BACKOFF_MAX_SECONDS = 4.0
PREMIUM_INTERVAL_MS = 60 * MINUTE_MS
PREMIUM_CACHE_TTL = 15 * 60.0
PREMIUM_CACHE_DIR = ".cache/premium_index"


class BinanceIngestError(RuntimeError):
    """Raised when Binance ingestion fails."""


class BinanceAccessDenied(BinanceIngestError):
    """Raised when Binance refuses further requests (403/418)."""


@dataclass(slots=True)
class MinuteBaseline:
    start_ms: int
    end_ms: int
    timestamps: List[int]


def _align_minute(ts: int) -> int:
    return (ts // MINUTE_MS) * MINUTE_MS


def _build_minutes(start_ms: int, end_ms: int) -> MinuteBaseline:
    if end_ms <= start_ms:
        aligned = _align_minute(start_ms)
        return MinuteBaseline(start_ms=aligned, end_ms=aligned, timestamps=[aligned])
    end_aligned = _align_minute(end_ms - 1)
    start_aligned = _align_minute(start_ms)
    if start_ms % MINUTE_MS != 0:
        start_aligned = _align_minute_up(start_ms)
    if end_aligned < start_aligned:
        start_aligned = end_aligned
    count = ((end_aligned - start_aligned) // MINUTE_MS) + 1
    timestamps = [start_aligned + index * MINUTE_MS for index in range(count)]
    return MinuteBaseline(start_ms=start_aligned, end_ms=end_aligned, timestamps=timestamps)


def _align_minute_up(ts: int) -> int:
    if ts % MINUTE_MS == 0:
        return ts
    return ((ts // MINUTE_MS) + 1) * MINUTE_MS


def _init_metrics() -> Dict[str, int]:
    return {"requests": 0, "retries": 0, "rate_limit_events": 0}


async def _request_with_backoff(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any],
    scope: str,
    trace: TraceContext | None,
    client: httpx.AsyncClient | None,
    timeout: float | httpx.Timeout | None,
    metrics: MutableMapping[str, int] | None = None,
) -> httpx.Response:
    delay = BACKOFF_BASE_SECONDS
    attempt = 0
    last_exc: Exception | None = None
    while attempt < MAX_RETRY_ATTEMPTS:
        attempt += 1
        if metrics is not None:
            metrics["requests"] = metrics.get("requests", 0) + 1
        try:
            response = await http_request(
                method,
                url,
                scope=scope,
                trace=trace,
                params=params,
                client=client,
                timeout=timeout,
                max_retries=0,
                retry_statuses=TRANSIENT_STATUSES,
                rate_limit_statuses=RATE_LIMIT_STATUSES,
            )
        except httpx.RequestError as exc:  # pragma: no cover - network errors
            last_exc = exc
            LOGGER.warning(
                "Binance request error",
                extra={"url": url, "params": dict(params), "attempt": attempt, "error": str(exc)},
            )
            if metrics is not None:
                metrics["retries"] = metrics.get("retries", 0) + 1
        else:
            status = response.status_code
            if status in (403, 418):
                raise BinanceAccessDenied(f"access denied (status {status})")
            if status == 429:
                message = ""
                try:
                    payload = response.json()
                    message = str(payload.get("msg") or "")
                except Exception:  # pragma: no cover - empty payload
                    try:
                        message = response.text
                    except Exception:  # pragma: no cover - I/O failure
                        message = ""
                LOGGER.warning(
                    "Binance rate limited",
                    extra={"url": url, "params": dict(params), "attempt": attempt, "message": message},
                )
                if attempt >= MAX_RETRY_ATTEMPTS:
                    return response
                if metrics is not None:
                    metrics["rate_limit_events"] = metrics.get("rate_limit_events", 0) + 1
                    metrics["retries"] = metrics.get("retries", 0) + 1
                await asyncio.sleep(delay + random.uniform(0.0, 0.2))
                delay = min(delay * 2, BACKOFF_MAX_SECONDS)
                continue
            return response
        await asyncio.sleep(delay + random.uniform(0.0, 0.2))
        delay = min(delay * 2, BACKOFF_MAX_SECONDS)
    if last_exc is not None:
        raise BinanceIngestError(f"request failed after {MAX_RETRY_ATTEMPTS} attempts") from last_exc
    raise BinanceIngestError(f"request failed after {MAX_RETRY_ATTEMPTS} attempts")


def _parse_kline_row(row: Sequence[Any]) -> Dict[str, Any] | None:
    try:
        open_time = int(row[0])
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        volume = float(row[5])
        close_time = int(row[6])
        quote_volume = float(row[7]) if row[7] is not None else 0.0
        trades = int(row[8])
        taker_buy_base = float(row[9])
        taker_buy_quote = float(row[10])
    except (IndexError, TypeError, ValueError):
        return None
    if open_time < 0 or close_time < 0:
        return None
    return {
        "openTime": open_time,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "closeTime": close_time,
        "quoteVolume": quote_volume,
        "trades": trades,
        "takerBuyBase": taker_buy_base,
        "takerBuyQuote": taker_buy_quote,
    }


def _validate_candle(candle: Mapping[str, Any], *, ts: int) -> None:
    open_price = float(candle["open"])
    close_price = float(candle["close"])
    high_price = float(candle["high"])
    low_price = float(candle["low"])
    if not (low_price <= open_price <= high_price):
        raise BinanceIngestError(f"invalid candle high/low at {ts}: open outside range")
    if not (low_price <= close_price <= high_price):
        raise BinanceIngestError(f"invalid candle high/low at {ts}: close outside range")
    if high_price < low_price:
        raise BinanceIngestError(f"invalid candle high/low ordering at {ts}")


def _normalise_minutes_payload(
    baseline: MinuteBaseline,
    candles_by_minute: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for ts in baseline.timestamps:
        candle = candles_by_minute.get(ts)
        if ts in seen:
            raise BinanceIngestError(f"duplicate minute detected: {ts}")
        seen.add(ts)
        if candle is None:
            series.append(
                {
                    "ts_min": ts,
                    "gap_mask": True,
                    "openTime": None,
                    "closeTime": None,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": 0.0,
                    "quoteVolume": 0.0,
                    "trades": 0,
                    "takerBuyBase": 0.0,
                    "takerBuyQuote": 0.0,
                }
            )
        else:
            enriched = dict(candle)
            enriched["ts_min"] = ts
            enriched["gap_mask"] = False
            series.append(enriched)
    return series


async def ingest_klines(
    *,
    symbol: str,
    start_ms: int,
    t0_ms: int,
    interval: str = "1m",
    client: httpx.AsyncClient | None = None,
    trace: TraceContext | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> Dict[str, Any]:
    """Fetch 1m klines from Binance Futures REST and align them to minute baseline."""

    if interval != "1m":
        raise ValueError("only 1m interval supported")
    if start_ms >= t0_ms:
        raise ValueError("start_ms must be less than t0_ms")

    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    baseline = _build_minutes(start_ms, t0_ms)
    cursor = baseline.start_ms
    candles: Dict[int, Dict[str, Any]] = {}
    metrics = _init_metrics()

    while cursor < t0_ms:
        window_end = min(t0_ms, cursor + KLINES_LIMIT * MINUTE_MS)
        params = {
            "symbol": symbol_clean,
            "interval": interval,
            "limit": str(KLINES_LIMIT),
            "startTime": str(cursor),
            "endTime": str(window_end - 1),
        }
        scope = "ingest.klines"
        response = await _request_with_backoff(
            "GET",
            KLINES_ENDPOINT,
            params=params,
            scope=scope,
            trace=trace,
            client=client,
            timeout=timeout,
            metrics=metrics,
        )
        status = response.status_code
        if status != 200:
            raise BinanceIngestError(f"kline request failed with status {status}")
        payload = response.json()
        if not isinstance(payload, list):
            raise BinanceIngestError("invalid kline payload")
        if not payload:
            break

        last_close_time = None
        for entry in payload:
            candle = _parse_kline_row(entry)
            if candle is None:
                continue
            open_time = int(candle["openTime"])
            if open_time < baseline.start_ms or open_time >= t0_ms:
                continue
            ts_min = _align_minute(open_time)
            _validate_candle(candle, ts=ts_min)
            candles[ts_min] = candle
            last_close_time = int(candle["closeTime"])

        if last_close_time is None:
            break
        next_cursor = last_close_time + 1
        if next_cursor <= cursor:
            cursor = cursor + MINUTE_MS
            break
        cursor = next_cursor

    series = _normalise_minutes_payload(baseline, candles)
    if len(series) != len(baseline.timestamps):
        raise BinanceIngestError("minute series length mismatch")

    return {
        "symbol": symbol_clean,
        "interval": interval,
        "range": {
            "start_ms": baseline.start_ms,
            "end_ms": baseline.end_ms,
            "count": len(series),
        },
        "minutes": series,
        "expected_count": len(baseline.timestamps),
        "metrics": metrics,
    }


def _iter_days(start_ms: int, end_ms: int) -> List[str]:
    start_day = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, tz=UTC).date()
    days: List[str] = []
    cursor = start_day
    while cursor <= end_day:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def _load_vision_archive(
    *,
    symbol: str,
    archives_root: Path,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    symbol_clean = symbol.upper()
    seen_ids: set[int] = set()
    for day in _iter_days(start_ms, end_ms):
        path = archives_root / symbol_clean / f"{symbol_clean}-aggTrades-{day}.zip"
        if not path.exists():
            LOGGER.warning("Vision archive missing", extra={"path": str(path)})
            continue
        with ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as handle:
                    text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
                    reader = csv.reader(text_stream, delimiter=",")
                    for row in reader:
                        if len(row) < 7:
                            continue
                        try:
                            agg_id = int(row[0].strip())
                            if agg_id in seen_ids:
                                continue
                            price = float(row[1].strip())
                            qty = float(row[2].strip())
                            first_id = int(row[3].strip())
                            last_id = int(row[4].strip())
                            ts = int(row[5].strip())
                        except (ValueError, TypeError, AttributeError):
                            continue
                        maker_text = row[6].strip().lower()
                        if maker_text not in {"true", "false"}:
                            continue
                        maker_flag = maker_text == "true"
                        if ts < start_ms or ts >= end_ms:
                            continue
                        seen_ids.add(agg_id)
                        records.append(
                            {
                                "a": agg_id,
                                "p": price,
                                "q": qty,
                                "f": first_id,
                                "l": last_id,
                                "T": ts,
                                "m": maker_flag,
                            }
                        )
    return records


def _normalise_live_trade(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    try:
        agg_id = int(entry["a"])
        price = float(entry["p"])
        qty = float(entry["q"])
        first_id = int(entry["f"])
        last_id = int(entry["l"])
        ts = ensure_ms_epoch(entry["T"])
        maker_flag = bool(entry.get("m"))
    except (KeyError, TypeError, ValueError):
        return None
    if ts is None:
        return None
    return {
        "a": agg_id,
        "p": price,
        "q": qty,
        "f": first_id,
        "l": last_id,
        "T": ts,
        "m": maker_flag,
    }


async def _fetch_live_trades(
    *,
    symbol: str,
    start_ms: int,
    t0_ms: int,
    from_id: int | None,
    trace: TraceContext | None,
    client: httpx.AsyncClient | None,
    timeout: float | httpx.Timeout | None,
    metrics: MutableMapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    symbol_clean = symbol.upper()
    window_ms = AGG_TRADES_WINDOW_MS
    cursor = start_ms
    last_from_id = from_id

    while cursor < t0_ms:
        window_end = min(t0_ms, cursor + window_ms)
        params: Dict[str, Any] = {
            "symbol": symbol_clean,
            "limit": str(AGG_TRADES_LIMIT),
        }
        if last_from_id is not None:
            params["fromId"] = str(last_from_id)
        else:
            params["startTime"] = str(cursor)
            params["endTime"] = str(window_end - 1)

        response = await _request_with_backoff(
            "GET",
            AGG_TRADES_ENDPOINT,
            params=params,
            scope="ingest.aggTrades",
            trace=trace,
            client=client,
            timeout=timeout,
            metrics=metrics,
        )
        status = response.status_code
        if status == 429:
            body_text = ""
            try:
                body = response.json()
                body_text = str(body.get("msg") or "")
            except Exception:  # pragma: no cover - parsing
                try:
                    body_text = response.text
                except Exception:
                    body_text = ""
            if "Too much request weight" in body_text and window_ms > AGG_TRADES_MIN_WINDOW_MS:
                window_ms = max(AGG_TRADES_MIN_WINDOW_MS, window_ms // 2)
                LOGGER.info(
                    "Reducing aggTrades window after weight warning",
                    extra={"window_ms": window_ms},
                )
                continue
        if status != 200:
            raise BinanceIngestError(f"aggTrades request failed with status {status}")
        payload = response.json()
        if not isinstance(payload, list):
            raise BinanceIngestError("invalid aggTrades payload")
        if not payload:
            cursor = window_end
            continue

        parsed_batch: List[Dict[str, Any]] = []
        for entry in payload:
            normalised = _normalise_live_trade(entry if isinstance(entry, Mapping) else {})
            if normalised is None:
                continue
            ts = normalised["T"]
            if ts < start_ms or ts >= t0_ms:
                continue
            parsed_batch.append(normalised)

        if parsed_batch:
            trades.extend(parsed_batch)
            last_trade = parsed_batch[-1]
            if last_from_id is not None:
                last_from_id = last_trade["a"] + 1
            else:
                next_cursor = last_trade["T"] + 1
                if next_cursor <= cursor:
                    cursor = window_end
                else:
                    cursor = next_cursor
        else:
            cursor = window_end

        if len(payload) < AGG_TRADES_LIMIT and last_from_id is None:
            cursor = window_end

        if last_from_id is not None and len(payload) < AGG_TRADES_LIMIT:
            break

    return trades


def _dedupe_trades(trades: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[int, Dict[str, Any]] = {}
    for trade in trades:
        try:
            agg_id = int(trade["a"])
        except (KeyError, TypeError, ValueError):
            continue
        if agg_id in deduped:
            continue
        try:
            ts = int(trade["T"])
            price = float(trade["p"])
            qty = float(trade["q"])
        except (KeyError, TypeError, ValueError):
            continue
        deduped[agg_id] = {
            "a": agg_id,
            "T": ts,
            "p": price,
            "q": qty,
            "f": int(trade.get("f", 0)),
            "l": int(trade.get("l", 0)),
            "m": bool(trade.get("m")),
        }
    return sorted(deduped.values(), key=lambda row: (row["T"], row["a"]))


def _aggregate_trades_to_minutes(
    trades: Sequence[Mapping[str, Any]],
    *,
    window_start: int,
    window_end: int,
) -> Tuple[List[Dict[str, Any]], float, bool]:
    per_minute: Dict[int, Dict[str, Any]] = {}
    baseline_cvd = 0.0
    baseline_used = False
    for trade in trades:
        try:
            ts = int(trade.get("T"))
            qty = float(trade.get("q", 0.0))
            price = float(trade.get("p", 0.0))
        except (TypeError, ValueError):
            continue
        maker_flag = bool(trade.get("m"))
        delta_contrib = -qty if maker_flag else qty
        if ts < window_start:
            baseline_cvd += delta_contrib
            baseline_used = True
            continue
        if ts >= window_end:
            continue
        ts_min = _align_minute(ts)
        stats = per_minute.get(ts_min)
        if stats is None:
            stats = {
                "buy_vol": 0.0,
                "sell_vol": 0.0,
                "trades_cnt": 0,
                "min_price": float("inf"),
                "max_price": float("-inf"),
                "qty_terms": [],
                "pxqty_terms": [],
            }
            per_minute[ts_min] = stats
        stats["qty_terms"].append(qty)
        stats["pxqty_terms"].append(price * qty)
        stats["trades_cnt"] += 1
        if maker_flag:
            stats["sell_vol"] += qty
        else:
            stats["buy_vol"] += qty
        stats["min_price"] = min(stats["min_price"], price)
        stats["max_price"] = max(stats["max_price"], price)

    minutes: List[Dict[str, Any]] = []
    running_cvd = baseline_cvd
    for ts_min in sorted(per_minute.keys()):
        stats = per_minute[ts_min]
        qty_terms = stats.pop("qty_terms", [])
        pxqty_terms = stats.pop("pxqty_terms", [])
        buy_vol = stats["buy_vol"]
        sell_vol = stats["sell_vol"]
        delta = buy_vol - sell_vol
        running_cvd += delta
        qty_sum = math.fsum(qty_terms) if qty_terms else 0.0
        pxqty_sum = math.fsum(pxqty_terms) if pxqty_terms else 0.0
        vwap = pxqty_sum / qty_sum if qty_sum > 0 else None
        if vwap is not None:
            min_price = stats["min_price"]
            max_price = stats["max_price"]
            epsilon = 1e-9
            if not (min_price - epsilon <= vwap <= max_price + epsilon):
                raise BinanceIngestError(f"VWAP outside trade range at {ts_min}")
        minutes.append(
            {
                "ts_min": ts_min,
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "vol": buy_vol + sell_vol,
                "delta": delta,
                "cvd": running_cvd,
                "vwap_min": vwap,
                "trades_cnt": stats["trades_cnt"],
            }
        )
    if not baseline_used and minutes:
        deltas = [entry["delta"] for entry in minutes[:10]]
        median_delta = statistics.median(deltas) if deltas else 0.0
        baseline_cvd = -median_delta
        if baseline_cvd != 0.0:
            for entry in minutes:
                entry["cvd"] += baseline_cvd
    return minutes, baseline_cvd, baseline_used


async def ingest_agg_trades(
    *,
    symbol: str,
    start_ms: int,
    t0_ms: int,
    archives_path: str | Path | None = None,
    from_id: int | None = None,
    client: httpx.AsyncClient | None = None,
    trace: TraceContext | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> Dict[str, Any]:
    """Load aggTrades from archives and live REST, aggregate into minute buckets."""

    if start_ms >= t0_ms:
        raise ValueError("start_ms must be less than t0_ms")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    baseline = _build_minutes(start_ms, t0_ms)
    window_start = baseline.timestamps[0]
    window_end = window_start + len(baseline.timestamps) * MINUTE_MS
    archives_root = None
    if archives_path is not None:
        archives_root = Path(archives_path)
    else:
        settings = get_settings().binance_vision
        archives_root = Path(settings.datasets.agg_trades)
    archives_root = archives_root.expanduser()

    vision_records: List[Dict[str, Any]] = []
    metrics = _init_metrics()
    if archives_root.exists():
        vision_records = _load_vision_archive(
            symbol=symbol_clean,
            archives_root=archives_root,
            start_ms=window_start,
            end_ms=window_end,
        )
    else:
        LOGGER.warning("AggTrades archives root missing", extra={"path": str(archives_root)})

    live_records = await _fetch_live_trades(
        symbol=symbol_clean,
        start_ms=window_start,
        t0_ms=window_end,
        from_id=from_id,
        trace=trace,
        client=client,
        timeout=timeout,
        metrics=metrics,
    )

    combined = _dedupe_trades(vision_records + live_records)
    minutes, cvd_baseline, baseline_used = _aggregate_trades_to_minutes(
        combined,
        window_start=window_start,
        window_end=window_end,
    )
    if baseline_used:
        baseline_source = "warm_start"
    elif minutes:
        baseline_source = "estimated"
    else:
        baseline_source = "cold_start"

    return {
        "symbol": symbol_clean,
        "range": {
            "start_ms": baseline.start_ms,
            "end_ms": baseline.end_ms,
            "count": len(minutes),
        },
        "minutes": minutes,
        "vision_records": len(vision_records),
        "expected_count": len(baseline.timestamps),
        "metrics": metrics,
        "cvd_baseline": cvd_baseline,
        "cvd_baseline_source": baseline_source,
    }


def _parse_premium_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        time_value_raw = int(payload.get("time"))
        mark_price = float(payload.get("markPrice"))
        index_price = float(payload.get("indexPrice"))
        funding_rate = float(payload.get("lastFundingRate"))
        next_funding_time = int(payload.get("nextFundingTime"))
    except (TypeError, ValueError) as exc:
        raise BinanceIngestError(f"invalid premiumIndex payload: {payload}") from exc
    time_value = _align_minute(time_value_raw)
    if time_value is None or next_funding_time is None:
        raise BinanceIngestError("premiumIndex payload missing timestamps")
    return {
        "time": time_value,
        "markPrice": mark_price,
        "indexPrice": index_price,
        "lastFundingRate": funding_rate,
        "nextFundingTime": next_funding_time,
    }


async def fetch_premium_index_series(
    *,
    symbol: str,
    start_ms: int,
    t0_ms: int,
    every_ms: int = PREMIUM_INTERVAL_MS,
    client: httpx.AsyncClient | None = None,
    trace: TraceContext | None = None,
    timeout: float | httpx.Timeout | None = None,
    cache: FileCache | None = None,
) -> Dict[str, Any]:
    """Fetch premium index snapshots on a fixed hourly grid."""

    if start_ms >= t0_ms:
        raise ValueError("start_ms must be less than t0_ms")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")
    if every_ms <= 0:
        raise ValueError("every_ms must be positive")

    cache_instance = cache or FileCache(PREMIUM_CACHE_DIR)
    samples: List[Dict[str, Any]] = []
    tick = (start_ms // every_ms) * every_ms
    if tick < start_ms:
        tick += every_ms
    metrics = _init_metrics()

    while tick < t0_ms:
        cache_key = f"{symbol_clean}:{tick // every_ms}"
        cached = cache_instance.get(cache_key)
        payload: Dict[str, Any] | None = None
        if cached is not None:
            try:
                payload = _parse_premium_payload(cached.payload)
            except BinanceIngestError:
                payload = None
        if payload is None:
            params = {"symbol": symbol_clean}
            response = await _request_with_backoff(
                "GET",
                f"{BINANCE_FAPI_BASE_URL}/premiumIndex",
                params=params,
                scope="ingest.premiumIndex",
                trace=trace,
                client=client,
                timeout=timeout,
                metrics=metrics,
            )
            if response.status_code != 200:
                raise BinanceIngestError(f"premiumIndex request failed with status {response.status_code}")
            body = response.json()
            if not isinstance(body, Mapping):
                raise BinanceIngestError("invalid premiumIndex payload structure")
            payload = _parse_premium_payload(body)
            cache_instance.set(cache_key, body, ttl_seconds=PREMIUM_CACHE_TTL)
        samples.append(payload)
        tick += every_ms

    return {
        "symbol": symbol_clean,
        "range": {"start_ms": start_ms, "end_ms": t0_ms, "count": len(samples)},
        "samples": samples,
        "expected_count": len(samples),
        "metrics": metrics,
    }


__all__ = [
    "ingest_klines",
    "ingest_agg_trades",
    "fetch_premium_index_series",
    "BinanceIngestError",
    "BinanceAccessDenied",
]
