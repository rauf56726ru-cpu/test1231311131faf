"""Orderflow aggregation helpers shared across pipeline stages."""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import aiohttp

from ..binance import (
    BinanceAPIException,
    BinanceRequestException,
    fetch_um_agg_trades,
)
from ...common.config import AppConfig
from ..http_client import RATE_LIMIT_STATUSES
from ..orderflow import compute_orderflow_aggregates
from ..pipeline_utils import (
    align_to_interval,
    build_expected_times,
    coerce_float,
    isoformat_utc,
    safe_float,
    safe_int,
)
from ..tracing import TraceContext
from ..vision_store import get_store

LOGGER = logging.getLogger(__name__)

AGG_TRADES_LIMIT = 1000
_AGG_TRADES_REQUEST_TIMEOUT = float(os.getenv("AGG_TRADES_REQUEST_TIMEOUT", "8.0"))
_AGG_TRADES_FAILURE_MAX_ATTEMPTS = max(1, int(os.getenv("AGG_TRADES_FAILURE_MAX_ATTEMPTS", "3")))
_AGG_TRADES_FAILURE_BUDGET_SECONDS = float(os.getenv("AGG_TRADES_FAILURE_BUDGET_SECONDS", "20.0"))
_AGG_TRADES_TOTAL_ATTEMPTS_MAX = max(1, int(os.getenv("AGG_TRADES_TOTAL_ATTEMPTS_MAX", "40")))
_AGG_TRADES_TOTAL_DURATION_SECONDS = float(os.getenv("AGG_TRADES_TOTAL_DURATION_SECONDS", "25.0"))

__all__ = [
    "AGG_TRADES_LIMIT",
    "OrderflowConfig",
    "aggregate_orderflow_series",
    "build_orderflow_block",
    "build_orderflow_per_bar",
    "bucket_trades_by_minute",
    "compute_atr_series",
    "compute_large_trade_threshold",
    "coerce_orderflow_per_bar",
    "coerce_trade_record",
    "download_agg_trades_async",
    "extract_trades",
    "load_orderflow_trades",
    "resolve_trade_sides",
    "summarise_aggregate_orderflow",
    "summarise_minute_orderflow",
]


@dataclass(slots=True)
class OrderflowConfig:
    """Configuration overrides for orderflow-derived metrics."""

    imbalance_ratio: float = 1.8
    absorption_ratio: float = 2.0
    atr_period: int = 14
    atr_band_k: float = 0.2
    large_trade_min_qty: float = 0.0
    large_trade_lookback_minutes: int = 2880
    large_trade_percentile: float = 0.99
    epsilon: float = 1e-9


def coerce_trade_record(entry: Mapping[str, Any], *, source: str) -> Dict[str, Any] | None:
    ts = safe_int(entry.get("t"))
    if ts is None:
        ts = safe_int(entry.get("T"))
    if ts is None:
        return None

    qty = safe_float(entry.get("q"))
    if qty is None or not math.isfinite(qty) or qty <= 0:
        return None

    price = safe_float(entry.get("p"))
    if price is not None and not math.isfinite(price):
        price = None

    side_raw = entry.get("side")
    side = str(side_raw).strip().lower() if isinstance(side_raw, str) else None
    if side not in {"buy", "sell"}:
        side = None

    maker_flag = entry.get("m")
    maker: bool | None
    if isinstance(maker_flag, bool):
        maker = maker_flag
    elif maker_flag in (0, 1):
        maker = bool(maker_flag)
    else:
        maker = None

    return {
        "t": int(ts),
        "qty": float(qty),
        "price": float(price) if price is not None else None,
        "side": side,
        "maker": maker,
        "source": source,
    }


def extract_trades(payload: Any, *, source: str = "snapshot") -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, Mapping):
        entries = payload.get("agg")
    else:
        entries = payload

    if not isinstance(entries, Sequence):
        return []

    records: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        record = coerce_trade_record(entry, source=source)
        if record is not None:
            records.append(record)
    return records


def resolve_trade_sides(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []

    sorted_records = sorted(
        (
            record
            for record in records
            if isinstance(record.get("t"), (int, float)) and record.get("qty") is not None
        ),
        key=lambda item: (int(item["t"]), float(item.get("price") or 0.0)),
    )

    resolved: List[Dict[str, Any]] = []
    seen: set[tuple[int, str, float, float]] = set()
    last_price: float | None = None
    last_side = "buy"

    for record in sorted_records:
        ts = int(record["t"])
        qty = float(record["qty"])
        price_value = record.get("price")
        side = record.get("side")
        reconstructed = False

        maker_flag = record.get("maker")
        if side not in {"buy", "sell"}:
            if isinstance(maker_flag, bool):
                side = "sell" if maker_flag else "buy"
            elif price_value is not None and last_price is not None:
                if price_value > last_price + 1e-9:
                    side = "buy"
                elif price_value < last_price - 1e-9:
                    side = "sell"
            if side not in {"buy", "sell"}:
                side = last_side
            reconstructed = True
        elif isinstance(maker_flag, bool):
            expected_side = "sell" if maker_flag else "buy"
            if side != expected_side:
                side = expected_side
                reconstructed = True

        price_for_key = price_value if price_value is not None else last_price
        key = (
            ts,
            side or "buy",
            round(qty, 12),
            round(price_for_key or 0.0, 8),
        )
        if key in seen:
            continue
        seen.add(key)

        resolved_side = side or last_side or "buy"
        price_output = price_value if price_value is not None else price_for_key
        if isinstance(maker_flag, bool):
            buyer_is_maker = maker_flag
        else:
            buyer_is_maker = resolved_side == "sell"

        source_ts = record.get("T")
        if isinstance(source_ts, (int, float)):
            upstream_ts = int(source_ts)
        else:
            upstream_ts = ts

        resolved_entry = {
            "t": ts,
            "T": upstream_ts,
            "q": qty,
            "qty": qty,
            "price": price_output,
            "side": resolved_side,
            "m": bool(buyer_is_maker),
            "maker": maker_flag if isinstance(maker_flag, bool) else None,
            "reconstructed": bool(reconstructed),
            "source": record.get("source"),
        }
        if price_output is not None:
            resolved_entry["p"] = price_output

        resolved.append(resolved_entry)

        if price_value is not None:
            last_price = price_value
        elif price_for_key is not None:
            last_price = price_for_key
        last_side = resolved_side

    return resolved


def bucket_trades_by_minute(
    trades: Sequence[Mapping[str, Any]],
    *,
    minute_interval: int,
    start_ms: int | None,
    end_ms: int | None,
) -> Dict[int, List[Mapping[str, Any]]]:
    buckets: Dict[int, List[Mapping[str, Any]]] = {}
    for trade in trades:
        ts = safe_int(trade.get("t"))
        if ts is None:
            continue
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts >= end_ms:
            continue
        bucket = align_to_interval(ts, minute_interval)
        buckets.setdefault(bucket, []).append(trade)
    return buckets


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 1:
        return max(values)
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(percentile * (len(ordered) - 1)))))
    return ordered[index]


def compute_large_trade_threshold(
    trades: Sequence[Mapping[str, Any]],
    *,
    cutoff_ts: int | None,
    lookback_minutes: int,
    minute_interval: int,
    base_threshold: float,
    percentile: float,
) -> float:
    if not trades:
        return max(0.0, base_threshold)

    reference_cutoff = None
    if cutoff_ts is not None:
        reference_cutoff = cutoff_ts - max(0, lookback_minutes - 1) * minute_interval

    quantities: List[float] = []
    for trade in trades:
        ts = safe_int(trade.get("t"))
        if ts is None:
            continue
        if reference_cutoff is not None and ts < reference_cutoff:
            continue
        qty = coerce_float(trade.get("q"))
        if math.isfinite(qty):
            quantities.append(float(qty))

    if not quantities:
        for trade in trades:
            qty = coerce_float(trade.get("q"))
            if math.isfinite(qty):
                quantities.append(float(qty))

    percentile_value = _percentile(quantities, percentile) if quantities else None
    threshold = max(0.0, base_threshold)
    if percentile_value is not None:
        threshold = max(threshold, float(percentile_value))
    return threshold


def compute_atr_series(
    candles: Sequence[Mapping[str, Any]],
    *,
    period: int,
) -> Dict[int, float]:
    if period <= 0 or not candles:
        return {}

    atr_values: Dict[int, float] = {}
    atr_window: List[float] = []

    prev_close = None
    for candle in sorted(candles, key=lambda item: safe_int(item.get("t")) or 0):
        ts = safe_int(candle.get("t"))
        if ts is None:
            continue

        high = safe_float(candle.get("h"))
        low = safe_float(candle.get("l"))
        close = safe_float(candle.get("c"))
        if high is None or low is None or close is None:
            continue

        tr_components = [high - low]
        if prev_close is not None:
            tr_components.extend(
                [
                    abs(high - prev_close),
                    abs(low - prev_close),
                ]
            )
        true_range = max(tr_components) if tr_components else 0.0

        atr_window.append(true_range)
        if len(atr_window) > period:
            atr_window.pop(0)

        if len(atr_window) == period:
            atr = sum(atr_window) / period
            atr_values[ts] = atr
        prev_close = close

    return atr_values


async def download_agg_trades_async(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    allow_network: bool,
    trace_ctx: TraceContext | None,
    budget: Any,
    page_span_ms: int,
    fetch_override: Any = None,
    budget_exceeded_exc: tuple[type[BaseException], ...] | type[BaseException] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "attempted": bool(allow_network),
        "downloaded": 0,
        "status": None,
        "batches": 0,
        "attempts": 0,
        "duration_ms": 0,
        "fallback_reason": None,
    }
    if start_ms >= end_ms:
        diag["duration_ms"] = 0
        return [], diag

    operation_started = time.perf_counter()
    failure_attempts = 0
    failure_first_at: float | None = None

    store = get_store()
    store_rows = await asyncio.to_thread(
        store.fetch_agg_trades,
        symbol,
        start_ms,
        end_ms,
    )
    if store_rows:
        normalised = extract_trades(store_rows, source="vision_store")
        diag.update(
            {
                "downloaded": len(normalised),
                "status": "vision_store",
                "attempted": False,
                "batches": 1,
            }
        )
        diag["duration_ms"] = round((time.perf_counter() - operation_started) * 1000.0, 2)
        return normalised, diag

    if not allow_network:
        diag["duration_ms"] = round((time.perf_counter() - operation_started) * 1000.0, 2)
        return [], diag

    symbol_upper = symbol.upper()
    if symbol_upper.startswith("TEST") or AppConfig.load().offline:
        diag["status"] = "network_skipped"
        diag["duration_ms"] = round((time.perf_counter() - operation_started) * 1000.0, 2)
        return [], diag

    if callable(fetch_override):
        result = fetch_override(symbol, start_ms, end_ms, AGG_TRADES_LIMIT)
        rows = await result if inspect.isawaitable(result) else result  # type: ignore[arg-type]
        normalised = extract_trades(rows, source="download")
        diag["downloaded"] = len(normalised)
        diag["status"] = 200 if normalised else None
        if diag["status"] is None and not normalised:
            diag["status"] = 204
        diag["duration_ms"] = round((time.perf_counter() - operation_started) * 1000.0, 2)
        return normalised, diag

    budget_exc_tuple: tuple[type[BaseException], ...]
    if budget_exceeded_exc is None:
        budget_exc_tuple = ()
    elif isinstance(budget_exceeded_exc, tuple):
        budget_exc_tuple = budget_exceeded_exc
    else:
        budget_exc_tuple = (budget_exceeded_exc,)

    trades: List[Dict[str, Any]] = []
    scope = "orderflow.aggTrades"
    cursor = start_ms

    while cursor < end_ms:
        if failure_attempts >= _AGG_TRADES_FAILURE_MAX_ATTEMPTS:
            diag["fallback_reason"] = diag.get("fallback_reason") or "retry_limit"
            LOGGER.warning(
                "Orderflow agg-trade download aborted after retry limit",
                extra={
                    "symbol": symbol,
                    "scope": scope,
                    "cursor": cursor,
                    "attempts": failure_attempts,
                    "window_start": start_ms,
                    "window_end": end_ms,
                },
            )
            break
        if failure_first_at is not None:
            elapsed_fail = time.perf_counter() - failure_first_at
            if elapsed_fail >= _AGG_TRADES_FAILURE_BUDGET_SECONDS:
                diag["fallback_reason"] = diag.get("fallback_reason") or "retry_timeout"
                LOGGER.warning(
                    "Orderflow agg-trade download aborted after timeout budget",
                    extra={
                        "symbol": symbol,
                        "scope": scope,
                        "cursor": cursor,
                        "attempts": failure_attempts,
                        "window_start": start_ms,
                        "window_end": end_ms,
                        "elapsed_s": round(elapsed_fail, 3),
                    },
                )
                break

        if budget_exc_tuple:
            try:
                budget.raise_if_exceeded("orderflow_trades_download")
            except budget_exc_tuple as exc:  # type: ignore[misc]
                LOGGER.warning(
                    "Orderflow trade download hit budget",
                    extra={
                        "symbol": symbol,
                        "cursor": cursor,
                        "window_start": start_ms,
                        "window_end": end_ms,
                        "stage": getattr(exc, "stage", None),
                    },
                )
                diag["status"] = "budget_exceeded"
                break
        else:
            budget.raise_if_exceeded("orderflow_trades_download")

        page_end = min(end_ms, cursor + page_span_ms)
        diag["attempts"] += 1
        total_elapsed = time.perf_counter() - operation_started
        if diag["attempts"] > _AGG_TRADES_TOTAL_ATTEMPTS_MAX:
            diag["fallback_reason"] = "attempt_limit"
            LOGGER.warning(
                "Orderflow agg-trade download hit attempt ceiling",
                extra={
                    "symbol": symbol,
                    "scope": scope,
                    "cursor": cursor,
                    "attempt": diag["attempts"],
                    "window_start": start_ms,
                    "window_end": end_ms,
                    "max_attempts": _AGG_TRADES_TOTAL_ATTEMPTS_MAX,
                },
            )
            break
        if total_elapsed >= _AGG_TRADES_TOTAL_DURATION_SECONDS:
            diag["fallback_reason"] = "time_budget"
            LOGGER.warning(
                "Orderflow agg-trade download exceeded time budget",
                extra={
                    "symbol": symbol,
                    "scope": scope,
                    "cursor": cursor,
                    "attempt": diag["attempts"],
                    "window_start": start_ms,
                    "window_end": end_ms,
                    "elapsed_s": round(total_elapsed, 3),
                    "budget_s": _AGG_TRADES_TOTAL_DURATION_SECONDS,
                },
            )
            break
        attempt_started = time.perf_counter()
        LOGGER.info(
            "Orderflow agg-trade request",
            extra={
                "symbol": symbol,
                "scope": scope,
                "attempt": diag["attempts"],
                "cursor": cursor,
                "page_end": page_end,
                "window_start": start_ms,
                "window_end": end_ms,
            },
        )
        try:
            payload = await asyncio.wait_for(
                fetch_um_agg_trades(
                    symbol,
                    start_time=cursor,
                    end_time=page_end,
                    limit=AGG_TRADES_LIMIT,
                ),
                timeout=_AGG_TRADES_REQUEST_TIMEOUT,
            )
            failure_attempts = 0
            failure_first_at = None
        except BinanceAPIException as exc:
            status = getattr(exc, "status_code", None)
            diag["status"] = status
            failure_attempts += 1
            failure_first_at = failure_first_at or time.perf_counter()
            reason = "rate_limited" if status in RATE_LIMIT_STATUSES else f"api_error_{status or 'unknown'}"
            diag["fallback_reason"] = diag.get("fallback_reason") or reason
            LOGGER.warning(
                "Orderflow agg-trade download failed",
                extra={
                    "symbol": symbol,
                    "scope": scope,
                    "cursor": cursor,
                    "page_end": page_end,
                    "status": status,
                    "attempt": diag["attempts"],
                    "reason": reason,
                },
                exc_info=False,
            )
            if status in RATE_LIMIT_STATUSES or (isinstance(status, int) and status >= 500):
                break
            raise
        except asyncio.TimeoutError:
            failure_attempts += 1
            failure_first_at = failure_first_at or time.perf_counter()
            diag["status"] = "timeout"
            diag["fallback_reason"] = diag.get("fallback_reason") or "request_timeout"
            LOGGER.warning(
                "Orderflow agg-trade download timed out",
                extra={
                    "symbol": symbol,
                    "scope": scope,
                    "cursor": cursor,
                    "page_end": page_end,
                    "attempt": diag["attempts"],
                    "timeout_s": _AGG_TRADES_REQUEST_TIMEOUT,
                },
            )
            break
        except (BinanceRequestException, aiohttp.ClientError) as exc:
            LOGGER.debug(
                "Failed to download agg trades",
                exc_info=exc,
                extra={"symbol": symbol, "cursor": cursor, "page_end": page_end},
            )
            failure_attempts += 1
            failure_first_at = failure_first_at or time.perf_counter()
            diag["status"] = "request_failed"
            diag["fallback_reason"] = diag.get("fallback_reason") or "request_error"
            break

        diag["batches"] += 1

        if not isinstance(payload, list):
            diag["status"] = "invalid_payload"
            break
        if not payload:
            cursor = page_end + 1
            continue

        trades.extend(extract_trades(payload, source="download"))
        diag["downloaded"] = len(trades)
        diag["status"] = 200

        last_trade_time = max(int(row.get("t", row.get("T", cursor))) for row in payload)
        cursor = last_trade_time + 1

        diag["last_fetch_ms"] = round((time.perf_counter() - attempt_started) * 1000.0, 2)

    diag["duration_ms"] = round((time.perf_counter() - operation_started) * 1000.0, 2)

    return trades, diag


async def load_orderflow_trades(
    symbol: str,
    start_ms: int,
    end_ms: int,
    trades_payload: Any,
    *,
    allow_network: bool,
    trace_ctx: TraceContext | None,
    budget: Any,
    page_span_ms: int,
    fetch_override: Any = None,
    budget_exceeded_exc: tuple[type[BaseException], ...] | type[BaseException] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    snapshot_records = extract_trades(trades_payload, source="snapshot")
    snapshot_filtered = [record for record in snapshot_records if start_ms <= record["t"] <= end_ms]

    download_trace = trace_ctx.child(stage="orderflow.trades") if trace_ctx is not None else None
    downloaded_records, download_diag = await download_agg_trades_async(
        symbol,
        start_ms,
        end_ms,
        allow_network=allow_network,
        trace_ctx=download_trace,
        budget=budget,
        page_span_ms=page_span_ms,
        fetch_override=fetch_override,
        budget_exceeded_exc=budget_exceeded_exc,
    )

    combined_records: List[Dict[str, Any]] = []
    combined_records.extend(snapshot_filtered)
    combined_records.extend(downloaded_records)

    resolved_trades = resolve_trade_sides(combined_records)
    filtered_trades = [trade for trade in resolved_trades if start_ms <= trade["t"] <= end_ms]

    diag: Dict[str, Any] = {
        "window": {"from": start_ms, "to": end_ms},
        "snapshot_trades": len(snapshot_filtered),
        "downloaded_trades": len(downloaded_records),
        "resolved_trades": len(filtered_trades),
        "sources": [],
        "download": download_diag,
    }
    diag["fallback_reason"] = download_diag.get("fallback_reason")
    if download_diag.get("status") is not None:
        diag["status"] = download_diag.get("status")
    if download_diag.get("attempts") is not None:
        diag["attempts"] = download_diag.get("attempts")
    if snapshot_filtered:
        diag["sources"].append("snapshot")
    if downloaded_records:
        diag["sources"].append("download")

    return filtered_trades, diag


def build_orderflow_per_bar(
    minute_index: Mapping[int, Mapping[str, Any]],
    trades_by_minute: Mapping[int, Sequence[Mapping[str, Any]]],
    expected_minutes: Sequence[int],
    *,
    config: OrderflowConfig,
    large_trade_threshold: float,
    atr_series: Mapping[int, float],
    minute_interval_ms: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    running_cvd = 0.0
    coverage_diag: Dict[str, Any] = {
        "expected_minutes": len(expected_minutes),
        "with_trades": 0,
        "missing_minutes": 0,
        "gaps": [],
        "reconstructed_minutes": 0,
    }
    gap_start: int | None = None
    any_trades_observed = False

    for ts in expected_minutes:
        minute_trades = list(trades_by_minute.get(ts, []))
        has_trades = bool(minute_trades)
        if has_trades:
            coverage_diag["with_trades"] += 1
            any_trades_observed = True
        else:
            coverage_diag["missing_minutes"] += 1

        if not has_trades and gap_start is None:
            gap_start = ts
        elif has_trades and gap_start is not None:
            coverage_diag["gaps"].append({"from": gap_start, "to": ts - minute_interval_ms})
            gap_start = None

        ask_volume = sum(
            coerce_float(trade.get("q")) for trade in minute_trades if str(trade.get("side")).lower() == "buy"
        )
        bid_volume = sum(
            coerce_float(trade.get("q")) for trade in minute_trades if str(trade.get("side")).lower() == "sell"
        )
        reconstructed_minutes = any(bool(trade.get("reconstructed")) for trade in minute_trades)
        if reconstructed_minutes:
            coverage_diag["reconstructed_minutes"] += 1

        delta = ask_volume - bid_volume
        running_cvd += delta

        candle = minute_index.get(ts)
        if candle:
            close_price = coerce_float(candle.get("c"))
            high_price = coerce_float(candle.get("h"))
            low_price = coerce_float(candle.get("l"))
        else:
            sorted_trades = sorted(minute_trades, key=lambda trade: safe_int(trade.get("t")) or ts)
            trade_prices = [coerce_float(trade.get("price")) for trade in sorted_trades if trade.get("price") is not None]
            if trade_prices:
                close_price = trade_prices[-1]
                high_price = max(trade_prices)
                low_price = min(trade_prices)
            else:
                close_price = high_price = low_price = 0.0

        atr_value = atr_series.get(ts)
        if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
            atr_value = max(high_price - low_price, 0.0)
        band = atr_value * config.atr_band_k

        close_near_high = abs(high_price - close_price) <= band if band > 0 else math.isclose(high_price, close_price)
        close_near_low = abs(close_price - low_price) <= band if band > 0 else math.isclose(low_price, close_price)

        imbalance_buy = False
        imbalance_sell = False
        if ask_volume > 0:
            imbalance_buy = (ask_volume / max(bid_volume, config.epsilon)) >= config.imbalance_ratio
        if bid_volume > 0:
            imbalance_sell = (bid_volume / max(ask_volume, config.epsilon)) >= config.imbalance_ratio

        absorption_high = (
            delta < 0
            and close_near_high
            and bid_volume > 0
            and (bid_volume / max(ask_volume, config.epsilon)) >= config.absorption_ratio
        )
        absorption_low = (
            delta > 0
            and close_near_low
            and ask_volume > 0
            and (ask_volume / max(bid_volume, config.epsilon)) >= config.absorption_ratio
        )

        threshold = max(0.0, large_trade_threshold)
        large_count = sum(1 for trade in minute_trades if coerce_float(trade.get("q")) >= threshold)

        series.append(
            {
                "ts": ts,
                "t": isoformat_utc(ts),
            "delta": delta,
            "cvd": running_cvd,
            "bid_vol": bid_volume,
            "ask_vol": ask_volume,
            "volume": ask_volume + bid_volume,
            "has_trades": has_trades,
            "reconstructed": bool(reconstructed_minutes),
            "large_trades_count": int(large_count),
            "absorption_high": bool(absorption_high),
            "absorption_low": bool(absorption_low),
            "imbalance_buy": bool(imbalance_buy),
            "imbalance_sell": bool(imbalance_sell),
            }
        )

    if gap_start is not None:
        coverage_diag["gaps"].append({"from": gap_start, "to": expected_minutes[-1]})

    if not any_trades_observed:
        coverage_diag["no_trades"] = True
        return [], coverage_diag

    return series, coverage_diag


def aggregate_orderflow_series(
    series: Sequence[Mapping[str, Any]],
    *,
    interval_ms: int,
    minute_interval: int,
) -> List[Dict[str, Any]]:
    if not series:
        return []

    buckets: Dict[int, List[Mapping[str, Any]]] = {}
    for item in series:
        if not isinstance(item, Mapping):
            continue
        ts_value = item.get("ts")
        try:
            ts_int = int(ts_value)
        except (TypeError, ValueError):
            continue
        bucket_start = align_to_interval(ts_int, interval_ms)
        buckets.setdefault(bucket_start, []).append(item)

    aggregated: List[Dict[str, Any]] = []
    running_cvd = 0.0

    for bucket_start in sorted(buckets):
        bucket_entries = sorted(
            buckets[bucket_start],
            key=lambda entry: int(entry.get("ts") or 0),
        )
        if not bucket_entries:
            continue
        ask_volume = sum(coerce_float(entry.get("ask_vol")) for entry in bucket_entries)
        bid_volume = sum(coerce_float(entry.get("bid_vol")) for entry in bucket_entries)
        delta = sum(coerce_float(entry.get("delta")) for entry in bucket_entries)
        if delta == 0.0:
            delta = ask_volume - bid_volume
        running_cvd += delta
        aggregated.append(
            {
                "ts": bucket_start,
                "t": isoformat_utc(bucket_start),
                "delta_sum": delta,
                "cvd_close": running_cvd,
                "vol_sum": ask_volume + bid_volume,
                "bars": len(bucket_entries),
            }
        )

    return aggregated


def summarise_minute_orderflow(series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_delta = sum(coerce_float(entry.get("delta")) for entry in series)
    total_volume = sum(coerce_float(entry.get("volume")) for entry in series)
    minutes_with_trades = sum(1 for entry in series if entry.get("has_trades"))
    reconstructed_minutes = sum(1 for entry in series if entry.get("reconstructed"))
    return {
        "minutes": len(series),
        "minutes_with_trades": minutes_with_trades,
        "delta_sum": total_delta,
        "volume_sum": total_volume,
        "reconstructed_minutes": reconstructed_minutes,
    }


def summarise_aggregate_orderflow(series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_delta = sum(coerce_float(entry.get("delta_sum")) for entry in series)
    total_volume = sum(coerce_float(entry.get("vol_sum")) for entry in series)
    bars_total = sum(int(entry.get("bars", 0)) for entry in series)
    return {
        "bars": len(series),
        "delta_sum": total_delta,
        "volume_sum": total_volume,
        "bars_total": bars_total,
    }


def coerce_orderflow_per_bar(
    orderflow_source: Mapping[str, Any],
    *,
    minute_interval: int,
    target_length: int,
) -> List[Dict[str, Any]]:
    rows = []
    per_bar = orderflow_source.get("per_bar")
    if isinstance(per_bar, Sequence) and not isinstance(per_bar, (str, bytes)):
        for entry in per_bar:
            if not isinstance(entry, Mapping):
                continue
            ts_value = entry.get("ts") or entry.get("t")
            ts_int = safe_int(ts_value)
            if ts_int is None:
                continue
            aligned_ts = align_to_interval(ts_int, minute_interval)
            rows.append(
                {
                    "ts": aligned_ts,
                    "delta": coerce_float(entry.get("delta")),
                    "cvd": coerce_float(entry.get("cvd")),
                    "bid_vol": coerce_float(entry.get("bid_vol")),
                    "ask_vol": coerce_float(entry.get("ask_vol")),
                    "imbalance": entry.get("imbalance"),
                    "absorption_high": bool(entry.get("absorption_high")),
                    "absorption_low": bool(entry.get("absorption_low")),
                    "large_trades_count": int(entry.get("large_trades_count", 0))
                    if isinstance(entry.get("large_trades_count"), (int, float))
                    else 0,
                }
            )

    if not rows:
        return []

    dedup: Dict[int, Dict[str, Any]] = {}
    for entry in rows:
        ts_int = int(entry.get("ts", 0))
        dedup[ts_int] = entry

    ordered_rows = [dedup[key] for key in sorted(dedup)]
    if len(ordered_rows) > target_length:
        ordered_rows = ordered_rows[-target_length:]

    running_cvd = 0.0
    for entry in ordered_rows:
        running_cvd += coerce_float(entry.get("delta"))
        entry.setdefault("cvd", running_cvd)
        entry.setdefault("bid_vol", 0.0)
        entry.setdefault("ask_vol", 0.0)
        entry.setdefault(
            "volume",
            coerce_float(entry.get("ask_vol")) + coerce_float(entry.get("bid_vol")),
        )
        entry.setdefault("has_trades", bool(entry.get("volume")))
        entry.setdefault("reconstructed", False)
        entry.setdefault(
            "imbalance_buy",
            bool(entry.get("imbalance")) and coerce_float(entry.get("imbalance")) > 1.0,
        )
        entry.setdefault(
            "imbalance_sell",
            bool(entry.get("imbalance")) and coerce_float(entry.get("imbalance")) < 1.0,
        )
        entry.setdefault("absorption_high", bool(entry.get("absorption_high")))
        entry.setdefault("absorption_low", bool(entry.get("absorption_low")))
        entry.setdefault(
            "large_trades_count",
            int(entry.get("large_trades_count", 0))
            if isinstance(entry.get("large_trades_count"), (int, float))
            else 0,
        )
        entry.setdefault("t", isoformat_utc(entry.get("ts", 0)))

    return ordered_rows


def build_proxy_orderflow_from_minutes(
    minute_candles: Sequence[Mapping[str, Any]],
    *,
    minute_interval: int,
    target_length: int,
) -> List[Dict[str, Any]]:
    if not minute_candles or minute_interval <= 0:
        return []
    proxy_series: List[Dict[str, Any]] = []
    running_cvd = 0.0
    sorted_minutes = sorted(
        (candle for candle in minute_candles if isinstance(candle, Mapping)),
        key=lambda candle: safe_int(candle.get("t")) or 0,
    )
    for candle in sorted_minutes:
        ts = safe_int(candle.get("t"))
        if ts is None:
            continue
        volume = coerce_float(candle.get("v"))
        if volume < 0.0:
            volume = 0.0
        open_price = coerce_float(candle.get("o")) or 0.0
        close_price = coerce_float(candle.get("c")) or 0.0
        delta_sign = 0
        if close_price > open_price:
            delta_sign = 1
        elif close_price < open_price:
            delta_sign = -1
        delta = volume * delta_sign
        ask_volume = max(0.0, (volume + delta) / 2.0)
        bid_volume = max(0.0, volume - ask_volume)
        running_cvd += delta
        proxy_series.append(
            {
                "ts": ts,
                "t": isoformat_utc(ts),
                "delta": delta,
                "cvd": running_cvd,
                "bid_vol": bid_volume,
                "ask_vol": ask_volume,
                "volume": ask_volume + bid_volume,
                "has_trades": volume > 0.0,
                "reconstructed": False,
                "imbalance": None,
                "absorption_high": False,
                "absorption_low": False,
                "imbalance_buy": ask_volume > bid_volume,
                "imbalance_sell": bid_volume > ask_volume,
                "large_trades_count": 0,
            }
        )
    if len(proxy_series) > target_length:
        proxy_series = proxy_series[-target_length:]
    elif proxy_series and len(proxy_series) < target_length:
        missing = target_length - len(proxy_series)
        first_ts = safe_int(proxy_series[0].get("ts")) or 0
        minute_ms = max(60_000, int(minute_interval) * 60_000 if minute_interval < 1_000 else int(minute_interval))
        if minute_interval >= 60_000:
            minute_ms = int(minute_interval)
        placeholders: List[Dict[str, Any]] = []
        for index in range(missing, 0, -1):
            ts = first_ts - index * minute_ms
            placeholders.append(
                {
                    "ts": ts,
                    "t": isoformat_utc(ts),
                    "delta": 0.0,
                    "cvd": 0.0,
                    "bid_vol": 0.0,
                    "ask_vol": 0.0,
                    "volume": 0.0,
                    "has_trades": False,
                    "reconstructed": True,
                    "imbalance": None,
                    "absorption_high": False,
                    "absorption_low": False,
                    "imbalance_buy": False,
                    "imbalance_sell": False,
                    "large_trades_count": 0,
                }
            )
        proxy_series = placeholders + proxy_series
    return proxy_series


async def build_orderflow_block(
    minute_candles: Sequence[Mapping[str, Any]],
    trades_payload: Any,
    *,
    orderflow_start_ms: int,
    orderflow_end_ms: int,
    config: OrderflowConfig,
    symbol: str,
    allow_network: bool,
    trace_ctx: TraceContext | None,
    budget: Any,
    orderflow_source: Mapping[str, Any] | None,
    window_minutes: int,
    page_span_ms: int,
    target_timeframes: Tuple[str, ...],
    aggregated_timeframes: Tuple[str, ...],
    minute_interval_ms: int,
    timeframe_to_ms: Mapping[str, int],
    target_length: int,
    fetch_override: Any = None,
    budget_exceeded_exc: tuple[type[BaseException], ...] | type[BaseException] | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    minute_interval = max(1, int(minute_interval_ms))
    window_minutes = max(window_minutes, 1)
    ordered_timeframes = ("1m",) + tuple(tf for tf in target_timeframes if tf != "1m")
    result: Dict[str, Dict[str, Any]] = {tf: {"per_bar": []} for tf in ordered_timeframes}
    diag: Dict[str, Any] = {
        "source": None,
        "delta_source": None,
        "series_lengths": {},
        "trimmed": {},
        "target_length": target_length,
        "window_minutes": int(window_minutes),
        "window": {"from": orderflow_start_ms, "to": orderflow_end_ms},
    }
    if minute_interval <= 0:
        diag["source"] = "invalid_interval"
        diag["delta_source"] = "none"
        return result, diag

    expected_minutes = build_expected_times(orderflow_start_ms, orderflow_end_ms, minute_interval)

    trades, trades_diag = await load_orderflow_trades(
        symbol,
        orderflow_start_ms,
        orderflow_end_ms,
        trades_payload,
        allow_network=allow_network,
        trace_ctx=trace_ctx,
        budget=budget,
        page_span_ms=page_span_ms,
        fetch_override=fetch_override,
        budget_exceeded_exc=budget_exceeded_exc,
    )
    diag["trades"] = trades_diag

    trades_by_minute = bucket_trades_by_minute(
        trades,
        minute_interval=minute_interval,
        start_ms=orderflow_start_ms,
        end_ms=orderflow_end_ms + minute_interval,
    )

    last_ts = orderflow_end_ms
    threshold = compute_large_trade_threshold(
        trades,
        cutoff_ts=last_ts,
        lookback_minutes=config.large_trade_lookback_minutes,
        minute_interval=minute_interval,
        base_threshold=config.large_trade_min_qty,
        percentile=config.large_trade_percentile,
    )

    minute_index = {
        safe_int(candle.get("t")): candle
        for candle in minute_candles
        if safe_int(candle.get("t")) is not None
    }
    atr_series = compute_atr_series(minute_candles, period=config.atr_period)

    minute_series, coverage_diag = build_orderflow_per_bar(
        minute_index,
        trades_by_minute,
        expected_minutes,
        config=config,
        large_trade_threshold=threshold,
        atr_series=atr_series,
        minute_interval_ms=minute_interval,
    )
    source_tag = "trades" if minute_series else None

    if not minute_series and orderflow_source is not None:
        minute_series = coerce_orderflow_per_bar(
            orderflow_source, minute_interval=minute_interval, target_length=target_length
        )
        if minute_series:
            source_tag = "orderflow_source"

    if not minute_series:
        minute_series = build_proxy_orderflow_from_minutes(
            minute_candles,
            minute_interval=minute_interval,
            target_length=target_length,
        )
        if minute_series:
            source_tag = "proxy"
            if trace_ctx is not None:
                trace_ctx.warn(
                    "orderflow.proxy",
                    scope="orderflow",
                    reason=trades_diag.get("fallback_reason") if isinstance(trades_diag, Mapping) else None,
                )
            LOGGER.warning(
                "Orderflow proxy built from minute candles",
                extra={
                    "symbol": symbol,
                    "window_from": orderflow_start_ms,
                    "window_to": orderflow_end_ms,
                    "fallback_reason": trades_diag.get("fallback_reason") if isinstance(trades_diag, Mapping) else None,
                },
            )

    if not minute_series and expected_minutes:
        placeholder_ts = expected_minutes[-1]
        minute_series = [
            {
                "ts": placeholder_ts,
                "t": isoformat_utc(placeholder_ts),
                "delta": 0.0,
                "cvd": 0.0,
                "bid_vol": 0.0,
                "ask_vol": 0.0,
                "volume": 0.0,
                "has_trades": False,
                "reconstructed": False,
                "large_trades_count": 0,
                "absorption_high": False,
                "absorption_low": False,
                "imbalance_buy": False,
                "imbalance_sell": False,
                "placeholder": True,
            }
        ]
        diag.setdefault("placeholders", []).append("1m")
        coverage_diag.setdefault("missing_minutes", len(expected_minutes))

    if not minute_series:
        raise RuntimeError("orderflow_missing_trades")

    if source_tag == "proxy":
        diag["delta_source"] = "proxy"
    elif source_tag == "orderflow_source":
        diag["delta_source"] = "external"
    elif source_tag == "trades":
        diag["delta_source"] = "trades"
    else:
        diag["delta_source"] = "none"
    diag["source"] = source_tag or ("trades" if trades else "none")

    minute_series.sort(key=lambda entry: safe_int(entry.get("ts")) or 0)

    footprint_rows: List[Dict[str, Any]] = []
    for item in minute_series:
        ts_value = safe_int(item.get("ts"))
        if ts_value is None:
            continue
        footprint_rows.append(
            {
                "ts": ts_value,
                "t": isoformat_utc(ts_value),
                "ask": coerce_float(item.get("ask_vol")),
                "bid": coerce_float(item.get("bid_vol")),
                "volume": coerce_float(item.get("volume")),
                "delta": coerce_float(item.get("delta")),
                "absorption_high": bool(item.get("absorption_high")),
                "absorption_low": bool(item.get("absorption_low")),
                "imbalance_buy": bool(item.get("imbalance_buy")),
                "imbalance_sell": bool(item.get("imbalance_sell")),
            }
        )

    diag["series_lengths"]["1m_raw"] = len(minute_series)
    diag["trimmed"]["1m"] = 0
    diag["series_lengths"]["1m"] = len(minute_series)
    diag["coverage"] = coverage_diag

    minute_summary = summarise_minute_orderflow(minute_series)
    result["1m"] = {"per_bar": minute_series, "summary": minute_summary}

    footprint_summary = compute_orderflow_aggregates(footprint_rows)
    result["footprint"] = {"per_bar": footprint_rows, "summary": footprint_summary}
    diag["series_lengths"]["footprint"] = len(footprint_rows)

    for tf in aggregated_timeframes:
        interval_ms = timeframe_to_ms.get(tf)
        if not interval_ms or int(interval_ms) <= minute_interval:
            continue
        aggregated = aggregate_orderflow_series(
            minute_series,
            interval_ms=int(interval_ms),
            minute_interval=minute_interval,
        )
        result.setdefault(tf, {})["per_bar"] = aggregated
        result[tf]["summary"] = summarise_aggregate_orderflow(aggregated)
        diag["series_lengths"][tf] = len(aggregated)

    for tf in target_timeframes:
        if tf == "1m":
            continue
        series = result.get(tf, {}).get("per_bar", [])
        diag["series_lengths"][tf] = len(series) if isinstance(series, Sequence) else 0

    return result, diag
