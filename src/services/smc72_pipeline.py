"""SMC 72-hour pipeline orchestration."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx

from .binance_ingest import ingest_agg_trades, ingest_klines, fetch_premium_index_series
from .cache import FileCache
from .coverage import compute_coverage
from .depth_checks import capture_depth_series
from .tracing import TraceContext

LOGGER = logging.getLogger(__name__)
MINUTE_MS = 60_000
WINDOW_MINUTES = 72 * 60
DELTA_WINDOW = 240  # 4 hours


@dataclass(slots=True)
class PipelineDiagnostics:
    rate_limit_events: int = 0
    retries: int = 0
    vision_backfills: int = 0
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    problem_windows: List[Dict[str, Any]] = field(default_factory=list)
    recon_mismatch_pct: float | None = None
    vwap_oob_pct: float | None = None
    recon_minutes: List[int] = field(default_factory=list)
    vwap_oob_minutes: List[int] = field(default_factory=list)
    silent_minutes: List[int] = field(default_factory=list)
    cvd_baseline: float = 0.0
    cvd_baseline_source: str = "cold_start"


def _align_minute(ts: int) -> int:
    return (ts // MINUTE_MS) * MINUTE_MS


def _align_minute_up(ts: int) -> int:
    if ts % MINUTE_MS == 0:
        return ts
    return ((ts // MINUTE_MS) + 1) * MINUTE_MS


def _build_minutes_baseline(start_ms: int, end_ms: int, hours: int = 72) -> List[Dict[str, Any]]:
    count = max(1, hours * 60)
    end_aligned = _align_minute(end_ms - 1)
    start_candidate = end_aligned - (count - 1) * MINUTE_MS
    minimum_start = _align_minute_up(start_ms)
    if start_candidate < minimum_start:
        start_candidate = minimum_start
        end_aligned = start_candidate + (count - 1) * MINUTE_MS
    timestamps = [start_candidate + index * MINUTE_MS for index in range(count)]
    return [{"ts_min": ts} for ts in timestamps]


def _index_by_ts(entries: Sequence[Mapping[str, Any]], key: str = "ts_min") -> Dict[int, Mapping[str, Any]]:
    index: Dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        ts = entry.get(key)
        if isinstance(ts, (int, float)):
            index[int(ts)] = entry
    return index


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _sliding_stats(series: Sequence[float], window: int) -> Tuple[List[float], List[float]]:
    means: List[float] = []
    stds: List[float] = []
    window_values: List[float] = []
    for index, value in enumerate(series):
        window_values.append(value)
        if len(window_values) > window:
            window_values.pop(0)
        if len(window_values) < window:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(_mean(window_values))
            stds.append(_std(window_values))
    return means, stds


def _reconcile_minutes(
    klines: Sequence[Mapping[str, Any]],
    agg_minutes: Sequence[Mapping[str, Any]],
    *,
    tolerance_pct: float = 0.1,
    backfill: Optional[Callable[[int, int], Sequence[Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[int], List[int]]:
    kline_index = _index_by_ts(klines)
    agg_index = _index_by_ts(agg_minutes)
    recon_flags = 0
    vwap_oob_flags = 0
    total_minutes = len(kline_index)
    recon_minutes: List[int] = []
    vwap_oob_minutes: List[int] = []

    reconciled: List[Dict[str, Any]] = []
    for ts, kline in sorted(kline_index.items()):
        agg = agg_index.get(ts, {})
        volume = float(kline.get("volume") or 0.0)
        agg_vol = float(agg.get("vol") or 0.0)
        vwap = agg.get("vwap_min")
        low = float(kline.get("low") or 0.0)
        high = float(kline.get("high") or 0.0)
        vol_diff_abs = abs(agg_vol - volume)
        denom = volume if volume > 0 else 1e-12
        vol_diff_pct = vol_diff_abs / denom
        recon_flag = volume > 0 and vol_diff_pct > tolerance_pct
        if recon_flag and backfill is not None:
            start_window = ts
            end_window = ts + 5 * MINUTE_MS
            try:
                refreshed = backfill(start_window, end_window)
            except Exception:
                refreshed = None
            if refreshed:
                refreshed_index = _index_by_ts(refreshed)
                refreshed_agg = refreshed_index.get(ts)
                if refreshed_agg is not None:
                    agg = refreshed_agg
                    agg_vol = float(agg.get("vol") or 0.0)
                    vol_diff_abs = abs(agg_vol - volume)
                    denom = volume if volume > 0 else 1e-12
                    vol_diff_pct = vol_diff_abs / denom
                    recon_flag = volume > 0 and vol_diff_pct > tolerance_pct
        if recon_flag:
            recon_flags += 1
            recon_minutes.append(ts)
        vwap_oob = bool(agg_vol > 0 and vwap is not None and (vwap < low or vwap > high))
        if vwap_oob:
            vwap_oob_flags += 1
            vwap_oob_minutes.append(ts)
        entry = dict(agg)
        entry["ts_min"] = ts
        entry["recon_flag"] = recon_flag
        entry["vol_diff_abs"] = vol_diff_abs
        entry["vol_diff_pct"] = vol_diff_pct
        entry["vwap_oob"] = vwap_oob
        entry["volume_kline"] = volume
        reconciled.append(entry)

    stats = {
        "recon_mismatch_pct": (recon_flags / total_minutes * 100.0) if total_minutes else 0.0,
        "vwap_oob_pct": (vwap_oob_flags / total_minutes * 100.0) if total_minutes else 0.0,
    }
    return reconciled, stats, recon_minutes, vwap_oob_minutes


def _join_timeseries(
    minutes_baseline: Sequence[Mapping[str, Any]],
    klines: Sequence[Mapping[str, Any]],
    agg_minutes: Sequence[Mapping[str, Any]],
    premium_samples: Sequence[Mapping[str, Any]],
    *,
    cvd_baseline: float,
) -> List[Dict[str, Any]]:
    kline_index = _index_by_ts(klines)
    agg_index = _index_by_ts(agg_minutes)
    premium_sorted = sorted(premium_samples, key=lambda entry: entry["time"])
    premium_cursor = 0
    joined: List[Dict[str, Any]] = []
    last_premium: Dict[str, Any] | None = None
    last_cvd = cvd_baseline
    for minute in minutes_baseline:
        ts = minute["ts_min"]
        kline = kline_index.get(ts)
        agg = agg_index.get(ts, {})
        while premium_cursor < len(premium_sorted) and premium_sorted[premium_cursor]["time"] <= ts:
            last_premium = premium_sorted[premium_cursor]
            premium_cursor += 1
        record = {
            "ts_min": ts,
            "gap_mask": bool(agg.get("gap_mask", False)),
            "recon_flag": bool(agg.get("recon_flag", False)),
            "vwap_oob": bool(agg.get("vwap_oob", False)),
        }
        if kline:
            record.update(
                {
                    "open": kline.get("open"),
                    "high": kline.get("high"),
                    "low": kline.get("low"),
                    "close": kline.get("close"),
                    "volume": kline.get("volume"),
                    "quote_volume": kline.get("quoteVolume"),
                    "trades": kline.get("trades"),
                    "takerBuyBase": kline.get("takerBuyBase"),
                    "takerBuyQuote": kline.get("takerBuyQuote"),
                }
            )
            record["gap_mask"] = record["gap_mask"] or bool(kline.get("gap_mask"))
        else:
            record.update(
                {
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": 0.0,
                    "quote_volume": 0.0,
                    "trades": 0,
                    "takerBuyBase": 0.0,
                    "takerBuyQuote": 0.0,
                }
            )
            record["gap_mask"] = True
        record.update(
            {
                "buy_vol": agg.get("buy_vol", 0.0),
                "sell_vol": agg.get("sell_vol", 0.0),
                "vol": agg.get("vol", 0.0),
                "delta": agg.get("delta", 0.0),
                "CVD": agg.get("cvd", last_cvd),
                "vwap_min": agg.get("vwap_min"),
                "trades_cnt": agg.get("trades_cnt", 0),
            }
        )
        record["imbalance_1m"] = (
            record["buy_vol"] / record["vol"] if record["vol"] else 0.0
        )
        if "cvd" in agg:
            last_cvd = float(agg.get("cvd", last_cvd))
            record["CVD"] = last_cvd
        else:
            record["CVD"] = last_cvd
        if record["vol"] == 0 and record.get("volume") and record["volume"] > 0:
            record["gap_mask"] = True
        if last_premium and last_premium["time"] == ts:
            record["markPrice"] = last_premium["markPrice"]
            record["indexPrice"] = last_premium["indexPrice"]
            record["lastFundingRate"] = last_premium["lastFundingRate"]
            record["nextFundingTime"] = last_premium["nextFundingTime"]
        else:
            record["markPrice"] = None
            record["indexPrice"] = None
            record["lastFundingRate"] = None
            record["nextFundingTime"] = None
        joined.append(record)
    # Forward-fill mark/index for up to 60 minutes without new data
    last_mark = None
    last_mark_ts: int | None = None
    last_index = None
    last_index_ts: int | None = None
    for entry in joined:
        ts = entry["ts_min"]
        mark = entry.get("markPrice")
        idx = entry.get("indexPrice")
        if mark is not None:
            last_mark = mark
            last_mark_ts = ts
        elif last_mark is not None and last_mark_ts is not None and ts - last_mark_ts <= 60 * MINUTE_MS:
            entry["markPrice"] = last_mark
        if idx is not None:
            last_index = idx
            last_index_ts = ts
        elif last_index is not None and last_index_ts is not None and ts - last_index_ts <= 60 * MINUTE_MS:
            entry["indexPrice"] = last_index
    return joined


def _apply_derived_metrics(series: List[Dict[str, Any]]) -> None:
    deltas = [float(entry.get("delta") or 0.0) for entry in series]
    cvds = [float(entry.get("CVD") or 0.0) for entry in series]
    means, stds = _sliding_stats(deltas, DELTA_WINDOW)
    for index, entry in enumerate(series):
        std = stds[index]
        mean_value = means[index]
        delta_value = deltas[index]
        if index + 1 < DELTA_WINDOW or std <= 1e-9:
            entry["delta_z"] = 0.0
        else:
            entry["delta_z"] = (delta_value - mean_value) / std
    window_values: List[float] = []
    window_min = 0.0
    window_max = 0.0
    for index, cvd in enumerate(cvds):
        window_values.append(cvd)
        if len(window_values) > DELTA_WINDOW:
            window_values.pop(0)
        window_min = min(window_values) if window_values else 0.0
        window_max = max(window_values) if window_values else 0.0
        denom = window_max - window_min
        entry = series[index]
        if index + 1 < DELTA_WINDOW or denom <= 1e-12:
            entry["cvd_norm"] = 0.0
        else:
            entry["cvd_norm"] = (cvd - window_min) / denom


async def collect_data(
    *,
    symbol: str,
    t0_ms: int,
    hours: int = 72,
    trace: TraceContext | None = None,
    client: httpx.AsyncClient | None = None,
    cache: FileCache | None = None,
) -> Dict[str, Any]:
    start_ms = t0_ms - hours * 60 * MINUTE_MS

    async def _perform(http_client: httpx.AsyncClient) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        klines_task = ingest_klines(symbol=symbol, start_ms=start_ms, t0_ms=t0_ms, client=http_client, trace=trace)
        agg_task = ingest_agg_trades(symbol=symbol, start_ms=start_ms, t0_ms=t0_ms, client=http_client, trace=trace)
        premium_task = fetch_premium_index_series(
            symbol=symbol,
            start_ms=start_ms,
            t0_ms=t0_ms,
            client=http_client,
            trace=trace,
            cache=cache,
        )
        return await asyncio.gather(klines_task, agg_task, premium_task)

    if client is None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as owned_client:
            klines_result, agg_result, premium_result = await _perform(owned_client)
            depth_series, depth_metrics = await capture_depth_series(symbol=symbol, client=owned_client, trace=trace)
    else:
        klines_result, agg_result, premium_result = await _perform(client)
        depth_series, depth_metrics = await capture_depth_series(symbol=symbol, client=client, trace=trace)

    baseline = _build_minutes_baseline(start_ms, t0_ms, hours)
    reconciled_minutes, recon_stats, recon_minutes, vwap_oob_minutes = _reconcile_minutes(
        klines_result["minutes"], agg_result["minutes"]
    )
    agg_baseline = float(agg_result.get("cvd_baseline", 0.0))
    joined = _join_timeseries(
        baseline,
        klines_result["minutes"],
        reconciled_minutes,
        premium_result["samples"],
        cvd_baseline=agg_baseline,
    )
    _apply_derived_metrics(joined)
    coverage, silent_minutes = compute_coverage(joined)

    klines_metrics = klines_result.get("metrics", {})
    agg_metrics = agg_result.get("metrics", {})
    premium_metrics = premium_result.get("metrics", {})

    diag = PipelineDiagnostics()
    diag.recon_mismatch_pct = recon_stats["recon_mismatch_pct"]
    diag.vwap_oob_pct = recon_stats["vwap_oob_pct"]
    diag.recon_minutes = recon_minutes
    diag.vwap_oob_minutes = vwap_oob_minutes
    diag.silent_minutes = silent_minutes
    diag.cvd_baseline = agg_baseline
    diag.cvd_baseline_source = agg_result.get("cvd_baseline_source", "cold_start")
    diag.rate_limit_events = (
        klines_metrics.get("rate_limit_events", 0)
        + agg_metrics.get("rate_limit_events", 0)
        + premium_metrics.get("rate_limit_events", 0)
        + depth_metrics.get("rate_limit_events", 0)
    )
    diag.retries = (
        klines_metrics.get("retries", 0)
        + agg_metrics.get("retries", 0)
        + premium_metrics.get("retries", 0)
        + depth_metrics.get("retries", 0)
    )

    gap_entries: List[Dict[str, Any]] = []
    for entry in joined:
        reasons: List[str] = []
        if entry.get("open") is None:
            reasons.append("missing_klines")
        agg_vol = float(entry.get("vol") or 0.0)
        kline_vol = float(entry.get("volume") or 0.0)
        if agg_vol <= 0.0 and kline_vol > 0.0:
            reasons.append("missing_aggTrades")
        if entry.get("gap_mask") and reasons:
            gap_entries.append({"ts_min": entry["ts_min"], "reasons": reasons})
    diag.gaps = gap_entries

    labels = ["start", "mid", "end"]
    label_targets = {
        "start": baseline[0]["ts_min"] if baseline else window_start,
        "mid": baseline[len(baseline) // 2]["ts_min"] if baseline else window_start + (len(baseline) // 2) * MINUTE_MS,
        "end": baseline[-1]["ts_min"] if baseline else window_end - MINUTE_MS,
    }
    depth_payload: List[Dict[str, Any]] = []
    for idx, snapshot in enumerate(depth_series):
        labeled = dict(snapshot)
        label = labels[idx] if idx < len(labels) else f"extra_{idx}"
        labeled["label"] = label
        label_ts = label_targets.get(label, window_start)
        labeled["label_ts"] = label_ts
        ts_req = labeled.get("ts_req") or labeled.get("ts")
        labeled["ts_requested"] = ts_req
        depth_payload.append(labeled)
        if ts_req is not None and label_ts is not None and abs(int(ts_req) - int(label_ts)) > 5_000:
            diag.problem_windows.append(
                {
                    "reason": "depth_timestamp_skew",
                    "label": label,
                    "ts_requested": ts_req,
                    "label_ts": label_ts,
                }
            )

    return {
        "minutes": joined,
        "klines": klines_result,
        "agg": agg_result,
        "premium": premium_result,
        "depth_checks": depth_payload,
        "coverage": coverage,
        "diagnostics": diag,
        "window": {"start_ms": joined[0]["ts_min"] if joined else start_ms, "end_ms": t0_ms},
        "metrics": {
            "klines": klines_metrics,
            "agg_trades": agg_metrics,
            "premium": premium_metrics,
            "depth": depth_metrics,
        },
    }


def _convert_to_arrow(recordset: Sequence[Mapping[str, Any]]) -> "pa.Table":
    import pyarrow as pa  # type: ignore

    def _column(name: str) -> List[Any]:
        return [record.get(name) for record in recordset]

    schema = pa.schema(
        [
            ("ts_min", pa.int64()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
            ("quote_volume", pa.float64()),
            ("trades", pa.int32()),
            ("takerBuyBase", pa.float64()),
            ("takerBuyQuote", pa.float64()),
            ("buy_vol", pa.float64()),
            ("sell_vol", pa.float64()),
            ("vol", pa.float64()),
            ("delta", pa.float64()),
            ("CVD", pa.float64()),
            ("vwap_min", pa.float64()),
            ("delta_z", pa.float64()),
            ("cvd_norm", pa.float64()),
            ("imbalance_1m", pa.float64()),
            ("gap_mask", pa.bool_()),
            ("recon_flag", pa.bool_()),
            ("vwap_oob", pa.bool_()),
            ("markPrice", pa.float64()),
            ("indexPrice", pa.float64()),
            ("lastFundingRate", pa.float64()),
            ("nextFundingTime", pa.int64()),
        ]
    )
    columns = {field.name: _column(field.name) for field in schema}
    return pa.Table.from_pydict(columns, schema=schema)


def _validate_timeseries(recordset: Sequence[Mapping[str, Any]], expected_count: int) -> None:
    if len(recordset) != expected_count:
        raise ValueError(f"expected {expected_count} minutes, got {len(recordset)}")
    for entry in recordset:
        ts = entry.get("ts_min")
        gap = bool(entry.get("gap_mask"))
        open_value = entry.get("open")
        high_value = entry.get("high")
        low_value = entry.get("low")
        close_value = entry.get("close")
        volume_value = entry.get("volume")
        if not gap:
            for value in (open_value, high_value, low_value, close_value, volume_value):
                if value is None:
                    raise ValueError(f"missing OHLCV data at {ts}")
                if isinstance(value, float) and math.isnan(value):
                    raise ValueError(f"NaN detected in OHLCV data at {ts}")
            low = float(low_value)
            high = float(high_value)
            open_val = float(open_value)
            close_val = float(close_value)
            if low > min(open_val, close_val) or high < max(open_val, close_val) or high < low:
                raise ValueError(f"invalid candle bounds at {ts}")
        vwap = entry.get("vwap_min")
        vol = float(entry.get("vol") or 0.0)
        if vol > 0.0 and vwap is None:
            raise ValueError(f"missing vwap_min for traded minute at {ts}")
        if isinstance(vwap, float) and math.isnan(vwap):
            raise ValueError(f"NaN vwap_min at {ts}")
    

def write_outputs(
    *,
    recordset: Sequence[Mapping[str, Any]],
    out_dir: Path,
    diagnostics: PipelineDiagnostics,
    symbol: str,
    window_start: int,
    window_end: int,
    depth_checks: Sequence[Mapping[str, Any]] | None = None,
    coverage: Mapping[str, float] | None = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_count = int((window_end - window_start) / MINUTE_MS) if window_end > window_start else len(recordset)
    _validate_timeseries(recordset, expected_count)
    table = _convert_to_arrow(recordset)
    parquet_path = out_dir / "timeseries_72h.parquet"
    import pyarrow.parquet as pq  # type: ignore

    pq.write_table(table, parquet_path)
    summary_path = out_dir / "summary_72h.json"
    minutes = len(recordset)
    coverage_payload = coverage or {
        "klines_pct": 0.0,
        "aggtrades_pct": 0.0,
        "recon_mismatch_pct": diagnostics.recon_mismatch_pct or 0.0,
        "vwap_oob_pct": diagnostics.vwap_oob_pct or 0.0,
    }
    payload = {
        "symbol": symbol,
        "window": {"start": window_start, "end": window_end, "minutes": minutes},
        "coverage": coverage_payload,
        "funding": [
            {
                "time": record.get("ts_min"),
                "lastFundingRate": record.get("lastFundingRate"),
                "nextFundingTime": record.get("nextFundingTime"),
            }
            for record in recordset
            if record.get("lastFundingRate") is not None
        ],
        "depth_checks": list(depth_checks or []),
        "diagnostics": {
            "rate_limit_events": diagnostics.rate_limit_events,
            "retries": diagnostics.retries,
            "vision_backfills": diagnostics.vision_backfills,
            "gaps": diagnostics.gaps,
            "problem_windows": diagnostics.problem_windows,
            "recon_minutes": diagnostics.recon_minutes,
            "vwap_oob_minutes": diagnostics.vwap_oob_minutes,
            "silent_minutes": diagnostics.silent_minutes,
            "cvd_baseline": diagnostics.cvd_baseline,
            "cvd_baseline_source": diagnostics.cvd_baseline_source,
        },
    }
    summary_path.write_text(json_dumps(payload), encoding="utf-8")
    return {"parquet": parquet_path, "summary": summary_path}


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "collect_data",
    "write_outputs",
    "PipelineDiagnostics",
]
