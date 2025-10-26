"""Fast session analysis built on UM minute ring/parquet snapshots."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

MINUTE_MS = 60_000

DEFAULT_ROOT = Path("var/um_ingest")


class SessionDataUnavailable(RuntimeError):
    """Raised when session data cannot be located or is incomplete."""


@dataclass(slots=True)
class SessionContext:
    symbol: str
    start_ms: int
    end_ms: int
    date: str
    frame: pd.DataFrame
    previous_frame: pd.DataFrame
    history_volumes: Sequence[float]


def _last_closed_session_bounds(reference_ms: int | None = None) -> tuple[int, int]:
    now = datetime.fromtimestamp((reference_ms or _utc_ms()) / 1000, tz=UTC)
    session_end = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(milliseconds=MINUTE_MS)
    session_start = session_end - timedelta(hours=24) + timedelta(minutes=1)
    start_ms = int(session_start.timestamp() * 1000)
    end_ms = int(session_end.timestamp() * 1000)
    return start_ms, end_ms


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _load_partition(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "ts_min" not in frame:
        return pd.DataFrame()
    frame = frame.copy()
    frame["ts_min"] = frame["ts_min"].astype("int64")
    frame.sort_values("ts_min", inplace=True)
    frame.drop_duplicates("ts_min", keep="last", inplace=True)
    return frame


def _resolve_partition(root: Path, symbol: str, date: datetime) -> pd.DataFrame:
    partition = date.strftime("date=%Y-%m-%d")
    path = root / symbol.lower() / partition / "minute.parquet"
    return _load_partition(path)


def _collect_history_volumes(root: Path, symbol: str, session_date: datetime, limit: int = 30) -> List[float]:
    volumes: List[float] = []
    cursor = session_date - timedelta(days=1)
    checked = 0
    while len(volumes) < limit and checked < limit * 3:
        frame = _resolve_partition(root, symbol, cursor)
        if frame.empty:
            cursor -= timedelta(days=1)
            checked += 1
            continue
        expected = 24 * 60
        observed = len(frame)
        if observed / max(expected, 1) < 0.6:
            cursor -= timedelta(days=1)
            checked += 1
            continue
        volumes.append(float(frame.get("volume", pd.Series(dtype=float)).sum()))
        cursor -= timedelta(days=1)
        checked += 1
    return volumes


def _load_session_frames(symbol: str, *, root: Path, start_ms: int, end_ms: int) -> SessionContext:
    session_date = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
    frame = _resolve_partition(root, symbol, session_date)
    if frame.empty:
        raise SessionDataUnavailable(f"no minute data found for {symbol} on {session_date.date()}")
    scoped = frame[(frame["ts_min"] >= start_ms) & (frame["ts_min"] <= end_ms)].copy()
    if scoped.empty:
        raise SessionDataUnavailable(f"minute data missing within session bounds ({symbol})")
    prev_frame = _resolve_partition(root, symbol, session_date - timedelta(days=1))
    history_volumes = _collect_history_volumes(root, symbol, session_date)
    return SessionContext(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        date=session_date.strftime("%Y-%m-%d"),
        frame=scoped,
        previous_frame=prev_frame,
        history_volumes=history_volumes,
    )


def _compute_coverage(frame: pd.DataFrame, context: SessionContext) -> Dict[str, Any]:
    expected = int((context.end_ms - context.start_ms) / MINUTE_MS) + 1
    available = len(frame)
    largest_gap = 0
    if available > 1:
        diffs = np.diff(frame["ts_min"].to_numpy())
        gap_minutes = np.maximum((diffs // MINUTE_MS) - 1, 0)
        largest_gap = int(gap_minutes.max(initial=0))
    streams: Dict[str, float | None] = {}
    for column, name in (
        ("has_kline", "kline"),
        ("has_aggtrade", "aggTrade"),
        ("has_bookticker", "bookTicker"),
    ):
        if column in frame:
            present = float(frame[column].astype(int).sum())
            streams[name] = round(present / max(expected, 1), 4)
    return {
        "minutes_expected": expected,
        "minutes_found": available,
        "coverage_pct": round(available / max(expected, 1), 4),
        "largest_gap_min": largest_gap,
        "streams": streams,
    }


def _compute_price_metrics(frame: pd.DataFrame, context: SessionContext) -> Dict[str, Any]:
    frame = frame.sort_values("ts_min")

    open_price = float(frame["open"].iloc[0])
    close_price = float(frame["close"].iloc[-1])
    high_price = float(frame["high"].max())
    low_price = float(frame["low"].min())
    session_range = high_price - low_price
    volume_total = float(frame["volume"].sum())

    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    numerator = (typical * frame["volume"]).sum()
    denominator = frame["volume"].sum()
    vwap = float(numerator / denominator) if denominator > 0 else close_price

    atr = _compute_atr(frame)

    previous_close = None
    previous_high = None
    previous_low = None
    if not context.previous_frame.empty:
        previous_close = float(context.previous_frame["close"].iloc[-1])
        previous_high = float(context.previous_frame["high"].max())
        previous_low = float(context.previous_frame["low"].min())

    ib_cutoff = context.start_ms + 60 * MINUTE_MS
    ib_frame = frame[frame["ts_min"] < ib_cutoff]
    ib_high = float(ib_frame["high"].max()) if not ib_frame.empty else None
    ib_low = float(ib_frame["low"].min()) if not ib_frame.empty else None

    rvol = None
    if context.history_volumes:
        baseline = sum(context.history_volumes) / len(context.history_volumes)
        if baseline > 0:
            rvol = round(volume_total / baseline, 4)

    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "range": session_range,
        "atr14": atr,
        "vwap": vwap,
        "volume": volume_total,
        "rvol": rvol,
        "pdh": previous_high,
        "pdl": previous_low,
        "pdc": previous_close,
        "ib_high": ib_high,
        "ib_low": ib_low,
    }


def _compute_atr(frame: pd.DataFrame, period: int = 14) -> float:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    previous_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=min(3, period)).mean()
    last = atr_series.dropna()
    if not last.empty:
        return float(last.iloc[-1])
    return float(tr.mean())


def _compute_orderflow(frame: pd.DataFrame) -> Dict[str, Any]:
    delta = frame.get("delta", pd.Series(dtype=float)).fillna(0.0)
    cvd = frame.get("cvd", pd.Series(dtype=float))
    cvd = cvd.ffill().fillna(0.0)
    delta_sum = float(delta.sum())
    delta_max = float(delta.max()) if not delta.empty else 0.0
    delta_min = float(delta.min()) if not delta.empty else 0.0
    cvd_start = float(cvd.iloc[0]) if not cvd.empty else 0.0
    cvd_end = float(cvd.iloc[-1]) if not cvd.empty else 0.0
    cvd_change = cvd_end - cvd_start

    window = 45
    rolling_mean = delta.rolling(window=window, min_periods=30).mean()
    rolling_std = delta.rolling(window=window, min_periods=30).std(ddof=0)
    zscore = (delta - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    impulse_df = pd.DataFrame(
        {
            "ts_min": frame["ts_min"],
            "delta": delta,
            "zscore": zscore,
        }
    )
    impulse_df["abs_z"] = impulse_df["zscore"].abs()
    impulse_candidates = impulse_df.nlargest(10, "abs_z")
    impulses: List[Dict[str, Any]] = []
    for _, row in impulse_candidates.iterrows():
        if not math.isfinite(row["zscore"]):
            continue
        side = "buy" if row["delta"] > 0 else "sell" if row["delta"] < 0 else "flat"
        impulses.append(
            {
                "ts_min": int(row["ts_min"]),
                "delta": float(row["delta"]),
                "zscore": round(float(row["zscore"]), 3),
                "side": side,
            }
        )

    return {
        "delta_sum": delta_sum,
        "delta_max": delta_max,
        "delta_min": delta_min,
        "delta_mean": float(delta.mean()) if not delta.empty else 0.0,
        "cvd_start": cvd_start,
        "cvd_end": cvd_end,
        "cvd_change": cvd_change,
        "impulses": impulses,
    }


def _compute_microstructure(frame: pd.DataFrame) -> Dict[str, Any]:
    spread_p50 = frame.get("spread_bp_p50", pd.Series(dtype=float)).dropna()
    spread_p95 = frame.get("spread_bp_p95", pd.Series(dtype=float)).dropna()
    imbalance = frame.get("l1_imbalance_p50", pd.Series(dtype=float)).dropna()
    summary = {
        "spread_p50_median": float(spread_p50.median()) if not spread_p50.empty else None,
        "spread_p95_median": float(spread_p95.median()) if not spread_p95.empty else None,
        "l1_imbalance_median": float(imbalance.median()) if not imbalance.empty else None,
    }

    thin_mask = pd.Series(False, index=frame.index)
    if not spread_p95.empty:
        thin_mask |= frame["spread_bp_p95"].fillna(0.0) > 12.0
    if "l1_imbalance_p50" in frame:
        thin_mask |= frame["l1_imbalance_p50"].abs() > 0.6
    thin_minutes = frame.loc[thin_mask, "ts_min"].astype(int).tolist()
    summary["thin_book_count"] = len(thin_minutes)
    summary["thin_book_ratio"] = round(len(thin_minutes) / max(len(frame), 1), 4)
    summary["thin_book_samples"] = thin_minutes[:5]
    return summary


def _compute_perp_context(frame: pd.DataFrame) -> Dict[str, Any]:
    basis = frame.get("basis_bp", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    funding = frame.get("funding_rate", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    mark_price = frame.get("mark_price", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    index_price = frame.get("index_price", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "basis_bp": {
            "mean": float(basis.mean()) if not basis.empty else None,
            "min": float(basis.min()) if not basis.empty else None,
            "max": float(basis.max()) if not basis.empty else None,
        },
        "funding_rate": {
            "mean": float(funding.mean()) if not funding.empty else None,
            "latest": float(funding.iloc[-1]) if not funding.empty else None,
        },
        "mark_price": float(mark_price.iloc[-1]) if not mark_price.empty else None,
        "index_price": float(index_price.iloc[-1]) if not index_price.empty else None,
    }


def _detect_events(context: SessionContext, price: Mapping[str, Any], orderflow: Mapping[str, Any]) -> List[Dict[str, Any]]:
    frame = context.frame.sort_values("ts_min")
    events: List[Dict[str, Any]] = []

    pdh = price.get("pdh")
    pdl = price.get("pdl")
    vwap_value = price.get("vwap")
    ib_high = price.get("ib_high")
    ib_low = price.get("ib_low")

    def _add(event: Dict[str, Any]) -> None:
        events.append(event)

    if pdh is not None:
        breakout_mask = (frame["high"] >= pdh) & (frame["high"].shift(1).fillna(pdh - 1) < pdh)
        idx = breakout_mask[breakout_mask].index
        if len(idx) > 0:
            row = frame.loc[idx[0]]
            delta_val = float(row.get("delta", 0.0))
            _add(
                {
                    "type": "breakout_pdh",
                    "ts_min": int(row["ts_min"]),
                    "price": float(row["high"]),
                    "delta": delta_val,
                }
            )
            retest_mask = (frame["ts_min"] > row["ts_min"]) & (
                (frame["close"] - pdh).abs() / max(pdh, 1.0) < 0.0005
            )
            retest_idx = frame.index[retest_mask]
            if len(retest_idx) > 0:
                retest_row = frame.loc[retest_idx[0]]
                _add(
                    {
                        "type": "retest_pdh",
                        "ts_min": int(retest_row["ts_min"]),
                        "price": float(retest_row["close"]),
                    }
                )
            if delta_val <= 0:
                _add(
                    {
                        "type": "false_break_pdh",
                        "ts_min": int(row["ts_min"]),
                        "price": float(row["high"]),
                        "delta": delta_val,
                    }
                )

    if pdl is not None:
        breakout_mask = (frame["low"] <= pdl) & (frame["low"].shift(1).fillna(pdl + 1) > pdl)
        idx = breakout_mask[breakout_mask].index
        if len(idx) > 0:
            row = frame.loc[idx[0]]
            delta_val = float(row.get("delta", 0.0))
            _add(
                {
                    "type": "breakout_pdl",
                    "ts_min": int(row["ts_min"]),
                    "price": float(row["low"]),
                    "delta": delta_val,
                }
            )
            retest_mask = (frame["ts_min"] > row["ts_min"]) & (
                (frame["close"] - pdl).abs() / max(pdl, 1.0) < 0.0005
            )
            retest_idx = frame.index[retest_mask]
            if len(retest_idx) > 0:
                retest_row = frame.loc[retest_idx[0]]
                _add(
                    {
                        "type": "retest_pdl",
                        "ts_min": int(retest_row["ts_min"]),
                        "price": float(retest_row["close"]),
                    }
                )
            if delta_val >= 0:
                _add(
                    {
                        "type": "false_break_pdl",
                        "ts_min": int(row["ts_min"]),
                        "price": float(row["low"]),
                        "delta": delta_val,
                    }
                )

    if vwap_value is not None:
        tolerance = max(vwap_value * 0.0003, 0.0)
        deviations = (frame["close"] - vwap_value).abs()
        below = deviations > tolerance
        touch_mask = (~below) & below.shift(1, fill_value=True)
        idx = frame.index[touch_mask]
        if len(idx) > 0:
            row = frame.loc[idx[0]]
            _add(
                {
                    "type": "retest_vwap",
                    "ts_min": int(row["ts_min"]),
                    "price": float(row["close"]),
                }
            )

    for level, label in ((ib_high, "retest_ib_high"), (ib_low, "retest_ib_low")):
        if level is None:
            continue
        tolerance = max(level * 0.0004, 0.0)
        mask = (frame["close"] - level).abs() <= tolerance
        idx = frame.index[mask]
        if len(idx) > 0:
            row = frame.loc[idx[0]]
            _add(
                {
                    "type": label,
                    "ts_min": int(row["ts_min"]),
                    "price": float(row["close"]),
                }
            )

    for impulse in orderflow.get("impulses", []):
        _add(
            {
                "type": "delta_impulse",
                "ts_min": impulse["ts_min"],
                "zscore": impulse["zscore"],
                "delta": impulse["delta"],
                "side": impulse["side"],
            }
        )

    events.sort(key=lambda item: item.get("ts_min", 0))
    return events[:10]


def analyze_session_fast(
    symbol: str,
    *,
    parquet_root: Path | str | None = None,
    now_ms: int | None = None,
) -> Dict[str, Any]:
    start_clock = time.perf_counter()
    if not symbol:
        raise ValueError("symbol is required")
    root = Path(parquet_root) if parquet_root else DEFAULT_ROOT
    start_ms, end_ms = _last_closed_session_bounds(now_ms)
    context = _load_session_frames(symbol.upper(), root=root, start_ms=start_ms, end_ms=end_ms)

    coverage = _compute_coverage(context.frame, context)
    price = _compute_price_metrics(context.frame, context)
    orderflow = _compute_orderflow(context.frame)
    microstructure = _compute_microstructure(context.frame)
    perp = _compute_perp_context(context.frame)
    events = _detect_events(context, price, orderflow)

    analysis_ms = int((time.perf_counter() - start_clock) * 1000)
    data_lag_ms = max(0, (_utc_ms() if now_ms is None else now_ms) - context.frame["ts_min"].max())

    payload = {
        "schema": "SMC_session_v1",
        "symbol": context.symbol,
        "session": {
            "date": context.date,
            "start_ms": context.start_ms,
            "end_ms": context.end_ms,
        },
        "coverage": coverage,
        "latency": {
            "analysis_ms": analysis_ms,
            "data_lag_ms": int(data_lag_ms),
        },
        "price": price,
        "orderflow": orderflow,
        "microstructure": microstructure,
        "perp_context": perp,
        "events": events,
    }
    return payload


__all__ = [
    "analyze_session_fast",
    "SessionDataUnavailable",
]
