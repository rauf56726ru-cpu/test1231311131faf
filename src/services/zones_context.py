"""72-hour zone detection, caching, and session touch context."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.bias import compute_bias
from src.common.config import AppConfig
from src.storage.parquet import ParquetStorage

LOGGER = logging.getLogger(__name__)

MINUTE_MS = 60_000
WINDOW_HOURS = 72
ATR_PERIOD = 14
GAP_FACTOR = 0.3
FILL_THRESHOLD = 0.8
RVOL_FACTOR = 1.5
SWING_K = 2
SWING_MIN_ATR = 0.3
EQL_TOL_BPS = 0.001
PROBE_TOL_ATR = 0.2
OB_LOOKBACK = 10
ZONE_MERGE_OVERLAP = 0.8
ZONE_MIN_STRENGTH = 0.5
TOP_N_DEFAULT = 20

CACHE_ROOT = Path("var/zones_cache")


def _utc_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _last_closed_session_bounds(reference_ms: int | None = None) -> tuple[int, int]:
    now_dt = datetime.fromtimestamp((reference_ms or _utc_ms()) / 1000, tz=UTC)
    session_end = now_dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(milliseconds=MINUTE_MS)
    session_start = session_end - timedelta(hours=24) + timedelta(minutes=1)
    return int(session_start.timestamp() * 1000), int(session_end.timestamp() * 1000)


@dataclass(slots=True)
class ZoneRecord:
    id: str
    type: str
    side: str
    low: float
    high: float
    created_ts: int
    last_seen_ts: int
    width_atr: float
    strength: float
    status: str
    touches: int = 0
    source: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.high - self.low)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["width_atr"] = round(self.width_atr, 4)
        payload["strength"] = round(self.strength, 4)
        payload["width"] = round(self.width, 8)
        return payload


@dataclass(slots=True)
class TouchRecord:
    zone_id: str
    at_ts: int
    touch_kind: str
    depth: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "at_ts": self.at_ts,
            "touch_kind": self.touch_kind,
            "depth": round(self.depth, 4),
        }


def _coerce_depth(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, 4)


def _empty_touch_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "filled": 0,
        "wick_only": 0,
        "first_ms": None,
        "last_ms": None,
        "max_depth": None,
        "last_kind": None,
        "sampled": 0,
        "truncated": False,
    }


def summarise_touch_records(
    records: Sequence[TouchRecord],
    *,
    recent_tail: int = 3,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Compress raw touch records into key checkpoints and summary stats."""

    if not records:
        return [], _empty_touch_stats()

    ordered = sorted(records, key=lambda item: item.at_ts)
    events: list[Dict[str, Any]] = [
        {
            "ts_ms": int(record.at_ts),
            "kind": str(record.touch_kind),
            "depth": _coerce_depth(record.depth),
        }
        for record in ordered
    ]

    total = len(events)
    filled = sum(1 for event in events if event["kind"] == "filled")
    wick_only = total - filled
    depths = [event["depth"] for event in events if event["depth"] is not None]
    max_depth = round(max(depths), 4) if depths else None
    first_ms = events[0]["ts_ms"]
    last_ms = events[-1]["ts_ms"]
    last_kind = events[-1]["kind"]

    keypoints: Dict[int, Dict[str, Any]] = {}

    def _mark(event: Dict[str, Any], label: str) -> None:
        ts = event["ts_ms"]
        item = keypoints.get(ts)
        if item is None:
            item = {
                "ts_ms": ts,
                "kind": event["kind"],
                "depth": event.get("depth"),
                "labels": [label],
            }
            keypoints[ts] = item
            return
        if label not in item["labels"]:
            item["labels"].append(label)
        if item["kind"] != "filled" and event["kind"] == "filled":
            item["kind"] = event["kind"]
        depth = event.get("depth")
        if depth is not None:
            current = item.get("depth")
            if current is None or depth > current:
                item["depth"] = depth

    _mark(events[0], "first")
    _mark(events[-1], "last")

    filled_indexes = [idx for idx, event in enumerate(events) if event["kind"] == "filled"]
    if filled_indexes:
        _mark(events[filled_indexes[0]], "filled_first")
        _mark(events[filled_indexes[-1]], "filled_last")

    if depths:
        deepest_idx = max(range(len(events)), key=lambda idx: events[idx]["depth"] or -1.0)
        _mark(events[deepest_idx], "max_depth")

    if recent_tail > 0:
        tail = max(1, int(recent_tail))
        for event in events[-tail:]:
            _mark(event, "recent")

    checkpoints = sorted(keypoints.values(), key=lambda item: item["ts_ms"])
    for entry in checkpoints:
        labels = entry.get("labels")
        if isinstance(labels, list):
            entry["labels"] = sorted(set(labels))

    stats = {
        "total": total,
        "filled": filled,
        "wick_only": wick_only,
        "first_ms": first_ms,
        "last_ms": last_ms,
        "max_depth": max_depth,
        "last_kind": last_kind,
        "sampled": len(checkpoints),
        "truncated": total > len(checkpoints),
    }

    return checkpoints, stats


class ZoneCache:
    """Persist open zones between runs."""

    def __init__(self, root: Path | str = CACHE_ROOT) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, symbol: str) -> Path:
        return self._root / f"{symbol.upper()}.json"

    def load(self, symbol: str) -> List[ZoneRecord]:
        path = self._path_for(symbol)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("zones.cache.load_failed", extra={"path": str(path)})
            return []
        records: List[ZoneRecord] = []
        for entry in raw if isinstance(raw, list) else []:
            try:
                records.append(
                    ZoneRecord(
                        id=str(entry["id"]),
                        type=str(entry["type"]),
                        side=str(entry["side"]),
                        low=float(entry["low"]),
                        high=float(entry["high"]),
                        created_ts=int(entry["created_ts"]),
                        last_seen_ts=int(entry.get("last_seen_ts", entry["created_ts"])),
                        width_atr=float(entry.get("width_atr", 0.0)),
                        strength=float(entry.get("strength", 0.0)),
                        status=str(entry.get("status", "open")),
                        touches=int(entry.get("touches", 0)),
                        source=entry.get("source"),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def store(self, symbol: str, zones: Sequence[ZoneRecord]) -> None:
        path = self._path_for(symbol)
        payload = [zone.to_dict() for zone in zones]
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("zones.cache.write_failed", extra={"path": str(path), "error": str(exc)})


class ZoneDetectionError(RuntimeError):
    """Raised when the detection pipeline cannot proceed."""


def _ensure_storage(storage: ParquetStorage | None) -> ParquetStorage:
    if storage is not None:
        return storage
    cfg = AppConfig.load()
    return ParquetStorage(root=cfg.data_dir, market=cfg.market, index_path=cfg.duckdb_path)


def build_zones_context(
    symbol: str,
    *,
    storage: ParquetStorage | None = None,
    now_ms: int | None = None,
    hours: int = WINDOW_HOURS,
    top_n: int = TOP_N_DEFAULT,
    cache: ZoneCache | None = None,
) -> Dict[str, Any]:
    if not symbol:
        raise ValueError("symbol is required")
    target_symbol = symbol.strip().upper()
    storage_instance = _ensure_storage(storage)

    end_ms = int(now_ms if now_ms is not None else _utc_ms())
    window_hours = max(1, int(hours))
    start_ms = end_ms - window_hours * 60 * MINUTE_MS

    frame = storage_instance.load_window(
        target_symbol,
        "1m",
        start_ms,
        end_ms,
        columns=("ts_open", "open", "high", "low", "close", "volume", "taker_buy_vol"),
    )
    if frame.empty:
        raise ZoneDetectionError(f"no data for {target_symbol} in {window_hours}h window")

    frame = frame.dropna(subset=["ts_open", "open", "high", "low", "close"]).reset_index(drop=True)
    if frame.empty:
        raise ZoneDetectionError("window contains no usable candles")

    frame["delta"] = (frame["taker_buy_vol"] * 2.0) - frame["volume"]
    frame["atr14"] = _compute_atr(frame["high"], frame["low"], frame["close"], period=ATR_PERIOD)
    volume_median = float(frame["volume"].median()) or 1e-9
    frame["rvol"] = frame["volume"] / volume_median

    session_start_ms, session_end_ms = _last_closed_session_bounds(end_ms)
    session_start_ms = max(session_start_ms, start_ms)

    detector = ZoneDetector(frame, session_start_ms=session_start_ms)
    zones = detector.detect()

    cache = cache or ZoneCache()
    cache.store(target_symbol, [zone for zone in zones if zone.status != "filled"])

    touches = detector.detect_session_touches(zones, session_end_ms=session_end_ms)
    zones_filtered = _filter_zones(zones)
    zones_sorted = sorted(zones_filtered, key=lambda z: (-z.strength, -z.created_ts))
    top_zones = zones_sorted[: max(1, min(int(top_n), 200))]

    coverage = _compute_coverage(frame, start_ms, end_ms)
    metrics = _compute_metrics(frame)
    frame_for_metrics = frame.assign(ts_open=frame["ts_open"].astype("int64"))
    bias_block = compute_bias(
        frame_for_metrics,
        timeframes=("1h", "4h", "1d"),
        neutral_pct=0.1,
    )
    if bias_block:
        metrics["bias"] = bias_block
    vwap_context = _build_vwap_context(frame_for_metrics[["ts_open", "open", "high", "low", "close", "volume"]])
    if vwap_context:
        metrics["vwap_context"] = vwap_context
    tpo_context = _build_tpo_context(frame_for_metrics[["ts_open", "open", "high", "low", "close", "volume"]])
    if tpo_context:
        metrics["tpo_context"] = tpo_context
    session_ib = _build_session_ib(frame_for_metrics[["ts_open", "high", "low"]], session_end_ms)
    if session_ib:
        metrics.setdefault("sessions", {})["last_closed"] = session_ib

    touch_map: Dict[str, List[TouchRecord]] = {}
    for touch in touches:
        touch_map.setdefault(touch.zone_id, []).append(touch)

    zones_payload: List[Dict[str, Any]] = []
    touches_payload: List[Dict[str, Any]] = []
    touch_stats_map: Dict[str, Dict[str, Any]] = {}

    for zone in top_zones:
        raw_records = touch_map.get(zone.id, [])
        checkpoints, stats = summarise_touch_records(raw_records or [])
        zone_payload = zone.to_dict()
        zone_payload["touch_count"] = zone_payload.get("touches", 0)
        zone_payload["touches"] = checkpoints
        zone_payload["touch_stats"] = stats
        for checkpoint in checkpoints:
            merged = dict(checkpoint)
            merged["zone_id"] = zone.id
            touches_payload.append(merged)
        zones_payload.append(zone_payload)
        touch_stats_map[zone.id] = stats

    payload = {
        "schema": "SMC_72h_ctx_v1",
        "symbol": target_symbol,
        "window": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "hours": window_hours,
        },
        "metrics": metrics,
        "coverage": coverage,
        "zones": zones_payload,
        "touches": touches_payload,
        "touch_stats": touch_stats_map,
    }

    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if payload_bytes > 300_000:
        LOGGER.warning(
            "zones.context.payload_too_large",
            extra={"bytes": payload_bytes, "symbol": target_symbol, "zones": len(top_zones)},
        )
    return payload


def _compute_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    atr_mean = float(frame["atr14"].mean(skipna=True))
    volume_mean = float(frame["volume"].mean(skipna=True))
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    vwap_num = (typical_price * frame["volume"]).sum()
    vwap_den = frame["volume"].sum()
    vwap = float(vwap_num / vwap_den) if vwap_den > 0 else float(frame["close"].iloc[-1])
    return {
        "atr14_mean": atr_mean,
        "avg_volume": volume_mean,
        "vwap": vwap,
    }


def _compute_coverage(frame: pd.DataFrame, start_ms: int, end_ms: int) -> Dict[str, Any]:
    expected = int((end_ms - start_ms) / MINUTE_MS) + 1
    timestamps = frame["ts_open"].to_numpy()
    deltas = np.diff(timestamps)
    gap_minutes = np.maximum((deltas // MINUTE_MS) - 1, 0)
    largest_gap = int(gap_minutes.max()) if gap_minutes.size else 0
    return {
        "minutes_expected": expected,
        "minutes_found": int(len(frame)),
        "coverage_pct": round(len(frame) / max(expected, 1), 4),
        "largest_gap_min": largest_gap,
    }


def _filter_zones(zones: Sequence[ZoneRecord]) -> List[ZoneRecord]:
    results: List[ZoneRecord] = []
    for zone in zones:
        if zone.status == "filled":
            continue
        if zone.strength < ZONE_MIN_STRENGTH:
            continue
        results.append(zone)
    return results


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, *, period: int) -> pd.Series:
    high_low = (high - low).abs()
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean().bfill()
    return atr.bfill().ffill()


def _build_vwap_context(frame: pd.DataFrame) -> Dict[str, Any]:
    from src.analysis.vwap_context import build_vwap_context as _build

    return _build(frame)


def _build_tpo_context(frame: pd.DataFrame) -> Dict[str, Any]:
    from src.analysis.tpo_context import build_tpo_context as _build

    return _build(frame)


def _build_session_ib(frame: pd.DataFrame, session_end_ms: int) -> Dict[str, Any]:
    from src.analysis.session_ib import compute_session_ib

    session_start_ms = session_end_ms - (24 * 60 * 60 * 1000)
    if session_start_ms < 0:
        session_start_ms = 0
    return compute_session_ib(
        frame,
        session_start_ms=session_start_ms,
        session_end_ms=session_end_ms,
        ib_minutes=60,
    )


class ZoneDetector:
    def __init__(self, frame: pd.DataFrame, *, session_start_ms: int) -> None:
        self.frame = frame.copy()
        self.session_start_ms = session_start_ms
        self.mean_abs_delta = float(self.frame["delta"].abs().mean()) or 1e-9

    def detect(self) -> List[ZoneRecord]:
        fvgs = self._detect_fvg()
        obs = self._detect_order_blocks()
        swings = self._detect_swings()
        eql = self._detect_equal_levels(swings)

        all_zones = fvgs + obs + swings + eql
        merged = self._merge_overlaps(all_zones)
        return merged

    def detect_session_touches(
        self,
        zones: Sequence[ZoneRecord],
        *,
        session_end_ms: int,
    ) -> List[TouchRecord]:
        if not zones:
            return []
        session_mask = (self.frame["ts_open"] >= self.session_start_ms) & (self.frame["ts_open"] <= session_end_ms)
        session_frame = self.frame.loc[session_mask].copy()
        if session_frame.empty:
            return []

        touches: List[TouchRecord] = []
        highs = session_frame["high"].to_numpy()
        lows = session_frame["low"].to_numpy()
        closes = session_frame["close"].to_numpy()
        atrs = session_frame["atr14"].replace(0.0, np.nan).ffill().bfill().to_numpy()
        timestamps = session_frame["ts_open"].to_numpy()

        for zone in zones:
            width = zone.width
            if width <= 0:
                touches.extend(self._touches_for_level(zone, closes, highs, lows, atrs, timestamps))
                continue

            overlaps = (highs >= zone.low) & (lows <= zone.high)
            if not overlaps.any():
                continue
            idxs = np.where(overlaps)[0]
            for idx in idxs:
                high = highs[idx]
                low = lows[idx]
                close = closes[idx]
                penetration = 0.0
                if zone.side in {"bull", "demand"}:
                    penetration = max(0.0, min(1.0, (zone.high - close) / width))
                    depth = max(0.0, min(1.0, (zone.high - low) / width))
                    touch_kind = "filled" if penetration >= FILL_THRESHOLD else "wick_only"
                elif zone.side in {"bear", "supply"}:
                    penetration = max(0.0, min(1.0, (close - zone.low) / width))
                    depth = max(0.0, min(1.0, (high - zone.low) / width))
                    touch_kind = "filled" if penetration >= FILL_THRESHOLD else "wick_only"
                else:
                    penetration = max(0.0, min(1.0, (zone.high - close) / width))
                    depth = max(0.0, min(1.0, (zone.high - low) / width))
                    touch_kind = "filled" if penetration >= FILL_THRESHOLD else "wick_only"

                touches.append(
                    TouchRecord(
                        zone_id=zone.id,
                        at_ts=int(timestamps[idx]),
                        touch_kind=touch_kind,
                        depth=depth,
                    )
                )
                zone.last_seen_ts = int(timestamps[idx])
                zone.touches += 1
                if penetration >= FILL_THRESHOLD:
                    zone.status = "filled"
                    break
        return touches

    def _touches_for_level(
        self,
        zone: ZoneRecord,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        atrs: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[TouchRecord]:
        tolerance = PROBE_TOL_ATR
        touches: List[TouchRecord] = []
        level = zone.high
        for idx in range(len(closes)):
            close = closes[idx]
            atr_val = atrs[idx] if idx < len(atrs) else atrs[-1] if len(atrs) else 0.0
            probe = tolerance * (atr_val if atr_val and atr_val > 0 else 1.0)
            if lows[idx] <= level <= highs[idx]:
                depth = min(1.0, abs(close - level) / max(probe, 1e-9))
                touch_kind = "filled" if abs(close - level) >= probe else "wick_only"
                touches.append(
                    TouchRecord(
                        zone_id=zone.id,
                        at_ts=int(timestamps[idx]),
                        touch_kind=touch_kind,
                        depth=depth,
                    )
                )
                zone.last_seen_ts = int(timestamps[idx])
                zone.touches += 1
                if touch_kind == "filled":
                    zone.status = "broken"
                    break
        return touches

    def _evaluate_zone_fill(
        self,
        idx: int,
        zone_low: float,
        zone_high: float,
        side: str,
    ) -> tuple[bool, int, int, float]:
        future = self.frame.iloc[idx + 1 :]
        width = zone_high - zone_low
        if width <= 0 or future.empty:
            ts_current = int(self.frame["ts_open"].iloc[idx])
            return False, 0, ts_current, 0.0

        touches = 0
        filled_pct = 0.0
        last_seen_ts = int(self.frame["ts_open"].iloc[idx])

        for future_row in future.itertuples():
            high_val = float(future_row.high)
            low_val = float(future_row.low)
            close_val = float(future_row.close)
            timestamp = int(future_row.ts_open)

            if zone_low <= close_val <= zone_high:
                touches += 1
                last_seen_ts = timestamp

            if side in {"bull", "demand"}:
                body_pen = max(0.0, (zone_high - close_val) / width)
                wick_pen = max(0.0, (zone_high - max(low_val, zone_low)) / width)
            else:
                body_pen = max(0.0, (close_val - zone_low) / width)
                wick_pen = max(0.0, (min(high_val, zone_high) - zone_low) / width)

            penetration = min(1.0, max(body_pen, wick_pen))
            filled_pct = max(filled_pct, penetration)

            if penetration >= FILL_THRESHOLD:
                last_seen_ts = timestamp
                return True, touches, last_seen_ts, filled_pct

        return False, touches, last_seen_ts, filled_pct

    def _detect_fvg(self) -> List[ZoneRecord]:
        lows = self.frame["low"].to_numpy()
        highs = self.frame["high"].to_numpy()
        atr = self.frame["atr14"].to_numpy()
        rvol = self.frame["rvol"].to_numpy()
        delta = self.frame["delta"].to_numpy()
        ts = self.frame["ts_open"].to_numpy()

        records: List[ZoneRecord] = []
        if len(self.frame) < 3:
            return records

        high_a = highs[:-2]
        low_a = lows[:-2]
        low_b = lows[1:-1]
        high_b = highs[1:-1]

        mean_abs_delta = self.mean_abs_delta

        for local_idx in range(len(low_b)):
            idx = local_idx + 1
            bull_gap = low_b[local_idx] > high_a[local_idx] and (low_b[local_idx] - high_a[local_idx]) >= GAP_FACTOR * atr[idx]
            bear_gap = high_b[local_idx] < low_a[local_idx] and (low_a[local_idx] - high_b[local_idx]) >= GAP_FACTOR * atr[idx]

            if not bull_gap and not bear_gap:
                continue

            if bull_gap:
                zone_low = float(high_a[local_idx])
                zone_high = float(low_b[local_idx])
                side = "bull"
            else:
                zone_low = float(high_b[local_idx])
                zone_high = float(low_a[local_idx])
                side = "bear"

            if zone_high <= zone_low:
                continue

            filled, touches, last_seen_ts, filled_pct = self._evaluate_zone_fill(idx, zone_low, zone_high, side)
            if filled:
                continue

            width = zone_high - zone_low
            width_atr = width / max(atr[idx], 1e-9)
            strength = width_atr * (1.0 + rvol[idx] + (abs(delta[idx]) / max(mean_abs_delta, 1e-9)))

            records.append(
                ZoneRecord(
                    id=f"fvg_{int(ts[idx])}",
                    type="fvg",
                    side=side,
                    low=zone_low,
                    high=zone_high,
                    created_ts=int(ts[idx]),
                    last_seen_ts=last_seen_ts,
                    width_atr=width_atr,
                    strength=float(strength),
                    status="open",
                    touches=touches,
                    source="fvg",
                )
            )

        return records

    def _detect_order_blocks(self) -> List[ZoneRecord]:
        opens = self.frame["open"].to_numpy()
        closes = self.frame["close"].to_numpy()
        atr = self.frame["atr14"].to_numpy()
        rvol = self.frame["rvol"].to_numpy()
        delta = self.frame["delta"].to_numpy()
        ts = self.frame["ts_open"].to_numpy()

        records: List[ZoneRecord] = []
        for idx in range(len(self.frame)):
            atr_val = atr[idx]
            if atr_val <= 0:
                continue
            body = abs(closes[idx] - opens[idx])
            if body < atr_val:
                continue
            if rvol[idx] < RVOL_FACTOR:
                continue
            is_bull = closes[idx] > opens[idx]
            search_range = range(max(0, idx - OB_LOOKBACK), idx)[::-1]
            base_idx = None
            for candidate in search_range:
                if is_bull and closes[candidate] < opens[candidate]:
                    base_idx = candidate
                    break
                if not is_bull and closes[candidate] > opens[candidate]:
                    base_idx = candidate
                    break
            if base_idx is None:
                continue
            base_open = opens[base_idx]
            base_close = closes[base_idx]
            extension = 0.25 * atr_val
            if is_bull:
                base_low = min(base_open, base_close)
                base_high = max(base_open, base_close)
                zone_low = float(base_low - extension)
                zone_high = float(base_high + extension)
                side = "demand"
            else:
                base_low = min(base_open, base_close)
                base_high = max(base_open, base_close)
                zone_low = float(base_low - extension)
                zone_high = float(base_high + extension)
                side = "supply"

            if zone_high <= zone_low:
                continue
            width = zone_high - zone_low
            width_atr = width / max(atr_val, 1e-9)
            filled, touches, last_seen_ts, filled_pct = self._evaluate_zone_fill(idx, zone_low, zone_high, side)
            if filled:
                continue
            strength = rvol[idx] * (abs(delta[idx]) / max(atr_val, 1e-9))
            records.append(
                ZoneRecord(
                    id=f"ob_{int(ts[idx])}",
                    type="ob",
                    side=side,
                    low=zone_low,
                    high=zone_high,
                    created_ts=int(ts[idx]),
                    last_seen_ts=last_seen_ts,
                    width_atr=width_atr,
                    strength=float(strength),
                    status="open",
                    touches=touches,
                    source="ob",
                )
            )
        return records

    def _detect_swings(self) -> List[ZoneRecord]:
        highs = self.frame["high"].to_numpy()
        lows = self.frame["low"].to_numpy()
        atr = self.frame["atr14"].to_numpy()
        ts = self.frame["ts_open"].to_numpy()

        records: List[ZoneRecord] = []
        length = len(self.frame)
        for idx in range(SWING_K, length - SWING_K):
            window_high = highs[idx - SWING_K : idx + SWING_K + 1]
            window_low = lows[idx - SWING_K : idx + SWING_K + 1]
            current_high = highs[idx]
            current_low = lows[idx]
            atr_val = atr[idx]
            if atr_val <= 0:
                continue
            # swing high
            if current_high == window_high.max():
                amplitude = current_high - window_low.min()
                if amplitude >= SWING_MIN_ATR * atr_val:
                    strength = amplitude / atr_val
                    records.append(
                        ZoneRecord(
                            id=f"swing_high_{int(ts[idx])}",
                            type="swing",
                            side="high",
                            low=float(current_high),
                            high=float(current_high),
                            created_ts=int(ts[idx]),
                            last_seen_ts=int(ts[idx]),
                            width_atr=0.0,
                            strength=float(strength),
                            status="open",
                            source="swing",
                        )
                    )
            # swing low
            if current_low == window_low.min():
                amplitude = window_high.max() - current_low
                if amplitude >= SWING_MIN_ATR * atr_val:
                    strength = amplitude / atr_val
                    records.append(
                        ZoneRecord(
                            id=f"swing_low_{int(ts[idx])}",
                            type="swing",
                            side="low",
                            low=float(current_low),
                            high=float(current_low),
                            created_ts=int(ts[idx]),
                            last_seen_ts=int(ts[idx]),
                            width_atr=0.0,
                            strength=float(strength),
                            status="open",
                            source="swing",
                        )
                    )

        # collapse nearby swings
        records.sort(key=lambda z: z.created_ts)
        collapsed: List[ZoneRecord] = []
        for zone in records:
            if not collapsed:
                collapsed.append(zone)
                continue
            last = collapsed[-1]
            if zone.side != last.side:
                collapsed.append(zone)
                continue
            distance = abs(zone.high - last.high)
            atr_ref = max(self.frame.loc[self.frame["ts_open"] == zone.created_ts, "atr14"].iloc[0], 1e-9)
            if distance < 2 * atr_ref:
                if zone.strength > last.strength:
                    collapsed[-1] = zone
            else:
                collapsed.append(zone)
        return collapsed

    def _detect_equal_levels(self, swings: Sequence[ZoneRecord]) -> List[ZoneRecord]:
        if not swings:
            return []
        swings_sorted = sorted(swings, key=lambda z: z.created_ts)
        highs = [z for z in swings_sorted if z.side == "high"]
        lows = [z for z in swings_sorted if z.side == "low"]

        eq_records: List[ZoneRecord] = []
        for collection, eq_type in ((highs, "eqh"), (lows, "eql")):
            for first, second in zip(collection, collection[1:]):
                price1 = first.high
                price2 = second.high
                if price1 <= 0:
                    continue
                diff = abs(price1 - price2) / price1
                if diff <= EQL_TOL_BPS:
                    level = (price1 + price2) / 2.0
                    touches = sum(
                        1
                        for swing in collection
                        if abs(swing.high - level) / max(level, 1e-9) <= EQL_TOL_BPS
                    )
                    strength = float(touches)
                    eq_records.append(
                        ZoneRecord(
                            id=f"{eq_type}_{int(second.created_ts)}",
                            type=eq_type,
                            side="high" if eq_type == "eqh" else "low",
                            low=float(level),
                            high=float(level),
                            created_ts=second.created_ts,
                            last_seen_ts=second.created_ts,
                            width_atr=0.0,
                            strength=strength,
                            status="intact",
                            source="equal",
                        )
                    )
        return eq_records

    def _merge_overlaps(self, zones: Sequence[ZoneRecord]) -> List[ZoneRecord]:
        if not zones:
            return []
        zones_sorted = sorted(zones, key=lambda z: (z.type, z.side, z.created_ts))
        merged: List[ZoneRecord] = []
        for zone in zones_sorted:
            if not merged:
                merged.append(zone)
                continue
            last = merged[-1]
            if zone.type != last.type or zone.side != last.side:
                merged.append(zone)
                continue
            if zone.width <= 0 or last.width <= 0:
                merged.append(zone)
                continue
            overlap_low = max(zone.low, last.low)
            overlap_high = min(zone.high, last.high)
            overlap = overlap_high - overlap_low
            if overlap <= 0:
                merged.append(zone)
                continue
            overlap_ratio = overlap / min(zone.width, last.width)
            if overlap_ratio >= ZONE_MERGE_OVERLAP:
                keep = zone if zone.strength >= last.strength else last
                merged[-1] = ZoneRecord(
                    id=keep.id,
                    type=keep.type,
                    side=keep.side,
                    low=min(zone.low, last.low),
                    high=max(zone.high, last.high),
                    created_ts=keep.created_ts,
                    last_seen_ts=max(zone.last_seen_ts, last.last_seen_ts),
                    width_atr=max(zone.width_atr, last.width_atr),
                    strength=max(zone.strength, last.strength),
                    status=keep.status,
                    touches=max(zone.touches, last.touches),
                    source=keep.source,
                )
            else:
                merged.append(zone)
        return merged


__all__ = ["build_zones_context", "ZoneDetectionError", "ZoneCache"]
