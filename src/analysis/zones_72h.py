"""High-performance detection of 72h supply/demand zones and FVGs."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

Side = Literal["demand", "supply"]
Source = Literal["swing", "fvg"]


@dataclass(slots=True)
class ZoneDetectionConfig:
    swing_k: int = 2
    atr_period: int = 14
    atr_alpha: float = 0.5
    fvg_gap_beta: float = 1.0
    tick_size: float = 0.1
    close_threshold: float = 0.80
    weight_width: float = 0.6
    weight_untouched: float = 0.3
    weight_volume: float = 0.1

    def __post_init__(self) -> None:
        if self.swing_k < 1:
            raise ValueError("swing_k must be >= 1")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if self.atr_alpha <= 0:
            raise ValueError("atr_alpha must be > 0")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if not 0 < self.close_threshold <= 1:
            raise ValueError("close_threshold must be within (0, 1]")


@dataclass(slots=True)
class Zone:
    id: str
    type: str
    side: Side
    symbol: str
    price_low: float
    price_high: float
    created_ts: int
    last_seen_ts: int
    touches: int
    filled_pct: float
    strength: float
    source: Source

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "side": self.side,
            "symbol": self.symbol,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "created_ts": self.created_ts,
            "last_seen_ts": self.last_seen_ts,
            "touches": self.touches,
            "filled_pct": round(self.filled_pct, 4),
            "strength": round(self.strength, 4),
            "source": self.source,
        }


def detect_zones_72h(frame: pd.DataFrame, *, config: ZoneDetectionConfig | None = None) -> List[Zone]:
    if frame.empty:
        return []
    config = config or ZoneDetectionConfig()

    zones: list[Zone] = []
    for symbol, df in _iter_symbol_frames(frame):
        df_sorted = df.sort_values("ts_open").reset_index(drop=True)
        df_prepped = _prepare_frame(df_sorted)
        atr = _compute_atr(df_prepped["high"], df_prepped["low"], df_prepped["close"], period=config.atr_period)
        swing_candidates = _find_swings(df_prepped, atr, config)
        fvg_candidates = _find_fvgs(df_prepped, config)
        zones.extend(_build_zones(df_prepped, atr, swing_candidates, symbol, "swing", config))
        zones.extend(_build_zones(df_prepped, atr, fvg_candidates, symbol, "fvg", config))

    zones.sort(key=lambda z: (z.symbol, z.created_ts, z.type))
    return zones


def export_open_zones_jsonl(zones: Iterable[Zone], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for zone in zones:
            handle.write(json.dumps(zone.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    LOGGER.info("zones.72h.export", extra={"path": str(output_path), "rows": count})
    return count


def infer_tick_size(frame: pd.DataFrame, *, default: float = 0.1) -> float:
    values = frame[["open", "high", "low", "close"]].to_numpy().ravel()
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return default
    diffs = np.diff(np.unique(np.sort(finite)))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return default
    return float(round(positive.min(), 12))


def _iter_symbol_frames(frame: pd.DataFrame):
    if "symbol" in frame.columns:
        for symbol, group in frame.groupby("symbol", sort=False):
            yield str(symbol), group
    else:
        yield "UNKNOWN", frame


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "taker_buy_vol" not in frame.columns:
        frame["taker_buy_vol"] = 0.0
    required = ["ts_open", "open", "high", "low", "close", "volume", "taker_buy_vol"]
    missing = [col for col in required if col not in frame]
    if missing:
        raise KeyError(f"Input frame missing columns: {missing}")
    return frame.astype(
        {
            "ts_open": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "taker_buy_vol": "float64",
        }
    )


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, *, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean().bfill()


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def _rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def _find_swings(df: pd.DataFrame, atr: pd.Series, config: ZoneDetectionConfig) -> List[dict]:
    k = config.swing_k
    records: list[dict] = []
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    atr_values = atr.to_numpy()

    length = len(df)
    for idx in range(k, length - k):
        prev_slice_high = highs[idx - k : idx]
        prev_slice_low = lows[idx - k : idx]
        next_slice_high = highs[idx + 1 : idx + 1 + k]
        next_slice_low = lows[idx + 1 : idx + 1 + k]
        if len(prev_slice_high) < k or len(next_slice_high) < k:
            continue

        prev_close = closes[idx - 1]
        atr_threshold = config.atr_alpha * atr_values[idx]

        is_supply = highs[idx] > prev_slice_high.max() and highs[idx] >= next_slice_high.max()
        is_demand = lows[idx] < prev_slice_low.min() and lows[idx] <= next_slice_low.min()

        if is_supply and abs(highs[idx] - prev_close) >= atr_threshold:
            records.append({"index": idx, "type": "supply"})
        elif is_demand and abs(prev_close - lows[idx]) >= atr_threshold:
            records.append({"index": idx, "type": "demand"})

    return records


def _find_fvgs(df: pd.DataFrame, config: ZoneDetectionConfig) -> List[dict]:
    gap = config.fvg_gap_beta * config.tick_size
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    records: list[dict] = []
    for idx in range(2, len(df)):
        low_now = lows[idx]
        high_now = highs[idx]
        ref_high = highs[idx - 2]
        ref_low = lows[idx - 2]
        if low_now > ref_high and (low_now - ref_high) >= gap:
            records.append({"index": idx, "type": "demand"})
        elif ref_low > high_now and (ref_low - high_now) >= gap:
            records.append({"index": idx, "type": "supply"})
    return records


def _build_zones(
    df: pd.DataFrame,
    atr: pd.Series,
    candidates: List[dict],
    symbol: str,
    source: Source,
    config: ZoneDetectionConfig,
) -> List[Zone]:
    zones: list[Zone] = []
    atr_values = atr.reset_index(drop=True)
    rolling_vol = df["volume"].rolling(window=config.atr_period, min_periods=1).mean().reset_index(drop=True)
    rolling_taker = df["taker_buy_vol"].rolling(window=config.atr_period, min_periods=1).mean().reset_index(drop=True)
    records = df.reset_index(drop=True)

    for entry in candidates:
        idx = entry["index"]
        side: Side = "demand" if entry["type"] == "demand" else "supply"
        row = records.loc[idx]

        if source == "swing":
            price_high = row["high"] if side == "supply" else max(row["open"], row["close"])
            price_low = min(row["open"], row["close"]) if side == "supply" else row["low"]
        else:
            if side == "demand":
                price_low = records.loc[idx - 2, "high"]
                price_high = row["low"]
            else:
                price_low = row["high"]
                price_high = records.loc[idx - 2, "low"]

        width = float(price_high - price_low)
        if width <= 0 or not math.isfinite(width):
            continue

        future = records.iloc[idx + 1 :]
        touches = 0
        filled_pct = 0.0
        first_touch_offset: int | None = None
        last_seen_ts = int(row["ts_open"])

        for offset, future_row in enumerate(future.itertuples()):
            close_val = future_row.close
            high_val = future_row.high
            low_val = future_row.low

            if side == "supply":
                body_pen = max(0.0, (close_val - price_low) / width)
                wick_pen = max(0.0, (min(high_val, price_high) - price_low) / width)
                if price_low <= close_val <= price_high:
                    touches += 1
                    if first_touch_offset is None:
                        first_touch_offset = offset
                    last_seen_ts = future_row.ts_open
                penetration = max(body_pen, wick_pen)
                filled_pct = max(filled_pct, min(1.0, penetration))
                if close_val > price_high or penetration >= config.close_threshold:
                    filled_pct = max(filled_pct, min(1.0, penetration))
                    break
            else:
                body_pen = max(0.0, (price_high - close_val) / width)
                wick_pen = max(0.0, (price_high - max(low_val, price_low)) / width)
                if price_low <= close_val <= price_high:
                    touches += 1
                    if first_touch_offset is None:
                        first_touch_offset = offset
                    last_seen_ts = future_row.ts_open
                penetration = max(body_pen, wick_pen)
                filled_pct = max(filled_pct, min(1.0, penetration))
                if close_val < price_low or penetration >= config.close_threshold:
                    filled_pct = max(filled_pct, min(1.0, penetration))
                    break

        if filled_pct >= config.close_threshold:
            continue

        untouched_bars = first_touch_offset if first_touch_offset is not None else len(future)
        atr_val = float(atr_values.iloc[idx]) if atr_values.iloc[idx] > 0 else 0.0
        strength_width = width / atr_val if atr_val > 0 else 0.0
        vol_spike = row["taker_buy_vol"] / max(rolling_taker.iloc[idx], 1e-9)
        strength = (
            config.weight_width * strength_width
            + config.weight_untouched * untouched_bars
            + config.weight_volume * float(vol_spike)
        )

        zones.append(
            Zone(
                id=f"{symbol}-{int(row['ts_open'])}",
                type="fvg" if source == "fvg" else "zone",
                side=side,
                symbol=symbol,
                price_low=float(price_low),
                price_high=float(price_high),
                created_ts=int(row["ts_open"]),
                last_seen_ts=int(last_seen_ts),
                touches=touches,
                filled_pct=filled_pct,
                strength=strength,
                source=source,
            )
        )

    return zones


__all__ = ["Zone", "ZoneDetectionConfig", "detect_zones_72h", "export_open_zones_jsonl", "infer_tick_size"]
