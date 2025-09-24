"""Feature engineering pipeline covering price, structural and regime signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import json

import numpy as np
import pandas as pd
from ..config import get_settings
from ..io.store import write_parquet
from ..utils.logging import get_logger
from ..utils.timeframes import interval_to_seconds
from .indicators import prepare_indicators
from .orderflow import build_orderflow_features

LOGGER = get_logger(__name__)


STRUCTURE_WINDOWS = (5, 15, 30)
SWING_THRESHOLD_ATR = 1.5


@dataclass
class FeatureSet:
    horizon: int
    X: pd.DataFrame
    y: pd.Series
    regimes: pd.DataFrame
    regime_labels: pd.Series
    metadata: Dict[str, object] = field(default_factory=dict)


def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _session_label(ts: pd.Timestamp) -> str:
    hour = ts.hour
    if 0 <= hour < 8:
        return "Asia"
    if 8 <= hour < 16:
        return "EU"
    return "US"


def _assign_sessions(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([_session_label(ts) for ts in index], index=index, name="session")


def _assign_vol_buckets(vol: pd.Series) -> tuple[pd.Series, Dict[str, float]]:
    if vol.empty:
        return pd.Series(index=vol.index, dtype=object), {"low": 0.0, "high": 0.0}
    low_q, high_q = vol.quantile([0.33, 0.66])
    bucket = pd.Series("med", index=vol.index, dtype=object)
    bucket[vol <= low_q] = "low"
    bucket[vol >= high_q] = "high"
    return bucket.rename("vol_bucket"), {"low": float(low_q), "high": float(high_q)}


def _time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    minutes = index.hour * 60 + index.minute
    hour_angle = 2 * np.pi * minutes / (24 * 60)
    dow = index.dayofweek
    dow_angle = 2 * np.pi * dow / 7
    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "dow_sin": np.sin(dow_angle),
            "dow_cos": np.cos(dow_angle),
        },
        index=index,
    ).astype(float)


def _interaction_features(features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}
    trend_cols = [col for col in features.columns if col.startswith("trend_")]
    vol_cols = [col for col in regimes.columns if col.startswith("vol_bucket_")]
    for trend_col in trend_cols:
        for vol_col in vol_cols:
            name = f"{trend_col}_x_{vol_col}"
            out[name] = features[trend_col] * regimes[vol_col]
    return pd.DataFrame(out, index=features.index)


def make_target(close: pd.Series, horizon: int, fee_buffer: pd.Series) -> pd.Series:
    """Build a three-class target {-1,0,1} with commission-aware flat region."""

    log_future = np.log(close.shift(-horizon)) - np.log(close)
    threshold = fee_buffer.reindex(close.index).fillna(method="ffill").fillna(0.0)
    signal = pd.Series(0, index=close.index, dtype="Int8")
    signal = signal.mask(log_future > threshold, 1).mask(log_future < -threshold, -1)
    return signal.astype("Int8")


def _filter_synthetic(candles: pd.DataFrame) -> pd.DataFrame:
    if "is_synthetic" not in candles.columns:
        return candles
    mask = candles["is_synthetic"].astype(bool)
    if mask.any():
        LOGGER.info("Dropping synthetic candles: removed %s rows", int(mask.sum()))
    return candles.loc[~mask].drop(columns=["is_synthetic"])


def _bos_features(candles: pd.DataFrame, *, tf_frame: Optional[str] = None) -> pd.DataFrame:
    """Break-of-structure flags for configurable lookbacks."""

    LOGGER.debug(
        "Computing BOS features tf_frame=%s rows=%s",
        tf_frame or "unknown",
        len(candles),
    )
    highs = candles["high"]
    lows = candles["low"]
    data: Dict[str, pd.Series] = {}
    for window in STRUCTURE_WINDOWS:
        prev_high = highs.shift(1).rolling(window, min_periods=1).max()
        prev_low = lows.shift(1).rolling(window, min_periods=1).min()
        data[f"bos_up_{window}"] = (highs > prev_high).astype(float)
        data[f"bos_down_{window}"] = (lows < prev_low).astype(float)
    return pd.DataFrame(data, index=candles.index)


def _fvg_features(candles: pd.DataFrame, *, tf_frame: Optional[str] = None) -> pd.DataFrame:
    """Fair value gap metrics capturing gaps and fill ratios."""

    LOGGER.debug(
        "Computing FVG features tf_frame=%s rows=%s",
        tf_frame or "unknown",
        len(candles),
    )
    high_prev = candles["high"].shift(1)
    low_prev = candles["low"].shift(1)
    up_gap = candles["low"] - high_prev
    down_gap = low_prev - candles["high"]
    gap_up = up_gap.where(up_gap > 0.0, 0.0)
    gap_down = down_gap.where(down_gap > 0.0, 0.0)
    gap_width = (candles["high"] - candles["low"]).replace(0.0, np.nan)
    up_fill = 1.0 - (candles["close"] - high_prev).clip(lower=0.0) / gap_up.replace(0.0, np.nan)
    down_fill = 1.0 - (low_prev - candles["close"]).clip(lower=0.0) / gap_down.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "fvg_up_width": gap_up.fillna(0.0),
            "fvg_down_width": gap_down.fillna(0.0),
            "fvg_up_fill": up_fill.clip(0.0, 1.0).fillna(0.0),
            "fvg_down_fill": down_fill.clip(0.0, 1.0).fillna(0.0),
        },
        index=candles.index,
    )


def _swing_features(candles: pd.DataFrame, *, tf_frame: Optional[str] = None) -> pd.DataFrame:
    close = candles["close"]
    LOGGER.debug(
        "Computing swing features tf_frame=%s rows=%s",
        tf_frame or "unknown",
        len(candles),
    )
    atr = (candles["high"] - candles["low"]).rolling(14, min_periods=1).mean().replace(0.0, np.nan)
    threshold = atr * SWING_THRESHOLD_ATR
    swing_high = pd.Series(index=close.index, dtype=float)
    swing_low = pd.Series(index=close.index, dtype=float)
    swing_time_high = pd.Series(index=close.index, dtype=float)
    swing_time_low = pd.Series(index=close.index, dtype=float)
    last_high = close.iloc[0]
    last_low = close.iloc[0]
    last_high_ts = close.index[0]
    last_low_ts = close.index[0]
    for ts, price in close.items():
        th = threshold.loc[ts]
        if np.isnan(th):
            th = 0.0
        if price >= last_high + th:
            last_high = price
            last_high_ts = ts
        if price <= last_low - th:
            last_low = price
            last_low_ts = ts
        swing_high.loc[ts] = last_high
        swing_low.loc[ts] = last_low
        swing_time_high.loc[ts] = (ts - last_high_ts).total_seconds()
        swing_time_low.loc[ts] = (ts - last_low_ts).total_seconds()
    distance_high = (swing_high - close).abs()
    distance_low = (close - swing_low).abs()
    return pd.DataFrame(
        {
            "swing_dist_high": distance_high.fillna(0.0),
            "swing_dist_low": distance_low.fillna(0.0),
            "swing_time_high": swing_time_high.fillna(0.0),
            "swing_time_low": swing_time_low.fillna(0.0),
        },
        index=close.index,
    )


def _impulse_features(candles: pd.DataFrame) -> pd.DataFrame:
    close = candles["close"].astype(float)
    volume = candles["volume"].astype(float)
    returns = close.pct_change().fillna(0.0)
    rolling_sigma = returns.rolling(30, min_periods=5).std().replace(0.0, np.nan)
    burst = (returns.abs() / rolling_sigma).clip(0.0, 10.0).fillna(0.0)
    volume_z = (volume - volume.rolling(60, min_periods=5).mean()) / volume.rolling(60, min_periods=5).std().replace(0.0, np.nan)
    atr = (candles["high"] - candles["low"]).rolling(14, min_periods=1).mean().fillna(0.0)
    atr_acc = atr.diff().fillna(0.0)
    return pd.DataFrame(
        {
            "burst_score": burst,
            "volume_zscore": volume_z.fillna(0.0),
            "atr_accel": atr_acc,
        },
        index=candles.index,
    )


def _liquidity_state(candles: pd.DataFrame) -> pd.DataFrame:
    close = candles["close"]
    ema_fast = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=55, adjust=False).mean()
    slope = ema_fast.diff().fillna(0.0)
    adx = (ema_fast - ema_slow).abs() / close.replace(0.0, np.nan)
    liquidity_proxy = candles["volume"].rolling(30, min_periods=5).mean().fillna(method="ffill").fillna(0.0)
    return pd.DataFrame(
        {
            "trend_slope": slope,
            "trend_adx_proxy": adx.fillna(0.0),
            "liquidity_proxy": liquidity_proxy,
        },
        index=candles.index,
    )


def _compute_fee_buffer(candles: pd.DataFrame, fee_bp: float, spread_model: str = "median") -> pd.Series:
    close = candles["close"].astype(float)
    high = candles.get("high", close)
    low = candles.get("low", close)
    spread = (high - low).abs()
    if spread_model == "median":
        spread_est = spread.rolling(50, min_periods=1).median()
    else:
        spread_est = spread.rolling(50, min_periods=1).mean()
    spread_ratio = (spread_est / close.replace(0.0, np.nan)).fillna(0.0)
    fee = fee_bp / 10_000.0
    buffer = fee + spread_ratio / 2.0
    return buffer.ffill().fillna(0.0)


def _cross_timeframe_features(
    candles: pd.DataFrame,
    cross_timeframes: Optional[Dict[str, pd.DataFrame]],
    tf_frame: Optional[str],
) -> pd.DataFrame:
    if not cross_timeframes:
        return pd.DataFrame(index=candles.index)
    features: Dict[str, pd.Series] = {}
    for tf, frame in cross_timeframes.items():
        if frame is None or frame.empty:
            continue
        freq = pd.to_timedelta(max(1.0, interval_to_seconds(tf)), unit="s")
        aligned_idx = candles.index.floor(freq)
        high_tf = frame.sort_index()
        for column in ["open", "high", "low", "close", "volume"]:
            if column not in high_tf.columns:
                high_tf[column] = 0.0
        aligned = high_tf.reindex(aligned_idx).ffill()
        aligned.index = candles.index
        prefix = f"{tf}_"
        LOGGER.debug(
            "Aligning cross-TF features base_tf=%s high_tf=%s rows=%s",
            tf_frame or "unknown",
            tf,
            len(aligned),
        )
        features[f"agg_{prefix}open"] = aligned["open"].astype(float)
        features[f"agg_{prefix}high"] = aligned["high"].astype(float)
        features[f"agg_{prefix}low"] = aligned["low"].astype(float)
        features[f"agg_{prefix}close"] = aligned["close"].astype(float)
        features[f"agg_{prefix}volume"] = aligned["volume"].astype(float)
        features[f"agg_{prefix}hl_range"] = (aligned["high"] - aligned["low"]).astype(float)
    return pd.DataFrame(features, index=candles.index)


def build_features(
    candles: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    book: Optional[pd.DataFrame] = None,
    horizons: Optional[Iterable[int]] = None,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    fee_bp: Optional[float] = None,
    spread_model: str = "median",
    *,
    cross_timeframes: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[int, FeatureSet]:
    if horizons is None:
        horizons = [get_settings().data.horizon_min]
    horizons = sorted({int(h) for h in horizons if h > 0})
    if not horizons:
        raise ValueError("At least one horizon must be provided")

    candles = _ensure_index(candles)
    candles = _filter_synthetic(candles)
    candles = candles.sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in candles.columns:
            candles[column] = candles[column].astype(float)
        else:
            candles[column] = 0.0

    indicator_df = prepare_indicators(candles)
    orderflow_df = build_orderflow_features(trades, book, candles.index)
    time_df = _time_features(candles.index)
    structure_df = _bos_features(candles, tf_frame=interval)
    fvg_df = _fvg_features(candles, tf_frame=interval)
    swing_df = _swing_features(candles, tf_frame=interval)
    impulse_df = _impulse_features(candles)
    liquidity_df = _liquidity_state(candles)

    base_features = pd.concat(
        [indicator_df, orderflow_df, time_df, structure_df, fvg_df, swing_df, impulse_df, liquidity_df],
        axis=1,
    )
    base_features = base_features.replace([np.inf, -np.inf], np.nan)

    cross_df = _cross_timeframe_features(candles, cross_timeframes, interval)
    if not cross_df.empty:
        base_features = pd.concat([base_features, cross_df], axis=1)

    vol_reference = indicator_df.get("realized_vol_60")
    vol_series = vol_reference if vol_reference is not None else pd.Series(index=candles.index, dtype=float)
    vol_bucket_series, vol_thresholds = _assign_vol_buckets(vol_series)
    regimes_raw = pd.DataFrame(
        {
            "session": _assign_sessions(candles.index),
            "vol_bucket": vol_bucket_series,
        },
        index=candles.index,
    )
    regimes = pd.get_dummies(regimes_raw, columns=["session", "vol_bucket"], drop_first=False, dtype=float)

    interactions = _interaction_features(base_features.fillna(0.0), regimes)

    features = pd.concat([base_features, regimes, interactions], axis=1)
    features = features.ffill(limit=5).bfill(limit=5)
    features = features.dropna()

    fee_bp = fee_bp if fee_bp is not None else get_settings().data.fee_bp
    fee_buffer = _compute_fee_buffer(candles.loc[features.index], fee_bp, spread_model=spread_model)

    results: Dict[int, FeatureSet] = {}
    for horizon in horizons:
        target = make_target(candles.loc[features.index, "close"], horizon, fee_buffer)
        y = target.loc[features.index]
        X = features.loc[y.index].astype(float)
        if len(X) == 0:
            LOGGER.warning("No valid rows for horizon=%s", horizon)
            continue
        regime_slice = regimes.loc[X.index]
        regime_labels = regimes_raw.loc[X.index, "vol_bucket"].astype(str)
        metadata = {
            "symbol": symbol,
            "interval": interval,
            "horizon": horizon,
            "feature_columns": list(X.columns),
            "n_rows": int(len(X)),
            "vol_thresholds": vol_thresholds,
        }
        results[horizon] = FeatureSet(
            horizon=horizon,
            X=X,
            y=y.astype(int),
            regimes=regime_slice,
            regime_labels=regime_labels,
            metadata=metadata,
        )

    if not results:
        raise ValueError("Feature pipeline produced no rows. Check input data availability.")

    return results


def save_feature_sets(
    feature_sets: Mapping[int, FeatureSet],
    base_path: Path | str = Path("data/features"),
) -> Dict[int, Path]:
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[int, Path] = {}
    for horizon, fs in feature_sets.items():
        path = base_path / f"features_h{horizon}.parquet"
        df = fs.X.copy()
        df["target"] = fs.y
        df["regime_label"] = fs.regime_labels
        for column in fs.regimes.columns:
            df[column] = fs.regimes[column]
        write_parquet(df, path)
        meta_path = path.with_suffix(".meta.json")
        meta = {**fs.metadata, "regime_columns": list(fs.regimes.columns)}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths[horizon] = path
        LOGGER.info("Saved feature set horizon=%s rows=%s path=%s", horizon, len(fs.X), path)
    return saved_paths
