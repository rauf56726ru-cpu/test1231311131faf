"""Technical indicator computation helpers used across the feature pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _ensure_float(series: pd.Series) -> pd.Series:
    return pd.Series(series.astype(float), index=series.index, name=series.name)


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    series = _ensure_float(series)
    return np.log(series / series.shift(periods))


def ema(series: pd.Series, span: int) -> pd.Series:
    series = _ensure_float(series)
    return series.ewm(span=span, adjust=False).mean()


def kama(series: pd.Series, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average implementation."""

    series = _ensure_float(series)
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum()
    efficiency_ratio = change / volatility.replace(0, np.nan)
    smoothing = (efficiency_ratio * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama_values = pd.Series(np.nan, index=series.index, dtype=float)
    prev = series.iloc[0]
    kama_values.iloc[0] = prev
    for idx in range(1, len(series)):
        alpha = float(smoothing.iloc[idx] if not np.isnan(smoothing.iloc[idx]) else 2 / (slow + 1))
        prev = prev + alpha * (series.iloc[idx] - prev)
        kama_values.iloc[idx] = prev
    return kama_values


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = _ensure_float(series)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    series = _ensure_float(series)
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    series = _ensure_float(series)
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std()
    upper = mean + num_std * std
    lower = mean - num_std * std
    width = (upper - lower) / mean.replace(0, np.nan)
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_width": width})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = _ensure_float(high)
    low = _ensure_float(low)
    close = _ensure_float(close)
    prev_close = close.shift(1)
    data = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return data.max(axis=1).fillna(0.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period).mean().bfill().fillna(0.0)


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    returns = log_return(series).fillna(0.0)
    return returns.rolling(window=window, min_periods=1).std().fillna(0.0)


def rogers_satchell(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = _ensure_float(high)
    low = _ensure_float(low)
    close = _ensure_float(close)
    open_price = close.shift(1).fillna(close.iloc[0])
    term1 = np.log(high / close) * np.log(high / open_price)
    term2 = np.log(low / close) * np.log(low / open_price)
    return (term1 + term2).rolling(window=14, min_periods=1).mean().fillna(0.0)


def zscore(series: pd.Series, window: int) -> pd.Series:
    series = _ensure_float(series)
    rolling_mean = series.rolling(window, min_periods=1).mean()
    rolling_std = series.rolling(window, min_periods=1).std().replace(0.0, np.nan)
    return ((series - rolling_mean) / rolling_std).fillna(0.0)


@dataclass
class IndicatorBundle:
    candles: pd.DataFrame
    cache: Dict[str, pd.Series] = field(default_factory=dict)

    def series(self, name: str) -> pd.Series:
        return self.cache[name]

    def store(self, name: str, series: pd.Series) -> None:
        self.cache[name] = _ensure_float(series)


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the core indicator matrix described in the specification."""

    required_cols = ["open", "high", "low", "close", "volume"]
    frame = df.copy()
    for col in required_cols:
        if col in frame.columns:
            frame[col] = frame[col].astype(float)
        else:
            frame[col] = 0.0
    bundle = IndicatorBundle(frame)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    indicators: Dict[str, pd.Series] = {}

    for horizon in (1, 3, 5, 15):
        indicators[f"logret_{horizon}"] = log_return(close, horizon)

    for span in (12, 26, 50, 200):
        indicators[f"ema_{span}"] = ema(close, span)

    kama_series = kama(close)
    indicators["kama"] = kama_series

    macd_df = macd(close)
    for column in macd_df.columns:
        indicators[column] = macd_df[column]

    indicators["rsi_14"] = rsi(close, 14)

    bb_df = bollinger_bands(close)
    for column in bb_df.columns:
        indicators[column] = bb_df[column]

    indicators["atr_14"] = atr(high, low, close, 14)

    for window in (15, 60):
        indicators[f"realized_vol_{window}"] = realized_vol(close, window)
    indicators["vol_30"] = realized_vol(close, 30)

    indicators["rsj"] = rogers_satchell(high, low, close)

    indicators["zscore_close_50"] = zscore(close, 50)
    indicators["zscore_close_200"] = zscore(close, 200)
    indicators["zscore_kama_50"] = zscore(kama_series, 50)

    # Trend strength differentials for later policy interactions
    indicators["trend_ema12_26"] = indicators["ema_12"] - indicators["ema_26"]
    indicators["trend_ema50_200"] = indicators["ema_50"] - indicators["ema_200"]

    volume = _ensure_float(volume)
    indicators["log_volume"] = np.log1p(volume)
    indicators["volume_zscore_50"] = zscore(volume, 50)

    indicator_df = pd.DataFrame(indicators).astype(float)
    indicator_df.index = frame.index
    return indicator_df


def indicator_columns() -> Iterable[str]:
    """Return the canonical ordering for indicator columns."""

    base = [
        "logret_1",
        "logret_3",
        "logret_5",
        "logret_15",
        "ema_12",
        "ema_26",
        "ema_50",
        "ema_200",
        "kama",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "atr_14",
        "realized_vol_15",
        "realized_vol_60",
        "rsj",
        "zscore_close_50",
        "zscore_close_200",
        "zscore_kama_50",
        "trend_ema12_26",
        "trend_ema50_200",
        "log_volume",
        "volume_zscore_50",
    ]
    return base
