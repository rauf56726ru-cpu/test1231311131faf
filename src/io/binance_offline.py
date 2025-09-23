"""Offline Binance futures fixture data for environments blocked from REST."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import cos, sin
from typing import Dict, List, Optional, Tuple

INTERVAL_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "10m": 600_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass(frozen=True)
class OfflineCandle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    def to_row(self) -> List[float]:
        return [
            int(self.open_time),
            float(f"{self.open:.2f}"),
            float(f"{self.high:.2f}"),
            float(f"{self.low:.2f}"),
            float(f"{self.close:.2f}"),
            float(f"{self.volume:.3f}"),
            int(self.close_time),
        ]


class OfflineDataUnavailable(RuntimeError):
    """Raised when the offline dataset cannot satisfy a request."""


class OfflineBinanceData:
    """Deterministic offline data generator to mirror Binance futures klines."""

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        base_interval: str = "1m",
        base_count: int = 50_000,
    ) -> None:
        if base_interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported base interval: {base_interval}")
        self.symbol = symbol.upper()
        self.base_interval = base_interval
        self.base_count = max(1, int(base_count))
        self._cache: Dict[Tuple[str, str], List[OfflineCandle]] = {}
        self._tick_sizes: Dict[str, float] = {self.symbol: 0.1}
        self._build_base()

    def _build_base(self) -> None:
        interval_ms = INTERVAL_TO_MS[self.base_interval]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Align end to interval boundary and keep one full interval behind "now".
        end_aligned = now_ms - (now_ms % interval_ms) - interval_ms
        if end_aligned <= 0:
            end_aligned = interval_ms * self.base_count
        start_ms = end_aligned - interval_ms * (self.base_count - 1)
        candles: List[OfflineCandle] = []
        for idx in range(self.base_count):
            ts = start_ms + idx * interval_ms
            base = 26_000 + sin(idx / 180.0) * 900 + cos(idx / 55.0) * 320
            open_price = base + sin(idx / 12.0) * 120
            close_price = base + cos((idx + 0.5) / 9.0) * 110
            upper = abs(sin(idx / 5.0)) * 60 + 25
            lower = abs(cos(idx / 7.0)) * 60 + 25
            high_price = max(open_price, close_price) + upper
            low_price = min(open_price, close_price) - lower
            volume = 500 + abs(sin(idx / 15.0)) * 220 + (idx % 10) * 12
            candles.append(
                OfflineCandle(
                    open_time=int(ts),
                    open=float(f"{open_price:.2f}"),
                    high=float(f"{high_price:.2f}"),
                    low=float(f"{low_price:.2f}"),
                    close=float(f"{close_price:.2f}"),
                    volume=float(f"{volume:.3f}"),
                    close_time=int(ts + interval_ms - 1),
                )
            )
        self._cache[(self.symbol, self.base_interval)] = candles

    def _ensure_interval(self, symbol: str, interval: str) -> List[OfflineCandle]:
        norm_symbol = symbol.upper()
        key = (norm_symbol, interval)
        if key in self._cache:
            return self._cache[key]
        if norm_symbol != self.symbol:
            raise OfflineDataUnavailable(f"Offline data only available for {self.symbol}")
        if interval == self.base_interval:
            return self._cache[key]
        if interval not in INTERVAL_TO_MS:
            raise OfflineDataUnavailable(f"Unsupported interval: {interval}")
        base = self._ensure_interval(norm_symbol, self.base_interval)
        interval_ms = INTERVAL_TO_MS[interval]
        base_ms = INTERVAL_TO_MS[self.base_interval]
        if interval_ms % base_ms != 0:
            raise OfflineDataUnavailable(f"Interval {interval} not aligned with {self.base_interval}")
        ratio = interval_ms // base_ms
        start_offset = 0
        for candidate in range(min(ratio, len(base))):
            if base[candidate].open_time % interval_ms == 0:
                start_offset = candidate
                break
        aggregated: List[OfflineCandle] = []
        for start in range(start_offset, len(base), ratio):
            window = base[start : start + ratio]
            if len(window) < ratio:
                break
            open_raw = window[0].open_time
            open_time = open_raw - (open_raw % interval_ms)
            close_time = open_time + interval_ms - 1
            aggregated.append(
                OfflineCandle(
                    open_time=open_time,
                    open=window[0].open,
                    high=max(bar.high for bar in window),
                    low=min(bar.low for bar in window),
                    close=window[-1].close,
                    volume=float(f"{sum(bar.volume for bar in window):.3f}"),
                    close_time=close_time,
                )
            )
        self._cache[key] = aggregated
        return aggregated

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List[float]]:
        candles = list(self._ensure_interval(symbol, interval))
        if not candles:
            return []
        start = int(start_time) if start_time is not None else None
        end = int(end_time) if end_time is not None else None

        def _within(candle: OfflineCandle) -> bool:
            if start is not None and candle.open_time < start:
                return False
            if end is not None and candle.open_time > end:
                return False
            return True

        filtered = [bar for bar in candles if _within(bar)]
        if start is None and end is None:
            subset = filtered[-limit:] if limit else filtered
        elif start is None and end is not None:
            subset = filtered[-limit:] if limit else filtered
        else:
            subset = filtered[:limit] if limit else filtered
        return [bar.to_row() for bar in subset]

    def tick_size(self, symbol: str) -> float:
        return float(self._tick_sizes.get(symbol.upper(), 0.1))

    def exchange_info(self, symbol: str) -> Dict[str, object]:
        symbol = symbol.upper()
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": float(f"{self.tick_size(symbol):.6f}")}
                    ],
                }
            ]
        }


OFFLINE_DATA = OfflineBinanceData()
__all__ = ["OFFLINE_DATA", "OfflineBinanceData", "OfflineDataUnavailable", "INTERVAL_TO_MS"]
