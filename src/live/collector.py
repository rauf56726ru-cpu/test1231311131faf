"""Synthetic stream manager used by the web interface."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logging import get_logger
from .buffer import FeatureBuffer
from .market import MarketDataProvider

LOGGER = get_logger(__name__)


@dataclass
class StreamState:
    symbol: str
    interval: str
    window: int
    provider: MarketDataProvider
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0
    last_price: float = 100.0
    last_timestamp: datetime = field(default_factory=datetime.utcnow)
    position: str = "flat"
    entry_price: Optional[float] = None
    pnl: float = 0.0
    balance: float = 10_000.0
    last_features: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.buffer = FeatureBuffer(window=self.window)
        self._history_window = max(self.window, 1200)
        self.candle_history: Deque[Dict[str, float]] = deque(maxlen=self._history_window)
        self.prediction_marks: Deque[Dict[str, Any]] = deque(maxlen=self._history_window)

    async def ensure_bootstrap(self, limit: int) -> None:
        if self.candle_history:
            return
        candles = await self.provider.history(self.symbol, self.interval, max(limit, self._history_window))
        for candle in candles:
            self._append_candle(candle)

    def _append_candle(self, candle: Dict[str, float]) -> Dict[str, float]:
        data = dict(candle)
        ts_ms = data.get("ts_ms_utc")
        if ts_ms is not None:
            try:
                data["ts_ms_utc"] = int(ts_ms)
            except (TypeError, ValueError):
                data["ts_ms_utc"] = int(datetime.utcnow().timestamp() * 1000)
        else:
            data["ts_ms_utc"] = int(datetime.utcnow().timestamp() * 1000)
        ts = data.get("timestamp")
        if not isinstance(ts, datetime):
            ts = datetime.fromtimestamp(data["ts_ms_utc"] / 1000.0, tz=timezone.utc)
            data["timestamp"] = ts
        prev: Optional[Dict[str, float]] = self.candle_history[-1] if self.candle_history else None
        prev_ts = int(prev["ts_ms_utc"]) if prev and "ts_ms_utc" in prev else None
        current_ts = int(data["ts_ms_utc"])
        if prev_ts is not None and current_ts <= prev_ts:
            while self.candle_history and int(self.candle_history[-1].get("ts_ms_utc", 0)) >= current_ts:
                self.candle_history.pop()
                if self.buffer.buffer:
                    self.buffer.buffer.pop()
            prev = self.candle_history[-1] if self.candle_history else None
        prev_close = float(prev["close"]) if prev else float(data.get("close", 0.0) or 0.0)
        close = float(data.get("close", prev_close))
        change = (close - prev_close) / prev_close if prev_close else 0.0
        momentum = close - prev_close
        volatility = float(abs(float(data.get("high", close)) - float(data.get("low", close))))
        volume = float(data.get("volume", 0.0))
        features = {
            "close": close,
            "return": float(change),
            "volume": volume,
            "momentum": float(momentum),
            "volatility": volatility,
        }
        self.buffer.add(features)
        self.candle_history.append(data)
        self.last_price = close
        self.last_timestamp = ts
        self.last_features = dict(features)
        return features

    async def step(
        self,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], pd.DataFrame]:
        async with self.lock:
            await self.ensure_bootstrap(self.window)
            last = self.candle_history[-1] if self.candle_history else None
            attempts = 0
            new_candle: Optional[Dict[str, float]] = None
            while attempts < 3:
                candidate = await self.provider.next_candle(self.symbol, self.interval, last)
                if candidate is None:
                    attempts += 1
                    if attempts < 3:
                        await asyncio.sleep(0.5)
                    continue
                new_candle = candidate
                break
            if new_candle is None:
                frame = self.buffer.to_frame()
                return None, None, frame
            features = self._append_candle(new_candle)
            frame = self.buffer.to_frame()
            return dict(self.candle_history[-1]), dict(features), frame

    def apply_decision(self, side: str, price: float) -> None:
        if side == self.position:
            return

        if self.position == "long" and self.entry_price is not None:
            self.pnl += price - self.entry_price
        elif self.position == "short" and self.entry_price is not None:
            self.pnl += self.entry_price - price

        if side in {"long", "short"}:
            self.position = side
            self.entry_price = price
        else:
            self.position = "flat"
            self.entry_price = None

        self.balance = 10_000.0 + self.pnl


class StreamManager:
    """Manage market data streams with reference counting and history cache."""

    def __init__(self, provider: MarketDataProvider, window: int = 300) -> None:
        self.window = window
        self.provider = provider
        self._streams: Dict[Tuple[str, str], StreamState] = {}
        self._lock = asyncio.Lock()

    async def ensure_stream(self, symbol: str, interval: str) -> StreamState:
        key = (symbol.upper(), interval)
        async with self._lock:
            state = self._streams.get(key)
            if state is None:
                LOGGER.info("Starting stream %s %s", symbol.upper(), interval)
                state = StreamState(symbol=symbol.upper(), interval=interval, window=self.window, provider=self.provider)
                self._streams[key] = state
            state.ref_count += 1
            return state

    async def release_stream(self, symbol: str, interval: str) -> None:
        key = (symbol.upper(), interval)
        async with self._lock:
            state = self._streams.get(key)
            if state is None:
                return
            state.ref_count = max(0, state.ref_count - 1)
            if state.ref_count == 0:
                LOGGER.info("Stopping stream %s %s", symbol.upper(), interval)
                self._streams.pop(key, None)

    async def next_step(
        self, symbol: str, interval: str
    ) -> Tuple[StreamState, Optional[Dict[str, float]], Optional[Dict[str, float]], pd.DataFrame]:
        key = (symbol.upper(), interval)
        async with self._lock:
            state = self._streams.get(key)
        if state is None:
            state = await self.ensure_stream(symbol, interval)
        candle, features, frame = await state.step()
        return state, candle, features, frame

    async def snapshot(self, symbol: str, interval: str) -> Optional[StreamState]:
        key = (symbol.upper(), interval)
        async with self._lock:
            return self._streams.get(key)

    async def history(self, symbol: str, interval: str, limit: int = 1000) -> List[Dict[str, float]]:
        key = (symbol.upper(), interval)
        async with self._lock:
            state = self._streams.get(key)
        if state is None:
            candles = await self.provider.history(symbol, interval, limit)
            return [dict(bar) for bar in candles[-limit:]]
        async with state.lock:
            await state.ensure_bootstrap(max(limit, state._history_window))
            return [dict(bar) for bar in list(state.candle_history)[-limit:]]
