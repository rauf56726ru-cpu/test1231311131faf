"""Aggregate Binance trades into OHLCV candles using deterministic rules.

This module implements the exact procedure described in the specification:

* Trades are received from the real-time ``@trade`` websocket stream.
* Missing trades are backfilled through the ``/aggTrades`` REST endpoint.
* Candles are keyed by ``bucket = floor(ts/Δ) * Δ`` where ``Δ`` is the candle
  interval in seconds and ``ts`` is the trade timestamp in seconds.
* Each candle tracks the open, high, low, close, base volume, quote volume,
  trade count and taker-buy volumes.
* Buckets are emitted when ``now >= bucket + Δ`` and gaps are filled with
  synthetic empty candles that keep the previous close price.

The implementation is intentionally self-contained so that it can be reused in
tests and during live streaming without introducing additional dependencies on
other parts of the codebase.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..utils.logging import get_logger

try:  # pragma: no cover - optional import for type checking
    from typing import Awaitable
except Exception:  # pragma: no cover - Python <3.11 compatibility
    Awaitable = Any  # type: ignore

from .binance_ws import fetch_agg_trades

LOGGER = get_logger(__name__)


@dataclass
class CandleState:
    """Mutable state accumulated for a single candle bucket."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    taker_buy_quote: float = 0.0


TradePayload = Mapping[str, Any]
AggTradeFetcher = Callable[[str, Any | None, Any | None, int], Awaitable[Sequence[Mapping[str, Any]]]]


class TradeCandleAggregator:
    """Convert Binance trades into OHLCV candles following the reference rules."""

    def __init__(
        self,
        symbol: str,
        interval_seconds: float,
        *,
        time_fn: Callable[[], float] | None = None,
        rest_fetcher: AggTradeFetcher | None = None,
    ) -> None:
        self.symbol = (symbol or "").upper()
        self.interval_seconds = max(1, int(interval_seconds))
        self._time_fn = time_fn or time.time
        self._rest_fetcher = rest_fetcher or fetch_agg_trades
        self._state: Dict[int, CandleState] = {}
        self._last_emitted_bucket: Optional[int] = None
        self._last_close: Optional[float] = None
        self.last_seen_time_ms: int = 0

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _bucket_from_ms(self, time_ms: int) -> int:
        ts_sec = max(0, int(time_ms) // 1000)
        return (ts_sec // self.interval_seconds) * self.interval_seconds

    # ------------------------------------------------------------- state ops
    def _ensure_state(self, bucket: int, price: float) -> CandleState:
        state = self._state.get(bucket)
        if state is None:
            if self._last_emitted_bucket is not None and bucket <= self._last_emitted_bucket:
                raise ValueError(f"Bucket {bucket} already emitted; late trade not allowed")
            state = CandleState(open=price, high=price, low=price, close=price)
            self._state[bucket] = state
        return state

    def _emit_empty(self, bucket: int) -> Optional[List[Any]]:
        if self._last_close is None:
            return None
        open_time_ms = bucket * 1000
        close_time_ms = (bucket + self.interval_seconds) * 1000 - 1
        price = float(self._last_close)
        return [
            int(open_time_ms),
            price,
            price,
            price,
            price,
            0.0,
            int(close_time_ms),
            0.0,
            0,
            0.0,
            0.0,
        ]

    def _emit_bucket(self, bucket: int, state: CandleState) -> List[List[Any]]:
        emitted: List[List[Any]] = []
        if self._last_emitted_bucket is not None:
            missing = self._last_emitted_bucket + self.interval_seconds
            while missing < bucket:
                empty = self._emit_empty(missing)
                if empty is not None:
                    emitted.append(empty)
                    self._last_emitted_bucket = missing
                missing += self.interval_seconds
        open_time_ms = bucket * 1000
        close_time_ms = (bucket + self.interval_seconds) * 1000 - 1
        record = [
            int(open_time_ms),
            float(state.open),
            float(state.high),
            float(state.low),
            float(state.close),
            float(state.volume),
            int(close_time_ms),
            float(state.quote_volume),
            int(state.trades),
            float(state.taker_buy_volume),
            float(state.taker_buy_quote),
        ]
        emitted.append(record)
        self._last_emitted_bucket = bucket
        self._last_close = float(state.close)
        return emitted

    def _close_ready(self, now_sec: int) -> List[List[Any]]:
        emitted: List[List[Any]] = []
        while self._state:
            oldest_bucket = min(self._state)
            if now_sec >= oldest_bucket + self.interval_seconds:
                state = self._state.pop(oldest_bucket)
                emitted.extend(self._emit_bucket(oldest_bucket, state))
            else:
                break
        return emitted

    # --------------------------------------------------------------- API
    def update_from_trade(
        self, trade: TradePayload, *, current_time_ms: Optional[int] = None
    ) -> List[List[Any]]:
        """Process a single trade payload and emit any completed candles.

        Args:
            trade: Mapping with keys ``T`` (trade time in milliseconds), ``p``
                (price), ``q`` (quantity) and ``m`` (``True`` if buyer is
                maker).
            current_time_ms: Optional override for the "now" timestamp used in
                the closing condition. If omitted the ``time_fn`` provided at
                construction time is used.

        Returns:
            A list of emitted candles, each matching the Binance kline payload
            structure ``[openTime, open, high, low, close, volume, closeTime,
            quoteVolume, trades, takerBuyBase, takerBuyQuote]``.
        """

        try:
            trade_time_ms = int(trade["T"])
        except (KeyError, TypeError, ValueError):
            LOGGER.debug("Skipping trade without valid timestamp: %s", trade)
            return []

        price = self._to_float(trade.get("p"))
        qty = self._to_float(trade.get("q"))
        if price is None or qty is None:
            LOGGER.debug("Skipping trade with invalid price/qty: %s", trade)
            return []

        bucket = self._bucket_from_ms(trade_time_ms)
        if self._last_emitted_bucket is not None and bucket < self._last_emitted_bucket:
            LOGGER.debug("Ignoring late trade for closed bucket %s: %s", bucket, trade)
            self.last_seen_time_ms = max(self.last_seen_time_ms, trade_time_ms)
            return []

        try:
            state = self._ensure_state(bucket, price)
        except ValueError:
            LOGGER.debug("Late trade for already emitted bucket %s", bucket)
            self.last_seen_time_ms = max(self.last_seen_time_ms, trade_time_ms)
            return []

        state.high = max(state.high, price)
        state.low = min(state.low, price)
        state.close = price
        state.volume += qty
        state.quote_volume += price * qty
        state.trades += 1
        is_buyer_maker = bool(trade.get("m", False))
        if not is_buyer_maker:
            state.taker_buy_volume += qty
            state.taker_buy_quote += price * qty

        self.last_seen_time_ms = max(self.last_seen_time_ms, trade_time_ms)

        now_ms = current_time_ms if current_time_ms is not None else int(self._time_fn() * 1000)
        now_sec = max(trade_time_ms // 1000, now_ms // 1000)
        return self._close_ready(now_sec)

    def force_close_older_than(self, cutoff_sec: int) -> List[List[Any]]:
        """Close all buckets whose open time is ``<= cutoff_sec``."""

        target = cutoff_sec + self.interval_seconds
        return self._close_ready(target)

    def close_ready(self, *, current_time_ms: Optional[int] = None) -> List[List[Any]]:
        """Close any buckets that satisfy the closing condition at ``now``."""

        now_ms = current_time_ms if current_time_ms is not None else int(self._time_fn() * 1000)
        now_sec = now_ms // 1000
        return self._close_ready(now_sec)

    def flush(self) -> List[List[Any]]:
        """Force close all remaining buckets regardless of ``now``."""

        if not self._state:
            return []
        last_bucket = max(self._state)
        target = last_bucket + self.interval_seconds
        return self._close_ready(target)

    async def recover_missing(
        self,
        *,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        fetcher: Optional[AggTradeFetcher] = None,
        limit: int = 1000,
    ) -> List[List[Any]]:
        """Fetch and process historical aggregated trades via REST.

        Args:
            start_time_ms: Timestamp (inclusive) to begin the recovery from.
                Defaults to ``last_seen_time_ms + 1``.
            end_time_ms: Timestamp (inclusive) to stop recovery. Defaults to
                the current ``time_fn`` timestamp.
            fetcher: Custom coroutine for fetching aggregated trades. Defaults
                to :func:`fetch_agg_trades`.
            limit: REST page size, clamped to Binance constraints.

        Returns:
            List of emitted candles produced while replaying the recovered
            trades.
        """

        fetch = fetcher or self._rest_fetcher
        if fetch is None:
            return []

        if start_time_ms is None:
            start_time_ms = self.last_seen_time_ms + 1 if self.last_seen_time_ms else None
        if start_time_ms is None:
            return []

        end_time = end_time_ms if end_time_ms is not None else int(self._time_fn() * 1000)
        cursor = max(0, int(start_time_ms))
        end_time = max(cursor, int(end_time))
        safe_limit = max(1, min(int(limit), 1000))
        emitted: List[List[Any]] = []
        safety = 0

        while cursor <= end_time and safety < 1000:
            safety += 1
            try:
                batch = await fetch(self.symbol, cursor, end_time, safe_limit)
            except Exception as exc:  # pragma: no cover - network safeguards
                LOGGER.warning("Failed to recover aggTrades for %s: %s", self.symbol, exc)
                break
            if not batch:
                break
            ordered = sorted(batch, key=lambda item: int(item.get("T", cursor)))
            for trade in ordered:
                trade_time = int(trade.get("T", cursor))
                emitted.extend(self.update_from_trade(trade, current_time_ms=trade_time))
            last_trade_time = int(ordered[-1].get("T", cursor))
            if last_trade_time <= cursor:
                break
            cursor = last_trade_time + 1
            if len(ordered) < safe_limit:
                break

        emitted.extend(self.close_ready(current_time_ms=end_time))
        return emitted

    # ----------------------------------------------------------- inspection
    @property
    def open_buckets(self) -> Dict[int, CandleState]:
        """Return a shallow copy of the currently open candle states."""

        return dict(self._state)


__all__ = ["TradeCandleAggregator", "CandleState"]
