"""Utilities for computing trade delta metrics per OHLC bar."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import aiohttp

from .binance import (
    BinanceAPIException,
    BinanceRequestException,
    fetch_um_agg_trades,
)
from .ohlc import TIMEFRAME_TO_MS, TIMEFRAME_WINDOWS, fetch_ohlcv


@dataclass(slots=True)
class AggTrade:
    """Normalized view of a Binance aggregated trade."""

    time: int
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass(slots=True)
class BarDelta:
    """Delta metrics calculated for a single OHLC bar."""

    t: int
    tf: str
    delta: float
    delta_max: float
    delta_min: float
    delta_pct: float
    cvd: float

    def as_dict(self) -> Dict[str, float | int | str]:
        return {
            "t": self.t,
            "tf": self.tf,
            "delta": self.delta,
            "deltaMax": self.delta_max,
            "deltaMin": self.delta_min,
            "deltaPct": self.delta_pct,
            "cvd": self.cvd,
        }


def _normalise_trade(row: Dict[str, object]) -> AggTrade | None:
    try:
        time = int(row["T"])  # millisecond timestamp
        price = float(row["p"])
        quantity = float(row["q"])
        is_buyer_maker = bool(row["m"])
    except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    return AggTrade(time=time, price=price, quantity=quantity, is_buyer_maker=is_buyer_maker)


async def _fetch_trades(symbol: str, start_ms: int, end_ms: int) -> List[AggTrade]:
    """Fetch aggregated trades from Binance futures between the given bounds."""

    trades: List[AggTrade] = []
    cursor = start_ms
    limit = 1000

    while cursor < end_ms:
        try:
            rows = await fetch_um_agg_trades(
                symbol,
                start_time=cursor,
                end_time=end_ms,
                limit=limit,
            )
        except (BinanceAPIException, BinanceRequestException, aiohttp.ClientError, asyncio.TimeoutError):
            # Propagate to caller for consistent error handling.
            raise

        if not rows:
            break

        batch_added = 0
        last_time: int | None = None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            trade = _normalise_trade(dict(row))
            if trade is None:
                continue
            if trade.time < start_ms or trade.time > end_ms:
                continue
            trades.append(trade)
            batch_added += 1
            last_time = trade.time

        if batch_added == 0:
            cursor += 1000
        else:
            if last_time is None:
                break
            if last_time >= end_ms:
                break
            cursor = last_time + 1

        if len(rows) < limit:
            break

    trades.sort(key=lambda trade: trade.time)
    return trades


def _iter_bar_bounds(candles: Sequence[Dict[str, object]], interval_ms: int) -> Iterable[tuple[int, int]]:
    for candle in candles:
        open_time = int(candle.get("t", 0))
        yield open_time, open_time + interval_ms


def _compute_bar_metrics(
    trades: Sequence[AggTrade],
    bar_bounds: Sequence[tuple[int, int]],
    timeframe: str,
) -> List[BarDelta]:
    deltas: List[BarDelta] = []
    trade_index = 0
    cumulative_delta = 0.0
    total_trades = len(trades)

    for open_ms, close_ms in bar_bounds:
        buy_volume = 0.0
        sell_volume = 0.0
        running_delta = 0.0
        bar_delta_max = 0.0
        bar_delta_min = 0.0
        first_trade_processed = False

        while trade_index < total_trades:
            trade = trades[trade_index]
            if trade.time >= close_ms:
                break
            if trade.time < open_ms:
                trade_index += 1
                continue
            signed_qty = trade.quantity if not trade.is_buyer_maker else -trade.quantity
            if signed_qty >= 0:
                buy_volume += trade.quantity
            else:
                sell_volume += trade.quantity
            running_delta += signed_qty
            if not first_trade_processed:
                bar_delta_max = bar_delta_min = running_delta
                first_trade_processed = True
            else:
                bar_delta_max = max(bar_delta_max, running_delta)
                bar_delta_min = min(bar_delta_min, running_delta)
            trade_index += 1

        net_delta = buy_volume - sell_volume
        volume = buy_volume + sell_volume
        delta_pct = (net_delta / volume * 100.0) if volume > 0 else 0.0
        if not first_trade_processed:
            bar_delta_max = 0.0
            bar_delta_min = 0.0
        cumulative_delta += net_delta
        deltas.append(
            BarDelta(
                t=open_ms,
                tf=timeframe,
                delta=net_delta,
                delta_max=bar_delta_max,
                delta_min=bar_delta_min,
                delta_pct=delta_pct,
                cvd=cumulative_delta,
            )
        )
    return deltas


async def fetch_bar_delta(symbol: str, timeframe: str) -> Dict[str, object]:
    """Compute delta metrics for the given symbol/timeframe."""

    timeframe = timeframe.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    ohlcv = await fetch_ohlcv(symbol, timeframe)
    candles = ohlcv.get("candles", []) if isinstance(ohlcv, dict) else []
    if not candles:
        return {"symbol": symbol.upper(), "bar_delta": []}

    interval_ms = TIMEFRAME_TO_MS[timeframe]
    bar_bounds = list(_iter_bar_bounds(candles, interval_ms))
    start_ms = bar_bounds[0][0]
    end_ms = bar_bounds[-1][1]
    trades = await _fetch_trades(symbol, start_ms, end_ms)
    deltas = _compute_bar_metrics(trades, bar_bounds, timeframe)
    return {
        "symbol": symbol.upper(),
        "bar_delta": [delta.as_dict() for delta in deltas],
    }


def fetch_bar_delta_sync(symbol: str, timeframe: str) -> Dict[str, object]:
    """Synchronous helper for scripts/tests."""

    return asyncio.run(fetch_bar_delta(symbol, timeframe))
