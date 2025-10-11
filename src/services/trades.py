"""Utilities for collecting side-marked aggregated trades."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import time
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

TradeDict = Mapping[str, object]
Snapshot = dict[str, object]


def _current_time_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class _NormalisedTrade:
    t: int
    p: float
    q: float
    side: str
    key: Tuple[int, str, str]

    def as_public_dict(self) -> dict[str, object]:
        return {"t": self.t, "p": self.p, "q": self.q, "side": self.side}


class AggTradeCollector:
    """Collect aggregated trades with side information in a rolling time window."""

    def __init__(
        self,
        symbol: str,
        *,
        max_age_ms: int = 86_400_000,
        min_age_ms: int = 28_800_000,
    ) -> None:
        self.symbol = (symbol or "UNKNOWN").upper()
        self._min_age_ms = max(0, int(min_age_ms))
        self.max_age_ms = max(self._min_age_ms, int(max_age_ms))
        self._clock_offset_ms = 0
        self._times: List[int] = []
        self._trades: List[dict[str, object]] = []
        self._keys: List[Tuple[int, str, str]] = []
        self._dedup: set[Tuple[int, str, str]] = set()

    def sync_clock(self, server_time_ms: int, *, received_at_ms: Optional[int] = None) -> int:
        """Update the clock offset using a server timestamp."""

        if received_at_ms is None:
            received_at_ms = _current_time_ms()
        self._clock_offset_ms = int(server_time_ms) - int(received_at_ms)
        return self._clock_offset_ms

    def ingest(self, trade: TradeDict, *, received_at_ms: Optional[int] = None) -> bool:
        """Add a single trade to the buffer."""

        normalised = self._normalise_trade(trade, received_at_ms)
        if normalised.key in self._dedup:
            return False

        insert_pos = bisect_right(self._times, normalised.t)
        self._times.insert(insert_pos, normalised.t)
        self._trades.insert(insert_pos, normalised.as_public_dict())
        self._keys.insert(insert_pos, normalised.key)
        self._dedup.add(normalised.key)
        self._expire_old()
        return True

    def ingest_many(self, trades: Iterable[TradeDict]) -> int:
        """Add multiple trades, returning the number of newly stored entries."""

        inserted = 0
        for trade in trades:
            if self.ingest(trade):
                inserted += 1
        return inserted

    def backfill_from_rest(
        self,
        fetcher: Callable[..., Sequence[TradeDict]],
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> int:
        """Recover missing trades using a REST fetcher."""

        if start_ms is None and end_ms is None:
            trades = fetcher()
        else:
            trades = fetcher(start_ms, end_ms)
        return self.ingest_many(trades)

    def get_snapshot(self) -> Snapshot:
        """Return the current buffered trades as a snapshot."""

        return {"symbol": self.symbol, "agg": [dict(trade) for trade in self._trades]}

    def __len__(self) -> int:
        return len(self._trades)

    def _normalise_trade(self, trade: TradeDict, received_at_ms: Optional[int]) -> _NormalisedTrade:
        if not isinstance(trade, Mapping):
            raise ValueError("trade must be a mapping")

        raw_ts = trade.get("t")
        if raw_ts is None:
            if received_at_ms is None:
                raise ValueError("trade does not contain 't' and received_at_ms was not provided")
            raw_ts = received_at_ms
        ts = int(raw_ts) + self._clock_offset_ms

        side_raw = str(trade.get("side", "")).strip().lower()
        if side_raw not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        price_key, price_value = self._normalise_number(trade.get("p"), "p")
        qty_key, qty_value = self._normalise_number(trade.get("q"), "q")

        return _NormalisedTrade(
            t=ts,
            p=price_value,
            q=qty_value,
            side=side_raw,
            key=(ts, price_key, qty_key),
        )

    def _normalise_number(self, value: object, field: str) -> Tuple[str, float]:
        if value is None:
            raise ValueError(f"trade missing '{field}'")
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"invalid numeric value for '{field}'") from exc
        if not decimal_value.is_finite():
            raise ValueError(f"non-finite numeric value for '{field}'")
        normalised = decimal_value.normalize()
        key = format(normalised, 'f')
        if key == '':
            key = '0'
        return key, float(decimal_value)

    def _expire_old(self) -> None:
        if not self._times:
            return
        reference_time = self._times[-1]
        cutoff = reference_time - self.max_age_ms
        while self._times and self._times[0] < cutoff:
            self._dedup.discard(self._keys.pop(0))
            self._trades.pop(0)
            self._times.pop(0)


__all__ = ["AggTradeCollector"]
