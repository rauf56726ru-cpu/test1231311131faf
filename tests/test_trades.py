from __future__ import annotations

import itertools

import pytest

from src.services import AggTradeCollector


def make_trade(t: int, p: float | str, q: float | str, side: str) -> dict[str, object]:
    return {"t": t, "p": p, "q": q, "side": side}


def test_side_preserved() -> None:
    collector = AggTradeCollector("btcusdt", min_age_ms=0)
    collector.ingest(make_trade(1_000, "100.0", "0.5", "BUY"))
    collector.ingest(make_trade(2_000, "101.5", "0.25", "sell"))

    snapshot = collector.get_snapshot()
    sides = [item["side"] for item in snapshot["agg"]]
    assert sides == ["buy", "sell"]


def test_dedup_by_time_price_quantity() -> None:
    collector = AggTradeCollector("ethusdt", min_age_ms=0)
    trade = make_trade(1_500, "10.0", "1.0", "buy")
    collector.ingest(trade)
    collector.ingest({**trade, "side": "sell"})  # duplicate t/p/q should be ignored

    snapshot = collector.get_snapshot()
    assert len(snapshot["agg"]) == 1


def test_rotation_by_time_window() -> None:
    collector = AggTradeCollector("bnbusdt", max_age_ms=2_000, min_age_ms=0)
    collector.ingest(make_trade(0, "1", "1", "buy"))
    collector.ingest(make_trade(3_500, "2", "1", "sell"))

    snapshot = collector.get_snapshot()
    times = [item["t"] for item in snapshot["agg"]]
    assert times == [3_500]


def test_rest_recovery_and_no_duplicates() -> None:
    collector = AggTradeCollector("adausdt", min_age_ms=0)
    collector.ingest(make_trade(1_000, "1", "1", "buy"))

    def fetcher(start_ms: int | None = None, end_ms: int | None = None) -> list[dict[str, object]]:
        assert start_ms == 0
        assert end_ms == 2_000
        return [
            make_trade(1_000, "1", "1", "buy"),  # duplicate should be ignored
            make_trade(1_200, "1.1", "0.5", "sell"),
            make_trade(1_800, "1.2", "0.25", "buy"),
        ]

    inserted = collector.backfill_from_rest(fetcher, start_ms=0, end_ms=2_000)
    assert inserted == 2

    snapshot = collector.get_snapshot()
    times = [item["t"] for item in snapshot["agg"]]
    assert times == [1_000, 1_200, 1_800]


@pytest.mark.parametrize(
    "server_time,received_at,trade_time,expected",
    [
        (1_000_000, 900_000, 1_000, 101_000),
        (500_000, 500_000, 2_000, 2_000),
    ],
)
def test_clock_offset_applied(
    server_time: int, received_at: int, trade_time: int, expected: int
) -> None:
    collector = AggTradeCollector("xrpusdt", min_age_ms=0)
    collector.sync_clock(server_time, received_at_ms=received_at)
    collector.ingest(make_trade(trade_time, "1", "1", "buy"))

    snapshot = collector.get_snapshot()
    assert snapshot["agg"][0]["t"] == expected
