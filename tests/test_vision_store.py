from __future__ import annotations

import sqlite3
from pathlib import Path

from src.services.vision_store import VisionStore


def _fetch_all(path: Path, query: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(query, params))
    finally:
        conn.close()


def test_insert_agg_trades_idempotent(tmp_path):
    db_path = tmp_path / "vision.sqlite"
    store = VisionStore(db_path)

    sample = [
        {
            "agg_id": 1,
            "ts": 1_700_000_000_000,
            "price": 100.5,
            "qty": 2.0,
            "side": "buy",
        }
    ]

    stats_first = store.insert_agg_trades("BTCUSDT", "2024-01-01", sample)
    stats_second = store.insert_agg_trades("BTCUSDT", "2024-01-01", sample)

    rows = _fetch_all(db_path, "SELECT COUNT(*) AS c FROM agg_trades")
    assert rows[0]["c"] == 1
    assert stats_first.inserted == 1
    assert stats_second.inserted == 0


def test_upsert_klines(tmp_path):
    db_path = tmp_path / "vision.sqlite"
    store = VisionStore(db_path)
    candles = [
        {
            "ts": 1_700_000_000_000,
            "open_time": 1_700_000_000_000,
            "close_time": 1_700_000_059_999,
            "o": 100,
            "h": 110,
            "l": 90,
            "c": 105,
            "v": 12,
            "quote_volume": 1200,
            "trades": 40,
        }
    ]

    stats = store.upsert_klines("BTCUSDT", "1m", "2024-01-01", candles)
    assert stats.inserted == 1

    candles[0]["c"] = 106
    stats_update = store.upsert_klines("BTCUSDT", "1m", "2024-01-01", candles)
    assert stats_update.inserted == 1

    rows = _fetch_all(
        db_path,
        "SELECT close FROM klines WHERE symbol=? AND interval=?",
        ("BTCUSDT", "1m"),
    )
    assert rows[0]["close"] == 106.0


def test_upsert_liquidations_hash(tmp_path):
    db_path = tmp_path / "vision.sqlite"
    store = VisionStore(db_path)
    rows = [
        {
            "ts": 1_700_000_100_000,
            "price": 100.0,
            "qty": 3.0,
            "side": "sell",
            "notional": 300.0,
        }
    ]

    store.upsert_liquidations("BTCUSDT", "2024-01-01", rows)
    store.upsert_liquidations("BTCUSDT", "2024-01-01", rows)

    rows = _fetch_all(db_path, "SELECT COUNT(*) AS c FROM liquidations")
    assert rows[0]["c"] == 1
