"""Local persistence for Binance Vision derived datasets."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from threading import RLock
from time import time
from typing import Dict, Iterable, List, Mapping, Sequence

from .settings import PROJECT_ROOT

DB_DIR = PROJECT_ROOT / "var"
DB_PATH = DB_DIR / "vision_store.sqlite"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS agg_trades (
    symbol TEXT NOT NULL,
    agg_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    side TEXT NOT NULL,
    buyer_maker INTEGER NOT NULL,
    first_id INTEGER,
    last_id INTEGER,
    day TEXT NOT NULL,
    PRIMARY KEY(symbol, agg_id)
);
CREATE TABLE IF NOT EXISTS klines (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open_time INTEGER NOT NULL,
    close_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL,
    trades INTEGER,
    PRIMARY KEY(symbol, interval, ts)
);
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    mark_price REAL,
    PRIMARY KEY(symbol, ts)
);
CREATE TABLE IF NOT EXISTS open_interest (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open_interest REAL NOT NULL,
    notional REAL,
    PRIMARY KEY(symbol, ts)
);
CREATE TABLE IF NOT EXISTS metrics_raw (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open_interest REAL,
    open_interest_value REAL,
    top_trader_sum REAL,
    top_trader_count REAL,
    trader_sum REAL,
    trader_count REAL,
    taker_vol_ratio REAL,
    PRIMARY KEY(symbol, ts)
);
CREATE TABLE IF NOT EXISTS liquidations (
    event_hash TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    price REAL,
    qty REAL,
    side TEXT,
    notional REAL
);
CREATE INDEX IF NOT EXISTS idx_liquidations_symbol_ts ON liquidations(symbol, ts);
CREATE TABLE IF NOT EXISTS depth_snapshots (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    bids_json TEXT,
    asks_json TEXT,
    meta_json TEXT,
    PRIMARY KEY(symbol, ts)
);
CREATE TABLE IF NOT EXISTS exchange_info (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL,
    fetched_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ingestion_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    symbol TEXT,
    interval TEXT,
    day TEXT,
    count INTEGER,
    bytes INTEGER,
    inserted INTEGER,
    ts INTEGER NOT NULL
);
"""

_LOCK = RLock()


def _canonicalise_symbol(symbol: str) -> str:
    clean = (symbol or "").strip().upper()
    if not clean:
        raise ValueError("symbol is required")
    return clean


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _normalise_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hash_liquidation(symbol: str, record: Mapping[str, object]) -> str:
    price = round(float(record.get("price") or 0.0), 8)
    qty = round(float(record.get("qty") or 0.0), 8)
    notional = record.get("notional")
    notional_round = round(float(notional), 8) if notional is not None else 0.0
    ts = int(record.get("ts") or 0)
    side = str(record.get("side") or "").lower()
    payload = f"{symbol}|{ts}|{price}|{qty}|{notional_round}|{side}".encode("utf-8")
    return sha1(payload).hexdigest()


@dataclass(slots=True)
class IngestionStats:
    """Return value describing the persistence outcome for a dataset."""

    dataset: str
    symbol: str | None
    interval: str | None
    day: str | None
    count: int
    inserted: int


class VisionStore:
    """Persist Binance Vision datasets with idempotent semantics."""

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self._path = Path(path)
        self._schema_applied = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        if self._schema_applied:
            return
        with _connect(self._path) as conn:
            conn.executescript(_SCHEMA)
        self._schema_applied = True

    def _execute_many(self, statement: str, rows: Sequence[Sequence[object]]) -> int:
        if not rows:
            return 0
        with _connect(self._path) as conn:
            cursor = conn.executemany(statement, rows)
            return cursor.rowcount

    def _record_metrics(
        self,
        dataset: str,
        *,
        symbol: str | None,
        interval: str | None,
        day: str | None,
        count: int,
        inserted: int,
        bytes_downloaded: int | None = None,
    ) -> None:
        timestamp_ms = int(time() * 1000)
        with _connect(self._path) as conn:
            conn.execute(
                """
                INSERT INTO ingestion_metrics(dataset, symbol, interval, day, count, bytes, inserted, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset,
                    symbol,
                    interval,
                    day,
                    int(count),
                    int(bytes_downloaded or 0),
                    int(inserted),
                    timestamp_ms,
                ),
            )

    # ------------------------------------------------------------------
    # Public fetch APIs
    # ------------------------------------------------------------------
    def fetch_agg_trades(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
        *,
        ascending: bool = True,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT agg_id, ts, price, qty, side, buyer_maker",
            "FROM agg_trades",
            "WHERE symbol = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol)]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts < ?")
            params.append(int(end_ms))
        order_clause = "ASC" if ascending else "DESC"
        query.append(f"ORDER BY ts {order_clause}")
        if limit is not None and limit > 0:
            query.append("LIMIT ?")
            params.append(int(limit))
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        trades: List[Dict[str, object]] = []
        for row in rows:
            trades.append(
                {
                    "agg_id": int(row["agg_id"]),
                    "t": int(row["ts"]),
                    "p": float(row["price"]),
                    "q": float(row["qty"]),
                    "side": (row["side"] or "").lower() or None,
                    "m": bool(row["buyer_maker"]),
                }
            )
        return trades

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> List[List[float | int | None]]:
        self._ensure_schema()
        query = [
            "SELECT ts, open_time, close_time, open, high, low, close, volume, quote_volume, trades",
            "FROM klines",
            "WHERE symbol = ? AND interval = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol), (interval or "").strip().lower()]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts < ?")
            params.append(int(end_ms))
        query.append("ORDER BY ts ASC")
        if limit is not None and limit > 0:
            query.append("LIMIT ?")
            params.append(int(limit))
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        result: List[List[float | int | None]] = []
        for row in rows:
            result.append(
                [
                    int(row["open_time"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    int(row["close_time"]),
                    float(row["quote_volume"]) if row["quote_volume"] is not None else None,
                    int(row["trades"]) if row["trades"] is not None else None,
                ]
            )
        return result

    def fetch_latest_kline(
        self,
        symbol: str,
        interval: str,
    ) -> Dict[str, object] | None:
        self._ensure_schema()
        query = """
            SELECT ts, open_time, close_time, open, high, low, close, volume, quote_volume, trades
            FROM klines
            WHERE symbol = ? AND interval = ?
            ORDER BY ts DESC
            LIMIT 1
        """
        params = (_canonicalise_symbol(symbol), (interval or "").strip().lower())
        with _connect(self._path) as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return {
            "ts": int(row["ts"]),
            "open_time": int(row["open_time"]),
            "close_time": int(row["close_time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quote_volume": float(row["quote_volume"]) if row["quote_volume"] is not None else None,
            "trades": int(row["trades"]) if row["trades"] is not None else None,
        }

    def fetch_funding_rates(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT ts, funding_rate, mark_price",
            "FROM funding_rates",
            "WHERE symbol = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol)]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts < ?")
            params.append(int(end_ms))
        query.append("ORDER BY ts ASC")
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        result: List[Dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "ts": int(row["ts"]),
                    "funding_rate": float(row["funding_rate"]),
                    "mark_price": float(row["mark_price"]) if row["mark_price"] is not None else None,
                }
            )
        return result

    def fetch_open_interest(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT ts, open_interest, notional",
            "FROM open_interest",
            "WHERE symbol = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol)]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts < ?")
            params.append(int(end_ms))
        query.append("ORDER BY ts ASC")
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        result: List[Dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "ts": int(row["ts"]),
                    "open_interest": float(row["open_interest"]),
                    "notional": float(row["notional"]) if row["notional"] is not None else None,
                }
            )
        return result

    def fetch_liquidations(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT ts, price, qty, side, notional",
            "FROM liquidations",
            "WHERE symbol = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol)]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts < ?")
            params.append(int(end_ms))
        query.append("ORDER BY ts ASC")
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        result: List[Dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "ts": int(row["ts"]),
                    "price": float(row["price"]) if row["price"] is not None else None,
                    "qty": float(row["qty"]) if row["qty"] is not None else None,
                    "side": (row["side"] or "").lower() or None,
                    "notional": float(row["notional"]) if row["notional"] is not None else None,
                }
            )
        return result

    def fetch_depth_snapshots(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        *,
        limit: int | None = 1,
        descending: bool = True,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT ts, bids_json, asks_json, meta_json",
            "FROM depth_snapshots",
            "WHERE symbol = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol)]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts <= ?")
            params.append(int(end_ms))
        order = "DESC" if descending else "ASC"
        query.append(f"ORDER BY ts {order}")
        if limit is not None and limit > 0:
            query.append("LIMIT ?")
            params.append(int(limit))
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        snapshots: List[Dict[str, object]] = []
        for row in rows:
            bids = json.loads(row["bids_json"]) if row["bids_json"] else None
            asks = json.loads(row["asks_json"]) if row["asks_json"] else None
            meta = json.loads(row["meta_json"]) if row["meta_json"] else None
            snapshots.append(
                {
                    "ts": int(row["ts"]),
                    "bids": bids,
                    "asks": asks,
                    "meta": meta or {},
                }
            )
        if descending:
            snapshots.reverse()
        return snapshots

    def fetch_ingestion_metrics(
        self,
        dataset: str,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        limit: int | None = 50,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT dataset, symbol, interval, day, count, inserted, bytes, ts",
            "FROM ingestion_metrics",
            "WHERE dataset = ?",
        ]
        params: List[object] = [dataset]
        if symbol is not None:
            query.append("AND symbol = ?")
            params.append(_canonicalise_symbol(symbol))
        if interval is not None:
            query.append("AND interval = ?")
            params.append(interval.strip().lower())
        query.append("ORDER BY ts DESC")
        if limit is not None and limit > 0:
            query.append("LIMIT ?")
            params.append(int(limit))
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        metrics: List[Dict[str, object]] = []
        for row in rows:
            metrics.append(
                {
                    "dataset": row["dataset"],
                    "symbol": row["symbol"],
                    "interval": row["interval"],
                    "day": row["day"],
                    "count": int(row["count"]),
                    "inserted": int(row["inserted"]),
                    "bytes": int(row["bytes"]) if row["bytes"] is not None else 0,
                    "ts": int(row["ts"]),
                }
            )
        return metrics

    # ------------------------------------------------------------------
    # Public insert APIs
    # ------------------------------------------------------------------
    def insert_agg_trades(
        self,
        symbol: str,
        day: str,
        trades: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)

        rows = []
        for trade in trades:
            agg_id = trade.get("agg_id")
            ts = trade.get("ts")
            price = trade.get("price")
            qty = trade.get("qty")
            side = str(trade.get("side") or "").lower()
            if agg_id is None or ts is None or price is None or qty is None:
                continue
            rows.append(
                (
                    symbol_clean,
                    int(agg_id),
                    int(ts),
                    float(price),
                    float(qty),
                    side if side in {"buy", "sell"} else "unknown",
                    1 if trade.get("buyer_maker") else 0,
                    trade.get("first_id"),
                    trade.get("last_id"),
                    day,
                )
            )

        inserted = self._execute_many(
            """
            INSERT OR IGNORE INTO agg_trades
            (symbol, agg_id, ts, price, qty, side, buyer_maker, first_id, last_id, day)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        self._record_metrics(
            "aggTrades",
            symbol=symbol_clean,
            interval=None,
            day=day,
            count=len(rows),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("aggTrades", symbol_clean, None, day, len(rows), inserted)

    def upsert_klines(
        self,
        symbol: str,
        interval: str,
        day: str,
        candles: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)
        interval_clean = (interval or "").strip().lower()
        rows = []
        for candle in candles:
            ts = candle.get("ts")
            open_time = candle.get("open_time", ts)
            close_time = candle.get("close_time", ts)
            if ts is None:
                continue
            try:
                row = (
                    symbol_clean,
                    interval_clean,
                    int(ts),
                    int(open_time),
                    int(close_time),
                    float(candle.get("o")),
                    float(candle.get("h")),
                    float(candle.get("l")),
                    float(candle.get("c")),
                    float(candle.get("v")),
                    _normalise_float(candle.get("quote_volume")),
                    int(candle.get("trades") or 0),
                )
            except (TypeError, ValueError):
                continue
            rows.append(row)

        inserted = self._execute_many(
            """
            INSERT INTO klines(symbol, interval, ts, open_time, close_time, open, high, low, close, volume, quote_volume, trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, ts) DO UPDATE SET
                open_time=excluded.open_time,
                close_time=excluded.close_time,
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                quote_volume=excluded.quote_volume,
                trades=excluded.trades
            """,
            rows,
        )

        self._record_metrics(
            "klines",
            symbol=symbol_clean,
            interval=interval_clean,
            day=day,
            count=len(rows),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("klines", symbol_clean, interval_clean, day, len(rows), inserted)

    def upsert_funding_rates(
        self,
        symbol: str,
        day: str,
        rows: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)
        payload = []
        for row in rows:
            ts = row.get("ts")
            rate = row.get("funding_rate")
            if ts is None or rate is None:
                continue
            payload.append(
                (symbol_clean, int(ts), float(rate), _normalise_float(row.get("mark_price")))
            )

        inserted = self._execute_many(
            """
            INSERT INTO funding_rates(symbol, ts, funding_rate, mark_price)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                funding_rate=excluded.funding_rate,
                mark_price=excluded.mark_price
            """,
            payload,
        )

        self._record_metrics(
            "fundingRate",
            symbol=symbol_clean,
            interval=None,
            day=day,
            count=len(payload),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("fundingRate", symbol_clean, None, day, len(payload), inserted)

    def upsert_open_interest(
        self,
        symbol: str,
        day: str,
        rows: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)
        payload = []
        for row in rows:
            ts = row.get("ts")
            value = row.get("open_interest")
            if ts is None or value is None:
                continue
            payload.append(
                (symbol_clean, int(ts), float(value), _normalise_float(row.get("notional")))
            )

        inserted = self._execute_many(
            """
            INSERT INTO open_interest(symbol, ts, open_interest, notional)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                open_interest=excluded.open_interest,
                notional=excluded.notional
            """,
            payload,
        )

        self._record_metrics(
            "openInterest",
            symbol=symbol_clean,
            interval=None,
            day=day,
            count=len(payload),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("openInterest", symbol_clean, None, day, len(payload), inserted)

    def upsert_metrics(
        self,
        symbol: str,
        day: str,
        rows: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)
        payload = []
        for row in rows:
            ts = row.get("ts")
            open_interest = row.get("open_interest")
            if ts is None or open_interest is None:
                continue
            payload.append(
                (symbol_clean, int(ts), float(open_interest), _normalise_float(row.get("notional")))
            )

        inserted = self._execute_many(
            """
            INSERT INTO open_interest(symbol, ts, open_interest, notional)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                open_interest=excluded.open_interest,
                notional=excluded.notional
            """,
            payload,
        )

        metric_rows = []
        for row in rows:
            ts = row.get("ts")
            if ts is None:
                continue
            metric_rows.append(
                (
                    symbol_clean,
                    int(ts),
                    _normalise_float(row.get("open_interest")),
                    _normalise_float(row.get("open_interest_value")),
                    _normalise_float(row.get("top_trader_sum")),
                    _normalise_float(row.get("top_trader_count")),
                    _normalise_float(row.get("trader_sum")),
                    _normalise_float(row.get("trader_count")),
                    _normalise_float(row.get("taker_vol_ratio")),
                )
            )

        if metric_rows:
            self._execute_many(
                """
                INSERT INTO metrics_raw(symbol, ts, open_interest, open_interest_value, top_trader_sum, top_trader_count, trader_sum, trader_count, taker_vol_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, ts) DO UPDATE SET
                    open_interest=excluded.open_interest,
                    open_interest_value=excluded.open_interest_value,
                    top_trader_sum=excluded.top_trader_sum,
                    top_trader_count=excluded.top_trader_count,
                    trader_sum=excluded.trader_sum,
                    trader_count=excluded.trader_count,
                    taker_vol_ratio=excluded.taker_vol_ratio
                """,
                metric_rows,
            )

        self._record_metrics(
            "metrics",
            symbol=symbol_clean,
            interval=None,
            day=day,
            count=len(payload),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("metrics", symbol_clean, None, day, len(payload), inserted)

    def fetch_metrics(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> List[Dict[str, object]]:
        self._ensure_schema()
        query = [
            "SELECT ts, open_interest, open_interest_value, top_trader_sum, top_trader_count, trader_sum, trader_count, taker_vol_ratio",
            "FROM metrics_raw",
            "WHERE symbol = ?",
        ]
        params: List[object] = [_canonicalise_symbol(symbol)]
        if start_ms is not None:
            query.append("AND ts >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            query.append("AND ts < ?")
            params.append(int(end_ms))
        query.append("ORDER BY ts ASC")
        sql = " ".join(query)
        with _connect(self._path) as conn:
            rows = conn.execute(sql, params).fetchall()

        result: List[Dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "ts": int(row["ts"]),
                    "open_interest": _normalise_float(row["open_interest"]),
                    "open_interest_value": _normalise_float(row["open_interest_value"]),
                    "top_trader_sum": _normalise_float(row["top_trader_sum"]),
                    "top_trader_count": _normalise_float(row["top_trader_count"]),
                    "trader_sum": _normalise_float(row["trader_sum"]),
                    "trader_count": _normalise_float(row["trader_count"]),
                    "taker_vol_ratio": _normalise_float(row["taker_vol_ratio"]),
                }
            )
        return result

    def upsert_liquidations(
        self,
        symbol: str,
        day: str,
        rows: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)
        payload = []
        for row in rows:
            ts = row.get("ts")
            if ts is None:
                continue
            hash_key = _hash_liquidation(symbol_clean, row)
            payload.append(
                (
                    hash_key,
                    symbol_clean,
                    int(ts),
                    _normalise_float(row.get("price")),
                    _normalise_float(row.get("qty")),
                    (row.get("side") or None),
                    _normalise_float(row.get("notional")),
                )
            )

        inserted = self._execute_many(
            """
            INSERT OR IGNORE INTO liquidations(event_hash, symbol, ts, price, qty, side, notional)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

        self._record_metrics(
            "liquidationOrders",
            symbol=symbol_clean,
            interval=None,
            day=day,
            count=len(payload),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("liquidationOrders", symbol_clean, None, day, len(payload), inserted)

    def upsert_depth_snapshots(
        self,
        symbol: str,
        day: str,
        rows: Iterable[Mapping[str, object]],
        *,
        bytes_downloaded: int | None = None,
    ) -> IngestionStats:
        self._ensure_schema()
        symbol_clean = _canonicalise_symbol(symbol)
        payload = []
        for row in rows:
            ts = row.get("ts")
            if ts is None:
                continue
            bids = row.get("bids")
            asks = row.get("asks")
            meta = {key: row[key] for key in row.keys() if key not in {"bids", "asks"}}
            payload.append(
                (
                    symbol_clean,
                    int(ts),
                    json.dumps(bids, ensure_ascii=False) if bids is not None else None,
                    json.dumps(asks, ensure_ascii=False) if asks is not None else None,
                    json.dumps(meta, ensure_ascii=False) if meta else None,
                )
            )

        inserted = self._execute_many(
            """
            INSERT INTO depth_snapshots(symbol, ts, bids_json, asks_json, meta_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                bids_json=excluded.bids_json,
                asks_json=excluded.asks_json,
                meta_json=excluded.meta_json
            """,
            payload,
        )

        self._record_metrics(
            "bookDepth",
            symbol=symbol_clean,
            interval=None,
            day=day,
            count=len(payload),
            inserted=inserted,
            bytes_downloaded=bytes_downloaded,
        )
        return IngestionStats("bookDepth", symbol_clean, None, day, len(payload), inserted)

    def store_exchange_info(self, payload: Mapping[str, object]) -> None:
        self._ensure_schema()
        with _connect(self._path) as conn:
            conn.execute(
                """
                INSERT INTO exchange_info(id, payload, fetched_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, fetched_at=excluded.fetched_at
                """,
                (json.dumps(payload, ensure_ascii=False), int(time() * 1000)),
            )


_DEFAULT_STORE: VisionStore | None = None


def get_store() -> VisionStore:
    """Return the shared VisionStore singleton."""

    global _DEFAULT_STORE
    with _LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = VisionStore()
        return _DEFAULT_STORE


__all__ = [
    "IngestionStats",
    "VisionStore",
    "get_store",
]
