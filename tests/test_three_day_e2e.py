from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, Sequence

import pytest

from src.api import app as app_module
from src.services import candles_repository, http_client, summary_collector

UTC = timezone.utc


@pytest.fixture
def anyio_backend():
    yield "asyncio"


@dataclass
class _StoredCandle:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class InMemoryRepository:
    """Minimal in-memory candle repository for strict 3-day scenarios."""

    def __init__(self) -> None:
        self._storage: Dict[tuple[str, str], Dict[int, _StoredCandle]] = defaultdict(dict)
        self.fetch_calls: List[tuple[str, str, int, int]] = []

    def _key(self, symbol: str, interval: str) -> tuple[str, str]:
        return (symbol.upper(), interval.lower())

    def _normalise_rows(
        self, candles: Iterable[Mapping[str, object]]
    ) -> List[_StoredCandle]:
        normalised: List[_StoredCandle] = []
        for row in candles:
            if not isinstance(row, Mapping):
                continue
            ts_raw = row.get("t") or row.get("time") or row.get("timestamp") or row.get("ts")
            try:
                ts = int(ts_raw) if ts_raw is not None else None
            except (TypeError, ValueError):
                ts = None
            if ts is None:
                continue
            try:
                open_value = float(row.get("o", row.get("open", 0.0)))
                high_value = float(row.get("h", row.get("high", 0.0)))
                low_value = float(row.get("l", row.get("low", 0.0)))
                close_value = float(row.get("c", row.get("close", 0.0)))
                volume_value = float(row.get("v", row.get("volume", 0.0)))
            except (TypeError, ValueError):
                continue
            normalised.append(
                _StoredCandle(
                    t=ts,
                    o=open_value,
                    h=high_value,
                    l=low_value,
                    c=close_value,
                    v=volume_value,
                )
            )
        return normalised

    def seed(self, symbol: str, interval: str, candles: Sequence[Mapping[str, object]]) -> None:
        bucket = self._storage[self._key(symbol, interval)]
        for candle in self._normalise_rows(candles):
            bucket[candle.t] = candle

    def fetch_candles(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
        key = self._key(symbol, interval)
        bucket = self._storage.get(key, {})
        self.fetch_calls.append((symbol.upper(), interval.lower(), start_ms, end_ms))
        rows = [
            candle
            for ts, candle in sorted(bucket.items())
            if start_ms <= ts <= end_ms
        ]
        return [
            {
                "t": row.t,
                "o": row.o,
                "h": row.h,
                "l": row.l,
                "c": row.c,
                "v": row.v,
            }
            for row in rows
        ]

    def fetch_open_times(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[int]:
        key = self._key(symbol, interval)
        bucket = self._storage.get(key, {})
        return [
            ts
            for ts in sorted(bucket)
            if start_ms <= ts <= end_ms
        ]

    def upsert_candles(
        self,
        symbol: str,
        interval: str,
        candles: Sequence[Mapping[str, object]] | Sequence[Sequence[object]],
        *,
        stage: str,
    ) -> candles_repository.UpsertStats:
        rows: List[Mapping[str, object]] = []
        for candle in candles:
            if isinstance(candle, Mapping):
                rows.append(candle)
            elif isinstance(candle, Sequence):
                try:
                    ts, o, h, l, c, v = candle[:6]
                except (TypeError, ValueError):
                    continue
                rows.append({"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v})
        self.seed(symbol, interval, rows)
        return candles_repository.UpsertStats(written=len(rows), dropped_ts=0, dropped_ohlc=0)

    # Gap progress helpers are no-ops for the in-memory store used in tests.
    def load_gap_progress(self, symbol: str, interval: str, gap_start_ms: int) -> int | None:  # pragma: no cover - unused path
        return None

    def save_gap_progress(self, symbol: str, interval: str, gap_start_ms: int, last_open_ms: int) -> None:  # pragma: no cover - unused path
        return None


@pytest.mark.anyio("asyncio")
async def test_three_day_pipeline_runs_offline(monkeypatch, caplog):
    symbol = "TESTUSDT"
    end_dt = datetime(2024, 1, 4, 0, 0, tzinfo=UTC)
    total_minutes = 72 * 60 + 2
    start_dt = end_dt - timedelta(minutes=total_minutes)
    minute_candles: List[Dict[str, object]] = []
    for index in range(total_minutes):
        moment = start_dt + timedelta(minutes=index)
        ts = int(moment.timestamp() * 1000)
        base_price = 100.0 + index * 0.01
        minute_candles.append(
            {
                "t": ts,
                "o": round(base_price, 3),
                "h": round(base_price + 1.0, 3),
                "l": round(base_price - 1.0, 3),
                "c": round(base_price + 0.25, 3),
                "v": round(5.0 + index * 0.01, 3),
            }
        )

    repo = InMemoryRepository()
    monkeypatch.setattr(candles_repository, "_DEFAULT_REPOSITORY", repo, raising=False)
    monkeypatch.setattr(candles_repository, "get_repository", lambda: repo)

    async def forbidden_request(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("network request attempted during strict pipeline")

    monkeypatch.setattr(http_client, "request", forbidden_request)

    current_run = {"value": 0}
    summary_calls = {"count": 0}

    async def fake_collect_recent_summary(symbol_arg: str, *_, **kwargs):
        assert symbol_arg == symbol
        summary_calls["count"] += 1
        repo.seed(symbol, "1m", minute_candles)
        start_ms = kwargs.get("start_ms")
        end_ms = kwargs.get("end_ms")
        if start_ms is None or end_ms is None:
            start_ms = minute_candles[0]["t"]
            end_ms = minute_candles[-1]["t"]
        sleep_duration = 0.03 if current_run["value"] == 0 else 0.005
        await asyncio.sleep(sleep_duration)
        interval_summary = summary_collector.IntervalSummary(
            gaps_total=0,
            gaps_filled=len(minute_candles),
            candles_written=len(minute_candles),
            dropped_candles=0,
            requests=1,
            remaining_gaps=[],
            expected_candles=len(minute_candles),
            coverage_pct=100.0,
        )
        return summary_collector.CollectionSummary(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            intervals={"1m": interval_summary},
            requests=1,
            candles_written=len(minute_candles),
            dropped_candles=0,
        )

    monkeypatch.setattr(summary_collector, "collect_recent_summary", fake_collect_recent_summary)
    monkeypatch.setattr(app_module, "collect_recent_summary", fake_collect_recent_summary)

    import src.services.ohlc as ohlc_module
    import src.services.check_all_datas as check_all_module

    async def ohlc_forbidden(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("external OHLCV fetch attempted in strict pipeline")

    def ohlc_forbidden_sync(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("external OHLCV fetch attempted in strict pipeline")

    monkeypatch.setattr(ohlc_module, "fetch_ohlcv", ohlc_forbidden)
    monkeypatch.setattr(ohlc_module, "fetch_ohlcv_sync", ohlc_forbidden_sync)
    monkeypatch.setattr(check_all_module, "fetch_ohlcv", ohlc_forbidden)
    monkeypatch.setattr(check_all_module, "fetch_ohlcv_sync", ohlc_forbidden_sync)

    agg_trades: List[Dict[str, object]] = []
    for candle in minute_candles[-120:]:
        trade_ts = int(candle["t"]) + 30_000
        side = "buy" if len(agg_trades) % 2 == 0 else "sell"
        agg_trades.append({"t": trade_ts, "q": 1.5, "side": side})

    snapshot = {
        "symbol": symbol,
        "tf": "1m",
        "candles": minute_candles[-180:],
        "selection": {"start": minute_candles[0]["t"], "end": minute_candles[-1]["t"]},
        "agg_trades": {"symbol": symbol, "agg": agg_trades},
        "frames": {"1m": {"tf": "1m", "candles": minute_candles[-360:]}},
        "meta": {"context": {}},
    }

    now_override = datetime.fromtimestamp(minute_candles[-1]["t"] / 1000, tz=UTC)

    caplog.set_level(logging.INFO, logger="src.services.tracing")

    current_run["value"] = 0
    payload_one, summary_payload_one, trace_one = await app_module._run_summary_workflow(
        snapshot,
        days=3,
        window_hours=72,
        now_override=now_override,
        branch_log={},
    )
    compact_one = app_module._prepare_summary_payload(payload_one, trace=trace_one)

    current_run["value"] = 1
    payload_two, summary_payload_two, trace_two = await app_module._run_summary_workflow(
        snapshot,
        days=3,
        window_hours=72,
        now_override=now_override,
        branch_log={},
    )
    compact_two = app_module._prepare_summary_payload(payload_two, trace=trace_two)

    assert summary_calls["count"] >= 4, "collect_recent_summary should handle both seeding stages"
    assert repo.fetch_calls, "repository should serve minute data without network"
    assert payload_one["status"] == "ok"
    assert payload_two["status"] == "ok"
    for timing_block in (payload_one["timing"], payload_two["timing"]):
        for key in ("fetch_ms", "db_ms", "compute_ms", "serialize_ms", "size_bytes"):
            assert key in timing_block
            assert timing_block[key] >= 0

    fetch_one = payload_one["timing"]["fetch_ms"]
    fetch_two = payload_two["timing"]["fetch_ms"]
    assert fetch_one > fetch_two
    assert fetch_one / max(fetch_two, 1.0) >= 1.5

    encoded_compact = json.dumps(compact_one, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert len(encoded_compact) < 4 * 1024 * 1024

    assert compact_one["status"] == "ok"
    assert compact_two["status"] in {"ok", "insufficient_data"}
    assert len(compact_one["ohlcv"]["1m_rollups"]) <= 180
    assert set(compact_one["ohlcv"]) >= {"15m", "1h"}

    per_bar = compact_one["orderflow"]["per_bar"]
    assert set(per_bar) <= {"15m", "1h"}
    for series in per_bar.values():
        assert len(series) <= 120
        for row in series:
            assert row["t"] >= minute_candles[-1]["t"] - 120 * 60_000
            assert "delta" in row and "cvd" in row

    delta_metrics = compact_one["orderflow"]["delta_cvd_compact"]
    assert set(delta_metrics) <= {"15m", "1h"}
    for metrics in delta_metrics.values():
        assert metrics["bars"] <= 120

    events: List[Mapping[str, object]] = []
    for record in caplog.records:
        if record.name.startswith("src.services.tracing"):
            try:
                events.append(json.loads(record.message))
            except json.JSONDecodeError:  # pragma: no cover - unexpected format
                continue
        assert "body" not in record.message
        assert "attempt" not in record.message
        assert "429" not in record.message

    pipeline_starts = [event for event in events if event.get("event") == "pipeline.start" and event.get("scope") == "pipeline"]
    pipeline_done = [event for event in events if event.get("event") == "pipeline.done" and event.get("scope") == "pipeline"]
    assert len(pipeline_starts) >= 2
    assert len(pipeline_done) >= 2

    publish_events = [event for event in events if event.get("event") == "output.publish"]
    assert publish_events, "output.publish events should be present"

    assert summary_payload_one is not None
    if summary_payload_two is not None:
        assert summary_payload_two.get("coverage")

    assert compact_one["timing"]["json_bytes"] < 4 * 1024 * 1024
