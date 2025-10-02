from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Sequence

import pytest

import src.services.check_all_datas as check_all_datas

UTC = timezone.utc


@pytest.fixture()
def stub_check_all_dependencies(monkeypatch):
    def fake_profile_config(symbol: str, meta: Dict[str, Any] | None) -> Dict[str, Any]:
        return {"tick_size": 0.5}

    def fake_profile_package(*args, **kwargs):
        return ([], [], [])

    def fake_detect_zones(*args, **kwargs):
        return {
            "zones": {
                "fvg": [],
                "fvl": [],
                "ob": [],
                "mb": [],
                "bb": [],
                "rb": [],
                "pb": [],
                "sr": [],
                "profile_levels": [],
            },
            "meta": {},
        }

    def fake_liquidity(*args, **kwargs):
        return {"eqh": [], "eql": [], "diagnostics": {"config": {}}}

    def fake_multi_tf(candles: Sequence[Dict[str, Any]], *, symbol: str | None = None):
        frame = {"candles": [dict(c) for c in candles]}
        return {
            "1m": frame,
            "3m": frame,
            "5m": frame,
            "15m": frame,
            "1h": frame,
            "4h": frame,
            "1d": frame,
        }

    def fake_aggregate(_candles: Sequence[Dict[str, Any]]):
        return []

    def fake_smc(*args, **kwargs):
        return ([], {})

    monkeypatch.setattr(check_all_datas, "resolve_profile_config", fake_profile_config)
    monkeypatch.setattr(check_all_datas, "build_profile_package", fake_profile_package)
    monkeypatch.setattr(check_all_datas, "detect_zones", fake_detect_zones)
    monkeypatch.setattr(check_all_datas, "build_liquidity_snapshot", fake_liquidity)
    monkeypatch.setattr(check_all_datas, "resolve_liquidity_tick_size", lambda *args, **kwargs: (0.5, "stub"))
    monkeypatch.setattr(check_all_datas, "build_multi_timeframe_ohlcv", fake_multi_tf)
    monkeypatch.setattr(check_all_datas, "aggregate_1m_to_1h", fake_aggregate)
    monkeypatch.setattr(check_all_datas, "detect_smc_blocks", fake_smc)
    monkeypatch.setattr(check_all_datas, "_build_expected_times", lambda *args, **kwargs: [])
    monkeypatch.setattr(check_all_datas, "_summarise_missing_times", lambda *args, **kwargs: [])
    yield


def _snapshot_with_candles(candles: Sequence[Dict[str, Any]], entry_price: float) -> Dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "tf": "1m",
        "frames": {"1m": {"tf": "1m", "candles": candles}},
        "selection": {"start": candles[0]["t"], "end": candles[-1]["t"]},
        "analysis": {"trade": {"entry_price": entry_price}},
    }


def test_last_price_prefers_minute_candle_over_analysis(stub_check_all_dependencies) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    candles = []
    for offset, close in enumerate((110.0, 116.0)):
        moment = base + timedelta(minutes=offset)
        candles.append(
            {
                "t": int(moment.timestamp() * 1000),
                "o": 100.0 + offset,
                "h": 120.0 + offset,
                "l": 90.0 + offset,
                "c": close,
                "v": 5.0,
            }
        )

    snapshot = _snapshot_with_candles(candles, entry_price=101.0)
    now_dt = base + timedelta(minutes=2)

    result = check_all_datas.build_check_all_datas(snapshot, now_utc=now_dt)

    assert result is not None
    meta = result["meta"]
    assert meta["last_price"] == pytest.approx(116.0)
    assert meta["last_tf"] == "1m"
    assert meta.get("insufficient_reason") is None
    assert meta["last_price_source"] == "ohlcv"
    assert meta["stale"] is False


def test_snapshot_staleness_marks_meta(stub_check_all_dependencies) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    stale_time = base
    candles = [
        {
            "t": int(stale_time.timestamp() * 1000),
            "o": 100.0,
            "h": 110.0,
            "l": 95.0,
            "c": 105.0,
            "v": 3.0,
        }
    ]
    snapshot = _snapshot_with_candles(candles, entry_price=99.0)
    now_dt = stale_time + timedelta(minutes=10)

    result = check_all_datas.build_check_all_datas(snapshot, now_utc=now_dt)

    assert result is not None
    meta = result["meta"]
    assert meta["last_price"] == pytest.approx(105.0)
    assert meta["snapshot_age_sec"] >= 600
    reason = meta.get("insufficient_reason")
    assert isinstance(reason, str) and reason.startswith("stale_snapshot_")
    assert meta["stale"] is True


def test_last_price_updates_when_minute_candles_change(stub_check_all_dependencies) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    first_candles = [
        {
            "t": int((base + timedelta(minutes=0)).timestamp() * 1000),
            "o": 100.0,
            "h": 110.0,
            "l": 95.0,
            "c": 105.0,
            "v": 3.0,
        },
        {
            "t": int((base + timedelta(minutes=1)).timestamp() * 1000),
            "o": 105.0,
            "h": 112.0,
            "l": 100.0,
            "c": 108.0,
            "v": 3.5,
        },
    ]
    snapshot = _snapshot_with_candles(first_candles, entry_price=100.0)
    now_dt = base + timedelta(minutes=2)

    initial = check_all_datas.build_check_all_datas(snapshot, now_utc=now_dt)
    assert initial["meta"]["last_price"] == pytest.approx(108.0)

    updated_candles = first_candles[:-1] + [
        {
            "t": int((base + timedelta(minutes=2)).timestamp() * 1000),
            "o": 108.0,
            "h": 118.0,
            "l": 105.0,
            "c": 123.0,
            "v": 4.0,
        }
    ]
    snapshot["frames"]["1m"]["candles"] = updated_candles
    snapshot["selection"]["end"] = updated_candles[-1]["t"]

    refreshed = check_all_datas.build_check_all_datas(snapshot, now_utc=base + timedelta(minutes=3))
    assert refreshed["meta"]["last_price"] == pytest.approx(123.0)
    assert refreshed["meta"]["last_tf"] == "1m"
    assert refreshed["meta"]["last_price_source"] == "ohlcv"


def test_stream_price_overrides_candle(stub_check_all_dependencies) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    candles = [
        {
            "t": int((base + timedelta(minutes=0)).timestamp() * 1000),
            "o": 100.0,
            "h": 110.0,
            "l": 95.0,
            "c": 105.0,
            "v": 3.0,
        },
    ]
    snapshot = _snapshot_with_candles(candles, entry_price=101.0)
    stream_ts = int((base + timedelta(minutes=1, seconds=2)).timestamp() * 1000)
    snapshot["stream"] = {"price": 107.5, "ts": stream_ts}

    result = check_all_datas.build_check_all_datas(snapshot, now_utc=base + timedelta(minutes=1, seconds=3))

    meta = result["meta"]
    assert meta["last_price"] == pytest.approx(107.5)
    assert meta["last_tf"] == "stream"
    assert meta["last_price_source"] == "stream"
    assert meta.get("stale") is False
    assert meta.get("insufficient_reason") is None


def test_stream_vs_ohlcv_mismatch_flag(stub_check_all_dependencies) -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    candle_ts = int(base.timestamp() * 1000)
    candles = [
        {
            "t": candle_ts,
            "o": 100.0,
            "h": 110.0,
            "l": 95.0,
            "c": 100.0,
            "v": 3.0,
        }
    ]
    snapshot = _snapshot_with_candles(candles, entry_price=100.0)
    stream_ts = candle_ts + 60_000 + 4_000
    snapshot["stream"] = {"price": 101.5, "ts": stream_ts}

    result = check_all_datas.build_check_all_datas(
        snapshot,
        now_utc=base + timedelta(minutes=1, seconds=10),
    )

    meta = result["meta"]
    assert meta["last_price_source"] == "stream"
    assert meta["last_price"] == pytest.approx(101.5)
    assert meta.get("stream_vs_ohlcv_mismatch") is True
    assert meta.get("insufficient_reason") is None
