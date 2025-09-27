from datetime import datetime, timedelta, timezone
from typing import Dict, List

from src.services import inspection
from src.services.inspection import build_htf_section


UTC = timezone.utc


def _generate_minute_candles(
    base: datetime,
    count: int,
    *,
    skip: set[int] | None = None,
    price_start: float = 100.0,
    price_step: float = 0.1,
    volume_base: float = 1.0,
) -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    for index in range(count):
        ts = int((base + timedelta(minutes=index)).timestamp() * 1000)
        if skip and ts in skip:
            continue
        open_price = price_start + index * price_step
        close_price = open_price + 0.25
        high_price = close_price + 0.1
        low_price = open_price - 0.1
        volume = volume_base + index * 0.01
        candles.append(
            {
                "t": ts,
                "o": round(open_price, 5),
                "h": round(high_price, 5),
                "l": round(low_price, 5),
                "c": round(close_price, 5),
                "v": round(volume, 5),
            }
        )
    return candles


def test_htf_downloads_missing_minutes() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    total_minutes = 24 * 60
    gap_start = 12 * 60
    gap_length = 15
    gap_times = {
        int((base + timedelta(minutes=gap_start + offset)).timestamp() * 1000)
        for offset in range(gap_length)
    }

    candles = _generate_minute_candles(base, total_minutes, skip=gap_times)
    frames = {"1m": {"tf": "1m", "candles": candles}}
    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]}

    calls: List[tuple[str, int, int, int]] = []

    def fetcher(symbol: str, start_ms: int, end_ms: int, limit: int):
        calls.append((symbol, start_ms, end_ms, limit))
        assert limit <= 1000
        rows = []
        cursor = start_ms
        while cursor <= end_ms:
            rows.append([cursor, 200.0, 201.0, 199.0, 200.5, 2.0])
            cursor += 60_000
        return rows

    htf_section, dq = build_htf_section("BTCUSDT", frames, selection, fetcher=fetcher)

    assert dq["minute_missing_before"] == gap_length
    assert dq["minute_missing_after"] == 0
    assert dq["downloaded_1m"] == gap_length
    assert dq["timeframes"]["15m"]["missing_after"] == 0
    assert len(htf_section["candles"]["15m"]) == 96
    assert calls


def test_htf_aligns_start_boundary() -> None:
    base = datetime(2024, 2, 1, tzinfo=UTC)
    total_minutes = 24 * 60
    leading_missing = 7
    skip_times = {
        int((base + timedelta(minutes=offset)).timestamp() * 1000)
        for offset in range(leading_missing)
    }

    candles = _generate_minute_candles(base, total_minutes, skip=skip_times)
    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]}
    frames = {"1m": {"tf": "1m", "candles": candles}}

    def fetcher(symbol: str, start_ms: int, end_ms: int, limit: int):
        rows = []
        cursor = start_ms
        while cursor <= end_ms:
            rows.append([cursor, 150.0, 151.0, 149.0, 150.5, 3.0])
            cursor += 60_000
        return rows

    htf_section, dq = build_htf_section("BTCUSDT", frames, selection, fetcher=fetcher)

    first_15m = htf_section["candles"]["15m"][0]
    assert first_15m["t"] == int(base.timestamp() * 1000)
    assert htf_section["window"]["start_ms"] == int(base.timestamp() * 1000)
    assert dq["minute_missing_before"] == leading_missing
    assert dq["minute_missing_after"] == 0
    assert dq["timeframes"]["15m"]["missing_before"] == 1
    assert dq["timeframes"]["15m"]["missing_after"] == 0


def test_htf_excludes_unclosed_candle(monkeypatch) -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    total_minutes = 24 * 60
    missing_tail = {int((base + timedelta(minutes=total_minutes - offset)).timestamp() * 1000) for offset in range(1, 11)}
    candles = _generate_minute_candles(base, total_minutes, skip=missing_tail)
    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]}
    frames = {"1m": {"tf": "1m", "candles": candles}}

    original_tfs = inspection.HTF_TIMEFRAMES
    monkeypatch.setattr(inspection, "HTF_TIMEFRAMES", ("15m",))
    try:
        htf_section, dq = build_htf_section("BTCUSDT", frames, selection, fetcher=lambda *args, **kwargs: [])
    finally:
        monkeypatch.setattr(inspection, "HTF_TIMEFRAMES", original_tfs)

    candles_15m = htf_section["candles"]["15m"]
    assert candles_15m
    last_candle = candles_15m[-1]
    expected_last = base + timedelta(hours=23, minutes=30)
    assert last_candle["t"] == int(expected_last.timestamp() * 1000)
    assert dq["timeframes"]["15m"]["missing_after"] >= 0


def test_build_inspection_payload_includes_htf(monkeypatch) -> None:
    base = datetime(2024, 4, 1, tzinfo=UTC)
    candles = _generate_minute_candles(base, 24 * 60)
    frames = {"1m": {"tf": "1m", "candles": candles}}
    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]}

    monkeypatch.setattr(inspection, "_DEFAULT_MINUTE_FETCHER", lambda *args, **kwargs: [])
    monkeypatch.setattr(inspection, "build_profile_package", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(inspection, "detect_zones", lambda *args, **kwargs: {"symbol": "BTCUSDT", "zones": {"fvg": [], "ob": [], "inducement": [], "cisd": []}})
    monkeypatch.setattr(
        inspection,
        "resolve_profile_config",
        lambda symbol, meta: {"preset": None, "target_tf_key": "1m", "preset_payload": None, "preset_required": False},
    )

    snapshot = {
        "id": "snap-1",
        "symbol": "BTCUSDT",
        "frames": frames,
        "selection": selection,
        "agg_trades": {"symbol": "BTCUSDT", "agg": []},
        "meta": {"source": {"kind": "test"}},
    }

    payload = inspection.build_inspection_payload(snapshot)

    htf_payload = payload["DATA"]["htf"]
    dq_htf = payload["DATA"]["meta"]["data_quality_htf"]

    assert set(htf_payload["candles"]).issuperset({"15m", "1h", "4h", "1d"})
    assert dq_htf["minute_missing_before"] == 0
    assert dq_htf["minute_missing_after"] == 0
    assert all(isinstance(series, list) for series in htf_payload["candles"].values())
