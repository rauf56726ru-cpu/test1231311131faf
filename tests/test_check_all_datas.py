"""Tests for the check-all-datas aggregation endpoint and service."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.services.check_all_datas import build_market_overview
from src.services.inspection import register_snapshot


client = TestClient(app)


def _build_sample_candles() -> list[dict[str, float]]:
    candles: list[dict[str, float]] = []
    base_day = datetime(2024, 1, 10, tzinfo=timezone.utc)

    for day_offset in range(3):
        day_base = base_day + timedelta(days=day_offset)
        base_price = 100.0 + day_offset * 10.0
        volume = 5.0 + day_offset
        for minute in range(3):
            open_price = base_price + minute
            close_price = open_price + 0.5
            ts = int((day_base + timedelta(minutes=minute)).timestamp() * 1000)
            candles.append(
                {
                    "t": ts,
                    "o": open_price,
                    "h": close_price + 0.25,
                    "l": open_price - 0.25,
                    "c": close_price,
                    "v": volume,
                }
            )

    window_start = datetime(2024, 1, 13, 7, 0, tzinfo=timezone.utc)
    for idx in range(240):
        ts = int((window_start + timedelta(minutes=idx)).timestamp() * 1000)
        base_price = 200.0 + idx * 0.2
        open_price = base_price
        close_price = base_price + 0.2
        volume = 1.5 + (idx % 4) * 0.1
        candles.append(
            {
                "t": ts,
                "o": open_price,
                "h": close_price + 0.1,
                "l": open_price - 0.1,
                "c": close_price,
                "v": volume,
            }
        )

    return candles


def _register_sample_snapshot() -> list[dict[str, float]]:
    candles = _build_sample_candles()
    snapshot_id = f"check-all-{uuid.uuid4().hex}"
    register_snapshot(
        {
            "id": snapshot_id,
            "symbol": "BTCUSDT",
            "tf": "1m",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "frames": {"1m": {"tf": "1m", "candles": candles}},
        }
    )
    return candles


def _compute_expected_vwap(bars: list[dict[str, float]]) -> float:
    pv = sum(bar["c"] * bar["v"] for bar in bars)
    volume = sum(bar["v"] for bar in bars)
    return round(pv / volume, 4) if volume else 0.0


def test_market_overview_aggregates_previous_days_and_window(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _register_sample_snapshot()
    now_utc = datetime(2024, 1, 13, 11, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("src.services.check_all_datas._now_utc", lambda: now_utc)

    overview = build_market_overview(hours=4)

    previous = list(overview.previous_days_summary)
    assert [entry.date_local for entry in previous] == [
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    assert previous[0].bars == 3
    assert previous[0].open == pytest.approx(100.0)
    assert previous[0].close == pytest.approx(102.5)
    assert previous[0].volume_sum == pytest.approx(15.0)
    assert previous[0].ret_close_pct == pytest.approx(2.5, rel=1e-3)

    aggregate = overview.previous_days_aggregate
    assert aggregate.days == 3
    assert aggregate.volume_sum == pytest.approx(54.0)
    assert aggregate.high_max == pytest.approx(122.75)
    assert aggregate.low_min == pytest.approx(99.75)
    assert aggregate.ret_close_pct_avg == pytest.approx(2.2853, rel=1e-4)

    detailed = overview.last_hours_detailed
    assert detailed.hours == 4
    assert detailed.window_utc.start == int(datetime(2024, 1, 13, 7, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert detailed.window_utc.end == int(now_utc.timestamp() * 1000)
    assert detailed.ohlc_window.o == pytest.approx(200.0, rel=1e-6)
    assert detailed.ohlc_window.c == pytest.approx(248.0, rel=1e-6)
    assert detailed.volume_window == pytest.approx(396.0)
    assert detailed.ret_window_pct == pytest.approx(24.0)
    assert detailed.max_drawdown_pct == pytest.approx(0.0)
    assert len(detailed.bars) == 240

    first_bar = detailed.bars[0]
    second_bar = detailed.bars[1]
    assert first_bar.ret_1m == pytest.approx(0.0)
    expected_ret = round((second_bar.c - first_bar.c) / first_bar.c * 100.0, 4)
    assert second_bar.ret_1m == pytest.approx(expected_ret)

    expected_vwap = _compute_expected_vwap(candles[-240:])
    assert all(bar.vwap_window == pytest.approx(expected_vwap) for bar in detailed.bars[:10])

    last_bar = detailed.bars[-1]
    expected_sma5 = round(
        sum(bar["c"] for bar in candles[-5:]) / 5,
        4,
    )
    assert last_bar.sma_5 == pytest.approx(expected_sma5)

    fifteenth_bar = detailed.bars[14]
    first_returns = [bar.ret_1m for bar in detailed.bars[:15]]
    mean_ret = sum(first_returns) / len(first_returns)
    variance = sum((value - mean_ret) ** 2 for value in first_returns) / len(first_returns)
    expected_volatility = round(math.sqrt(variance), 4)
    assert fifteenth_bar.volatility_15m == pytest.approx(expected_volatility)


def test_check_all_datas_content_negotiation(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_sample_snapshot()
    now_utc = datetime(2024, 1, 13, 11, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("src.services.check_all_datas._now_utc", lambda: now_utc)

    response_json = client.get(
        "/inspection/check-all-datas?hours=3",
        headers={"accept": "application/json"},
    )
    assert response_json.status_code == 200
    body = response_json.json()
    assert body["timezone"] == "Europe/Berlin"
    assert body["last_hours_detailed"]["hours"] == 3

    response_html = client.get(
        "/inspection/check-all-datas?hours=3",
        headers={"accept": "text/html"},
    )
    assert response_html.status_code == 200
    assert "Previous days summary" in response_html.text


def test_check_all_datas_rejects_unsupported_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_sample_snapshot()
    now_utc = datetime(2024, 1, 13, 11, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("src.services.check_all_datas._now_utc", lambda: now_utc)

    response = client.get(
        "/inspection/check-all-datas?hours=4",
        headers={"accept": "application/xml"},
    )
    assert response.status_code == 406


def test_check_all_datas_hours_validation() -> None:
    response = client.get(
        "/inspection/check-all-datas?hours=1",
        headers={"accept": "application/json"},
    )
    assert response.status_code == 422


def test_check_all_datas_returns_204_when_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_sample_snapshot()
    future_now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("src.services.check_all_datas._now_utc", lambda: future_now)

    response = client.get(
        "/inspection/check-all-datas?hours=4",
        headers={"accept": "application/json"},
    )
    assert response.status_code == 204
