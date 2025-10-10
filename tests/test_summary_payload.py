from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.api.app import _prepare_summary_payload


def test_prepare_summary_payload_uses_data_window_for_orderflow_and_zones() -> None:
    base_dt = datetime(2024, 1, 4, 0, 0, tzinfo=timezone.utc)
    base_ts = int(base_dt.timestamp() * 1000)

    payload = {
        "status": "ok",
        "meta": {},
        "data": {
            "ohlcv": {},
            "orderflow": {
                "1m": {
                    "per_bar": [
                        {"ts": base_ts - 2 * 60_000, "delta": 1.0, "cvd": 1.0},
                        {"ts": base_ts - 60_000, "delta": 2.0, "cvd": 3.0},
                    ]
                },
                "15m": {
                    "per_bar": [
                        {
                            "ts": base_ts - 45 * 60_000,
                            "delta_sum": 3.0,
                            "cvd_close": 50.0,
                            "vol_sum": 120.0,
                        },
                        {
                            "ts": base_ts - 15 * 60_000,
                            "delta_sum": 4.0,
                            "cvd_close": 54.0,
                            "vol_sum": 150.0,
                        },
                    ]
                },
                "1h": {
                    "per_bar": [
                        {
                            "ts": base_ts - 90 * 60_000,
                            "delta_sum": -1.5,
                            "cvd_close": 20.0,
                            "vol_sum": 80.0,
                        },
                        {
                            "ts": base_ts - 30 * 60_000,
                            "delta_sum": 2.5,
                            "cvd_close": 22.5,
                            "vol_sum": 60.0,
                        },
                    ]
                },
            },
            "zones": {
                "ob": [
                    {
                        "status": "open",
                        "formed_at_utc": (base_dt - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                        "last_touched_utc": (base_dt - timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
                        "open": 100.0,
                        "close": 110.0,
                    }
                ]
            },
        },
        "availability": {},
    }

    compact = _prepare_summary_payload(payload)

    per_bar = compact["orderflow"]["per_bar"]
    assert set(per_bar.keys()) == {"1m", "3m", "5m", "15m"}
    assert per_bar["1m"], "Expected recent 1m orderflow entries"
    assert per_bar["1m"][-1]["cvd"] == 3.0
    assert per_bar["3m"] == []
    assert per_bar["5m"] == []
    assert per_bar["15m"]
    assert per_bar["15m"][-1]["cvd_close"] == 54.0

    meta = compact["orderflow"]["meta"]
    assert meta["partial"] is True
    assert meta["available_minutes"] == 2
    assert meta["available_aggregates"]["15m"] == 2
    assert meta["available_aggregates"]["1h"] == 2

    delta_compact = compact["orderflow"]["delta_cvd_compact"]
    assert delta_compact["15m"][-1]["delta_sum"] == 4.0
    assert delta_compact["1h"][-1]["cvd_close"] == 22.5

    zones_top = compact["zones"]["top"]
    assert zones_top, "Expected zones to be retained within the 72h window"
    expected_date = (base_dt - timedelta(hours=1)).date().isoformat()
    assert zones_top[0]["formed_at_utc"].startswith(expected_date)
    counts = compact["zones"]["counts"]
    assert counts.get("ob") == 1
    filter_diag = compact["zones"].get("diag", {}).get("filter")
    assert filter_diag
    assert filter_diag.get("allowed_statuses") == ["open", "fresh", "tapped"]


def test_prepare_summary_payload_filters_zones_by_status_and_counts() -> None:
    base_dt = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)
    base_ts = int(base_dt.timestamp() * 1000)

    payload = {
        "status": "ok",
        "meta": {"last_ts_utc": base_dt.isoformat().replace("+00:00", "Z")},
        "data": {
            "ohlcv": {
                "1m": {
                    "candles": [
                        {"t": base_ts - 60_000, "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 12},
                        {"t": base_ts, "o": 100.5, "h": 101.5, "l": 100.1, "c": 101.0, "v": 10},
                    ]
                },
                "15m": {"candles": []},
                "1h": {"candles": []},
                "4h": {"candles": []},
                "1d": {"candles": []},
            },
            "orderflow": {
                "1m": {
                    "per_bar": [
                        {"ts": base_ts - 2 * 60_000, "delta": 1.0, "cvd": 1.0},
                        {"ts": base_ts - 60_000, "delta": 2.0, "cvd": 3.0},
                    ]
                },
                "15m": {
                    "per_bar": [
                        {
                            "ts": base_ts - 45 * 60_000,
                            "delta_sum": 3.0,
                            "cvd_close": 50.0,
                            "vol_sum": 120.0,
                        },
                        {
                            "ts": base_ts - 15 * 60_000,
                            "delta_sum": 4.0,
                            "cvd_close": 54.0,
                            "vol_sum": 150.0,
                        },
                    ]
                },
                "1h": {
                    "per_bar": [
                        {
                            "ts": base_ts - 90 * 60_000,
                            "delta_sum": -1.5,
                            "cvd_close": 20.0,
                            "vol_sum": 80.0,
                        },
                        {
                            "ts": base_ts - 30 * 60_000,
                            "delta_sum": 2.5,
                            "cvd_close": 22.5,
                            "vol_sum": 60.0,
                        },
                    ]
                },
            },
            "zones": {
                "ob": [
                    {
                        "status": "fresh",
                        "tf": "15m",
                        "formed_at_utc": (base_dt - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                        "last_touched_utc": (base_dt - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                        "open": 100.0,
                        "close": 105.0,
                    },
                    {
                        "status": "tapped",
                        "tf": "15m",
                        "formed_at_utc": (base_dt - timedelta(hours=4)).isoformat().replace("+00:00", "Z"),
                        "last_touched_utc": (base_dt - timedelta(hours=3)).isoformat().replace("+00:00", "Z"),
                        "open": 103.0,
                        "close": 106.0,
                    },
                    {
                        "status": "invalidated",
                        "tf": "15m",
                        "formed_at_utc": (base_dt - timedelta(hours=10)).isoformat().replace("+00:00", "Z"),
                        "last_touched_utc": (base_dt - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                        "open": 110.0,
                        "close": 111.0,
                    },
                    {
                        "status": "invalidated",
                        "tf": "15m",
                        "formed_at_utc": (base_dt - timedelta(hours=90)).isoformat().replace("+00:00", "Z"),
                        "last_touched_utc": (base_dt - timedelta(hours=80)).isoformat().replace("+00:00", "Z"),
                        "open": 120.0,
                        "close": 121.0,
                    },
                ],
                "mb": [
                    {
                        "status": "fresh",
                        "tf": "15m",
                        "last_touched_utc": (base_dt - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                        "open": 95.0,
                        "close": 97.0,
                    }
                ],
            },
        },
        "availability": {},
    }

    compact = _prepare_summary_payload(payload)

    zones_top = compact["zones"]["top"]
    assert len(zones_top) == 2
    statuses = {item["status"] for item in zones_top}
    assert statuses == {"fresh", "tapped"}
    counts = compact["zones"]["counts"]
    assert counts.get("ob") == 2
    assert counts.get("mb") == 0


def test_prepare_summary_payload_requires_delta_cvd_for_ok_status() -> None:
    base_dt = datetime(2024, 1, 7, 8, 0, tzinfo=timezone.utc)
    base_ts = int(base_dt.timestamp() * 1000)

    minute_count = 72 * 60
    start_ts = base_ts - (minute_count - 1) * 60_000
    minute_candles = [
        {
            "t": start_ts + index * 60_000,
            "o": 1.0 + index * 0.001,
            "h": 1.1 + index * 0.001,
            "l": 0.9 + index * 0.001,
            "c": 1.05 + index * 0.001,
            "v": 10.0 + index * 0.01,
        }
        for index in range(minute_count)
    ]

    fifteen_candles = [
        {
            "t": start_ts + index * 15 * 60_000,
            "o": 1.0 + index * 0.01,
            "h": 1.2 + index * 0.01,
            "l": 0.8 + index * 0.01,
            "c": 1.1 + index * 0.01,
            "v": 150.0 + index,
        }
        for index in range(72 * 4)
    ]

    hourly_candles = [
        {
            "t": start_ts + index * 60 * 60_000,
            "o": 1.0 + index * 0.05,
            "h": 1.3 + index * 0.05,
            "l": 0.7 + index * 0.05,
            "c": 1.15 + index * 0.05,
            "v": 600.0 + index * 5,
        }
        for index in range(72)
    ]

    payload = {
        "status": "ok",
        "meta": {"last_ts_utc": base_dt.isoformat().replace("+00:00", "Z")},
        "data": {
            "ohlcv": {
                "1m": {"candles": minute_candles},
                "15m": {"candles": fifteen_candles},
                "1h": {"candles": hourly_candles},
            },
            "orderflow": {
                "1m": {
                    "per_bar": [
                        {"ts": base_ts - 60_000, "delta": 1.0, "cvd": 1.0},
                        {"ts": base_ts, "delta": 2.0, "cvd": 3.0},
                    ]
                }
            },
            "zones": {},
        },
        "availability": {
            "orderflow": {
                "timeframes": {
                    "1m": {"has_data": True, "bars": len(minute_candles)},
                    "15m": {"has_data": False, "bars": 0},
                    "1h": {"has_data": False, "bars": 0},
                }
            }
        },
    }

    compact = _prepare_summary_payload(payload)

    assert compact["status"] == "partial"
    assert compact["orderflow"]["meta"]["partial"] is True
    missing_required = set(compact["orderflow"]["meta"].get("missing_required", []))
    assert missing_required == {"15m", "1h"}


def test_prepare_summary_payload_marks_partial_when_fallback_meta_present() -> None:
    base_dt = datetime(2024, 1, 7, 0, 0, tzinfo=timezone.utc)
    base_ts = int(base_dt.timestamp() * 1000)

    payload = {
        "status": "ok",
        "meta": {},
        "data": {
            "ohlcv": {},
            "orderflow": {
                "1m": {
                    "per_bar": [
                        {"ts": base_ts - 60_000, "delta": 1.0, "cvd": 1.0},
                        {"ts": base_ts, "delta": -0.5, "cvd": 0.5},
                    ]
                },
                "15m": {
                    "per_bar": [
                        {
                            "ts": base_ts - 15 * 60_000,
                            "delta_sum": 0.5,
                            "cvd_close": 10.0,
                            "vol_sum": 40.0,
                        }
                    ]
                },
                "1h": {
                    "per_bar": [
                        {
                            "ts": base_ts - 60 * 60_000,
                            "delta_sum": 0.5,
                            "cvd_close": 9.5,
                            "vol_sum": 120.0,
                        }
                    ]
                },
                "meta": {
                    "partial": True,
                    "fallback": {"status": 429, "preserved_trades": 120},
                },
            },
            "zones": {},
        },
        "availability": {},
    }

    compact = _prepare_summary_payload(payload)

    orderflow_meta = compact["orderflow"]["meta"]
    assert orderflow_meta["partial"] is True
    assert orderflow_meta.get("fallback") == {"status": 429, "preserved_trades": 120}


def test_prepare_summary_payload_marks_insufficient_when_coverage_low() -> None:
    base_dt = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
    base_ts = int(base_dt.timestamp() * 1000)

    minute_candles = [
        {"t": base_ts - index * 60_000, "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 12.0}
        for index in range(60)
    ]

    payload = {
        "status": "ok",
        "meta": {"last_ts_utc": base_dt.isoformat().replace("+00:00", "Z")},
        "data": {
            "ohlcv": {
                "1m": {"candles": list(reversed(minute_candles))},
                "15m": {"candles": []},
                "1h": {"candles": []},
                "4h": {"candles": []},
                "1d": {"candles": []},
            },
            "orderflow": {
                "1m": {
                    "per_bar": [
                        {"ts": base_ts - 60_000, "delta": 1.0, "cvd": 1.0},
                        {"ts": base_ts, "delta": -0.5, "cvd": 0.5},
                    ]
                },
                "15m": {
                    "per_bar": [
                        {
                            "ts": base_ts - 15 * 60_000,
                            "delta_sum": 1.5,
                            "cvd_close": 5.0,
                            "vol_sum": 40.0,
                        }
                    ]
                },
                "1h": {
                    "per_bar": [
                        {
                            "ts": base_ts - 60 * 60_000,
                            "delta_sum": -2.0,
                            "cvd_close": 3.0,
                            "vol_sum": 80.0,
                        }
                    ]
                },
            },
            "zones": {},
        },
        "availability": {},
    }

    compact = _prepare_summary_payload(payload)

    assert compact["status"] == "insufficient_data"
    assert compact["orderflow"]["meta"]["partial"] is True


def test_prepare_summary_payload_status_ok_when_full_dataset_available() -> None:
    base_dt = datetime(2024, 1, 9, 0, 0, tzinfo=timezone.utc)
    base_ts = int(base_dt.timestamp() * 1000)

    minute_count = 72 * 60
    start_ts = base_ts - (minute_count - 1) * 60_000
    minute_candles = [
        {
            "t": start_ts + index * 60_000,
            "o": 100.0 + index * 0.01,
            "h": 100.5 + index * 0.01,
            "l": 99.5 + index * 0.01,
            "c": 100.25 + index * 0.01,
            "v": 5.0 + index * 0.005,
        }
        for index in range(minute_count)
    ]

    fifteen_candles = [
        {
            "t": start_ts + index * 15 * 60_000,
            "o": 100.0 + index * 0.05,
            "h": 100.7 + index * 0.05,
            "l": 99.3 + index * 0.05,
            "c": 100.4 + index * 0.05,
            "v": 75.0 + index * 0.5,
        }
        for index in range(72 * 4)
    ]

    hourly_candles = [
        {
            "t": start_ts + index * 60 * 60_000,
            "o": 100.0 + index * 0.2,
            "h": 101.0 + index * 0.2,
            "l": 99.0 + index * 0.2,
            "c": 100.6 + index * 0.2,
            "v": 300.0 + index,
        }
        for index in range(72)
    ]

    four_hour_candles = [
        {
            "t": start_ts + index * 4 * 60 * 60_000,
            "o": 100.0 + index * 0.5,
            "h": 101.5 + index * 0.5,
            "l": 98.5 + index * 0.5,
            "c": 100.8 + index * 0.5,
            "v": 1200.0 + index * 5,
        }
        for index in range(18)
    ]

    daily_candles = [
        {
            "t": start_ts + index * 24 * 60 * 60_000,
            "o": 100.0 + index,
            "h": 102.0 + index,
            "l": 98.0 + index,
            "c": 100.9 + index,
            "v": 2000.0 + index * 10,
        }
        for index in range(3)
    ]

    per_bar_rows = [
        {
            "ts": base_ts - (119 - index) * 60_000,
            "delta": 0.5 + index * 0.01,
            "cvd": (index + 1) * 0.5,
        }
        for index in range(120)
    ]

    fifteen_per_bar = [
        {
            "ts": base_ts - (72 * 4 - 1 - index) * 15 * 60_000,
            "delta_sum": 1.0 + index * 0.05,
            "cvd_close": 10.0 + index * 0.5,
            "vol_sum": 40.0 + index,
        }
        for index in range(72 * 4)
    ]

    hourly_per_bar = [
        {
            "ts": base_ts - (72 - 1 - index) * 60 * 60_000,
            "delta_sum": -2.0 + index * 0.1,
            "cvd_close": 20.0 + index * 0.3,
            "vol_sum": 80.0 + index * 2,
        }
        for index in range(72)
    ]

    payload = {
        "status": "ok",
        "meta": {"last_ts_utc": base_dt.isoformat().replace("+00:00", "Z")},
        "data": {
            "ohlcv": {
                "1m": {"candles": minute_candles},
                "15m": {"candles": fifteen_candles},
                "1h": {"candles": hourly_candles},
                "4h": {"candles": four_hour_candles},
                "1d": {"candles": daily_candles},
            },
            "orderflow": {
                "1m": {"per_bar": per_bar_rows},
                "15m": {"per_bar": fifteen_per_bar},
                "1h": {"per_bar": hourly_per_bar},
            },
            "zones": {},
        },
        "availability": {
            "orderflow": {
                "timeframes": {
                    "1m": {"has_data": True, "bars": 120},
                    "15m": {"has_data": True, "bars": len(fifteen_per_bar)},
                    "1h": {"has_data": True, "bars": len(hourly_per_bar)},
                }
            }
        },
    }

    compact = _prepare_summary_payload(payload)

    assert compact["status"] == "ok"
    orderflow_meta = compact["orderflow"]["meta"]
    assert orderflow_meta["partial"] is False
    assert orderflow_meta.get("missing_minutes") is None
    assert orderflow_meta.get("missing_aggregates") is None
