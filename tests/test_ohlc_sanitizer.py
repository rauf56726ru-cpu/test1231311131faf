from src.services.ohlc_sanitizer import sanitize_candles


def test_sanitize_candles_normalises_timestamps() -> None:
    candles = [
        {"t": 1_690_000_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5"},
        {"time": 1_690_000_060_000, "open": 1.6, "high": 1.8, "low": 1.5, "close": 1.7},
    ]
    result = sanitize_candles(candles, stage="test")
    assert result.invalid_ts == 0
    assert result.invalid_ohlc == 0
    assert result.candles[0]["t"] == 1_690_000_000_000
    assert result.candles[1]["t"] == 1_690_000_060_000
    assert result.candles[0]["o"] == 1.0
    assert result.candles[0]["c"] == 1.5


def test_sanitize_candles_counts_invalid_entries() -> None:
    candles = [
        {"t": 0, "o": 1, "h": 2, "l": 0.5, "c": 1.5},
        {"t": 1_690_000_000_000, "o": 1, "h": 0.8, "l": 1.2, "c": 0.9},
        {"t": 1_690_000_060_000, "o": 1.0, "h": 1.5, "l": 0.9, "c": 1.2},
    ]
    result = sanitize_candles(candles, stage="test_invalid")
    assert result.invalid_ts == 1
    assert result.invalid_ohlc == 1
    assert len(result.candles) == 1
    assert result.candles[0]["t"] == 1_690_000_060_000
