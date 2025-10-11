from datetime import datetime, timezone, timedelta

from src.services.timeutils import ensure_ms_epoch, safe_datetime_from_ms


def test_ensure_ms_epoch_normalises_seconds() -> None:
    seconds_value = 1_690_000_000
    ms_value = ensure_ms_epoch(seconds_value)
    assert ms_value == seconds_value * 1000


def test_ensure_ms_epoch_handles_microseconds() -> None:
    micro_value = 1_690_000_000_000_000
    ms_value = ensure_ms_epoch(micro_value)
    assert ms_value == micro_value // 1000


def test_ensure_ms_epoch_rejects_out_of_range() -> None:
    assert ensure_ms_epoch(0) is None
    assert ensure_ms_epoch(-1) is None
    future = datetime.now(timezone.utc) + timedelta(days=2)
    assert ensure_ms_epoch(int(future.timestamp() * 1000)) is None


def test_safe_datetime_from_ms_returns_datetime() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ms_value = int(now.timestamp() * 1000)
    dt = safe_datetime_from_ms(ms_value, timezone.utc)
    assert dt == now


def test_safe_datetime_from_ms_handles_overflow() -> None:
    assert safe_datetime_from_ms(10 ** 20, timezone.utc) is None
