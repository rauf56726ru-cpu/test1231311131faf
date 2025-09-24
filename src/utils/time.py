"""Time helper utilities."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_unix(ts: datetime) -> int:
    return int(ts.replace(tzinfo=timezone.utc).timestamp())


def from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def floor_to_interval(ts: datetime, interval: str) -> datetime:
    """Floor datetime to pandas offset like '1min'."""
    offset = pd.tseries.frequencies.to_offset(interval)
    floored = (pd.Timestamp(ts).floor(offset)).to_pydatetime()
    return floored.replace(tzinfo=timezone.utc)
