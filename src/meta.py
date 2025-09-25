"""Meta configuration for chart analytics sessions."""
from __future__ import annotations

from datetime import time
from typing import Iterable, Tuple


class Meta:
    """Holds application-wide metadata such as VWAP sessions."""

    VWAP_LOOKBACK_DAYS: int = 5
    _VWAP_SESSIONS: Tuple[Tuple[str, time, time], ...] = (
        ("asia", time(hour=0, minute=0), time(hour=8, minute=0)),
        ("london", time(hour=8, minute=0), time(hour=16, minute=0)),
        ("ny", time(hour=13, minute=0), time(hour=21, minute=0)),
    )

    @classmethod
    def iter_vwap_sessions(cls) -> Iterable[Tuple[str, time, time]]:
        """Return the configured VWAP sessions as an iterable."""

        return cls._VWAP_SESSIONS
