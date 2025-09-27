"""Meta configuration for chart analytics sessions."""
from __future__ import annotations

from datetime import time
from typing import Dict, Iterable, Tuple


# Hardcoded tick sizes for popular perpetual pairs used as liquidity fallbacks.
HARDCODED_TICK_SIZES: Dict[str, float] = {
    "BTCUSDT": 0.1,
    "ETHUSDT": 0.01,
    "SOLUSDT": 0.001,
}


class Meta:
    """Holds application-wide metadata such as VWAP sessions."""

    VWAP_LOOKBACK_DAYS: int = 5
    _VWAP_SESSIONS: Tuple[Tuple[str, time, time], ...] = (
        ("asia", time(hour=0, minute=0), time(hour=8, minute=0)),
        ("london", time(hour=8, minute=0), time(hour=12, minute=0)),
        ("ny", time(hour=12, minute=0), time(hour=20, minute=0)),
    )

    @classmethod
    def iter_vwap_sessions(cls) -> Iterable[Tuple[str, time, time]]:
        """Return the configured VWAP sessions as an iterable."""

        return cls._VWAP_SESSIONS
