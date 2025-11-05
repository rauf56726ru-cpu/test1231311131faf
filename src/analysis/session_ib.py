"""Session IB/high-low calculation helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd

MINUTE_MS = 60_000


def compute_session_ib(
    frame: pd.DataFrame,
    *,
    session_start_ms: int,
    session_end_ms: int,
    ib_minutes: int = 60,
) -> Dict[str, float | int] | Dict[str, None]:
    """Return session high/low and initial balance metrics."""

    if frame is None or frame.empty:
        return {}

    required = {"ts_open", "high", "low"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise KeyError(f"compute_session_ib requires columns: {missing}")

    session_mask = (frame["ts_open"] >= session_start_ms) & (frame["ts_open"] < session_end_ms)
    session_frame = frame.loc[session_mask]
    if session_frame.empty:
        return {}

    session_high = float(session_frame["high"].max())
    session_low = float(session_frame["low"].min())

    ib_cutoff = session_start_ms + max(ib_minutes, 1) * MINUTE_MS
    ib_mask = session_frame["ts_open"] < ib_cutoff
    ib_frame = session_frame.loc[ib_mask]
    if ib_frame.empty:
        ib_high = ib_low = None
    else:
        ib_high = float(ib_frame["high"].max())
        ib_low = float(ib_frame["low"].min())

    return {
        "session_start_ms": int(session_start_ms),
        "session_end_ms": int(session_end_ms),
        "high": session_high,
        "low": session_low,
        "ib_high": ib_high,
        "ib_low": ib_low,
    }


__all__ = ["compute_session_ib"]
