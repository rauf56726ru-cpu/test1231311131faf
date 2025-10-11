"""Utilities for streaming detailed progress updates to clients."""
from __future__ import annotations

from inspect import isawaitable
from typing import Any, Awaitable, Callable, Dict, Optional

ProgressPayload = Dict[str, Any]
ProgressReporter = Callable[[str, ProgressPayload], Awaitable[None] | None]


async def emit_progress(
    reporter: Optional[ProgressReporter],
    event: str,
    /,
    **payload: Any,
) -> None:
    """Dispatch a progress event if a reporter callback is provided."""

    if reporter is None:
        return

    message: ProgressPayload = dict(payload)
    try:
        result = reporter(event, message)
    except Exception:  # pragma: no cover - defensive guard
        return
    if result is None:
        return
    if isawaitable(result):
        await result
