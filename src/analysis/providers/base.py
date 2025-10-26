"""Base protocol for external analysis providers."""

from __future__ import annotations

from typing import Protocol


class AnalysisProvider(Protocol):
    """Generic analysis provider that consumes a payload and returns structured feedback."""

    name: str
    timeout_ms: int

    async def analyze(self, payload: dict) -> dict:
        """Run analysis and return a structured JSON-compatible payload."""
        ...


__all__ = ["AnalysisProvider"]

