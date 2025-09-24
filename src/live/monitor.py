"""Monitoring helpers for live trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Monitor:
    def emit(self, event: str, payload: Dict[str, object]) -> None:
        LOGGER.info("%s | %s", event, payload)
