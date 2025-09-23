"""Logging helpers."""
from __future__ import annotations

import logging
from logging import Logger


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> Logger:
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
