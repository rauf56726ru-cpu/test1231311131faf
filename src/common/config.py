"""Application-wide configuration helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

LOGGER = logging.getLogger(__name__)


def _parse_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class AppConfig:
    offline: bool = False
    ingest_policy: str = "LOCAL_THEN_REMOTE"
    fixture_dirs: List[str] = field(default_factory=lambda: ["tests/fixtures", "fixtures", "cache/vision"])
    data_dir: str = "./data"
    duckdb_path: str = "./meta/index.duckdb"
    market: str = "um"

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "AppConfig":
        offline_env = os.getenv("OFFLINE")
        if offline_env is not None and offline_env.strip():
            LOGGER.warning(
                "config.offline_ignored",
                extra={"requested": offline_env},
            )
        offline = False
        ingest_policy = os.getenv("INGEST_POLICY", "LOCAL_THEN_REMOTE").strip() or "LOCAL_THEN_REMOTE"
        fixture_dirs = _parse_list(os.getenv("FIXTURE_DIRS")) or ["tests/fixtures", "fixtures", "cache/vision"]
        data_dir = os.getenv("DATA_DIR", "./data").strip() or "./data"
        duckdb_path = os.getenv("DUCKDB_PATH", "./meta/index.duckdb").strip() or "./meta/index.duckdb"
        market_env = os.getenv("MARKET", "um")
        market = (market_env.strip() or "um").lower()
        if market != "um":
            raise ValueError("Futures UM only. Spot is not supported.")

        config = cls(
            offline=offline,
            ingest_policy=ingest_policy,
            fixture_dirs=fixture_dirs,
            data_dir=data_dir,
            duckdb_path=duckdb_path,
            market=market,
        )
        try:
            LOGGER.info(
                "config.loaded",
                extra={
                    "offline": config.offline,
                    "ingest_policy": config.ingest_policy,
                    "fixture_dirs": config.fixture_dirs,
                    "data_dir": config.data_dir,
                    "duckdb_path": config.duckdb_path,
                    "market": config.market,
                },
            )
        except Exception:  # pragma: no cover - logging safeguard
            pass
        return config


__all__ = ["AppConfig"]
