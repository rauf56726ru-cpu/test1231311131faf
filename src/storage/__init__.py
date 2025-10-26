"""Storage backends for OHLCV time series."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from .base import Storage, StorageError
from .json_storage import JSONStorage
from .parquet import ParquetStorage

__all__ = [
    "Storage",
    "StorageError",
    "JSONStorage",
    "ParquetStorage",
    "get_storage",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STORAGE: ParquetStorage | None = None
_STORAGE_LOCK = RLock()


def _resolve_market() -> str:
    try:
        from src.services.settings import get_settings

        return get_settings().binance_vision.market_source
    except Exception:  # pragma: no cover - fallback when settings unavailable
        return "default"


def get_storage() -> ParquetStorage:
    """Return a process-wide Parquet storage instance."""

    global _DEFAULT_STORAGE
    with _STORAGE_LOCK:
        if _DEFAULT_STORAGE is None:
            market = _resolve_market()
            _DEFAULT_STORAGE = ParquetStorage(
                root=_PROJECT_ROOT / "data",
                market=market,
                index_path=_PROJECT_ROOT / "meta" / "index.duckdb",
            )
        return _DEFAULT_STORAGE
