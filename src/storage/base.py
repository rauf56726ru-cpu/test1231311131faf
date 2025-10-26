"""Abstract storage interface for OHLCV datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


class StorageError(RuntimeError):
    """Raised when a storage backend cannot satisfy the requested operation."""


@dataclass(slots=True)
class StorageWriteStats:
    """Statistics returned after persisting a batch of bars."""

    symbol: str
    interval: str
    rows: int
    path: str
    start_ts: int
    end_ts: int


class Storage(ABC):
    """Interface implemented by all OHLCV storage backends."""

    @abstractmethod
    def load_window(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with rows between ``start_ts`` and ``end_ts`` inclusive."""

    def write_rows(
        self,
        symbol: str,
        interval: str,
        rows: Iterable[Mapping[str, object]],
    ) -> Sequence[StorageWriteStats]:
        """Persist a batch of rows. Optional for read-only backends."""
        raise StorageError(f"{self.__class__.__name__} does not support write_rows()")

    def rebuild_index(self) -> None:
        """Rebuild fast lookup metadata for the backend."""
        # Optional hook.
        return None
