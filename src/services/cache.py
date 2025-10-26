"""Filesystem-backed cache with optional TTL semantics."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class CacheEntry:
    payload: Any
    expires_at: float | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at


class FileCache:
    """Simple JSON cache backed by files."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"

    def get(self, key: str) -> CacheEntry | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expires_at = data.get("expires_at")
        entry = CacheEntry(payload=data.get("payload"), expires_at=float(expires_at) if expires_at is not None else None)
        if entry.is_expired():
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return entry

    def set(self, key: str, payload: Any, ttl_seconds: float | None = None) -> None:
        path = self._path_for(key)
        expires_at: float | None = None
        if ttl_seconds is not None and ttl_seconds > 0:
            expires_at = time.time() + ttl_seconds
        entry = {"payload": payload, "expires_at": expires_at}
        try:
            path.write_text(json.dumps(entry), encoding="utf-8")
        except OSError:
            pass


__all__ = ["FileCache", "CacheEntry"]
