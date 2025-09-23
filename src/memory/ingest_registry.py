"""Utilities for tracking ingested knowledge folders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


class IngestRegistry:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: Dict[str, str] = {}

    def load(self) -> None:
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._entries = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, folder: str) -> str | None:
        return self._entries.get(folder)

    def update(self, folder: str, digest: str) -> None:
        self._entries[folder] = digest
        self.save()

    def data(self) -> Dict[str, str]:
        return dict(self._entries)


def compute_folder_hash(path: Path) -> str:
    entries = []
    for item in sorted(Path(path).rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        entries.append(f"{item.relative_to(path)}:{stat.st_mtime_ns}:{stat.st_size}")
    joined = "\n".join(entries)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
