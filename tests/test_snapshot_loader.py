"""Snapshot persistence regression tests."""

from __future__ import annotations

import json
from collections import OrderedDict

from src.services import inspection


def test_snapshot_loader_skips_corrupted_files(tmp_path, monkeypatch, caplog):
    storage_dir = tmp_path / "snapshots"
    storage_dir.mkdir()

    valid = {"id": "valid", "symbol": "BTCUSDT", "tf": "1m", "frames": {}}
    (storage_dir / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    (storage_dir / "corrupt.json").write_bytes(b"\xff\xfe\x00")

    monkeypatch.setattr(inspection, "SNAPSHOT_STORAGE_DIR", storage_dir)
    monkeypatch.setattr(inspection, "_SNAPSHOT_STORE", OrderedDict())

    caplog.set_level("WARNING", logger=inspection.LOGGER.name)

    inspection._load_existing_snapshots()

    assert list(inspection._SNAPSHOT_STORE.keys()) == ["valid"]
    assert any("persisted snapshot" in record.getMessage() for record in caplog.records)
