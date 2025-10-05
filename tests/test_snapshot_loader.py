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


def test_snapshot_loader_normalises_microsecond_timestamps(tmp_path, monkeypatch):
    storage_dir = tmp_path / "snapshots"
    storage_dir.mkdir()

    micro_ts = 1_759_394_400_000_000
    snapshot_payload = {
        "id": "micro",
        "symbol": "BTCUSDT",
        "tf": "1m",
        "frames": {
            "1m": {
                "tf": "1m",
                "candles": [
                    {"t": micro_ts, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 10.0},
                    {"t": micro_ts + 60_000_000, "o": 1.05, "h": 1.2, "l": 1.0, "c": 1.15, "v": 8.0},
                    {"t": micro_ts + 120_000_000, "o": 1.15, "h": 1.25, "l": 1.1, "c": 1.2, "v": 7.0},
                ],
            }
        },
    }
    (storage_dir / "micro.json").write_text(json.dumps(snapshot_payload), encoding="utf-8")

    monkeypatch.setattr(inspection, "SNAPSHOT_STORAGE_DIR", storage_dir)
    monkeypatch.setattr(inspection, "_SNAPSHOT_STORE", OrderedDict())

    inspection._load_existing_snapshots()

    snapshot = inspection._SNAPSHOT_STORE.get("micro")
    assert snapshot is not None
    candles = snapshot["frames"]["1m"]["candles"]
    assert candles[0]["t"] == 1_759_394_400_000
    payload = inspection.build_inspection_payload(snapshot)
    frame_candles = payload["DATA"]["frames"]["1m"]["candles"]
    assert frame_candles
    assert frame_candles[-1]["t"] == candles[-1]["t"]
