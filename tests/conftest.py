from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services import collection_state


@pytest.fixture(autouse=True)
def _reset_collection_state(tmp_path, monkeypatch):
    """Isolate the collection state file between tests."""

    state_dir = tmp_path / "collection_state"
    state_file = state_dir / "collection_state.json"
    monkeypatch.setattr(collection_state, "STATE_DIR", state_dir)
    monkeypatch.setattr(collection_state, "STATE_FILE", state_file)
    collection_state.reset_state()
    yield
    collection_state.reset_state()
