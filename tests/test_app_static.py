"""Integration smoke-tests for serving the chart UI from FastAPI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_index_served_with_inspection_button(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "<div id=\"chart\"" in body
    assert "id=\"show-inspection\"" in body


def test_static_assets_are_available(client: TestClient) -> None:
    response = client.get("/public/app.js")
    assert response.status_code == 200
    assert "window.LightweightCharts" in response.text
