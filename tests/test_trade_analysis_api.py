import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import httpx

from src.api.app import app
import src.api.app as app_module
import src.services.analysis as analysis
import src.services.check_all_datas as check_all_datas
import src.services.inspection as inspection
from src.services import presets
from src.services.analysis import TradeAnalysisResult


UTC = timezone.utc


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def preset_storage(tmp_path, monkeypatch):
    storage_path = tmp_path / "presets.json"
    monkeypatch.setattr(presets, "_PRESET_STORAGE_PATH", storage_path)
    presets._PRESET_CACHE.clear()
    presets._STORAGE_LOADED = False
    yield
    presets._PRESET_CACHE.clear()
    presets._STORAGE_LOADED = False


@pytest.fixture(autouse=True)
def snapshot_storage(tmp_path, monkeypatch):
    storage_dir = tmp_path / "snapshots"
    monkeypatch.setattr(inspection, "SNAPSHOT_STORAGE_DIR", storage_dir)
    inspection._SNAPSHOT_STORE.clear()
    inspection._ensure_storage_dir()
    inspection._load_existing_snapshots()
    yield storage_dir
    inspection._SNAPSHOT_STORE.clear()


@pytest.fixture(autouse=True)
def stub_binance_minutes(monkeypatch):
    def filler(symbol: str, start_ms: int, end_ms: int, gaps):
        candles = []
        for gap in gaps:
            cursor = int(gap["from"])
            limit = int(gap["to"])
            while cursor <= limit:
                candles.append(
                    {
                        "t": cursor,
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1.0,
                    }
                )
                cursor += 60_000
        return candles

    monkeypatch.setattr(check_all_datas, "_download_missing_minutes", filler)

    def filler_htf(symbol: str, gaps, *, fetcher, target):
        inserted = 0
        for gap in gaps:
            cursor = int(gap.get("from", 0))
            limit = int(gap.get("to", cursor))
            while cursor <= limit:
                candle = {
                    "t": cursor,
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1.0,
                }
                if cursor not in target:
                    inserted += 1
                target[cursor] = candle
                cursor += 60_000
        return inserted

    monkeypatch.setattr(inspection, "_download_missing_minutes", filler_htf)
    yield


@pytest.fixture()
def analysis_env(tmp_path, monkeypatch):
    upload_dir = tmp_path / "analysis"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_ID", "gpt-5")
    monkeypatch.setenv("ANALYSIS_UPLOAD_DIR", str(upload_dir))
    return upload_dir


@pytest.fixture()
def anyio_backend():  # pragma: no cover - used by pytest-anyio to limit backends
    return "asyncio"


def _build_snapshot_payload(base: datetime, count: int = 12) -> dict:
    candles = []
    for index in range(count):
        moment = base + timedelta(minutes=index)
        timestamp_ms = int(moment.timestamp() * 1000)
        open_price = 100.0 + index
        close_price = open_price + 0.5
        high_price = close_price + 0.25
        low_price = open_price - 0.25
        volume = 5.0 + index * 0.1
        candles.append(
            {
                "t": timestamp_ms,
                "o": round(open_price, 2),
                "h": round(high_price, 2),
                "l": round(low_price, 2),
                "c": round(close_price, 2),
                "v": round(volume, 3),
            }
        )

    selection = {"start": candles[0]["t"], "end": candles[-1]["t"]} if candles else None
    payload = {"symbol": "BTCUSDT", "tf": "1m", "candles": candles}
    if selection:
        payload["selection"] = selection
    return payload


def _install_openai_stub(
    monkeypatch,
    *,
    status: str = "ok",
    trade: dict | None = None,
    raw: str | None = None,
    expected_key: str | None = None,
    expected_model: str | None = None,
):
    captured: dict[str, Any] = {}

    async def fake_call(**kwargs):
        captured["kwargs"] = kwargs
        if expected_key is not None:
            assert kwargs.get("api_key") == expected_key
        if expected_model is not None:
            assert kwargs.get("model") == expected_model
        file_path: Path = kwargs["file_path"]
        data = file_path.read_bytes()
        digest = analysis.sha256(data).hexdigest()
        trade_payload = trade if trade is not None else {"symbol": "BTCUSDT", "status": "ok"}
        raw_text = raw if raw is not None else json.dumps(trade_payload)
        return TradeAnalysisResult(
            status=status,
            request_id="resp_test",
            trade_json=trade_payload if status == "ok" else None,
            raw_text=raw_text,
            file_path=file_path,
            latency_ms=125,
            attachment_size=len(data),
            attachment_sha256=digest,
        )

    monkeypatch.setattr(analysis, "call_openai_with_attachment", fake_call)
    return captured


@pytest.mark.anyio
async def test_call_openai_uses_input_text(monkeypatch, tmp_path) -> None:
    attachment = tmp_path / "sample.json"
    attachment.write_text("{}", encoding="utf-8")

    captured: dict[str, Any] = {}

    async def fake_post(client, url, *, headers=None, data=None, files=None, json_payload=None):
        if url.endswith("/v1/files"):
            assert files is not None
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={"id": "file_uploaded"})

        if url.endswith("/v1/responses"):
            captured["payload"] = json_payload
            request = httpx.Request("POST", url)
            body = {
                "id": "resp_123",
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"symbol": "BTCUSDT"}),
                            }
                        ]
                    }
                ],
            }
            return httpx.Response(200, request=request, json=body)

        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(analysis, "_post_with_retry", fake_post)

    result = await analysis.call_openai_with_attachment(
        api_key="sk-test",
        model="gpt-5",
        file_path=attachment,
        symbol="BTCUSDT",
        period="3d_overview_4h_detail",
    )

    assert result.status == "ok"
    payload = captured["payload"]
    assert payload["input"][0]["content"][0]["type"] == "input_text"
    assert payload["input"][1]["content"][0]["type"] == "input_text"

def test_analyze_from_inspection_returns_payload(client: TestClient, analysis_env: Path, monkeypatch) -> None:
    base = datetime(2024, 6, 1, 12, tzinfo=UTC)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    captured = _install_openai_stub(
        monkeypatch,
        expected_key="test-key",
        expected_model="gpt-5",
    )

    selection = payload["selection"]
    body = {
        "snapshot_id": snapshot_id,
        "selection_start": selection["start"],
        "selection_end": selection["end"],
        "hours": 2,
        "period": "3d_overview_4h_detail",
    }

    response = client.post("/api/analyze-from-inspection", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["request_id"] == "resp_test"
    assert data["trade_json"]["symbol"] == "BTCUSDT"
    debug = data["debug"]
    assert debug["period"] == "3d_overview_4h_detail"
    assert debug["attachment_file"]
    file_path = analysis_env / debug["attachment_file"]
    assert file_path.exists()
    stored = json.loads(file_path.read_text(encoding="utf-8"))
    assert stored["SYMBOL"] == "BTCUSDT"
    assert stored["PERIOD"] == "3d_overview_4h_detail"
    assert stored["DATA"]["snapshot_id"]
    assert debug["model"] == "gpt-5"
    assert captured["kwargs"]["api_key"] == "test-key"


def test_analyze_from_inspection_handles_insufficient(client: TestClient, analysis_env: Path, monkeypatch) -> None:
    base = datetime(2024, 6, 2, 8, tzinfo=UTC)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    captured = _install_openai_stub(
        monkeypatch,
        status="insufficient_data",
        trade=None,
        raw="not-json",
        expected_key="test-key",
        expected_model="gpt-5",
    )

    selection = payload["selection"]
    body = {
        "snapshot_id": snapshot_id,
        "selection_start": selection["start"],
        "selection_end": selection["end"],
        "hours": 1,
        "period": "custom_period",
    }

    response = client.post("/api/analyze-from-inspection", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["trade_json"] is None
    assert "raw_text" in data["debug"]
    assert data["debug"]["period"] == "custom_period"
    assert data["debug"]["model"] == "gpt-5"
    assert captured["kwargs"]["api_key"] == "test-key"


def test_analyze_from_inspection_accepts_api_key_override(
    client: TestClient,
    analysis_env: Path,
    monkeypatch,
) -> None:
    base = datetime(2024, 6, 3, 10, tzinfo=UTC)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured = _install_openai_stub(
        monkeypatch,
        expected_key="override-key",
        expected_model="gpt-5",
    )

    selection = payload["selection"]
    body = {
        "snapshot_id": snapshot_id,
        "selection_start": selection["start"],
        "selection_end": selection["end"],
        "hours": 2,
        "period": "override_period",
        "api_key": "override-key",
    }

    response = client.post("/api/analyze-from-inspection", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["debug"]["model"] == "gpt-5"
    assert captured["kwargs"]["api_key"] == "override-key"


def test_analyze_from_inspection_requires_api_key_when_missing(
    client: TestClient,
    analysis_env: Path,
    monkeypatch,
) -> None:
    base = datetime(2024, 6, 4, 7, tzinfo=UTC)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    selection = payload["selection"]
    body = {
        "snapshot_id": snapshot_id,
        "selection_start": selection["start"],
        "selection_end": selection["end"],
        "hours": 1,
        "period": "needs_key",
    }

    response = client.post("/api/analyze-from-inspection", json=body)
    assert response.status_code == 400
    assert response.json()["detail"] == "OpenAI API key is required"


def test_analyze_from_inspection_surfaces_openai_error(
    client: TestClient,
    analysis_env: Path,
    monkeypatch,
) -> None:
    base = datetime(2024, 6, 5, 11, tzinfo=UTC)
    payload = _build_snapshot_payload(base)

    create_response = client.post("/inspection/snapshot", json=payload)
    assert create_response.status_code == 200
    snapshot_id = create_response.json()["snapshot_id"]

    async def failing_dispatch(*args, **kwargs):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        error_payload = {
            "error": {
                "message": "Attachment validation failed",
                "type": "invalid_request_error",
                "code": "attachment_invalid",
            }
        }
        response = httpx.Response(
            status_code=400,
            request=request,
            content=json.dumps(error_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        raise httpx.HTTPStatusError("Bad Request", request=request, response=response)

    monkeypatch.setattr(app_module, "dispatch_trade_analysis", failing_dispatch)

    selection = payload["selection"]
    body = {
        "snapshot_id": snapshot_id,
        "selection_start": selection["start"],
        "selection_end": selection["end"],
        "hours": 1,
        "period": "error_case",
    }

    response = client.post("/api/analyze-from-inspection", json=body)
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["status_code"] == 400
    assert detail["openai_error"]["message"] == "Attachment validation failed"
    assert detail["openai_error"]["type"] == "invalid_request_error"
