import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

joblib = pytest.importorskip("joblib")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import json
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def prepared_env():
    Path("models/xgb").mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)

    index = pd.date_range("2023-01-01", periods=10, freq="T")
    X = pd.DataFrame({"ema_50": np.linspace(1, 2, 10), "ema_200": np.linspace(0.5, 1.5, 10)}, index=index)
    y = pd.Series([0, 1] * 5, index=index)
    df = X.copy()
    df["target"] = y
    df.to_parquet("data/features/features.parquet")
    (Path("data/features") / "feature_list.json").write_text(json.dumps(list(X.columns)))

    scaler = StandardScaler().fit(X.values)
    model = LogisticRegression().fit(scaler.transform(X.values), y.values)
    joblib.dump(model, "models/xgb/model.pkl")
    joblib.dump(scaler, "models/xgb/scaler.pkl")
    (Path("models/xgb") / "feature_list.json").write_text(json.dumps(list(X.columns)))
    yield


@pytest.fixture(scope="module")
def client(prepared_env):
    from src.api.app import app

    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint(client: TestClient):
    resp = client.get("/predict")
    assert resp.status_code == 200
    data = resp.json()
    assert "prob_up" in data and "side" in data


def test_explain_endpoint(client: TestClient):
    resp = client.get("/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert "memory_hits" in data


def test_paper_order(client: TestClient):
    resp = client.post("/paper/order", json={"symbol": "BTCUSDT", "side": "buy", "size": 1})
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


def test_backfill_endpoint(client: TestClient, monkeypatch):
    provider = client.app.state.market_data

    async def fake_fetch_gap(symbol, interval, start_ms, end_ms):
        return [
            {
                "ts_ms_utc": int(start_ms),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10.0,
            }
        ]

    monkeypatch.setattr(provider, "fetch_gap", fake_fetch_gap)
    resp = client.post(
        "/ohlc/backfill",
        json={"symbol": "BTCUSDT", "interval": "1m", "start_ms": 0, "end_ms": 60_000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "candles" in data and data["candles"]
    assert data["range"]["end_ms"] == 60_000

