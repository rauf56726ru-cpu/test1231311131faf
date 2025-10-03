from typing import List

from src.services.candles_repository import CandleRepository


def test_upsert_candles_uses_batches(monkeypatch):
    repo = CandleRepository(":memory:")
    monkeypatch.setattr(repo, "_ensure_schema", lambda: None)

    batch_sizes: List[int] = []

    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def executemany(self, statement, payload):
            batch_sizes.append(len(payload))

    monkeypatch.setattr(repo, "_connect", lambda: DummyConnection())

    candles = [
        {"t": 1_700_000_000_000 + i * 60_000, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 1.0}
        for i in range(1_500)
    ]

    stats = repo.upsert_candles("BTCUSDT", "1m", candles, stage="test")

    assert stats.written == 1_500
    assert batch_sizes == [1000, 500]
