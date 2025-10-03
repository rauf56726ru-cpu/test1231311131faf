import pytest

from src.services.session_collector import collect_last_session_detailed

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_collect_last_session_reports_progress():
    events: list[str] = []

    async def reporter(event: str, payload):
        events.append(event)

    result = await collect_last_session_detailed("BTCUSDT", progress=reporter)

    assert result.symbol == "BTCUSDT"
    assert events[0] == "session_collector:start"
    assert "session_collector:finished" in events
