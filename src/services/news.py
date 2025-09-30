"""News feed aggregator for Binance assets."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List


async def fetch_news(symbol: str, window_hours: int) -> List[Dict[str, object]]:
    """Return mocked news events within the requested window."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=window_hours)
    topics = [
        "Ecosystem upgrade",
        "Partnership announcement",
        "Liquidity mining update",
        "Perpetual funding adjustment",
        "Regulatory headline",
    ]
    impacts = ["low", "medium", "high"]
    tags = ["announcement", "upgrade", "listing", "macro"]

    events: List[Dict[str, object]] = []
    cursor = start
    while cursor < now:
        cursor += timedelta(hours=random.randint(6, 18))
        if cursor >= now:
            break
        events.append(
            {
                "symbol": symbol_clean,
                "time_utc": cursor.isoformat().replace("+00:00", "Z"),
                "title": random.choice(topics),
                "impact": random.choice(impacts),
                "tag": random.choice(tags),
            }
        )

    return events
