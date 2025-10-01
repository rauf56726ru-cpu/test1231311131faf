"""News feed aggregator for Binance assets."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import httpx

LOGGER = logging.getLogger(__name__)
COINGECKO_EVENTS = "https://api.coingecko.com/api/v3/events"
_NEWS_CACHE: Dict[str, Tuple[datetime, List[Dict[str, object]]]] = {}


def _symbol_to_keyword(symbol: str) -> str:
    base = symbol.upper().replace("USDT", "").replace("BUSD", "")
    mapping = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    return mapping.get(base, base.lower())


async def _fetch_coingecko_events(keyword: str, window_hours: int) -> List[Dict[str, object]]:
    now = datetime.now(timezone.utc)
    min_time = now - timedelta(hours=window_hours)
    params = {"upcoming_events_only": "false", "page": "1", "type": "event"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        response = await client.get(COINGECKO_EVENTS, params=params)
        response.raise_for_status()
        payload = response.json()
    entries = payload.get("data", []) if isinstance(payload, dict) else []
    events: List[Dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "")
        description = str(entry.get("description") or "")
        if keyword.lower() not in (title + " " + description).lower():
            continue
        start_time = entry.get("start_date") or entry.get("start_date_detail", {}).get("date")
        if not start_time:
            continue
        try:
            event_time = datetime.fromisoformat(str(start_time))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            event_time = event_time.astimezone(timezone.utc)
        except ValueError:
            continue
        if event_time < min_time:
            continue
        events.append(
            {
                "title": title,
                "source": entry.get("screenshot") or entry.get("website"),
                "time_utc": event_time.isoformat().replace("+00:00", "Z"),
                "impact": (entry.get("importance") or "low").lower(),
            }
        )
        if len(events) >= 10:
            break
    return events


async def fetch_news(symbol: str, window_hours: int) -> List[Dict[str, object]]:
    """Fetch crypto events for the requested symbol within the time window."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    cache_key = f"{symbol_clean}:{window_hours}"
    now = datetime.now(timezone.utc)
    cached = _NEWS_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    keyword = _symbol_to_keyword(symbol_clean)
    try:
        events = await _fetch_coingecko_events(keyword, window_hours)
    except Exception as exc:  # pragma: no cover - external dependency guard
        LOGGER.warning("Falling back to empty news feed: %s", exc)
        events = []

    expiry = now + timedelta(hours=24)
    _NEWS_CACHE[cache_key] = (expiry, events)
    return events
