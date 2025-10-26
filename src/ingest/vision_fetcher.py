"""Helpers for ensuring Binance Vision archive availability."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import httpx

LOGGER = logging.getLogger(__name__)

BASE_URLS = {
    "um": "https://data.binance.vision/data/futures/um/daily/klines",
}


def _build_url(market: str, symbol: str, interval: str, day: str) -> str:
    market_key = market.lower()
    base = BASE_URLS.get(market_key)
    if base is None:
        raise ValueError(f"Unsupported market '{market}' for Vision download (UM futures only)")
    symbol_part = symbol.upper()
    interval_part = interval.lower()
    return f"{base}/{symbol_part}/{interval_part}/{symbol_part}-{interval_part}-{day}.zip"


def _local_path(cache_dir: str, market: str, symbol: str, interval: str, day: str) -> Path:
    return (
        Path(cache_dir)
        / market.lower()
        / symbol.upper()
        / interval.lower()
        / f"{symbol.upper()}-{interval.lower()}-{day}.zip"
    )


async def _download_file(
    client: httpx.AsyncClient,
    url: str,
    destination: Path,
    *,
    timeout_s: int,
) -> bool:
    try:
        response = await client.get(url, timeout=timeout_s)
        if response.status_code == 200:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)
            LOGGER.info(
                "vision.fetcher.downloaded",
                extra={"url": url, "path": str(destination), "bytes": len(response.content)},
            )
            return True
        LOGGER.warning(
            "vision.fetcher.unexpected_status",
            extra={"url": url, "status": response.status_code},
        )
    except Exception as exc:  # pragma: no cover - network edge cases
        LOGGER.exception(
            "vision.fetcher.download_failed",
            extra={"url": url, "error": str(exc)},
        )
    return False


async def ensure_vision_days(
    symbol: str,
    interval: str,
    market: str,
    days: Iterable[str],
    cache_dir: str = "cache/vision",
    concurrency: int = 3,
    timeout_s: int = 20,
) -> List[str]:
    """Ensure Binance Vision daily archives exist locally for the requested days."""

    symbol_upper = symbol.upper()
    interval_lower = interval.lower()
    requested_days = list(dict.fromkeys(days))

    cache_paths: List[Path] = []
    missing: List[tuple[str, Path]] = []
    for day in requested_days:
        destination = _local_path(cache_dir, market, symbol_upper, interval_lower, day)
        if destination.exists():
            cache_paths.append(destination)
        else:
            missing.append((day, destination))

    if not missing:
        return [str(path) for path in cache_paths]

    semaphore = asyncio.Semaphore(max(1, concurrency))
    downloaded: List[Path] = []
    base_url_errors = 0

    async with httpx.AsyncClient() as client:
        async def worker(day_item: tuple[str, Path]) -> None:
            nonlocal base_url_errors
            day, destination = day_item
            url: str
            try:
                url = _build_url(market, symbol_upper, interval_lower, day)
            except ValueError:
                base_url_errors += 1
                LOGGER.error(
                    "vision.fetcher.unsupported_market",
                    extra={"market": market, "symbol": symbol_upper, "interval": interval_lower},
                )
                return
            async with semaphore:
                if await _download_file(client, url, destination, timeout_s=timeout_s):
                    downloaded.append(destination)

        await asyncio.gather(*(worker(item) for item in missing))

    all_paths = cache_paths + downloaded
    LOGGER.info(
        "vision.fetcher.summary",
        extra={
            "symbol": symbol_upper,
            "interval": interval_lower,
            "market": market,
            "requested_days": requested_days,
            "cached": len(cache_paths),
            "downloaded": len(downloaded),
            "failed": len(missing) - len(downloaded),
            "errors": base_url_errors,
        },
    )
    return [str(path) for path in all_paths if path.exists()]


__all__ = ["ensure_vision_days"]
