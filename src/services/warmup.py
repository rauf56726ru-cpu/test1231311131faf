"""Utility helpers to pre-warm inspection caches before starting the API."""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Iterable, Sequence

from datetime import datetime, timezone

from .inspection_cache_service import ensure_inspection_daily_cache

# We re-use the internal summary collector to materialise the three-day snapshot.
# The helper lives in ``src.api.app``; importing lazily avoids heavy dependencies
# when the warmup module is not used from scripts.

LOGGER = logging.getLogger(__name__)


async def warmup_three_day_context(
    symbols: Sequence[str],
    *,
    days: int = 3,
    timeout: float | None = 600.0,
    allow_network_summary: bool = False,
) -> None:
    """Pre-fetch Binance Vision archives and build the inspection summary cache.

    Parameters
    ----------
    symbols:
        Iterable of symbols (e.g. ["BTCUSDT"]).
    days:
        Number of calendar days to backfill. Defaults to 3.
    timeout:
        Per-day timeout forwarded to ``ensure_inspection_daily_cache``.
    allow_network_summary:
        Whether the summary pipeline is allowed to hit Binance REST after the
        archives are cached. Defaults to ``False`` so the warmup works fully
        offline once archives are in place.
    """

    if not symbols:
        raise ValueError("At least one symbol is required")

    from src.api.app import _collect_summary_for_symbol  # type: ignore

    for symbol in symbols:
        symbol_clean = (symbol or "").strip().upper()
        if not symbol_clean:
            continue
        LOGGER.info("warmup.start", extra={"symbol": symbol_clean, "days": days})
        await ensure_inspection_daily_cache(
            symbol_clean,
            days=days,
            network_backfill=True,
            collection_timeout=timeout,
        )
        summary_payload, _, _ = await _collect_summary_for_symbol(
            symbol_clean,
            days=days,
            now_override=None,
            progress=None,
            allow_network=allow_network_summary,
        )
        LOGGER.info(
            "warmup.complete",
            extra={
                "symbol": symbol_clean,
                "days": days,
                "summary_ready": bool(summary_payload),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-warm inspection caches")
    parser.add_argument("--symbols", required=True, help="Comma-separated list of symbols (e.g. BTCUSDT,ETHUSDT)")
    parser.add_argument("--days", type=int, default=3, help="Number of days to backfill (default: 3)")
    parser.add_argument(
        "--summary-network",
        action="store_true",
        help="Allow the summary stage to access Binance REST after the archives are cached",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-symbol timeout in seconds for archive collection (default: 600)",
    )
    return parser.parse_args(argv)


def _split_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


async def _async_main(args: argparse.Namespace) -> None:
    symbols = _split_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No valid symbols provided")
    await warmup_three_day_context(
        symbols,
        days=max(1, args.days),
        timeout=args.timeout,
        allow_network_summary=bool(args.summary_network),
    )


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
