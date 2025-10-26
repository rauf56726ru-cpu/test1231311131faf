from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import AppConfig
from src.common.logging_setup import install_root_logging
from src.ingest.vision_fetcher import ensure_vision_days
from src.ingest.vision_materialize import materialize_klines_from_vision_zip

LOGGER = logging.getLogger("scripts.bootstrap_vision_cache")


def _parse_days(value: int, explicit: Iterable[str] | None = None) -> List[str]:
    today = datetime.now(timezone.utc).date()
    days: List[str] = []
    for offset in range(value):
        cursor = today - timedelta(days=offset + 1)
        days.append(cursor.isoformat())
    if explicit:
        for item in explicit:
            try:
                datetime.fromisoformat(item)
            except ValueError:
                LOGGER.warning("bootstrap.invalid_day", extra={"day": item})
                continue
            if item not in days:
                days.append(item)
    return sorted(set(days))


async def _bootstrap_symbol(
    symbol: str,
    *,
    interval: str,
    market: str,
    days: Iterable[str],
    cache_dir: str,
    data_dir: str,
) -> None:
    archives = await ensure_vision_days(
        symbol=symbol,
        interval=interval,
        market=market,
        days=days,
        cache_dir=cache_dir,
    )
    if not archives:
        LOGGER.warning(
            "bootstrap.no_archives",
            extra={"symbol": symbol.upper(), "market": market, "interval": interval},
        )
        return
    materialised: List[str] = []
    for archive in archives:
        created = materialize_klines_from_vision_zip(
            archive,
            symbol=symbol,
            interval=interval,
            market=market,
            data_dir=data_dir,
        )
        materialised.extend(created)
    LOGGER.info(
        "bootstrap.symbol_complete",
        extra={
            "symbol": symbol.upper(),
            "market": market,
            "interval": interval,
            "archives": len(archives),
            "parquet_files": len(materialised),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Binance Vision cache and Parquet storage.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval (default: 1m)")
    parser.add_argument("--market", default="um", choices=["um"], help="Market type (UM futures only)")
    parser.add_argument("--days", type=int, default=3, help="Number of past UTC days to fetch (default: 3)")
    parser.add_argument("--cache-dir", default="cache/vision", help="Vision cache directory")
    parser.add_argument(
        "--day",
        action="append",
        dest="explicit_days",
        help="Explicit YYYY-MM-DD to download (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    install_root_logging()
    args = parse_args()
    cfg = AppConfig.load()

    target_days = _parse_days(max(1, args.days), args.explicit_days)
    LOGGER.info(
        "bootstrap.start",
        extra={
            "symbols": args.symbols,
            "interval": args.interval,
            "market": args.market,
            "days": target_days,
            "cache_dir": args.cache_dir,
            "data_dir": cfg.data_dir,
        },
    )

    async def runner() -> None:
        await asyncio.gather(
            *(
                _bootstrap_symbol(
                    symbol=symbol,
                    interval=args.interval,
                    market=args.market,
                    days=target_days,
                    cache_dir=args.cache_dir,
                    data_dir=cfg.data_dir,
                )
                for symbol in args.symbols
            )
        )

    asyncio.run(runner())
    LOGGER.info("bootstrap.complete")


if __name__ == "__main__":
    main()
