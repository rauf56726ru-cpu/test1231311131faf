"""Command-line entrypoint for Binance Vision ingestion."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Sequence

from src.common.logging_setup import install_root_logging
from src.services.vision_ingest import ingest_binance_vision

LOGGER = logging.getLogger("ingest.binance_vision")


def _parse_timestamp(value: str) -> datetime:
    lowered = value.strip().lower()
    if lowered == "now":
        return datetime.now(timezone.utc)
    if lowered.isdigit():
        return datetime.fromtimestamp(int(lowered) / 1000, tz=timezone.utc)
    normalised = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_ms(moment: datetime) -> int:
    return int(moment.astimezone(timezone.utc).timestamp() * 1000)


def _normalise_datasets(datasets: Sequence[str] | None) -> Sequence[str] | None:
    if not datasets:
        return None
    cleaned = []
    for entry in datasets:
        if not entry:
            continue
        cleaned.append(entry.strip())
    return tuple(cleaned) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Binance Vision archives into local storage")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTCUSDT")
    parser.add_argument(
        "--end",
        default="now",
        help="End timestamp (ms or ISO8601). Use 'now' to align with current UTC time.",
    )
    parser.add_argument(
        "--start",
        help="Start timestamp (ms or ISO8601). Defaults to end - hours.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Window length in hours when --start is omitted",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of datasets to ingest (default: project presets)",
    )
    parser.add_argument(
        "--klines-intervals",
        nargs="*",
        help="Intervals for kline ingestion when klines dataset is included",
    )
    parser.add_argument(
        "--skip-exchange-info",
        action="store_true",
        help="Do not refresh the exchange info dataset",
    )
    parser.add_argument(
        "--vision-cache",
        type=str,
        default="cache/vision",
        help="Directory containing Binance Vision cache archives",
    )
    return parser.parse_args()


async def _run_ingest(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    datasets: Sequence[str] | None,
    intervals: Sequence[str] | None,
    include_exchange_info: bool,
) -> dict[str, object]:
    return await ingest_binance_vision(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        datasets=datasets,
        klines_intervals=intervals,
        include_exchange_info=include_exchange_info,
        source="cli.binance_ingest",
    )


def main() -> None:
    install_root_logging()
    args = parse_args()
    symbol = args.symbol.upper()

    if args.vision_cache:
        existing = os.environ.get("FIXTURE_DIRS")
        cache_entry = args.vision_cache
        if existing:
            dirs = [entry.strip() for entry in existing.split(",") if entry.strip()]
            if cache_entry not in dirs:
                dirs.append(cache_entry)
            os.environ["FIXTURE_DIRS"] = ",".join(dirs)
        else:
            os.environ["FIXTURE_DIRS"] = cache_entry

    end_dt = _parse_timestamp(args.end)
    if args.start:
        start_dt = _parse_timestamp(args.start)
    else:
        start_dt = end_dt - timedelta(hours=max(args.hours, 1))

    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)
    datasets = _normalise_datasets(args.datasets)
    intervals = _normalise_datasets(args.klines_intervals)
    include_exchange_info = not args.skip_exchange_info

    LOGGER.info(
        "vision.ingest.cli.start",
        extra={
            "symbol": symbol,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "datasets": datasets,
            "intervals": intervals,
            "include_exchange_info": include_exchange_info,
        },
    )

    try:
        summary = asyncio.run(
            _run_ingest(
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                datasets=datasets,
                intervals=intervals,
                include_exchange_info=include_exchange_info,
            )
        )
    except Exception:
        LOGGER.exception("vision.ingest.cli.error", extra={"symbol": symbol})
        raise SystemExit(1)

    status = summary.get("status")
    LOGGER.info(
        "vision.ingest.cli.complete",
        extra={"symbol": symbol, "status": status, "summary": summary},
    )

    if status != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
