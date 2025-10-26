from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.common.logging_setup import install_root_logging
from src.common.config import AppConfig
from src.storage.parquet import ParquetStorage
from src.services.zones_72h_service import collect_open_zones
from src.services.session_last_service import collect_last_sessions

LOGGER = logging.getLogger("zones.72h")


def _coerce_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise argparse.ArgumentTypeError("symbol cannot be empty")
    return cleaned


def _parse_symbols(values: Sequence[str]) -> Sequence[str]:
    cleaned = [_coerce_symbol(symbol) for symbol in values if symbol]
    if not cleaned:
        raise argparse.ArgumentTypeError("at least one symbol must be provided")
    return cleaned


def _parse_timestamp(value: str) -> datetime:
    lowered = value.strip().lower()
    if lowered == "now":
        return datetime.now(timezone.utc)
    if lowered.isdigit():
        return datetime.fromtimestamp(int(lowered) / 1000, tz=timezone.utc)
    normalised = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect open supply/demand zones over the last 72 hours")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to analyse, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--hours", type=int, default=72, help="Window length in hours (default: 72)")
    parser.add_argument(
        "--end",
        default="now",
        help="Window end timestamp (ms or ISO8601). Defaults to 'now'.",
    )
    parser.add_argument(
        "--market",
        type=str,
        help="Override storage market (defaults to AppConfig market)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("zones_72h.json"),
        help="Output JSON Lines path (default: zones_72h.json)",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        help="Explicit tick size to use for all symbols (optional, inferred otherwise)",
    )
    parser.add_argument(
        "--session-out",
        type=Path,
        default=Path("session_last.json"),
        help="Output JSON Lines path for session metrics (default: session_last.json)",
    )
    return parser.parse_args()


def main() -> None:
    install_root_logging()
    args = parse_args()

    symbols = _parse_symbols(args.symbols)
    hours = max(1, args.hours)
    end_dt = _parse_timestamp(args.end)
    end_ms = _to_ms(end_dt)
    start_ms = end_ms - hours * 60 * 60_000

    storage = None
    if args.market:
        app_config = AppConfig.load()
        storage = ParquetStorage(
            root=app_config.data_dir,
            market=args.market,
            index_path=app_config.duckdb_path,
        )

    tick_override = {symbol: args.tick_size for symbol in symbols} if args.tick_size else None
    start_time = time.perf_counter()
    zones = collect_open_zones(
        symbols,
        end_ms=end_ms,
        hours=hours,
        storage=storage,
        tick_size_overrides=tick_override,
        export_path=args.out,
    )
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    LOGGER.info(
        "zones.72h.complete",
        extra={"symbols": symbols, "zones_detected": len(zones), "elapsed_ms": elapsed_ms, "output": str(args.out)},
    )

    session_metrics = collect_last_sessions(
        symbols,
        end_ms=end_ms,
        storage=storage,
        zones=zones,
        export_path=args.session_out,
    )
    LOGGER.info(
        "session.last.complete",
        extra={"symbols": symbols, "sessions": len(session_metrics), "output": str(args.session_out)},
    )


if __name__ == "__main__":
    main()
