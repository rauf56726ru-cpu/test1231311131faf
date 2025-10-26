"""CLI helper to build the 72h zone context payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.services.zones_context import build_zones_context, ZoneDetectionError, ZoneCache
from src.storage.parquet import ParquetStorage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute 72h zone context for a symbol.")
    parser.add_argument("symbol", help="Symbol to analyse, e.g. BTCUSDT")
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Lookback window in hours (default: 72)",
    )
    parser.add_argument(
        "--top",
        dest="top_n",
        type=int,
        default=20,
        help="Number of zones to return (default: 20)",
    )
    parser.add_argument(
        "--root",
        dest="parquet_root",
        type=Path,
        help="Override parquet root directory",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    storage = ParquetStorage(root=args.parquet_root) if args.parquet_root else None

    try:
        payload = build_zones_context(
            args.symbol,
            storage=storage,
            hours=args.hours,
            top_n=args.top_n,
            cache=ZoneCache(),
        )
    except ZoneDetectionError as exc:
        parser.error(str(exc))
        return 2
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    json_kwargs = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs["indent"] = 2
        json_kwargs["sort_keys"] = True

    print(json.dumps(payload, **json_kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
