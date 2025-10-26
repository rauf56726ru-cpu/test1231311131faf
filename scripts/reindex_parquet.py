#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from src.common.logging_setup import install_root_logging
from src.storage import ParquetStorage

LOGGER = logging.getLogger("scripts.reindex_parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild DuckDB index for Parquet OHLCV partitions.")
    parser.add_argument("--root", default="data", help="Root directory containing Parquet partitions.")
    parser.add_argument("--market", default="futures_um", help="Market segment inside the storage root.")
    parser.add_argument(
        "--index",
        default=None,
        help="Override path to DuckDB index (defaults to <root>/../meta/index.duckdb).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_root_logging()
    start = perf_counter()
    storage = ParquetStorage(
        root=Path(args.root),
        market=args.market,
        index_path=Path(args.index) if args.index else None,
    )
    storage.rebuild_index()
    elapsed_ms = int((perf_counter() - start) * 1000)
    LOGGER.info(
        "reindex.complete",
        extra={"root": str(args.root), "market": args.market, "elapsed_ms": elapsed_ms},
    )


if __name__ == "__main__":
    main()
