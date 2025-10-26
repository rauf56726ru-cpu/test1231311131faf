#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.common.logging_setup import install_root_logging
from src.storage import JSONStorage, ParquetStorage

LOGGER = logging.getLogger("scripts.json_to_parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert legacy JSON OHLCV archives into Parquet storage.")
    parser.add_argument("--json-root", default="data", help="Root directory containing legacy JSON partitions.")
    parser.add_argument("--parquet-root", default="data", help="Destination root for Parquet partitions.")
    parser.add_argument("--market", default="futures_um", help="Market segment to convert.")
    return parser.parse_args()


def _discover_json_files(root: Path, market: str) -> list[Path]:
    base = root / market
    if not base.exists():
        LOGGER.warning("JSON root does not exist", extra={"base": str(base)})
        return []
    return sorted(base.rglob("*.json"))


def _normalise_rows(payload: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in JSONStorage._normalise_payload(payload):  # type: ignore[attr-defined]
        normalised = JSONStorage._normalise_row(entry)  # type: ignore[attr-defined]
        if normalised is not None:
            rows.append(normalised)
    return rows


def main() -> None:
    args = parse_args()
    install_root_logging()

    json_root = Path(args.json_root)
    parquet_root = Path(args.parquet_root)
    market = args.market

    json_files = _discover_json_files(json_root, market)
    if not json_files:
        LOGGER.info("No JSON files discovered for conversion", extra={"json_root": str(json_root), "market": market})
        return

    parquet_storage = ParquetStorage(root=parquet_root, market=market)
    converted = 0
    rows_written = 0

    for json_file in json_files:
        relative_parts = json_file.relative_to(json_root).parts
        if len(relative_parts) < 6:
            LOGGER.warning("Skipping unexpected layout", extra={"path": str(json_file)})
            continue
        symbol = relative_parts[-5]
        interval = relative_parts[-4]
        try:
            raw = json_file.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.error("Failed to read JSON file", extra={"path": str(json_file), "error": str(exc)})
            continue

        payload = json.loads(raw) if raw else []
        rows = _normalise_rows(payload)
        if not rows:
            LOGGER.debug("Skipping empty JSON file", extra={"path": str(json_file)})
            continue

        stats = parquet_storage.write_rows(symbol, interval, rows)
        converted += 1
        rows_written += sum(item.rows for item in stats)
        LOGGER.info(
            "converted.json_file",
            extra={
                "path": str(json_file),
                "symbol": symbol,
                "interval": interval,
                "rows": sum(item.rows for item in stats),
            },
        )

    LOGGER.info(
        "conversion.complete",
        extra={
            "json_root": str(json_root),
            "parquet_root": str(parquet_root),
            "market": market,
            "files_converted": converted,
            "rows_written": rows_written,
        },
    )


if __name__ == "__main__":
    main()
