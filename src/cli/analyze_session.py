"""CLI wrapper for the fast session analyzer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.services.session_fast import analyze_session_fast, SessionDataUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse last closed UTC session quickly.")
    parser.add_argument("symbol", help="Symbol to analyse, e.g. BTCUSDT")
    parser.add_argument(
        "--root",
        dest="parquet_root",
        type=Path,
        help="Override UM ingest parquet root (defaults to var/um_ingest)",
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
    try:
        payload = analyze_session_fast(
            args.symbol,
            parquet_root=args.parquet_root,
        )
    except SessionDataUnavailable as exc:
        parser.error(str(exc))
        return 2
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    json_kwargs = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs["indent"] = 2
        json_kwargs["sort_keys"] = True
    output = json.dumps(payload, **json_kwargs)
    print(output)
    if len(output.encode("utf-8")) > 200_000:
        print("warning: payload exceeds 200KB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
