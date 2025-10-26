#!/usr/bin/env python3
"""Benchmark 1m quick analysis pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.api.quick_analyze import main as quick_analyze_main
from src.common.logging_setup import install_root_logging

LOGGER = logging.getLogger("benchmark.1m")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the 1m quick analysis pipeline")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="Comma-separated list of symbols")
    parser.add_argument("--window", default="72h", help="Window size (default: 72h)")
    parser.add_argument("--out", type=Path, default=Path("out/benchmark"), help="Output directory")
    parser.add_argument("--market", help="Override market")
    parser.add_argument("--llm-providers", default="none", help="LLM providers to invoke (default: none)")
    parser.add_argument("--timeout-ms", type=int, default=40000, help="Overall timeout for providers")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    install_root_logging()
    args = parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    payload_path = quick_analyze_main(
        [
            "--symbols",
            args.symbols,
            "--interval",
            "1m",
            "--window",
            args.window,
            "--out",
            str(args.out),
            "--llm-providers",
            args.llm_providers,
            "--timeout-ms",
            str(args.timeout_ms),
            *(["--market", args.market] if args.market else []),
        ]
    )
    total_ms = int((time.perf_counter() - start) * 1000)

    payload = json.loads(payload_path.read_text())
    zones_info = payload.get("zones", {})
    sessions_info = payload.get("sessions", {})
    artifacts = payload.get("artifacts", {})
    llm_info = payload.get("llm", {})

    benchmark_log = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbols": payload.get("symbols"),
        "window_hours": payload.get("window_hours"),
        "zones_count": sum(len(items) for items in zones_info.values()),
        "sessions_count": len(sessions_info),
        "artifacts": artifacts,
        "llm_status": llm_info.get("status"),
        "total_ms": total_ms,
    }

    LOGGER.info("benchmark.result", extra=benchmark_log)
    print(
        f"symbols={benchmark_log['symbols']} window={benchmark_log['window_hours']}h "
        f"zones={benchmark_log['zones_count']} sessions={benchmark_log['sessions_count']} "
        f"total_ms={benchmark_log['total_ms']} llm_status={benchmark_log['llm_status']}"
    )


if __name__ == "__main__":
    main()

