from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.common.logging_setup import install_root_logging
from src.services.cache import FileCache
from src.services.smc72_pipeline import collect_data, write_outputs

LOGGER = logging.getLogger("smc72")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMC 72h pipeline")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTCUSDT")
    parser.add_argument("--t0", help="End timestamp ISO8601 (UTC) or 'now'", default="now")
    parser.add_argument("--hours", type=int, default=72, help="Window length in hours")
    parser.add_argument("--out", type=Path, default=Path("./out"), help="Output directory")
    parser.add_argument("--archives", type=Path, help="Path to Binance Vision archives", default=None)
    parser.add_argument("--save-parquet", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--rps", type=int, default=8, help="Target requests per second")
    parser.add_argument("--retry", type=int, default=7, help="Max retry attempts")
    parser.add_argument("--strict", action="store_true", help="Fail if coverage thresholds are missed")
    parser.add_argument(
        "--ingest-policy",
        choices=["LOCAL_ONLY", "LOCAL_THEN_REMOTE"],
        help="Override ingestion policy"
    )
    parser.add_argument("--market", choices=["um"], default="um", help="Data market to use (UM futures only)")
    parser.add_argument(
        "--bootstrap-vision",
        action="store_true",
        help="Bootstrap Binance Vision cache before running"
    )
    parser.add_argument(
        "--bootstrap-days",
        type=int,
        default=3,
        help="Number of historical UTC days to download when bootstrapping"
    )
    parser.add_argument(
        "--vision-cache",
        type=Path,
        default=Path("cache/vision"),
        help="Directory containing Binance Vision cache"
    )
    return parser.parse_args()


async def run_pipeline(args: argparse.Namespace) -> int:
    symbol = args.symbol.upper()
    if args.t0 == "now":
        t0_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    else:
        parsed = datetime.fromisoformat(args.t0.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        t0_ms = int(parsed.timestamp() * 1000)

    cache = FileCache(".cache/live")
    LOGGER.info("Collecting data for %s", symbol)
    payload = await collect_data(symbol=symbol, t0_ms=t0_ms, hours=args.hours, cache=cache)
    minutes = payload["minutes"]
    diagnostics = payload["diagnostics"]
    coverage = payload.get("coverage", {})
    depth_checks = payload.get("depth_checks", [])
    window_info = payload.get("window", {"start_ms": t0_ms - args.hours * 60 * 60_000, "end_ms": t0_ms})
    window_start = window_info.get("start_ms", t0_ms)
    window_end = window_info.get("end_ms", t0_ms)

    expected_minutes = args.hours * 60
    reasons: list[str] = []
    exit_code = 0

    if len(minutes) != expected_minutes:
        reasons.append(f"minutes mismatch: expected {expected_minutes}, got {len(minutes)}")
        exit_code = 3

    funding_points = [row for row in minutes if row.get("lastFundingRate") is not None]
    if len(funding_points) < 7:
        reasons.append(f"funding points below 7 (actual {len(funding_points)})")
        exit_code = max(exit_code, 2)

    if len(depth_checks) != 3:
        reasons.append(f"depth_checks count != 3 (actual {len(depth_checks)})")
        exit_code = max(exit_code, 2)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    valid_labels = {"start", "mid", "end"}
    labels_seen = set()
    for entry in depth_checks:
        label = entry.get("label")
        label_ts = entry.get("label_ts")
        ts_req = int(entry.get("ts_req", entry.get("ts", now_ms)))
        if label in valid_labels:
            labels_seen.add(label)
        if label_ts is None:
            reasons.append(f"depth label {label} missing label_ts")
            exit_code = max(exit_code, 2)
        if abs(ts_req - now_ms) > 5_000:
            reasons.append(
                f"depth {label} ts_req too far from now (delta {abs(ts_req - now_ms)} ms)"
            )
            exit_code = max(exit_code, 2)
    if len(labels_seen) != 3:
        reasons.append(f"depth labels missing expected set {valid_labels}")
        exit_code = max(exit_code, 2)

    if exit_code != 3 and (args.save_parquet or args.save_json):
        try:
            paths = write_outputs(
                recordset=minutes,
                out_dir=args.out,
                diagnostics=diagnostics,
                symbol=symbol,
                window_start=window_start,
                window_end=window_end,
                depth_checks=depth_checks,
                coverage=coverage,
            )
        except ValueError as exc:
            print(f"WHY: {exc}", file=sys.stderr)
            return 3
        if args.save_parquet:
            LOGGER.info("Saved Parquet: %s", paths["parquet"])
        if args.save_json:
            LOGGER.info("Saved summary: %s", paths["summary"])

    LOGGER.info(
        "Coverage: klines=%.2f%% aggTrades=%.2f%% recon_mismatch=%.2f%% vwap_oob=%.2f%%",
        coverage.get("klines_pct", 0.0),
        coverage.get("aggtrades_pct", 0.0),
        coverage.get("recon_mismatch_pct", diagnostics.recon_mismatch_pct or 0.0),
        coverage.get("vwap_oob_pct", diagnostics.vwap_oob_pct or 0.0),
    )
    LOGGER.info(
        "Diagnostics: rate_limit_events=%s retries=%s gaps=%s cvd_baseline=%s source=%s",
        diagnostics.rate_limit_events,
        diagnostics.retries,
        len(diagnostics.gaps),
        diagnostics.cvd_baseline,
        diagnostics.cvd_baseline_source,
    )

    klines_pct = coverage.get("klines_pct", 0.0)
    agg_pct = coverage.get("aggtrades_pct", 0.0)
    recon_pct = coverage.get("recon_mismatch_pct", diagnostics.recon_mismatch_pct or 0.0)
    vwap_pct = coverage.get("vwap_oob_pct", diagnostics.vwap_oob_pct or 0.0)
    rate_limit_events = diagnostics.rate_limit_events
    min_coverage = min(klines_pct, agg_pct)

    if args.strict:
        if klines_pct < 98.0:
            reasons.append(f"klines_pct below 98% (actual {klines_pct:.2f}%)")
            exit_code = max(exit_code, 3)
        if agg_pct < 98.0:
            reasons.append(f"aggtrades_pct below 98% (actual {agg_pct:.2f}%)")
            exit_code = max(exit_code, 3)
        if recon_pct > 2.0:
            reasons.append(f"recon_mismatch_pct above 2% (actual {recon_pct:.2f}%)")
            exit_code = max(exit_code, 3)
        if vwap_pct > 1.0:
            reasons.append(f"vwap_oob_pct above 1% (actual {vwap_pct:.2f}%)")
            exit_code = max(exit_code, 3)
    else:
        if klines_pct < 98.0:
            reasons.append(f"klines_pct below 98% (actual {klines_pct:.2f}%)")
            exit_code = max(exit_code, 2)
        if agg_pct < 98.0:
            reasons.append(f"aggtrades_pct below 98% (actual {agg_pct:.2f}%)")
            exit_code = max(exit_code, 2)
        if recon_pct > 2.0:
            reasons.append(f"recon_mismatch_pct above 2% (actual {recon_pct:.2f}%)")
            exit_code = max(exit_code, 2)
        if vwap_pct > 1.0:
            reasons.append(f"vwap_oob_pct above 1% (actual {vwap_pct:.2f}%)")
            exit_code = max(exit_code, 2)

    if rate_limit_events > 0 and min_coverage < 99.0:
        reasons.append(
            f"rate_limit_events={rate_limit_events} with coverage < 99% (min coverage {min_coverage:.2f}%)"
        )
        if exit_code < 3:
            exit_code = max(exit_code, 2)

    for reason in reasons:
        print(f"WHY: {reason}", file=sys.stderr)

    return exit_code


def main() -> None:
    install_root_logging()
    args = parse_args()

    os.environ.setdefault("MARKET", args.market)

    if args.ingest_policy:
        os.environ["INGEST_POLICY"] = args.ingest_policy

    if args.bootstrap_vision:
        cmd = [
            sys.executable,
            "-m",
            "scripts.bootstrap_vision_cache",
            "--symbols",
            args.symbol,
            "--interval",
            "1m",
            "--market",
            args.market,
            "--days",
            str(max(1, args.bootstrap_days)),
            "--cache-dir",
            str(args.vision_cache),
        ]
        subprocess.run(cmd, check=True)

    try:
        exit_code = asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
