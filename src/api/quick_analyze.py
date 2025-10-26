from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from src.common.logging_setup import install_root_logging
from src.common.config import AppConfig
from src.storage.parquet import ParquetStorage
from src.services.zones_72h_service import collect_open_zones
from src.services.session_last_service import collect_last_sessions
from src.analysis.zones_72h import Zone
from src.analysis.session_last import SessionMetrics
from src.analysis.providers import build_http_providers, run_providers

LOGGER = logging.getLogger("quick.analyze")


def _parse_symbols(value: str) -> Sequence[str]:
    parts = []
    for chunk in value.split(","):
        symbol = chunk.strip().upper()
        if symbol:
            parts.append(symbol)
    if not parts:
        raise argparse.ArgumentTypeError("At least one symbol must be provided")
    return parts


def _parse_window(value: str) -> int:
    cleaned = value.strip().lower()
    if cleaned.endswith("h"):
        cleaned = cleaned[:-1]
    if not cleaned.isdigit():
        raise argparse.ArgumentTypeError("Window must be specified in hours, e.g. 72h")
    hours = int(cleaned)
    if hours <= 0:
        raise argparse.ArgumentTypeError("Window must be positive")
    return hours


def _parse_timestamp(value: str | None) -> int:
    if value is None:
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "now":
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    if cleaned.isdigit():
        return int(cleaned)
    parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick 72h analysis and export payload for LLM consumption.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--interval", default="1m", help="Candle interval (only 1m is supported)")
    parser.add_argument("--window", default="72h", help="Window length in hours, e.g. 72h")
    parser.add_argument("--end", help="Optional end timestamp (ms or ISO8601, defaults to now)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory path")
    parser.add_argument("--llm-providers", default="none", help="Comma-separated LLM target endpoints or 'none'")
    parser.add_argument("--timeout-ms", type=int, default=40000, help="LLM request timeout placeholder")
    parser.add_argument("--market", help="Override storage market when loading candles")
    parser.add_argument("--zone-export", type=Path, help="Optional path to write zones_72h.json")
    parser.add_argument("--session-export", type=Path, help="Optional path to write session_last.json")
    return parser.parse_args(argv)


def _zones_to_dict(zones: Iterable[Zone]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for zone in zones:
        grouped.setdefault(zone.symbol, []).append(zone.to_dict())
    return grouped


def _sessions_to_dict(sessions: Iterable[SessionMetrics]) -> dict[str, dict]:
    return {entry.symbol: entry.to_dict() for entry in sessions}


def _resolve_llm_providers(raw: str) -> list[str]:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() == "none":
        return []
    return [entry.strip() for entry in cleaned.split(",") if entry.strip()]


def main(argv: Sequence[str] | None = None) -> Path:
    install_root_logging()
    args = parse_args(argv)

    symbols = _parse_symbols(args.symbols)
    interval = args.interval.strip().lower()
    if interval != "1m":
        raise ValueError("Only 1m interval is supported in quick analyze")

    window_hours = _parse_window(args.window)
    end_ms = _parse_timestamp(args.end)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    app_config = AppConfig.load()
    storage = None
    if args.market:
        storage = ParquetStorage(
            root=app_config.data_dir,
            market=args.market,
            index_path=app_config.duckdb_path,
        )

    zone_export_path = args.zone_export or (out_dir / "zones_72h.json")
    session_export_path = args.session_export or (out_dir / "session_last.json")

    pipeline_start = time.perf_counter()

    zones = collect_open_zones(
        symbols,
        end_ms=end_ms,
        hours=window_hours,
        storage=storage,
        export_path=zone_export_path,
    )
    sessions = collect_last_sessions(
        symbols,
        end_ms=end_ms,
        storage=storage,
        zones=zones,
        export_path=session_export_path,
    )

    zones_by_symbol = _zones_to_dict(zones)
    sessions_by_symbol = _sessions_to_dict(sessions)
    zone_ids = [zone.id for zone in zones]
    provider_specs = _resolve_llm_providers(args.llm_providers)

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "symbols": symbols,
        "interval": interval,
        "window_hours": window_hours,
        "end_ms": end_ms,
        "zones": zones_by_symbol,
        "sessions": sessions_by_symbol,
        "llm": {
            "providers": provider_specs,
            "timeout_ms": args.timeout_ms,
            "status": "pending",
            "results": [],
        },
        "artifacts": {
            "zones_path": str(zone_export_path),
            "session_path": str(session_export_path),
        },
    }

    providers = build_http_providers(provider_specs, default_timeout_ms=args.timeout_ms)
    summary_entries: list[dict] = []
    summary_path: Path | None = None

    if providers:
        metric_keys = ["vwap", "atr", "volume", "taker_buy_delta", "range", "body_ratio"]
        summary_entries = asyncio.run(
            run_providers(
                providers,
                payload,
                output_dir=out_dir,
                timestamp=int(end_ms),
                zone_ids=zone_ids,
                metric_keys=metric_keys,
                overall_timeout_ms=args.timeout_ms,
            )
        )
        summary_path = out_dir / f"analysis_summary_{int(end_ms)}.json"
        summary_path.write_text(json.dumps(summary_entries, ensure_ascii=False, indent=2))
        payload["llm"]["providers"] = [provider.name for provider in providers]
        payload["llm"]["status"] = "completed"
        payload["llm"]["results"] = summary_entries
        payload["artifacts"]["analysis_summary_path"] = str(summary_path)
    else:
        payload["llm"]["providers"] = provider_specs
        payload["llm"]["status"] = "disabled" if not provider_specs else "unavailable"

    payload_path = out_dir / f"payload_{int(end_ms)}.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    pipeline_total_ms = int((time.perf_counter() - pipeline_start) * 1000)
    LOGGER.info(
        "quick_analyze.complete",
        extra={
            "symbols": symbols,
            "pipeline_total_ms": pipeline_total_ms,
            "zones": sum(len(entries) for entries in zones_by_symbol.values()),
            "sessions": len(sessions_by_symbol),
            "providers": len(providers),
            "payload": str(payload_path),
            "analysis_summary": str(summary_path) if summary_path else None,
        },
    )

    return payload_path


if __name__ == "__main__":
    main()
