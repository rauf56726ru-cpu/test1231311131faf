from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Sequence

import uvloop

from src.common.logging_setup import install_root_logging
from src.ingest.binance_stream import BinanceFuturesStreamIngestor, BinanceStreamConfig

LOGGER = logging.getLogger("binance.stream")


def _parse_symbols(values: Sequence[str]) -> Sequence[str]:
    cleaned = [_coerce_symbol(symbol) for symbol in values if symbol]
    if not cleaned:
        raise argparse.ArgumentTypeError("at least one symbol must be provided")
    return cleaned


def _coerce_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise argparse.ArgumentTypeError("symbol cannot be empty")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance Futures realtime ingestion")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Trading symbols (e.g. BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=1.0,
        help="Micro-batch flush interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=72,
        help="Ring buffer retention window in hours (default: 72)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10_000,
        help="Max pending websocket messages (default: 10000)",
    )
    parser.add_argument(
        "--include-agg-trades",
        action="store_true",
        help="Subscribe to aggTrade streams in addition to klines",
    )
    parser.add_argument(
        "--websocket-url",
        type=str,
        default="wss://fstream.binance.com/stream",
        help="Binance Futures websocket endpoint",
    )
    parser.add_argument(
        "--rest-timeout",
        type=float,
        default=10.0,
        help="REST request timeout in seconds",
    )
    parser.add_argument(
        "--rest-retries",
        type=int,
        default=5,
        help="Maximum REST retry attempts",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    symbols = _parse_symbols(args.symbols)
    config = BinanceStreamConfig(
        symbols=symbols,
        interval="1m",
        flush_interval_seconds=max(0.1, args.flush_interval),
        window_hours=max(1, int(args.window_hours)),
        websocket_url=args.websocket_url,
        include_agg_trades=bool(args.include_agg_trades),
        queue_size=max(100, args.queue_size),
        rest_timeout=max(1.0, float(args.rest_timeout)),
        rest_max_retries=max(1, int(args.rest_retries)),
    )

    ingestor = BinanceFuturesStreamIngestor(config)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_shutdown() -> None:
        LOGGER.info("Shutdown requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(sig, lambda *_: stop_event.set())

    await ingestor.start()
    LOGGER.info("Binance futures stream running for symbols: %s", ", ".join(symbols))
    try:
        await stop_event.wait()
    finally:
        await ingestor.stop()
        LOGGER.info("Binance futures stream stopped")


def main() -> None:
    install_root_logging()
    uvloop.install()
    args = parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:  # pragma: no cover - signal path
        LOGGER.info("Interrupted by user")


if __name__ == "__main__":
    main()

