# Changelog

## Unreleased

- Added quick analysis CLI (`python -m src.api.quick_analyze`) that orchestrates 72h zones, last session metrics, JSONL exports, and LLM payloads with end-to-end timing logs.
- Exposed `/zones/72h` and `/sessions/last` FastAPI endpoints for lightweight zone/session snapshots with optional JSONL exports.
- Integrated asynchronous LLM providers with HTTP adapter, timeouts, JSON exports, and resilient logging.
- Added multi-timeframe OHLCV ingestion with Binance integration and caching.
- Introduced orderflow footprint, CVD, derivatives, liquidity map, orderbook and news services.
- Enhanced inspection API with validation, analysis batching and dashboard data.
- Refreshed frontend with modal confirmation, progress indicators, dashboard, and export tools.
- Updated dependencies to include python-binance and pydantic-settings for data sourcing and configuration.
- Optimised the strict offline pipeline: proxy orderflow now consumes the reconstructed minute window, keeps a 12h tail for aggregates, and reports detailed coverage diagnostics.
- Vision-store powered fetches favour local caches before hitting Binance, including footprint snapshots that hydrate from the latest stored trades.
- Added a minimal asyncio bridge for integration tests so they run without `pytest-asyncio`; documented the offline execution flow and ignored `.venv/` in git.
- Added rotating file logging (`var/logs/pipeline.log` by default) so console traces are mirrored on disk; configurable via `PIPELINE_LOG_DIR`.
- Derivatives now backfill funding/liquidation/open interest из Binance Vision `metrics` и кэшируются в стор; инспекционный валидатор допускает нулевые значения.
