"""Service layer exports for the chart backend."""

from .binance import (
    BINANCE_FAPI_BASE_URL,
    BINANCE_FAPI_REST,
    fetch_fapi_endpoint,
)
from .binance_ingest import ingest_agg_trades, ingest_klines, BinanceIngestError, BinanceAccessDenied, fetch_premium_index_series
from .coverage import compute_coverage
from .depth_checks import capture_depth_series
from .smc72_pipeline import collect_data as collect_smc72_data, write_outputs as write_smc72_outputs, PipelineDiagnostics
from .check_all_datas import (
    build_check_all_datas,
    build_check_all_datas_async,
    build_inspection_error_payload,
    DataQualityError,
)
from .inspection import (
    build_inspection_payload,
    build_placeholder_snapshot,
    DEFAULT_SYMBOL,
    get_latest_snapshot,
    get_snapshot,
    list_snapshots,
    register_snapshot,
    render_inspection_page,
)
from .profile import (
    build_profile_package,
    build_volume_profile,
    compute_session_profiles,
    flatten_profile,
    split_by_sessions,
)
from .presets import (
    DEFAULT_PRESETS,
    delete_preset,
    get_preset,
    list_presets as list_presets_configs,
    preset_to_payload,
    resolve_profile_config,
    resolve_or_prompt,
    save_preset,
    update_preset,
)
from .liquidity import build_liquidity_snapshot
from .zones import Config as ZonesConfig, detect_zones
from .enrichment import apply_enrichment_to_payload, enrich_inspection_snapshot
from .collection_state import get_last_collection_time, set_last_collection_time
from .zones_72h_service import collect_open_zones as collect_zones_72h
from .session_last_service import collect_last_sessions
from .shared_candles_store import (
    clear_shared_candles,
    get_shared_candles,
    merge_shared_candles,
)
from .summary_collector import collect_recent_summary, CollectionSummary
from .session_collector import collect_last_session_detailed, SessionCollectionResult
from .session_analysis import ZoneDetectionConfig, build_session_snapshot, build_72h_context, build_smc_session_v1
from .session_fast import analyze_session_fast, SessionDataUnavailable
from .zones_context import build_zones_context, ZoneCache, ZoneDetectionError
from .zones_ctx72 import build_smc_72h_ctx_v1
from .ohlc import (
    TIMEFRAME_WINDOWS,
    fetch_ohlcv,
    fetch_ohlcv_sync,
    aggregate_1m_to_1h,
    build_multi_timeframe_ohlcv,
    normalise_ohlcv,
    normalise_ohlcv_sync,
)
from .vwap import fetch_daily_vwap, fetch_daily_vwap_sync
from .trades import AggTradeCollector
from .um_ingest import UMIngestConfig, UMIngestService, MinuteRecord

__all__ = [
    "BINANCE_FAPI_BASE_URL",
    "BINANCE_FAPI_REST",
    "fetch_fapi_endpoint",
    "ingest_klines",
    "ingest_agg_trades",
    "fetch_premium_index_series",
    "BinanceIngestError",
    "BinanceAccessDenied",
    "compute_coverage",
    "capture_depth_series",
    "collect_smc72_data",
    "write_smc72_outputs",
    "PipelineDiagnostics",
    "build_inspection_payload",
    "build_liquidity_snapshot",
    "build_check_all_datas",
    "build_placeholder_snapshot",
    "build_check_all_datas_async",
    "build_inspection_error_payload",
    "DEFAULT_SYMBOL",
    "get_snapshot",
    "get_latest_snapshot",
    "list_snapshots",
    "fetch_ohlcv",
    "fetch_ohlcv_sync",
    "normalise_ohlcv",
    "normalise_ohlcv_sync",
    "register_snapshot",
    "render_inspection_page",
    "TIMEFRAME_WINDOWS",
    "AggTradeCollector",
    "fetch_daily_vwap",
    "fetch_daily_vwap_sync",
    "build_volume_profile",
    "compute_session_profiles",
    "flatten_profile",
    "split_by_sessions",
    "build_profile_package",
    "DEFAULT_PRESETS",
    "delete_preset",
    "get_preset",
    "list_presets_configs",
    "preset_to_payload",
    "resolve_profile_config",
    "resolve_or_prompt",
    "save_preset",
    "update_preset",
    "DataQualityError",
    "aggregate_1m_to_1h",
    "build_multi_timeframe_ohlcv",
    "ZonesConfig",
    "detect_zones",
    "get_last_collection_time",
    "set_last_collection_time",
    "get_shared_candles",
    "merge_shared_candles",
    "clear_shared_candles",
    "enrich_inspection_snapshot",
    "apply_enrichment_to_payload",
    "collect_zones_72h",
    "collect_last_sessions",
    "collect_recent_summary",
    "CollectionSummary",
    "collect_last_session_detailed",
    "SessionCollectionResult",
    "build_session_snapshot",
    "build_72h_context",
    "build_smc_session_v1",
    "build_smc_72h_ctx_v1",
    "ZoneDetectionConfig",
    "UMIngestConfig",
    "UMIngestService",
    "MinuteRecord",
    "analyze_session_fast",
    "SessionDataUnavailable",
    "build_zones_context",
    "ZoneCache",
    "ZoneDetectionError",
]
