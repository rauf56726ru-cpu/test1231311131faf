"""Application-wide configuration helpers for Binance Vision ingestion and analytics."""
from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_CANDIDATES: tuple[str, ...] = (
    "config/settings.json",
    "settings.json",
    "config/app-settings.json",
    "var/settings.json",
)


def _discover_settings_path() -> Optional[Path]:
    explicit = os.environ.get("APP_SETTINGS_PATH")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
    for relative in _DEFAULT_CONFIG_CANDIDATES:
        candidate = (PROJECT_ROOT / relative).expanduser()
        if candidate.is_file():
            return candidate
    return None


@dataclass(slots=True)
class ZoneThresholds:
    """Thresholds controlling zone validation heuristics."""

    tolerance_bps: float = 2.0
    min_separation_bars: int = 1
    max_fill_percent: float = 60.0
    min_displacement: float = 0.25
    min_separation_bars_by_tf: Dict[str, int] = field(default_factory=dict)
    tolerance_bps_by_tf: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EqualHighLowSettings:
    """Configuration for equal highs/lows detection."""

    tolerance_bps: float = 5.0
    min_separation_bars: int = 5
    symmetry: bool = True
    tolerance_bps_by_tf: Dict[str, float] = field(default_factory=dict)
    min_separation_bars_by_tf: Dict[str, int] = field(default_factory=dict)
    pivot_radius_by_tf: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class BinanceVisionDatasetPaths:
    """Directory structure for Binance Vision archives."""

    agg_trades: str = "data/futures/um/daily/aggTrades"
    klines: str = "data/futures/um/daily/klines"
    funding_rate: str = "data/futures/um/daily/fundingRate"
    open_interest: str = "data/futures/um/daily/openInterest"
    liquidation_orders: str = "data/futures/um/daily/liquidationOrders"
    depth_snapshots: str = "data/futures/um/daily/bookDepth"
    metrics: str = "data/futures/um/daily/metrics"
    exchange_info: str = "data/exchangeInfo.json"


@dataclass(slots=True)
class BinanceVisionSettings:
    """Settings for working with Binance Vision archive data."""

    base_url: str = "https://data.binance.vision"
    market_source: str = "futures_um"
    ingest_hours: int = 72
    default_intervals: tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")
    max_parallel_downloads: int = 4
    request_timeout_seconds: float = 30.0
    datasets: BinanceVisionDatasetPaths = field(default_factory=BinanceVisionDatasetPaths)
    allow_partial_results: bool = False


@dataclass(slots=True)
class FrontendFlags:
    """Feature flags guiding the front-end bootstrap."""

    preload_delta: bool = True
    preload_sessions: bool = True
    preload_zones: bool = True
    show_partial_badge: bool = True


@dataclass(slots=True)
class AppSettings:
    """Container for all tunable settings loaded from disk/environment."""

    binance_vision: BinanceVisionSettings = field(default_factory=BinanceVisionSettings)
    zone_thresholds: ZoneThresholds = field(default_factory=ZoneThresholds)
    eql_settings: EqualHighLowSettings = field(default_factory=EqualHighLowSettings)
    frontend: FrontendFlags = field(default_factory=FrontendFlags)


def _coerce_mapping(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if payload is None:
        return {}
    return dict(payload)


def _merge_dataclass(instance: Any, payload: Mapping[str, Any]) -> Any:
    """Recursively merge dictionary payload into a dataclass instance."""

    updates: Dict[str, Any] = {}
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current):
            nested_mapping = _coerce_mapping(value if isinstance(value, Mapping) else {})
            updates[key] = _merge_dataclass(current, nested_mapping)
        else:
            updates[key] = value

    for key, value in updates.items():
        setattr(instance, key, value)
    return instance


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        doc = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(doc, Mapping):
        return {}
    return dict(doc)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return application settings, merging optional overrides from disk."""

    settings = AppSettings()

    path = _discover_settings_path()
    if path:
        overrides = _load_payload(path)
        if overrides:
            _merge_dataclass(settings, overrides)

    return settings


__all__ = [
    "AppSettings",
    "BinanceVisionSettings",
    "BinanceVisionDatasetPaths",
    "EqualHighLowSettings",
    "FrontendFlags",
    "ZoneThresholds",
    "get_settings",
]
