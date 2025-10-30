"""Lightweight symbol configuration store used by tests and local tooling."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

DEFAULT_SYMBOL_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "preset": "summary_72h",
    },
    "orderflow": {
        "window_hours": 72,
        "reconstruct_min_volume": 0.0,
    },
    "liquidity": {
        "sweep": {
            "min_volume": 20.0,
            "min_delta": 1.0,
        },
    },
}

CONFIG_ENV_VAR = "SYMBOL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("config") / "symbol_configs.json"

__all__ = [
    "apply_symbol_config",
    "load_symbol_config",
    "save_symbol_config",
]


def _config_path() -> Path:
    env_override = os.getenv(CONFIG_ENV_VAR)
    if env_override:
        return Path(env_override)
    return DEFAULT_CONFIG_PATH


def _load_store() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _save_store(store: Mapping[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, Mapping):
            merged[key] = _deep_merge(value, {})  # copy nested dicts
        else:
            merged[key] = deepcopy(value)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalise_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def load_symbol_config(symbol: str) -> Dict[str, Any]:
    """Read configuration for ``symbol`` combining defaults with stored overrides."""

    store = _load_store()
    symbol_key = _normalise_symbol(symbol)
    overrides = store.get(symbol_key, {})
    if not isinstance(overrides, Mapping):
        overrides = {}
    defaults = deepcopy(DEFAULT_SYMBOL_CONFIG)
    effective = _deep_merge(defaults, overrides)
    return {
        "symbol": symbol_key,
        "defaults": defaults,
        "overrides": deepcopy(overrides),
        "effective": effective,
    }


def save_symbol_config(symbol: str, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist user overrides for ``symbol`` and return the updated record."""

    symbol_key = _normalise_symbol(symbol)
    store = _load_store()
    store[symbol_key] = deepcopy(overrides)
    _save_store(store)
    return load_symbol_config(symbol_key)


def _merge_into(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            node = target.setdefault(key, {})
            if isinstance(node, MutableMapping):
                _merge_into(node, value)
            else:
                target[key] = deepcopy(value)
        else:
            target[key] = deepcopy(value)


def apply_symbol_config(meta: MutableMapping[str, Any], symbol: str) -> None:
    """Merge the effective configuration for ``symbol`` into ``meta`` in-place."""

    record = load_symbol_config(symbol)
    effective = record.get("effective", {})
    if isinstance(effective, Mapping):
        _merge_into(meta, effective)
