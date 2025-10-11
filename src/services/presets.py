"""Preset storage and validation helpers for TPO/profile parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


_DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "var"
_PRESET_STORAGE_PATH = _DEFAULT_STORAGE_ROOT / "presets.json"


@dataclass(slots=True)
class ProfileBinning:
    """Binning configuration for the volume profile builder."""

    mode: str = "adaptive"
    tick_size: float | None = None
    atr_multiplier: float = 0.5
    target_bins: int = 80


@dataclass(slots=True)
class ProfileExtras:
    """Optional post-processing parameters for profile visualisation."""

    clip_low_volume_tail: float = 0.005
    smooth_window: int = 1


@dataclass(slots=True)
class TPOPreset:
    """Validated preset describing how to build the TPO/profile payload."""

    symbol: str
    tf: str = "1m"
    last_n: int = 3
    value_area_pct: float = 0.7
    binning: ProfileBinning = field(default_factory=ProfileBinning)
    extras: ProfileExtras = field(default_factory=ProfileExtras)
    builtin: bool = False


DEFAULT_PRESETS: Dict[str, TPOPreset] = {
    "BTCUSDT": TPOPreset(
        symbol="BTCUSDT",
        tf="1m",
        last_n=3,
        value_area_pct=0.7,
        binning=ProfileBinning(
            mode="adaptive",
            tick_size=0.01,
            atr_multiplier=0.5,
            target_bins=80,
        ),
        extras=ProfileExtras(clip_low_volume_tail=0.005, smooth_window=1),
        builtin=True,
    ),
    "ETHUSDT": TPOPreset(
        symbol="ETHUSDT",
        tf="1m",
        last_n=3,
        value_area_pct=0.7,
        binning=ProfileBinning(
            mode="adaptive",
            tick_size=0.01,
            atr_multiplier=0.5,
            target_bins=90,
        ),
        extras=ProfileExtras(clip_low_volume_tail=0.005, smooth_window=1),
        builtin=True,
    ),
    "SOLUSDT": TPOPreset(
        symbol="SOLUSDT",
        tf="1m",
        last_n=3,
        value_area_pct=0.7,
        binning=ProfileBinning(
            mode="adaptive",
            tick_size=0.001,
            atr_multiplier=0.5,
            target_bins=100,
        ),
        extras=ProfileExtras(clip_low_volume_tail=0.005, smooth_window=1),
        builtin=True,
    ),
}


_PRESET_CACHE: Dict[str, TPOPreset] = {}
_CACHE_LOCK = RLock()
_STORAGE_LOADED = False


def _ensure_storage_dir(path: Path) -> None:
    directory = path.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Persistence is optional; failure should not break request handling.
        pass


def _coerce_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _clamp(value: float, *, lower: float, upper: float, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    if value != value:  # NaN guard
        return default
    return max(lower, min(upper, float(value)))


def _coerce_positive_int(value: Any, *, lower: int, upper: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    if numeric < lower:
        return default
    if numeric > upper:
        return upper
    return numeric


def _validate_binning(data: Mapping[str, Any] | None) -> ProfileBinning:
    if not isinstance(data, Mapping):
        return ProfileBinning()

    mode = str(data.get("mode") or "adaptive").lower()
    if mode not in {"adaptive", "tick"}:
        mode = "adaptive"

    tick_size: float | None = None
    if mode == "tick":
        try:
            tick_value = float(data.get("tick_size"))
        except (TypeError, ValueError):
            tick_value = float("nan")
        if tick_value > 0:
            tick_size = tick_value
        else:
            mode = "adaptive"

    atr_multiplier = _clamp(
        float(data.get("atr_multiplier", 0.5)),
        lower=0.1,
        upper=2.0,
        default=0.5,
    )

    target_bins = _coerce_positive_int(
        data.get("target_bins", 80),
        lower=40,
        upper=200,
        default=80,
    )

    return ProfileBinning(
        mode=mode,
        tick_size=tick_size,
        atr_multiplier=atr_multiplier,
        target_bins=target_bins,
    )


def _validate_extras(data: Mapping[str, Any] | None) -> ProfileExtras:
    if not isinstance(data, Mapping):
        return ProfileExtras()

    clip = _clamp(
        float(data.get("clip_low_volume_tail", 0.005)),
        lower=0.0,
        upper=0.05,
        default=0.005,
    )
    smooth = _coerce_positive_int(
        data.get("smooth_window", 1),
        lower=1,
        upper=5,
        default=1,
    )
    return ProfileExtras(clip_low_volume_tail=clip, smooth_window=smooth)


def _validate_preset(symbol: str, data: Mapping[str, Any]) -> TPOPreset:
    tf = str(data.get("tf") or "1m").lower()
    if tf not in {"1m"}:
        tf = "1m"

    last_n = _coerce_positive_int(data.get("last_n", 3), lower=1, upper=5, default=3)
    value_area_pct = _clamp(
        float(data.get("value_area_pct", 0.7)), lower=0.01, upper=0.99, default=0.7
    )

    binning_data = data.get("binning") if isinstance(data, Mapping) else None
    extras_data = data.get("extras") if isinstance(data, Mapping) else None

    return TPOPreset(
        symbol=symbol,
        tf=tf,
        last_n=last_n,
        value_area_pct=value_area_pct,
        binning=_validate_binning(binning_data if isinstance(binning_data, Mapping) else None),
        extras=_validate_extras(extras_data if isinstance(extras_data, Mapping) else None),
        builtin=False,
    )


def _serialise_preset(preset: TPOPreset) -> Dict[str, Any]:
    payload = asdict(preset)
    payload.pop("builtin", None)
    return payload


def _load_storage_locked() -> None:
    global _STORAGE_LOADED
    if _STORAGE_LOADED:
        return

    if not _PRESET_STORAGE_PATH.exists():
        _STORAGE_LOADED = True
        return

    try:
        raw = _PRESET_STORAGE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        _STORAGE_LOADED = True
        return

    if isinstance(data, Mapping):
        for symbol, payload in data.items():
            if not isinstance(payload, Mapping):
                continue
            symbol_key = _coerce_symbol(symbol)
            if not symbol_key:
                continue
            preset = _validate_preset(symbol_key, payload)
            _PRESET_CACHE[symbol_key] = preset

    _STORAGE_LOADED = True


def _persist_storage_locked() -> None:
    if not _PRESET_CACHE:
        # Remove file if no custom presets remain.
        try:
            _PRESET_STORAGE_PATH.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return
        return

    storage: MutableMapping[str, Any] = {}
    for symbol, preset in _PRESET_CACHE.items():
        storage[symbol] = _serialise_preset(preset)

    _ensure_storage_dir(_PRESET_STORAGE_PATH)
    tmp_path = _PRESET_STORAGE_PATH.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(storage, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(_PRESET_STORAGE_PATH)
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def list_presets(*, include_defaults: bool = True) -> Iterable[TPOPreset]:
    """Return stored presets, optionally extending the list with built-ins."""

    with _CACHE_LOCK:
        _load_storage_locked()
        presets = list(_PRESET_CACHE.values())
        if include_defaults:
            for preset in DEFAULT_PRESETS.values():
                presets.append(preset)
        # Deduplicate by symbol keeping explicit overrides first.
        seen: Dict[str, TPOPreset] = {}
        for preset in presets:
            seen.setdefault(preset.symbol, preset)
        return list(seen.values())


def get_preset(symbol: str) -> Optional[TPOPreset]:
    """Fetch a preset for the given symbol (built-in or stored)."""

    key = _coerce_symbol(symbol)
    if not key:
        return None

    builtin = DEFAULT_PRESETS.get(key)

    with _CACHE_LOCK:
        _load_storage_locked()
        stored = _PRESET_CACHE.get(key)
        if stored:
            return stored
        if builtin:
            return builtin
    return None


def save_preset(symbol: str, data: Mapping[str, Any]) -> TPOPreset:
    """Persist a preset for the symbol, replacing any existing override."""

    key = _coerce_symbol(symbol)
    if not key:
        raise ValueError("Symbol is required for preset storage")

    preset = _validate_preset(key, data)

    with _CACHE_LOCK:
        _load_storage_locked()
        _PRESET_CACHE[key] = preset
        _persist_storage_locked()

    return preset


def update_preset(symbol: str, data: Mapping[str, Any]) -> TPOPreset:
    """Apply a partial update to an existing preset (or create one)."""

    key = _coerce_symbol(symbol)
    if not key:
        raise ValueError("Symbol is required for preset updates")

    with _CACHE_LOCK:
        _load_storage_locked()
        base = _PRESET_CACHE.get(key)
        if base is None:
            base = DEFAULT_PRESETS.get(key)
        if base is None:
            base = TPOPreset(symbol=key)
        merged: Dict[str, Any] = _serialise_preset(base)
        merged.update({k: v for k, v in data.items() if v is not None})
        preset = _validate_preset(key, merged)
        _PRESET_CACHE[key] = preset
        _persist_storage_locked()

    return preset


def delete_preset(symbol: str) -> None:
    """Remove a stored preset override (built-ins remain available)."""

    key = _coerce_symbol(symbol)
    if not key:
        return

    with _CACHE_LOCK:
        _load_storage_locked()
        if key in _PRESET_CACHE:
            _PRESET_CACHE.pop(key, None)
            _persist_storage_locked()


def preset_to_payload(preset: TPOPreset) -> Dict[str, Any]:
    """Serialise a preset into a JSON-friendly structure."""

    payload = {
        "symbol": preset.symbol,
        "tf": preset.tf,
        "last_n": preset.last_n,
        "value_area_pct": preset.value_area_pct,
        "binning": {
            "mode": preset.binning.mode,
            "tick_size": preset.binning.tick_size,
            "atr_multiplier": preset.binning.atr_multiplier,
            "target_bins": preset.binning.target_bins,
        },
        "extras": {
            "clip_low_volume_tail": preset.extras.clip_low_volume_tail,
            "smooth_window": preset.extras.smooth_window,
        },
        "builtin": preset.builtin,
    }
    return payload


def resolve_or_prompt(symbol: str) -> tuple[Optional[TPOPreset], bool]:
    """Return a preset and a flag indicating whether user input is required."""

    preset = get_preset(symbol)
    if preset:
        return preset, False
    return None, True


def resolve_profile_config(symbol: str, meta: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Resolve runtime profile parameters using presets and optional snapshot meta."""

    raw_defaults: Mapping[str, Any] | None = None
    if isinstance(meta, Mapping):
        candidate = meta.get("profile")
        if isinstance(candidate, Mapping):
            raw_defaults = candidate

    preset, preset_required = resolve_or_prompt(symbol)
    preset_payload = preset_to_payload(preset) if preset else None

    config: Dict[str, Any] = {
        "preset": preset,
        "preset_required": bool(preset_required),
        "preset_payload": preset_payload,
        "raw_defaults": raw_defaults,
        "target_tf_key": str(preset.tf if preset else "1m").lower(),
        "last_n": 3,
        "value_area_pct": 0.7,
        "adaptive_bins": True,
        "tick_size": None,
        "atr_multiplier": 0.5,
        "target_bins": 80,
        "clip_threshold": 0.0,
        "smooth_window": 1,
    }

    if preset:
        config["value_area_pct"] = float(preset.value_area_pct or 0.7)
        last_n = int(preset.last_n or 3)
        config["last_n"] = max(1, min(5, last_n))
        binning = preset.binning
        config["adaptive_bins"] = binning.mode != "tick"
        tick_value = binning.tick_size
        if isinstance(tick_value, (int, float)) and float(tick_value) > 0:
            config["tick_size"] = float(tick_value)
        else:
            config["tick_size"] = None
        config["atr_multiplier"] = float(binning.atr_multiplier or 0.5)
        config["target_bins"] = int(binning.target_bins or 80)
        config["clip_threshold"] = float(preset.extras.clip_low_volume_tail)
        config["smooth_window"] = int(preset.extras.smooth_window or 1)
    elif raw_defaults:
        try:
            config["value_area_pct"] = float(raw_defaults.get("value_area_pct", config["value_area_pct"]))
        except (TypeError, ValueError):
            pass
        try:
            last_n = int(raw_defaults.get("last_n", config["last_n"]))
            config["last_n"] = max(1, min(5, last_n))
        except (TypeError, ValueError):
            pass
        adaptive = raw_defaults.get("adaptive_bins")
        if isinstance(adaptive, bool):
            config["adaptive_bins"] = adaptive
        if "tick_size" in raw_defaults:
            try:
                tick_value = float(raw_defaults.get("tick_size"))
            except (TypeError, ValueError):
                tick_value = float("nan")
            if math.isfinite(tick_value) and tick_value > 0:
                config["tick_size"] = tick_value

    return config

