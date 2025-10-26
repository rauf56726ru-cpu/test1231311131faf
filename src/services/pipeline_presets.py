"""Pipeline preset definitions for the check-all data builder."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import time as dtime
from typing import Any, Dict, Mapping, MutableMapping, Tuple

ZONE_KEYS: Tuple[str, ...] = ("fvg", "fvl", "ob", "mb", "bb", "rb", "pb", "sr", "profile_levels")
MODULE_KEYS: Tuple[str, ...] = ("rollups", "orderflow", "zones", "liquidity", "vwap_tpo", "showcases")


@dataclass(slots=True, frozen=True)
class SessionWindow:
    """Session boundaries used for VWAP/TPO calculations."""

    name: str
    open: dtime
    close: dtime


@dataclass(slots=True, frozen=True)
class ZoneSettings:
    """Configuration bucket that controls zone detection staging."""

    focus_window_hours: int
    top_n: Mapping[str, int]
    min_bars_per_tf: Mapping[str, int]
    min_bars_strict: Mapping[str, int] | None
    warmup_bars_per_tf: Mapping[str, int]
    pivot_overrides: Mapping[str, int]
    resample_when_sparse: bool = True
    atr_adaptation: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderflowSettings:
    """Orderflow pipeline tuning."""

    window_hours: int
    timeframes: Tuple[str, ...]
    page_span_minutes: int
    allow_network: bool = True


@dataclass(slots=True, frozen=True)
class PipelinePreset:
    """Top-level preset applied by the check-all pipeline."""

    name: str
    rollup_timeframes: Tuple[str, ...]
    modules: Mapping[str, bool]
    orderflow: OrderflowSettings
    zones: ZoneSettings
    sessions: Tuple[SessionWindow, ...]
    resample_when_sparse: bool = True
    summary_window_hours: int = 72

    def validate(self) -> None:
        if not self.rollup_timeframes:
            raise ValueError(f"{self.name}: rollup timeframes must not be empty")
        for key in MODULE_KEYS:
            if key not in self.modules:
                raise ValueError(f"{self.name}: missing module flag '{key}'")
        if self.orderflow.window_hours <= 0:
            raise ValueError(f"{self.name}: orderflow window_hours must be positive")
        if not self.orderflow.timeframes:
            raise ValueError(f"{self.name}: orderflow timeframes must not be empty")
        for zone_key in self.zones.top_n.keys():
            if zone_key not in ZONE_KEYS:
                raise ValueError(f"{self.name}: unsupported zone key in top_n: {zone_key}")
        for zone_key in self.zones.min_bars_per_tf.keys():
            if zone_key not in {"15m", "1h", "4h", "1d"}:
                raise ValueError(f"{self.name}: unexpected timeframe in min_bars_per_tf: {zone_key}")
        if self.zones.min_bars_strict:
            for zone_key in self.zones.min_bars_strict.keys():
                if zone_key not in {"15m", "1h", "4h", "1d"}:
                    raise ValueError(f"{self.name}: unexpected timeframe in min_bars_strict: {zone_key}")
        if not self.sessions:
            raise ValueError(f"{self.name}: at least one session window is required")
        for session in self.sessions:
            if session.open >= session.close:
                raise ValueError(f"{self.name}: session '{session.name}' must have open < close")

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "PipelinePreset":
        if not overrides:
            return self
        data: Dict[str, Any] = {
            "rollup_timeframes": tuple(overrides.get("rollup_timeframes", self.rollup_timeframes)),
            "modules": {**self.modules, **overrides.get("modules", {})},
            "resample_when_sparse": overrides.get("resample_when_sparse", self.resample_when_sparse),
            "summary_window_hours": overrides.get("summary_window_hours", self.summary_window_hours),
        }

        zone_overrides: MutableMapping[str, Any] = dict(overrides.get("zones", {}))
        strict_map: Dict[str, int] = dict(self.zones.min_bars_strict or {})
        strict_override = zone_overrides.get("min_bars_strict")
        if isinstance(strict_override, Mapping):
            strict_map.update({str(key): int(value) for key, value in strict_override.items()})
        zones = ZoneSettings(
            focus_window_hours=int(zone_overrides.get("focus_window_hours", self.zones.focus_window_hours)),
            top_n={**self.zones.top_n, **zone_overrides.get("top_n", {})},
            min_bars_per_tf={**self.zones.min_bars_per_tf, **zone_overrides.get("min_bars_per_tf", {})},
            min_bars_strict=strict_map or None,
            warmup_bars_per_tf={**self.zones.warmup_bars_per_tf, **zone_overrides.get("warmup_bars_per_tf", {})},
            pivot_overrides={**self.zones.pivot_overrides, **zone_overrides.get("pivot_overrides", {})},
            resample_when_sparse=bool(zone_overrides.get("resample_when_sparse", self.zones.resample_when_sparse)),
            atr_adaptation={**self.zones.atr_adaptation, **zone_overrides.get("atr_adaptation", {})},
        )

        orderflow_overrides: MutableMapping[str, Any] = dict(overrides.get("orderflow", {}))
        orderflow = OrderflowSettings(
            window_hours=int(orderflow_overrides.get("window_hours", self.orderflow.window_hours)),
            timeframes=tuple(orderflow_overrides.get("timeframes", self.orderflow.timeframes)),
            page_span_minutes=int(orderflow_overrides.get("page_span_minutes", self.orderflow.page_span_minutes)),
            allow_network=bool(orderflow_overrides.get("allow_network", self.orderflow.allow_network)),
        )

        sessions_override = overrides.get("sessions")
        if sessions_override:
            sessions = tuple(
                SessionWindow(
                    name=str(item.get("name")),
                    open=_parse_time(item.get("open")),
                    close=_parse_time(item.get("close")),
                )
                for item in sessions_override
                if isinstance(item, Mapping)
            )
        else:
            sessions = self.sessions

        preset = PipelinePreset(
            name=self.name,
            rollup_timeframes=data["rollup_timeframes"],
            modules=data["modules"],
            orderflow=orderflow,
            zones=zones,
            sessions=sessions,
            resample_when_sparse=data["resample_when_sparse"],
            summary_window_hours=int(data["summary_window_hours"]),
        )
        preset.validate()
        return preset


def _parse_time(value: Any) -> dtime:
    if isinstance(value, dtime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Invalid time value: {value!r}")
    parts = value.strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time format: {value!r}")
    hour = max(0, min(23, int(parts[0])))
    minute = max(0, min(59, int(parts[1])))
    return dtime(hour=hour, minute=minute)


SUMMARY_72H_PRESET = PipelinePreset(
    name="summary_72h",
    rollup_timeframes=("1m", "3m", "5m", "15m", "1h", "4h", "1d"),
    modules={key: True for key in MODULE_KEYS},
    orderflow=OrderflowSettings(
        window_hours=72,
        timeframes=("1m", "3m", "5m", "15m", "1h"),
        page_span_minutes=72 * 60,
        allow_network=True,
    ),
    zones=ZoneSettings(
        focus_window_hours=72,
        top_n={"fvg": 6, "ob": 6},
        min_bars_per_tf={"15m": 200, "1h": 60, "4h": 24},
        min_bars_strict={"15m": 120, "1h": 50, "4h": 18},
        warmup_bars_per_tf={"15m": 60, "1h": 30, "4h": 18},
        pivot_overrides={"15m": 1, "1h": 2},
        resample_when_sparse=True,
        atr_adaptation={
            "pct_cap": 0.05,
            "cluster_multiplier": 3.5,
            "cluster_max": 0.002,
            "sweep_base": 0.3,
            "sweep_multiplier": 6.0,
            "sweep_min": 0.25,
            "sweep_max": 0.75,
            "min_move_base": 0.25,
            "min_move_multiplier": 10.0,
            "min_move_min": 0.2,
            "min_move_max": 1.0,
            "epsilon_multiplier": 2.5,
            "epsilon_min": 0.0002,
            "epsilon_max": 0.02,
        },
    ),
    sessions=(
        SessionWindow("asia", dtime(hour=0, minute=0), dtime(hour=3, minute=0)),
        SessionWindow("london", dtime(hour=7, minute=0), dtime(hour=10, minute=0)),
        SessionWindow("ny", dtime(hour=13, minute=30), dtime(hour=16, minute=30)),
    ),
    resample_when_sparse=True,
    summary_window_hours=72,
)
SUMMARY_72H_PRESET.validate()

_PIPELINE_PRESETS: Dict[str, PipelinePreset] = {
    SUMMARY_72H_PRESET.name: SUMMARY_72H_PRESET,
}


def resolve_pipeline_preset(name: str, overrides: Mapping[str, Any] | None = None) -> PipelinePreset:
    """Return a validated pipeline preset with optional overrides applied."""

    key = (name or "summary_72h").strip().lower()
    preset = _PIPELINE_PRESETS.get(key)
    if preset is None:
        available = ", ".join(sorted(_PIPELINE_PRESETS))
        raise ValueError(f"Unknown pipeline preset '{name}'. Available: {available}")
    return preset.with_overrides(overrides)


__all__ = [
    "PipelinePreset",
    "SessionWindow",
    "ZoneSettings",
    "OrderflowSettings",
    "resolve_pipeline_preset",
    "SUMMARY_72H_PRESET",
    "ZONE_KEYS",
    "MODULE_KEYS",
]
