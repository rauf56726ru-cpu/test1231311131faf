"""Progress tracking utilities for self-training orchestration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

SCHEMA_VERSION = 3


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class MetricsBundle(BaseModel):
    MCC: float = Field(0.0, alias="mcc")
    ECE: float = Field(1.0, alias="ece")
    SharpeProxy: float = Field(0.0, alias="sharpe_proxy")
    HitRate: float = Field(0.0, alias="hit_rate")

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


class GateBundle(BaseModel):
    ece_pass: bool = False
    mcc_pass: bool = False
    active: bool = False

    class Config:
        extra = "ignore"


class ResumeKey(BaseModel):
    D: str = ""
    TF: str = ""
    stage: str = "idle"

    class Config:
        extra = "ignore"


class StatusRecord(BaseModel):
    symbol: str
    day: str
    tf: str
    stage: str
    rows_in_day: int = 0
    rows_total_tf: int = 0
    metrics_day: MetricsBundle = Field(default_factory=MetricsBundle)
    metrics_cum: MetricsBundle = Field(default_factory=MetricsBundle)
    gate: GateBundle = Field(default_factory=GateBundle, alias="gates")
    resume_key: ResumeKey = Field(default_factory=ResumeKey)
    schema_version: int = SCHEMA_VERSION
    mode: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


class HistoryRecord(BaseModel):
    symbol: str
    day: str
    tf: str
    stage: str
    rows_in_day: int
    metrics: MetricsBundle
    gate: GateBundle = Field(default_factory=GateBundle, alias="gates")
    completed_at: str = Field(default_factory=_now_iso)
    schema_version: int = SCHEMA_VERSION

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


class ProgressState(BaseModel):
    schema_version: int = SCHEMA_VERSION
    status: Dict[str, StatusRecord] = Field(default_factory=dict)
    history: Dict[str, List[HistoryRecord]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


@dataclass
class ProgressStore:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def load(self) -> ProgressState:
        if not self.path.exists():
            return ProgressState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            return ProgressState()
        if isinstance(raw, dict) and "status" in raw:
            return ProgressState.parse_obj(raw)
        # legacy payload migration
        state = ProgressState()
        entries: Dict[str, dict] = {}
        if isinstance(raw, dict):
            entries = raw.get("entries", {}) if isinstance(raw.get("entries"), dict) else {}
        for key, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            symbol, _, interval = key.partition(":")
            day = payload.get("current_day") or payload.get("last_completed_day") or ""
            stage = payload.get("stage") or "idle"
            resume = ResumeKey(D=day, TF=payload.get("interval") or interval, stage=stage)
            status_record = StatusRecord(
                symbol=symbol or payload.get("symbol") or "",
                day=day,
                tf=payload.get("interval") or interval or "",
                stage=stage,
                rows_in_day=int(payload.get("rows") or 0),
                rows_total_tf=int(payload.get("rows") or 0),
                metrics_day=MetricsBundle(**payload.get("last_metrics", {})),
                metrics_cum=MetricsBundle(**payload.get("cumulative_metrics", {})),
                gate=GateBundle(active=bool(payload.get("active"))),
                resume_key=resume,
            )
            state.status[symbol or key] = status_record
        state.schema_version = SCHEMA_VERSION
        return state

    async def save(self, state: ProgressState) -> None:
        async with self._lock:
            payload = state.dict(by_alias=True)
            payload["updated_at"] = _now_iso()
            payload["schema_version"] = SCHEMA_VERSION
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ProgressTracker:
    """Stateful helper that wraps :class:`ProgressStore`."""

    def __init__(self, store: ProgressStore) -> None:
        self.store = store
        self.state = ProgressState()
        self._loaded = False
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            self.state = await self.store.load()
            self._loaded = True

    async def status_snapshot(self) -> Tuple[int, Optional[StatusRecord]]:
        await self.ensure_loaded()
        if not self.state.status:
            return SCHEMA_VERSION, None
        # return the most recent by updated resume key order
        latest = max(
            self.state.status.values(),
            key=lambda record: (record.resume_key.D, record.resume_key.TF, record.resume_key.stage, record.symbol),
        )
        return self.state.schema_version, latest

    async def history_snapshot(self) -> Tuple[int, List[HistoryRecord]]:
        await self.ensure_loaded()
        records: List[HistoryRecord] = []
        for entries in self.state.history.values():
            records.extend(entries)
        records.sort(key=lambda item: (item.day, item.tf, item.stage))
        return self.state.schema_version, records

    async def initialize_symbol(self, symbol: str, day: str, tf: str) -> StatusRecord:
        await self.ensure_loaded()
        record = StatusRecord(
            symbol=symbol,
            day=day,
            tf=tf,
            stage="queued",
            resume_key=ResumeKey(D=day, TF=tf, stage="queued"),
        )
        self.state.status[symbol] = record
        await self.store.save(self.state)
        return record

    async def update_stage(
        self,
        symbol: str,
        *,
        day: str,
        tf: str,
        stage: str,
        rows_in_day: int = 0,
        rows_total_tf: int = 0,
        metrics_day: Optional[Dict[str, float]] = None,
        gate: Optional[Dict[str, bool]] = None,
        resume_stage: Optional[str] = None,
    ) -> StatusRecord:
        await self.ensure_loaded()
        async with self._lock:
            record = self.state.status.get(symbol)
            if record is None:
                record = StatusRecord(symbol=symbol, day=day, tf=tf, stage=stage)
            record.day = day
            record.tf = tf
            record.stage = stage
            record.rows_in_day = int(rows_in_day)
            record.rows_total_tf = int(rows_total_tf)
            resume_value = resume_stage or stage
            record.resume_key = ResumeKey(D=day, TF=tf, stage=resume_value)
            if metrics_day is not None:
                record.metrics_day = MetricsBundle(**metrics_day)
            if gate is not None:
                current = record.gate.dict()
                current.update(gate)
                record.gate = GateBundle(**current)
            record.schema_version = SCHEMA_VERSION
            self.state.status[symbol] = record
            self.state.schema_version = SCHEMA_VERSION
            await self.store.save(self.state)
            return record

    async def record_history(
        self,
        symbol: str,
        *,
        day: str,
        tf: str,
        rows_in_day: int,
        metrics: Dict[str, float],
        gate: Dict[str, bool],
        stage: str = "gate",
    ) -> None:
        await self.ensure_loaded()
        history_record = HistoryRecord(
            symbol=symbol,
            day=day,
            tf=tf,
            stage=stage,
            rows_in_day=int(rows_in_day),
            metrics=MetricsBundle(**metrics),
            gate=GateBundle(**gate),
        )
        entries = self.state.history.setdefault(symbol, [])
        entries = [item for item in entries if not (item.day == day and item.tf == tf)]
        entries.append(history_record)
        entries.sort(key=lambda item: (item.day, item.tf, item.stage))
        self.state.history[symbol] = entries
        # recompute cumulative metrics for status entry
        record = self.state.status.get(symbol)
        if record is not None:
            record.metrics_cum = self._cumulative_metrics(symbol, tf)
            record.gate = GateBundle(**gate)
            self.state.status[symbol] = record
        await self.store.save(self.state)

    def _cumulative_metrics(self, symbol: str, tf: str) -> MetricsBundle:
        entries = self.state.history.get(symbol, [])
        totals: Dict[str, float] = {"MCC": 0.0, "ECE": 0.0, "SharpeProxy": 0.0, "HitRate": 0.0}
        total_rows = 0
        for record in entries:
            if record.tf != tf:
                continue
            rows = max(1, int(record.rows_in_day))
            total_rows += rows
            metrics = record.metrics.dict()
            for key in totals:
                totals[key] += float(metrics.get(key, 0.0)) * rows
        if total_rows <= 0:
            return MetricsBundle()
        averaged = {key: value / total_rows for key, value in totals.items()}
        return MetricsBundle(**averaged)

    async def resume_pointer(self, symbol: str) -> Optional[ResumeKey]:
        await self.ensure_loaded()
        record = self.state.status.get(symbol)
        if record is None:
            return None
        return record.resume_key

