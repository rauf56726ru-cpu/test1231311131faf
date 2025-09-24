"""FastAPI application powering the trading UI."""
from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
import zipfile
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import multipart  # type: ignore  # noqa: F401
except Exception as exc:  # pragma: no cover - fail fast for missing dependency
    raise RuntimeError(
        'python-multipart не установлен. Добавь "python-multipart" в requirements.txt и установи.'
    ) from exc

import numpy as np
import pandas as pd
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketState

from pydantic import BaseModel, Field, validator

from ..backtest.metrics import atr, rolling_volatility
from ..config import get_settings
from ..live.collector import StreamManager
from ..io.binance_ws import BINANCE_SOURCE
from ..live.candles_store import CandleStore
from ..live.market import MarketDataProvider
from ..live.jobs import JobManager
from ..services import bootstrap as bootstrap_service
from ..services import ensure_bootstrap as ensure_frames_bootstrap
from ..services import ensure_bootstrap as ensure_frames_bootstrap, reset_bootstrap
from ..memory.embedder import ImageEmbedder, TextEmbedder
from ..memory.faiss_store import FaissStore
from ..memory.notion_ingest import load_export
from ..memory.ingest_registry import IngestRegistry, compute_folder_hash
from ..memory.retriever import MemoryRetriever
from ..models import lgbm_predict, tcn_predict, xgb_predict
from ..models.model_registry import ModelRegistry, get_registry
from ..policy.compiler import load_and_compile
from ..policy.runtime import PolicyDecision, PolicyRuntime, propose_entry_plan
from ..self_train import SelfTrainManager
from ..utils.logging import configure_logging, get_logger
from ..utils.timeframes import compute_refresh_seconds, interval_to_seconds

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

configure_logging()
LOGGER = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "public"


class SelfTrainRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    date_from: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    date_to: Optional[str] = Field(None, regex=r"^\d{4}-\d{2}-\d{2}$")
    resume: bool = Field(True)

    @validator("symbol", pre=True)
    def _normalize_symbol(cls, value: object) -> str:
        symbol = str(value or "").strip().upper()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        return symbol

    @validator("date_from")
    def _validate_date_from(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("date_from must be YYYY-MM-DD") from exc
        return value

    @validator("date_to")
    def _validate_date_to(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("date_to must be YYYY-MM-DD") from exc
        return value


class GapRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    interval: str = Field(..., min_length=1)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)

    @validator("symbol", pre=True)
    def _normalize_symbol(cls, value: object) -> str:
        symbol = str(value or "").strip().upper()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        return symbol

    @validator("interval")
    def _validate_interval(cls, value: str) -> str:
        interval = value.strip()
        if not interval:
            raise ValueError("interval must be non-empty")
        return interval

class ModelRunner:
    """Wrapper around a trained model to provide predict interface."""

    def __init__(self, family: str, name: str, path: Path):
        self.family = family
        self.name = name
        self.path = Path(path)
        self.model: Optional[object] = None
        self.feature_list: Optional[List[str]] = None
        self.scaler: Optional[object] = None
        self.calibrator: Optional[object] = None
        self.window: int = 1
        self.ready: bool = False
        self.metadata: Dict[str, object] = {}
        self.default_class_order = (-1, 0, 1)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if self.family == "xgb":
            model, features, scaler, calibrator, metadata = xgb_predict.load_artifacts(self.path)
            self.model = model
            self.feature_list = features
            self.scaler = scaler
            self.calibrator = calibrator
            self.window = 1
            self.metadata = metadata or {}
        elif self.family == "lgbm":
            booster, features, scaler, calibrator, metadata = lgbm_predict.load_artifacts(self.path)
            self.model = booster
            self.feature_list = list(features) if features is not None else None
            self.scaler = scaler
            self.calibrator = calibrator
            self.window = 1
            self.metadata = metadata or {}
        elif self.family == "tcn":
            try:
                model, scaler, feature_list, window = tcn_predict.load_artifacts(self.path)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOGGER.warning("Failed to load TCN artifacts from %s: %s", self.path, exc)
                self.model = None
                self.feature_list = None
                self.scaler = None
                self.window = 1
            else:
                self.model = model
                self.feature_list = feature_list
                self.scaler = scaler
                self.window = window
        else:
            raise ValueError(f"Unknown model family: {self.family}")

        self.ready = self.model is not None and self.feature_list is not None

    def refresh_if_needed(self) -> None:
        if not self.ready:
            self._load_artifacts()

    def predict(
        self,
        features_dict: Dict[str, float],
        frame: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, object]]:
        if not self.ready:
            return None

        if self.family in {"xgb", "lgbm"}:
            assert self.feature_list is not None  # for type checkers
            model = self.model
            if model is None:
                return None
            if self.family == "xgb":
                distribution, metadata = xgb_predict.predict_distribution(features_dict, self.path)
            else:
                distribution, metadata = lgbm_predict.predict_distribution(features_dict, self.path)
            calibrator = self.calibrator
            bucket = None
            if calibrator is not None:
                bucket = calibrator.infer_bucket(features_dict)
                grey = calibrator.grey_zone_bounds(features=features_dict)
                ci = calibrator.confidence_interval(float(distribution.max()), features=features_dict)
            else:
                grey = (0.45, 0.55)
                ci = (float(distribution.max()), float(distribution.max()))
            class_order = metadata.get("class_order") or metadata.get("metadata", {}).get("class_order")
            if not class_order:
                class_order = list(self.default_class_order)
            label_map = {idx: cls for idx, cls in enumerate(class_order)}
            pred_idx = int(np.argmax(distribution))
            pred_dir_value = label_map.get(pred_idx, 0)
            dir_label = {1: "up", 0: "flat", -1: "down"}.get(int(pred_dir_value), "flat")
            return {
                "probs": {
                    "down": float(distribution[0]) if distribution.size >= 1 else None,
                    "flat": float(distribution[1]) if distribution.size >= 2 else None,
                    "up": float(distribution[2]) if distribution.size >= 3 else None,
                },
                "pred_dir": dir_label,
                "pred_conf": float(distribution[pred_idx]),
                "confidence_interval": ci,
                "grey_zone": grey,
                "regime_bucket": bucket,
                "class_order": class_order,
            }

        if self.family == "tcn":
            if torch is None:
                LOGGER.warning("PyTorch is required for TCN inference; skipping prediction")
                return None
            if frame is None:
                frame = pd.DataFrame([features_dict])
            data = frame.copy()
            assert self.feature_list is not None
            for col in self.feature_list:
                if col not in data.columns:
                    data[col] = 0.0
            data = data[self.feature_list]
            if len(data) < self.window:
                padding = pd.DataFrame(0.0, index=range(self.window - len(data)), columns=self.feature_list)
                data = pd.concat([padding, data], ignore_index=True)
            else:
                data = data.tail(self.window).reset_index(drop=True)
            scaler = self.scaler
            if scaler is not None:
                arr = scaler.transform(data.values)
            else:
                arr = data.values
            tensor = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0)
            model = self.model
            if model is None:
                return None
            with torch.no_grad():
                logits = model(tensor)
                prob = torch.sigmoid(logits).item()
            return {
                "probs": {"up": float(prob), "down": float(1.0 - prob), "flat": 0.0},
                "pred_dir": "up" if prob >= 0.5 else "down",
                "pred_conf": float(prob if prob >= 0.5 else 1.0 - prob),
                "confidence_interval": (max(0.0, float(prob) - 0.1), min(1.0, float(prob) + 0.1)),
                "grey_zone": (0.45, 0.55),
                "regime_bucket": None,
                "class_order": ["down", "flat", "up"],
            }

        LOGGER.warning("Unsupported model family: %s", self.family)
        return None


class ModelCatalog:
    """Bridge between the on-disk model registry and runtime."""

    def __init__(self, base_dir: Path, registry: ModelRegistry):
        self.base_dir = Path(base_dir)
        self.registry = registry
        self._overrides: Dict[str, str] = {}
        self._cache: Dict[Tuple[str, str], ModelRunner] = {}
        self._missing_warned: Set[Tuple[str, str]] = set()
        self._migrate_legacy_layouts()

    def _refresh_registry(self) -> None:
        """Reload registry from disk to pick up updates from training jobs."""

        try:
            self.registry = get_registry()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to refresh model registry: %s", exc)

    def _migrate_legacy_layouts(self) -> None:
        xgb_dir = self.base_dir / "xgb"
        legacy_model = xgb_dir / "model.pkl"
        if legacy_model.exists() and not any(child.is_dir() for child in xgb_dir.iterdir()):
            version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            version_dir = xgb_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            for name in ["model.pkl", "feature_list.json", "scaler.pkl"]:
                src = xgb_dir / name
                if src.exists():
                    dst = version_dir / name
                    src.replace(dst)
            feature_path = version_dir / "feature_list.json"
            if not feature_path.exists():
                source = Path("data/features/features.parquet")
                if source.exists():
                    try:
                        df = pd.read_parquet(source)
                        columns = [col for col in df.columns if col != "target"]
                        with feature_path.open("w", encoding="utf-8") as f:
                            json.dump(columns, f, indent=2)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.warning("Failed to backfill feature list during migration: %s", exc)
            if not (version_dir / "scaler.pkl").exists():
                LOGGER.info("No scaler found during migration; continuing without creating one")
            if self.registry.get("xgb", version) is None:
                self.registry.register("xgb", version, str(version_dir), metrics={})
            LOGGER.info("Migrated legacy XGB artifacts into %s", version_dir)

    def _list_dir_versions(self, family: str) -> List[str]:
        family_dir = self.base_dir / family
        if not family_dir.exists():
            return []
        versions: Set[str] = set()
        for horizon_dir in family_dir.iterdir():
            if not horizon_dir.is_dir():
                continue
            model_artifact = horizon_dir / "model.pkl"
            if model_artifact.exists():
                versions.add(horizon_dir.name)
                continue
            for candidate in horizon_dir.iterdir():
                if candidate.is_dir() and (candidate / "model.pkl").exists():
                    versions.add(candidate.name)
        return sorted(versions)

    def list_models(self, family: str) -> List[str]:
        self._refresh_registry()
        known = set(self.registry.list_models(family))
        known.update(self._list_dir_versions(family))
        return sorted(known)

    def active_name(self, family: str) -> Optional[str]:
        self._refresh_registry()
        if family in self._overrides:
            return self._overrides[family]
        active = self.registry.active_version(family)
        if active:
            return active
        discovered = self._list_dir_versions(family)
        if discovered:
            return discovered[-1]
        return None

    def _resolve_path(self, family: str, name: str) -> Path:
        record = self.registry.get(family, name)
        if record is not None:
            return Path(record.path)
        return self.base_dir / family / name

    def load_active(self, family: str, name: Optional[str] = None) -> ModelRunner:
        resolved_name = name or self.active_name(family) or (name or "")
        path = self._resolve_path(family, resolved_name)
        cache_key = (family, resolved_name)
        runner = self._cache.get(cache_key)
        if runner is None:
            runner = ModelRunner(family=family, name=resolved_name, path=path)
            self._cache[cache_key] = runner
        else:
            runner.refresh_if_needed()

        if not runner.ready and cache_key not in self._missing_warned:
            model_path = self.base_dir / family
            if resolved_name:
                model_path = model_path / resolved_name
            LOGGER.warning("Missing artifacts at %s. Streaming OHLC only.", model_path.as_posix())
            self._missing_warned.add(cache_key)

        return runner

    def activate(self, family: str, name: str) -> None:
        self._refresh_registry()
        available = set(self.list_models(family))
        if name not in available:
            raise ValueError(f"Unknown model name '{name}' for family '{family}'")
        if self.registry.get(family, name) is not None:
            self.registry.set_active(family, name)
            self._overrides.pop(family, None)
        else:
            self._overrides[family] = name
        self._cache = {key: value for key, value in self._cache.items() if key[0] != family}


class EventBroker:
    """Simple pub/sub broker for UI notifications."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def publish(self, event: Dict[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            await queue.put(event)

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)


class AutoPipeline:
    """Coordinate collection, feature building and training jobs."""

    def __init__(
        self,
        jobs: JobManager,
        streams: StreamManager,
        catalog: ModelCatalog,
        events: EventBroker,
        settings,
    ) -> None:
        self.jobs = jobs
        self.streams = streams
        self.catalog = catalog
        self.events = events
        self.settings = settings
        self._status: Dict[str, str] = {}
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._current_jobs: Dict[str, Dict[str, str]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _make_key(symbol: str, interval: str, family: str) -> str:
        return f"{symbol.upper()}::{interval}::{family}"

    def status(self, symbol: str, interval: str, family: str) -> str:
        key = self._make_key(symbol, interval, family)
        return self._status.get(key, "missing")

    def job_key(self, symbol: str, interval: str, family: str) -> str:
        return self._make_key(symbol, interval, family)

    async def ensure(
        self, symbol: str, interval: str, family: str, force: bool = False
    ) -> Tuple[str, Optional[str]]:
        key = self._make_key(symbol, interval, family)
        params = {
            "symbol": symbol,
            "interval": interval,
            "family": family,
            "min_bars": self.settings.auto.min_bars_for_train,
        }
        async with self._lock:
            current_status = self._status.get(key)
            if not force and current_status in {"collecting", "training"}:
                jobs = self._current_jobs.get(key, {})
                return current_status or "collecting", jobs.get("collect")
            if not force and current_status == "ready":
                return current_status, None
            self._status[key] = "collecting"
            self._contexts.setdefault(key, {})

        job_id = await self.jobs.enqueue("collect", key, params)
        async with self._lock:
            self._current_jobs[key] = {"collect": job_id}
        await self.events.publish({
            "event": "job",
            "type": "collect",
            "status": "queued",
            "job_id": job_id,
            "key": key,
        })
        await self.events.publish({"event": "job", "type": "pipeline", "status": "collecting", "key": key})
        asyncio.create_task(self._watch_collect(key, params, job_id))
        return "collecting", job_id

    async def trigger_training(self, symbol: str, interval: str, family: str) -> Tuple[str, bool]:
        key = self._make_key(symbol, interval, family)
        async with self._lock:
            jobs = self._current_jobs.get(key, {})
            for stage in ("train", "build_features", "collect"):
                job_id = jobs.get(stage)
                if job_id:
                    job = self.jobs.get(job_id)
                    if job and job.status in {"queued", "in_progress"}:
                        return job.id, True
        status, job_id = await self.ensure(symbol, interval, family, force=True)
        return job_id or "", False

    async def _watch_collect(self, key: str, params: Dict[str, Any], job_id: str) -> None:
        job = await self.jobs.wait(job_id)
        await self.events.publish({
            "event": "job",
            "type": "collect",
            "status": job.status,
            "job_id": job_id,
            "key": key,
        })
        if job.status != "done" or not job.result:
            async with self._lock:
                self._status[key] = "error"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "error", "key": key})
            return
        result = dict(job.result)
        async with self._lock:
            ctx = self._contexts.setdefault(key, {})
            ctx.update(result)

        next_params = dict(params)
        next_params.update(result)
        bf_job_id = await self.jobs.enqueue("build_features", key, next_params)
        async with self._lock:
            self._current_jobs.setdefault(key, {})["build_features"] = bf_job_id
        await self.events.publish({
            "event": "job",
            "type": "build_features",
            "status": "queued",
            "job_id": bf_job_id,
            "key": key,
        })
        asyncio.create_task(self._watch_build_features(key, next_params, bf_job_id))

    async def _watch_build_features(
        self, key: str, params: Dict[str, Any], job_id: str
    ) -> None:
        job = await self.jobs.wait(job_id)
        await self.events.publish({
            "event": "job",
            "type": "build_features",
            "status": job.status,
            "job_id": job_id,
            "key": key,
        })
        if job.status != "done" or not job.result:
            async with self._lock:
                self._status[key] = "error"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "error", "key": key})
            return
        result = dict(job.result)
        async with self._lock:
            ctx = self._contexts.setdefault(key, {})
            ctx.update(result)
            self._status[key] = "training"
        await self.events.publish({"event": "job", "type": "pipeline", "status": "training", "key": key})

        rows = int(result.get("rows", 0))
        min_rows = max(20, min(self.settings.auto.min_bars_for_train, 500))
        if rows < min_rows:
            async with self._lock:
                self._status[key] = "ready"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "ready", "key": key})
            async with self._lock:
                self._status[key] = "collecting"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "collecting", "key": key})
            collect_params = {
                "symbol": params["symbol"],
                "interval": params["interval"],
                "family": params.get("family", "xgb"),
                "min_bars": self.settings.auto.min_bars_for_train,
            }
            collect_job_id = await self.jobs.enqueue("collect", key, collect_params)
            async with self._lock:
                self._current_jobs[key] = {"collect": collect_job_id}
            await self.events.publish({
                "event": "job",
                "type": "collect",
                "status": "queued",
                "job_id": collect_job_id,
                "key": key,
            })
            asyncio.create_task(self._watch_collect(key, collect_params, collect_job_id))
            return

        train_type = f"train_{params['family']}"
        train_params = dict(params)
        train_params.update(result)
        train_job_id = await self.jobs.enqueue(train_type, key, train_params)
        async with self._lock:
            self._current_jobs.setdefault(key, {})["train"] = train_job_id
        await self.events.publish({
            "event": "job",
            "type": "training",
            "status": "queued",
            "job_id": train_job_id,
            "key": key,
        })
        asyncio.create_task(self._watch_train(key, train_params, train_job_id))

    async def _watch_train(self, key: str, params: Dict[str, Any], job_id: str) -> None:
        job = await self.jobs.wait(job_id)
        await self.events.publish({
            "event": "job",
            "type": "training",
            "status": job.status,
            "job_id": job_id,
            "key": key,
        })
        if job.status != "done" or not job.result:
            async with self._lock:
                self._status[key] = "error"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "error", "key": key})
            return
        result = dict(job.result)
        async with self._lock:
            ctx = self._contexts.setdefault(key, {})
            ctx.update(result)

        status = result.get("status")
        if status == "trained":
            family = params.get("family", "xgb")
            version = result.get("version")
            if version:
                try:
                    self.catalog.activate(family, version)
                except ValueError:
                    LOGGER.warning("Failed to activate model %s:%s", family, version)
            warm_params = dict(params)
            warm_params.update({"version": version})
            warm_job_id = await self.jobs.enqueue("warmup_predictor", key, warm_params)
            async with self._lock:
                self._current_jobs.setdefault(key, {})["warmup"] = warm_job_id
            await self.events.publish({
                "event": "job",
                "type": "warmup",
                "status": "queued",
                "job_id": warm_job_id,
                "key": key,
            })
            asyncio.create_task(self._watch_warmup(key, warm_params, warm_job_id))
        elif status == "cooldown":
            async with self._lock:
                self._status[key] = "ready"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "ready", "key": key})
        else:
            async with self._lock:
                self._status[key] = "collecting"
            await self.events.publish({"event": "job", "type": "pipeline", "status": "collecting", "key": key})

    async def _watch_warmup(self, key: str, params: Dict[str, Any], job_id: str) -> None:
        job = await self.jobs.wait(job_id)
        ready = job.status == "done" and job.result and job.result.get("status") == "ready"
        await self.events.publish({
            "event": "job",
            "type": "warmup",
            "status": job.status,
            "job_id": job_id,
            "key": key,
        })
        version = None
        if job.result:
            version = job.result.get("version") or params.get("version")
        async with self._lock:
            self._status[key] = "ready" if ready else "missing"
            self._current_jobs.pop(key, None)
        await self.events.publish({
            "event": "job",
            "type": "pipeline",
            "status": self._status[key],
            "key": key,
        })
        if ready and version:
            await self.events.publish({"event": "model_reloaded", "version": version, "family": params.get("family", "xgb")})
def format_reasoning(decision: PolicyDecision) -> List[str]:
    reason = decision.reason
    output: List[str] = []
    for guard in reason.get("guards", []):
        status = "ok" if guard.get("ok") else "blocked"
        output.append(f"guard:{guard.get('name')} {status}")
    for flt in reason.get("filters", []):
        action = flt.get("action", "")
        entry = f"filter:{flt.get('name')}"
        if action:
            entry = f"{entry} {action}"
        output.append(entry)
    if reason.get("side"):
        output.append(f"side:{reason['side']}")
    if reason.get("size"):
        output.append(f"size:{reason['size']}")
    return output


async def ingest_documents(
    base_path: Path,
    store: FaissStore,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder,
) -> Tuple[int, int]:
    documents = load_export(base_path)
    text_docs = [doc for doc in documents if doc.kind == "text" and doc.content.strip()]
    image_docs = [doc for doc in documents if doc.kind == "image"]

    added_docs = 0
    added_images = 0

    if text_docs:
        vectors = text_embedder.embed([doc.content for doc in text_docs])
        metadata = [
            {
                "title": doc.title,
                "path": str(doc.path),
                "tags": doc.tags,
                "kind": doc.kind,
            }
            for doc in text_docs
        ]
        store.add_text(vectors, metadata)
        added_docs = len(text_docs)

    if image_docs:
        vectors = image_embedder.embed([doc.path for doc in image_docs])
        metadata = [
            {
                "title": doc.title,
                "path": str(doc.path),
                "tags": doc.tags,
                "kind": doc.kind,
                "ocr": doc.content,
            }
            for doc in image_docs
        ]
        store.add_images(vectors, metadata)
        added_images = len(image_docs)

    return added_docs, added_images


async def build_predict_payload(
    app: FastAPI,
    symbol: str,
    interval: str,
    model: ModelRunner,
    paper_mode: bool,
    model_status: str,
) -> Dict[str, object]:
    state, candle, features, frame = await app.state.streams.next_step(symbol, interval)
    if candle is None:
        last = state.candle_history[-1] if getattr(state, "candle_history", None) else None
        if last is None:
            raise RuntimeError("No market data available for stream")
        candle = dict(last)
    else:
        candle = dict(candle)

    if features is None:
        features_dict = dict(getattr(state, "last_features", {}))
    else:
        features_dict = dict(features)

    context = {
        "features": features_dict,
        "state": {
            "balance": state.balance,
            "position": state.position,
            "pnl": state.pnl,
        },
        "memory_hits": {},
    }

    candles_df = pd.DataFrame(list(state.candle_history)) if getattr(state, "candle_history", None) else pd.DataFrame([candle])
    context["candles"] = candles_df

    stub_mode = False
    prediction: Optional[Dict[str, object]] = None
    warnings: List[str] = []

    if not model.ready or not features_dict:
        stub_mode = True
        prediction = None
    else:
        try:
            prediction = model.predict(features_dict, frame)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.warning("Prediction failed for %s:%s: %s", model.family, model.name, exc)
            warnings.append(str(exc))

    if prediction is None:
        stub_mode = True
        prediction = {
            "probs": {"up": 0.33, "down": 0.33, "flat": 0.34},
            "pred_dir": "flat",
            "pred_conf": 0.33,
            "confidence_interval": (0.3, 0.7),
            "grey_zone": (0.45, 0.55),
            "regime_bucket": None,
        }

    decision = app.state.policy.apply(prediction, context)

    state.apply_decision(decision.side, candle["close"])

    plan = propose_entry_plan(
        context,
        decision.side,
        prediction.get("probs", {}).get("up") if prediction else None,
        settings=app.state.settings,
    )
    target_price = plan.get("target_price") if plan else None
    entry_plan = plan.get("entry_plan") if plan else None

    reasoning = format_reasoning(decision)
    status_out = model_status
    if stub_mode:
        status_out = "stub"
    elif model.ready:
        status_out = "ready"

    ts_candidate = candle.get("ts_ms_utc")
    if ts_candidate is None:
        timestamp_obj = candle.get("timestamp")
        if isinstance(timestamp_obj, datetime):
            if timestamp_obj.tzinfo is None:
                timestamp_obj = timestamp_obj.replace(tzinfo=timezone.utc)
            ts_ms = int(timestamp_obj.timestamp() * 1000)
        else:
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    else:
        try:
            ts_ms = int(ts_candidate)
        except (TypeError, ValueError):
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    candle["ts_ms_utc"] = ts_ms
    if not isinstance(candle.get("timestamp"), datetime):
        candle["timestamp"] = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

    prediction_signal = prediction.get("pred_dir", "flat") if prediction else "flat"
    structure_hits = list(decision.reason.get("structure", [])) if isinstance(decision.reason, dict) else []
    knowledge_raw = decision.reason.get("memory_hits", {}) if isinstance(decision.reason, dict) else {}
    knowledge_hits = sorted(knowledge_raw.keys()) if isinstance(knowledge_raw, dict) else []
    pattern_hits = {"structure": structure_hits, "knowledge": knowledge_hits}
    prediction_entry = {
        "ts_ms_utc": ts_ms,
        "prob_up": float(prediction.get("probs", {}).get("up", 0.0)),
        "prob_down": float(prediction.get("probs", {}).get("down", 0.0)),
        "prob_flat": float(prediction.get("probs", {}).get("flat", 0.0)),
        "confidence": float(prediction.get("pred_conf", 0.0)),
        "confidence_interval": prediction.get("confidence_interval"),
        "grey_zone": prediction.get("grey_zone"),
        "signal": prediction_signal,
        "pred_dir": prediction_signal,
        "structure": structure_hits,
        "knowledge": knowledge_hits,
        "regime_bucket": prediction.get("regime_bucket"),
    }

    prediction_history = getattr(state, "prediction_marks", None)
    if prediction_history is None:
        prediction_history = deque(maxlen=getattr(state, "_history_window", 1200))
        setattr(state, "prediction_marks", prediction_history)
    if not prediction_history or int(prediction_history[-1].get("ts_ms_utc", -1)) != ts_ms:
        prediction_history.append(dict(prediction_entry))
    else:
        prediction_history[-1] = dict(prediction_entry)

    payload: Dict[str, Any] = {
        "server_time_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        "symbol": symbol.upper(),
        "interval": interval,
        "ohlc": {
            "ts_ms_utc": ts_ms,
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle.get("volume", 0.0)),
        },
        "prob_up": float(prediction.get("probs", {}).get("up", 0.0)),
        "prob_down": float(prediction.get("probs", {}).get("down", 0.0)),
        "prob_flat": float(prediction.get("probs", {}).get("flat", 0.0)),
        "pred_conf": float(prediction.get("pred_conf", 0.0)),
        "confidence_interval": prediction.get("confidence_interval"),
        "grey_zone": prediction.get("grey_zone"),
        "regime_bucket": prediction.get("regime_bucket"),
        "side": decision.side or "flat",
        "size": float(decision.size or 0.0),
        "tp": app.state.settings.risk.take_profit,
        "sl": app.state.settings.risk.stop_loss,
        "position": state.position,
        "pnl": float(state.pnl),
        "balance": float(state.balance),
        "paper_mode": paper_mode,
        "reasoning": reasoning,
        "guards": decision.reason.get("guards", []),
        "memory_hits": decision.reason.get("memory_hits", {}),
        "model_status": status_out,
        "model_family": model.family,
        "model_name": model.name,
        "version": model.name,
        "target_price": target_price,
        "entry_plan": entry_plan,
        "prediction": prediction_entry,
        "pattern_hits": pattern_hits,
        "pred_candles": [dict(mark) for mark in list(prediction_history)[-500:]] if prediction_history else [],
        "explain": {
            "rules": reasoning,
            "memory_hits": decision.reason.get("memory_hits", {}),
        },
    }
    payload["rule_tags"] = reasoning if reasoning else [f"side:{payload['side']}"]
    if warnings:
        payload["warnings"] = warnings
    return payload


def _build_candle_payload(
    interval: str,
    candles: List[Dict[str, float]],
    readiness: Optional[Dict[str, bool]] = None,
) -> Dict[str, object]:
    server_now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval_ms = int(max(1.0, interval_to_seconds(interval)) * 1000)
    prepared: List[Dict[str, float]] = []
    max_ts = 0
    for bar in candles:
        try:
            ts_ms = int(bar["ts_ms_utc"])
            open_price = float(bar["open"])
            high_price = float(bar["high"])
            low_price = float(bar["low"])
            close_price = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            continue
        prepared.append(
            {
                "ts_ms_utc": ts_ms,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )
        max_ts = max(max_ts, ts_ms)
    if prepared:
        assert max_ts <= server_now_ms + interval_ms
    payload: Dict[str, object] = {
        "source": BINANCE_SOURCE,
        "server_now_utc": server_now_ms,
        "max_kline_ts_ms": max_ts,
        "interval": interval,
        "candles": prepared,
    }
    if readiness is not None:
        payload["frames_ready"] = readiness
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    candles_store = CandleStore(Path(settings.data.candles_store_path))
    faiss_store = FaissStore(Path("data/memory"))
    faiss_store.load(text_dim=settings.memory.faiss_dim_text, image_dim=settings.memory.faiss_dim_image)

    text_embedder = TextEmbedder(
        model_name=settings.memory.text_model,
        allowed_models=settings.memory.allowed_text_models,
    )
    image_embedder = ImageEmbedder()
    retriever = MemoryRetriever(store=faiss_store, text_embedder=text_embedder, image_embedder=image_embedder)
    policy = PolicyRuntime(load_and_compile("configs/rules.yaml", retriever=retriever))
    market_data = MarketDataProvider(
        store=candles_store,
        fetch_fraction=settings.data.market_fetch_fraction,
        fetch_min_seconds=settings.data.market_fetch_min_seconds,
        fetch_max_seconds=settings.data.market_fetch_max_seconds,
    )
    streams = StreamManager(provider=market_data, window=settings.data.window)
    bootstrap_frames = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"]
    app.state.bootstrap_frames = bootstrap_frames
    app.state.bootstrap_ready = asyncio.Event()

    async def _initial_bootstrap() -> None:
        try:
            preload_counts = await market_data.preload_cache(
                settings.data.symbol,
                bootstrap_frames,
            )
            symbol_key = settings.data.symbol.upper()
            for frame, count in preload_counts.items():
                if count:
                    bootstrap_service._BOOT_TRACKER.add((symbol_key, frame))
            await ensure_frames_bootstrap(
                market_data,
                settings.data.symbol,
                bootstrap_frames,
                settings.data.lookback_days,
                active=settings.data.interval,
                force_all=False,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            LOGGER.warning("Initial bootstrap failed: %s", exc)
        finally:
            app.state.bootstrap_ready.set()

    asyncio.create_task(_initial_bootstrap())
    registry = get_registry()
    model_catalog = ModelCatalog(base_dir=Path("models"), registry=registry)
    jobs = JobManager(settings.auto.retrain_cooldown_min)
    events = EventBroker()
    pipeline = AutoPipeline(jobs=jobs, streams=streams, catalog=model_catalog, events=events, settings=settings)
    self_train = SelfTrainManager(Path("data/self_train/progress.json"))

    ingest_registry = IngestRegistry(Path("data/memory/ingest_registry.json"))
    ingest_registry.load()

    live_dir = Path("data/live")
    feature_dir = Path("data/features")
    live_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    async def handle_collect(params: Dict[str, Any]) -> Dict[str, Any]:
        symbol = params["symbol"]
        interval = params["interval"]
        family = params.get("family", "xgb")
        min_bars = int(params.get("min_bars", settings.auto.min_bars_for_train))
        await streams.ensure_stream(symbol, interval)
        candles: List[Dict[str, Any]] = []
        try:
            while len(candles) < min_bars:
                _, candle, _, _ = await streams.next_step(symbol, interval)
                if candle is None:
                    break
                candles.append(
                    {
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle.get("volume", 0.0),
                        "timestamp": candle["timestamp"],
                    }
                )
        finally:
            await streams.release_stream(symbol, interval)

        if len(candles) < min_bars:
            history_bars = await streams.history(symbol, interval, min_bars)
            candles = [
                {
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar.get("volume", 0.0),
                    "timestamp": bar["timestamp"],
                }
                for bar in history_bars[-min_bars:]
            ]

        df = pd.DataFrame(candles)
        path = live_dir / f"{symbol.upper()}_{interval}_{family}.parquet"
        df.to_parquet(path)
        return {"candles_path": str(path), "rows": int(len(df))}

    async def handle_build_features(params: Dict[str, Any]) -> Dict[str, Any]:
        candles_path = params.get("candles_path")
        if not candles_path:
            return {"rows": 0}
        symbol = params["symbol"]
        interval = params["interval"]
        family = params.get("family", "xgb")
        df = pd.read_parquet(candles_path)
        if df.empty:
            return {"rows": 0}

        df["return"] = df["close"].pct_change().fillna(0.0)
        df["momentum"] = df["close"].diff().fillna(0.0)
        df["atr"] = atr(df["high"], df["low"], df["close"], settings.auto.target_atr_period)
        df["volatility"] = rolling_volatility(df["close"], window=20)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int).fillna(0).astype(int)
        features = df[["close", "return", "momentum", "volatility", "volume", "atr", "target"]].fillna(0.0)

        features_path = feature_dir / f"{symbol.upper()}_{interval}_{family}.parquet"
        features.to_parquet(features_path)
        return {"features_path": str(features_path), "rows": int(len(features))}

    async def handle_train_xgb(params: Dict[str, Any]) -> Dict[str, Any]:
        from ..models.xgb_train import run_training as run_xgb_training

        features_path = params.get("features_path")
        if not features_path:
            return {"status": "training_skipped", "reason": "no_features"}
        summary = run_xgb_training(features_path=Path(features_path))
        trained_entries = list(summary.get("trained", [])) if isinstance(summary, dict) else []
        if trained_entries:
            latest = trained_entries[-1]
            version = latest.get("version")
            payload: Dict[str, Any] = {
                "status": "trained",
                "version": version,
                "trained": trained_entries,
            }
            if latest.get("metrics"):
                payload["metrics"] = latest["metrics"]
            if summary.get("status"):
                payload["summary_status"] = summary["status"]
            return payload
        status = "error"
        if isinstance(summary, dict):
            status = str(summary.get("status") or "error")
        return {"status": status, "summary": summary}

    async def handle_train_tcn(params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "training_skipped", "reason": "tcn_not_implemented"}

    async def handle_warmup(params: Dict[str, Any]) -> Dict[str, Any]:
        family = params.get("family", "xgb")
        version = params.get("version")
        runner = model_catalog.load_active(family, version)
        status = "ready" if runner.ready else "missing"
        return {"status": status, "version": runner.name, "family": family}

    jobs.register_handler("collect", handle_collect)
    jobs.register_handler("build_features", handle_build_features)
    jobs.register_handler("train_xgb", handle_train_xgb)
    jobs.register_handler("train_tcn", handle_train_tcn)
    jobs.register_handler("warmup_predictor", handle_warmup)

    app.state.settings = settings
    app.state.store = faiss_store
    app.state.text_embedder = text_embedder
    app.state.image_embedder = image_embedder
    app.state.retriever = retriever
    app.state.policy = policy
    app.state.streams = streams
    app.state.model_catalog = model_catalog
    app.state.jobs = jobs
    app.state.events = events
    app.state.pipeline = pipeline
    app.state.market_data = market_data
    app.state.candles_store = candles_store
    app.state.ingest_registry = ingest_registry
    app.state.active_model_family = settings.model.kind
    app.state.active_model_name = model_catalog.active_name(settings.model.kind) or ""
    app.state.active_symbol = settings.data.symbol
    app.state.active_interval = settings.data.interval
    app.state.self_train = self_train

    notion_root = Path("notion_export")

    async def bootstrap_knowledge() -> None:
        if not notion_root.exists():
            return
        for folder in sorted(child for child in notion_root.iterdir() if child.is_dir()):
            digest = compute_folder_hash(folder)
            key = folder.relative_to(notion_root).as_posix()
            if ingest_registry.get(key) == digest:
                continue
            LOGGER.info("Ingesting knowledge from %s", folder)
            await ingest_documents(folder, store, text_embedder, image_embedder)
            ingest_registry.update(key, digest)
            symbol = app.state.active_symbol or settings.data.symbol
            interval = app.state.active_interval or settings.data.interval
            family = app.state.active_model_family or settings.model.kind
            await pipeline.trigger_training(symbol, interval, family)

    await bootstrap_knowledge()

    try:
        yield
    finally:
        await self_train.stop()


app = FastAPI(title="Autotrader API", lifespan=lifespan)
app.mount("/public", StaticFiles(directory=STATIC_DIR), name="public")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    settings = app.state.settings
    catalog: ModelCatalog = app.state.model_catalog
    default_family = app.state.active_model_family or settings.model.kind
    model_names = catalog.list_models(default_family)
    context = {
        "request": request,
        "symbol": settings.data.symbol,
        "interval": settings.data.interval,
        "model_families": ["lgbm", "xgb", "tcn"],
        "model_family": default_family,
        "model_names": model_names,
        "model_name": app.state.active_model_name,
        "text_model": app.state.text_embedder.current_name(),
        "allowed_text_models": settings.memory.allowed_text_models,
        "paper_mode": True,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    symbol: str,
    interval: str,
    model_family: str = "xgb",
    model_name: str = "",
    text_embedder: str = "",
    paper: int = 0,
) -> HTMLResponse:
    settings = app.state.settings
    catalog: ModelCatalog = app.state.model_catalog
    pipeline: AutoPipeline = app.state.pipeline
    status, _ = await pipeline.ensure(symbol, interval, model_family)
    job_key = pipeline.job_key(symbol, interval, model_family)
    app.state.active_symbol = symbol
    app.state.active_interval = interval
    context = {
        "request": request,
        "symbol": symbol,
        "interval": interval,
        "model_family": model_family,
        "model_name": model_name,
        "text_embedder": text_embedder or app.state.text_embedder.current_name(),
        "paper": int(paper),
        "allowed_text_models": settings.memory.allowed_text_models,
        "model_families": ["lgbm", "xgb", "tcn"],
        "model_names": catalog.list_models(model_family),
        "model_status": status,
        "job_key": job_key,
        "chart_timeframes": getattr(app.state, "bootstrap_frames", [interval]),
    }
    return templates.TemplateResponse("predict.html", context)


@app.get("/models")
async def list_models(family: str) -> Dict[str, List[str]]:
    catalog: ModelCatalog = app.state.model_catalog
    models = catalog.list_models(family)
    return {"models": models}


@app.get("/jobs")
async def list_jobs(key: str) -> Dict[str, object]:
    jobs: JobManager = app.state.jobs
    return {"jobs": jobs.status_for_key(key)}


@app.get("/predict/data")
async def get_predict_data(symbol: str, tf_active: str) -> Dict[str, object]:
    settings = app.state.settings
    provider: MarketDataProvider = app.state.market_data
    frames: List[str] = getattr(app.state, "bootstrap_frames", [tf_active])
    readiness, candles = await ensure_frames_bootstrap(
        provider,
        symbol,
        frames,
        settings.data.lookback_days,
        active=tf_active,
    )
    return _build_candle_payload(tf_active, candles, readiness)


@app.get("/ohlc")
async def get_ohlc(
    symbol: str,
    interval: str,
    limit: int = 1000,
    refresh: bool = False,
) -> Dict[str, object]:
    limit = max(1, min(limit, 2000))
    frames = [interval]
    if refresh:
        reset_bootstrap(symbol, frames)
    readiness, candles = await ensure_frames_bootstrap(
        app.state.market_data,
        symbol,
        frames,
        app.state.settings.data.lookback_days,
        active=interval,
        force_all=bool(refresh),
    )
    sliced = candles[-limit:]
    return _build_candle_payload(interval, sliced, readiness)


@app.post("/ohlc/gap")
async def fetch_ohlc_gap(request: GapRequest) -> Dict[str, object]:
    provider: MarketDataProvider = app.state.market_data
    candles = await provider.fetch_gap(
        request.symbol,
        request.interval,
        int(request.start_ms),
        int(request.end_ms),
    )
    payload = _build_candle_payload(request.interval, candles, readiness=None)
    payload["range"] = {
        "start_ms": int(min(request.start_ms, request.end_ms)),
        "end_ms": int(max(request.start_ms, request.end_ms)),
    }
    payload["symbol"] = request.symbol
    return payload


@app.post("/ohlc/backfill")
async def backfill_ohlc(request: GapRequest) -> Dict[str, object]:
    provider: MarketDataProvider = app.state.market_data
    candles = await provider.fetch_gap(
        request.symbol,
        request.interval,
        int(request.start_ms),
        int(request.end_ms),
    )
    payload = _build_candle_payload(request.interval, candles, readiness=None)
    payload["range"] = {
        "start_ms": int(min(request.start_ms, request.end_ms)),
        "end_ms": int(max(request.start_ms, request.end_ms)),
    }
    payload["symbol"] = request.symbol
    return payload


@app.get("/settings")
async def get_settings_info() -> Dict[str, object]:
    catalog: ModelCatalog = app.state.model_catalog
    return {
        "text_model": app.state.text_embedder.current_name(),
        "allowed_text_models": app.state.settings.memory.allowed_text_models,
        "active_model_family": app.state.active_model_family,
        "active_model_name": app.state.active_model_name,
        "available_models": {
            family: catalog.list_models(family) for family in ["xgb", "tcn"]
        },
    }


@app.post("/settings/text_model")
async def set_text_model(payload: Dict[str, str] = Body(...)) -> Dict[str, object]:
    name = (payload or {}).get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Text model name is required")
    embedder: TextEmbedder = app.state.text_embedder
    try:
        embedder.set_text_model(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    app.state.retriever.text_embedder = embedder
    return {"ok": True, "text_model": embedder.current_name()}


@app.post("/settings/active_model")
async def set_active_model(payload: Dict[str, str] = Body(...)) -> Dict[str, object]:
    family = (payload or {}).get("family", "").strip()
    name = (payload or {}).get("name", "").strip()
    if not family or not name:
        raise HTTPException(status_code=400, detail="family and name are required")
    catalog: ModelCatalog = app.state.model_catalog
    try:
        catalog.activate(family, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    app.state.active_model_family = family
    app.state.active_model_name = name
    return {"ok": True, "family": family, "name": name}


@app.post("/upload")
async def upload(
    export_name: str = Form(...),
    source_dir: Optional[str] = Form(default=""),
    files: Optional[List[UploadFile]] = File(default=None),
) -> Dict[str, object]:
    export_name = export_name.strip()
    if not export_name:
        raise HTTPException(status_code=400, detail="export_name is required")

    safe_name = Path(export_name).name
    if safe_name != export_name or safe_name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid export_name")

    source_dir = (source_dir or "").strip()
    uploads: List[UploadFile] = [upload for upload in (files or []) if upload is not None]
    if not source_dir and not uploads:
        raise HTTPException(status_code=400, detail="Provide source_dir or files")

    target_root = Path("data/memory/notion_export")
    target_root.mkdir(parents=True, exist_ok=True)
    target_folder = target_root / safe_name
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []

    def _copy_tree(source: Path) -> None:
        for src in source.rglob("*"):
            if not src.is_file():
                continue
            relative = src.relative_to(source)
            destination = target_folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, destination)

    copied_any = False

    if source_dir:
        source_path = Path(source_dir).expanduser()
        if source_path.exists() and source_path.is_dir():
            _copy_tree(source_path)
            copied_any = True
        else:
            warnings.append(f"source_dir not found: {source_dir}")

    if uploads:
        def _sanitize(name: str) -> Path:
            relative = Path(name.strip().strip("/"))
            if not relative.name:
                raise HTTPException(status_code=400, detail="Invalid file name")
            if relative.is_absolute() or ".." in relative.parts:
                raise HTTPException(status_code=400, detail=f"Invalid path: {name}")
            return relative

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            upload_root = base_dir / "incoming"
            upload_root.mkdir(parents=True, exist_ok=True)
            saved_files: List[Path] = []

            for upload in uploads:
                filename = upload.filename or "upload"
                try:
                    relative = _sanitize(filename)
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, str) else "Invalid path"
                    warnings.append(str(detail))
                    continue
                data = await upload.read()
                destination = upload_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                saved_files.append(destination)

            ingest_root = upload_root
            if len(saved_files) == 1 and saved_files[0].suffix.lower() == ".zip":
                extract_dir = base_dir / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(saved_files[0], "r") as archive:
                        archive.extractall(extract_dir)
                    ingest_root = extract_dir
                except zipfile.BadZipFile:
                    warnings.append("Failed to extract ZIP archive")

            if saved_files:
                _copy_tree(ingest_root)
                copied_any = True

    if not copied_any:
        raise HTTPException(status_code=400, detail="No data was imported")

    added_docs, added_images = await ingest_documents(
        target_folder,
        app.state.store,
        app.state.text_embedder,
        app.state.image_embedder,
    )

    registry: IngestRegistry = app.state.ingest_registry
    digest = compute_folder_hash(target_folder)
    registry.update(target_folder.relative_to(target_root).as_posix(), digest)

    pipeline: AutoPipeline = app.state.pipeline
    active_family = app.state.active_model_family or app.state.settings.model.kind
    symbol = app.state.active_symbol or app.state.settings.data.symbol
    interval = app.state.active_interval or app.state.settings.data.interval
    job_id, reused = await pipeline.trigger_training(symbol, interval, active_family)

    response: Dict[str, object] = {
        "export_name": safe_name,
        "added": int(added_docs + added_images),
        "added_docs": added_docs,
        "added_images": added_images,
        "job_id": job_id,
        "reused": reused,
        "saved_to": f"memory/notion_export/{safe_name}",
    }
    if warnings:
        response["warnings"] = warnings
    return response


@app.post("/train")
async def trigger_train(payload: Dict[str, str] = Body(...), response: Response = None) -> Dict[str, object]:
    data = payload or {}
    symbol = data.get("symbol") or app.state.active_symbol or app.state.settings.data.symbol
    interval = data.get("interval") or app.state.active_interval or app.state.settings.data.interval
    family = data.get("family") or app.state.active_model_family or app.state.settings.model.kind
    pipeline: AutoPipeline = app.state.pipeline
    job_id, reused = await pipeline.trigger_training(symbol, interval, family)
    if response is not None and reused:
        response.status_code = 202
    return {"job_id": job_id, "reused": reused, "status": pipeline.status(symbol, interval, family)}


@app.post("/self_train/start")
async def start_self_train(req: SelfTrainRequest) -> Dict[str, object]:
    await app.state.self_train.start(req.symbol, req.date_from, req.date_to, resume=req.resume)
    snapshot = await app.state.self_train.status()
    return {
        "status": "started",
        "symbol": req.symbol,
        "date_from": req.date_from,
        "date_to": req.date_to,
        "resume": req.resume,
        "schema_version": snapshot.get("schema_version"),
    }


@app.get("/self_train/status")
async def self_train_status() -> Dict[str, object]:
    return await app.state.self_train.status()


@app.get("/self_train/history")
async def self_train_history() -> Dict[str, object]:
    return await app.state.self_train.history()


@app.post("/self_train/stop")
async def stop_self_train() -> Dict[str, object]:
    await app.state.self_train.stop()
    return {"status": "stopped"}


@app.post("/predict/leave")
async def predict_leave(payload: Optional[Dict[str, str]] = Body(default=None)) -> Dict[str, bool]:
    data = payload or {}
    symbol = data.get("symbol")
    interval = data.get("interval")
    if symbol and interval:
        await app.state.streams.release_stream(symbol, interval)
    return {"ok": True}


@app.get("/events")
async def events_stream(request: Request) -> StreamingResponse:
    broker: EventBroker = app.state.events
    queue = await broker.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {}\n\n"
        finally:
            await broker.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws/predict")
async def ws_predict(
    ws: WebSocket,
    symbol: str,
    interval: str,
    model_family: str = "xgb",
    model_name: str = "",
    text_embedder: str = "",
    paper: int = 0,
) -> None:
    await ws.accept()
    streams: StreamManager = app.state.streams
    await streams.ensure_stream(symbol, interval)
    broker: EventBroker = app.state.events
    queue = await broker.subscribe()
    base_settings = getattr(app.state, "settings", get_settings())

    async def send_if_open(message: Dict[str, object]) -> bool:
        if ws.application_state != WebSocketState.CONNECTED:
            return False
        try:
            await ws.send_json(message)
        except (WebSocketDisconnect, RuntimeError):
            return False
        return True

    async def forward_events() -> None:
        try:
            while True:
                event = await queue.get()
                if not await send_if_open(event):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to forward event to WS: %s", exc)

    forward_task = asyncio.create_task(forward_events())
    try:
        if text_embedder:
            try:
                app.state.text_embedder.set_text_model(text_embedder)
                app.state.retriever.text_embedder = app.state.text_embedder
            except ValueError as exc:
                await ws.send_json({"error": str(exc)})
        paper_mode = str(paper) in {"1", "true", "True"}
        requested_name = model_name.strip() or None
        catalog: ModelCatalog = app.state.model_catalog
        pipeline: AutoPipeline = app.state.pipeline
        status, _ = await pipeline.ensure(symbol, interval, model_family)
        app.state.active_symbol = symbol
        app.state.active_interval = interval
        history = await streams.history(symbol, interval, limit=1000)
        snapshot = await streams.snapshot(symbol, interval)
        pred_history: List[Dict[str, object]] = []
        if snapshot is not None:
            marks = getattr(snapshot, "prediction_marks", None)
            if marks:
                pred_history = [dict(mark) for mark in marks]
        runner = catalog.load_active(model_family, requested_name)
        app.state.active_model_family = model_family
        app.state.active_model_name = runner.name

        bootstrap = {
            "event": "bootstrap",
            "symbol": symbol.upper(),
            "interval": interval,
            "ohlc": [
                {
                    "ts_ms_utc": int(bar.get("ts_ms_utc") or int(datetime.now(timezone.utc).timestamp() * 1000)),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                }
                for bar in history
            ],
            "model_status": status,
            "version": runner.name,
            "pred_candles": pred_history,
        }
        if not await send_if_open(bootstrap):
            return

        while True:
            runner = catalog.load_active(model_family, requested_name)
            app.state.active_model_family = model_family
            app.state.active_model_name = runner.name
            status = pipeline.status(symbol, interval, model_family)
            try:
                payload = await build_predict_payload(app, symbol, interval, runner, paper_mode, status)
            except RuntimeError as exc:
                LOGGER.debug("Skipping tick for %s %s: %s", symbol.upper(), interval, exc)
                await asyncio.sleep(1.0)
                continue
            payload["event"] = "tick"
            payload.setdefault("version", runner.name)
            if not await send_if_open(payload):
                break
            current_settings = getattr(app.state, "settings", base_settings)
            data_settings = current_settings.data
            tick_seconds = compute_refresh_seconds(
                interval,
                fraction=data_settings.stream_tick_fraction,
                min_seconds=data_settings.stream_tick_min_seconds,
                max_seconds=data_settings.stream_tick_max_seconds,
            )
            await asyncio.sleep(tick_seconds)
    except WebSocketDisconnect:
        pass
    finally:
        if forward_task and not forward_task.done():
            forward_task.cancel()
            with contextlib.suppress(Exception):
                await forward_task
        await broker.unsubscribe(queue)
        await streams.release_stream(symbol, interval)
