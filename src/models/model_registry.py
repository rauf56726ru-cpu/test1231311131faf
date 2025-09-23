"""Simple file-based model registry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

REGISTRY_PATH = Path("models/registry.json")


@dataclass
class ModelRecord:
    model_type: str
    version: str
    path: str
    horizon: str
    metrics: Dict[str, float]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "path": self.path,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class ModelRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Dict[str, object]]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"models": {}, "active": {}}

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def register(
        self,
        model_type: str,
        version: str,
        path: str,
        *,
        horizon: str = "default",
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, object]] = None,
        activate: bool = True,
    ) -> ModelRecord:
        horizon_key = str(horizon)
        record = ModelRecord(
            model_type=model_type,
            version=version,
            path=path,
            horizon=horizon_key,
            metrics=metrics or {},
            metadata=metadata or {},
        )
        model_bucket = self._data.setdefault("models", {}).setdefault(model_type, {})
        horizon_bucket = model_bucket.setdefault(horizon_key, {})
        horizon_bucket[version] = record.to_dict()

        if activate:
            active_section = self._data.setdefault("active", {})
            raw_active = active_section.get(model_type)
            if isinstance(raw_active, str):
                LOGGER.debug("Migrating legacy active entry for model %s", model_type)
                if raw_active:
                    active_map: Dict[str, str] = {"default": raw_active}
                else:
                    active_map = {}
            elif isinstance(raw_active, dict):
                active_map = raw_active
            else:
                active_map = {}
            active_map[horizon_key] = version
            active_section[model_type] = active_map
        self._save()
        LOGGER.info("Registered %s model version=%s (activate=%s)", model_type, version, activate)
        return record

    def set_active(self, model_type: str, version: str, *, horizon: str = "default") -> None:
        models = self._data.get("models", {}).get(model_type, {}).get(str(horizon), {})
        if version not in models:
            raise ValueError(f"Unknown version {version} for model {model_type} horizon={horizon}")
        self._data.setdefault("active", {}).setdefault(model_type, {})[str(horizon)] = version
        self._save()

    def list_models(self, model_type: str, *, horizon: str = "default") -> List[str]:
        models = self._data.get("models", {}).get(model_type, {}).get(str(horizon), {})
        return sorted(models.keys())

    def horizons(self, model_type: str) -> List[str]:
        models = self._data.get("models", {}).get(model_type, {})
        return sorted(models.keys())

    def latest(self, model_type: str, *, horizon: str = "default") -> Optional[tuple[str, str]]:
        models = self._data.get("models", {}).get(model_type, {}).get(str(horizon), {})
        if not models:
            return None
        version = max(models.keys())
        path = models[version]["path"]
        return version, path

    def get(self, model_type: str, version: str, *, horizon: str = "default") -> Optional[ModelRecord]:
        models = self._data.get("models", {}).get(model_type, {}).get(str(horizon), {})
        info = models.get(version)
        if not info:
            return None
        return ModelRecord(**info)

    def active_version(self, model_type: str, *, horizon: str = "default") -> Optional[str]:
        active = self._data.get("active", {}).get(model_type, {})
        if isinstance(active, str):
            # backward compatibility with legacy layout
            return active if horizon == "default" else None
        return active.get(str(horizon))

    def get_active(self, model_type: str, *, horizon: str = "default") -> Optional[ModelRecord]:
        active_version = self.active_version(model_type, horizon=horizon)
        if not active_version:
            return None
        return self.get(model_type, active_version, horizon=horizon)


def get_registry() -> ModelRegistry:
    return ModelRegistry()
