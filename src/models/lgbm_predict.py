"""Inference helpers for LightGBM models with regime calibration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .calibration import RegimeCalibrator

LOGGER = get_logger(__name__)

MODEL_DIR = Path("models/lgbm")
CLASS_ORDER = (-1, 0, 1)

try:  # pragma: no cover - optional dependency guard
    import lightgbm as lgb
except Exception:  # pragma: no cover - handled by caller
    lgb = None  # type: ignore


def _artifacts_exist(model_dir: Path) -> bool:
    required = [model_dir / "model.txt", model_dir / "metadata.json"]
    return all(path.exists() for path in required)


def load_artifacts(
    model_dir: Path | str = MODEL_DIR,
) -> tuple[Optional[object], Optional[Sequence[str]], Optional[object], Optional[RegimeCalibrator], Dict[str, object]]:
    """Load LightGBM booster, calibrator and metadata if they exist."""

    model_dir = Path(model_dir)
    model_path = model_dir / "model.txt"
    metadata_path = model_dir / "metadata.json"
    calibrator_path = model_dir / "calibrator.pkl"

    booster: Optional[object] = None
    feature_list: Optional[Sequence[str]] = None
    metadata: Dict[str, object] = {}
    calibrator: Optional[RegimeCalibrator] = None

    if model_path.exists() and lgb is not None:
        try:
            booster = lgb.Booster(model_file=str(model_path))
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to load LightGBM booster from %s: %s", model_path, exc)
            booster = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_list = metadata.get("feature_list")
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to read metadata from %s: %s", metadata_path, exc)
            metadata = {}
            feature_list = None
    if calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to load calibrator from %s: %s", calibrator_path, exc)
            calibrator = None
    return booster, feature_list, None, calibrator, metadata


def _prepare_matrix(
    rows: Iterable[Dict[str, float]],
    feature_list: Optional[Sequence[str]],
) -> np.ndarray:
    if feature_list is None:
        feature_list = []
    matrix = []
    for row in rows:
        ordered = []
        for name in feature_list:
            value = row.get(name, 0.0)
            try:
                ordered.append(float(value))
            except (TypeError, ValueError):
                ordered.append(0.0)
        matrix.append(ordered)
    if not matrix:
        return np.zeros((0, len(feature_list) or 1), dtype=float)
    return np.asarray(matrix, dtype=float)


def _expand_probabilities(raw: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    num_classes = len(CLASS_ORDER)
    if raw.shape[1] == num_classes:
        return raw
    present = metadata.get("class_order")
    if present and len(present) == num_classes:
        return raw
    full = np.zeros((raw.shape[0], num_classes), dtype=float)
    cols = min(raw.shape[1], num_classes)
    full[:, :cols] = raw[:, :cols]
    denom = full.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return full / denom


def predict_distribution(
    feature_row: Dict[str, float],
    model_dir: Optional[Path | str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    target_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    if not _artifacts_exist(target_dir):
        raise RuntimeError(f"Missing LightGBM artifacts in {target_dir}")
    booster, features, _, calibrator, metadata = load_artifacts(target_dir)
    if booster is None or features is None:
        raise RuntimeError("LightGBM artifacts are incomplete")
    matrix = _prepare_matrix([feature_row], features)
    raw = booster.predict(matrix)
    expanded = _expand_probabilities(np.asarray(raw, dtype=float), metadata)
    if calibrator is not None:
        calibrated = calibrator.calibrate(expanded, features=feature_row)
    else:
        calibrated = expanded
    return calibrated[0], metadata


def batch_predict(df: pd.DataFrame, model_dir: Optional[Path | str] = None) -> pd.DataFrame:
    target_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    if not _artifacts_exist(target_dir) or lgb is None:
        default = np.full((len(df), 3), 1 / 3)
        return pd.DataFrame(default, index=df.index, columns=["prob_down", "prob_flat", "prob_up"])

    booster, features, _, calibrator, metadata = load_artifacts(target_dir)
    if booster is None or features is None:
        default = np.full((len(df), 3), 1 / 3)
        return pd.DataFrame(default, index=df.index, columns=["prob_down", "prob_flat", "prob_up"])

    rows = df[features].astype(float).to_dict(orient="records")
    matrix = _prepare_matrix(rows, features)
    raw = booster.predict(matrix)
    expanded = _expand_probabilities(np.asarray(raw, dtype=float), metadata)
    if calibrator is not None:
        calibrated = calibrator.calibrate(expanded, features=df.to_dict(orient="records"))
    else:
        calibrated = expanded
    return pd.DataFrame(
        calibrated,
        index=df.index,
        columns=["prob_down", "prob_flat", "prob_up"],
    )

