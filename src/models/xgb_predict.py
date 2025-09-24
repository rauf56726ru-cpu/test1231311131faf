"""XGBoost inference helper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

MODEL_DIR = Path("models/xgb")
_RSI_COLUMNS = ("rsi", "rsi_14")
_RSI_SIGMA = 10.0
CLASS_ORDER = (-1, 0, 1)


def _expand_with_metadata(raw: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    present = metadata.get("present_class_indices")
    if present:
        indices = [int(idx) for idx in present]
        full = np.zeros((raw.shape[0], len(CLASS_ORDER)), dtype=float)
        for col_idx, original_idx in enumerate(indices):
            if 0 <= original_idx < len(CLASS_ORDER):
                full[:, original_idx] = raw[:, col_idx]
        missing = [idx for idx in range(len(CLASS_ORDER)) if idx not in indices]
        if missing:
            epsilon = 1e-6
            full[:, missing] = epsilon
            denom = full.sum(axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            full = full / denom
        return full
    n_cols = raw.shape[1]
    if n_cols == len(CLASS_ORDER):
        return raw
    if n_cols == 2:
        full = np.zeros((raw.shape[0], len(CLASS_ORDER)), dtype=float)
        full[:, 0] = raw[:, 0]
        full[:, 2] = raw[:, 1]
        epsilon = 1e-6
        full[:, 1] = epsilon
        denom = full.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return full / denom
    if n_cols == 1:
        full = np.zeros((raw.shape[0], len(CLASS_ORDER)), dtype=float)
        full[:, 2] = raw[:, 0]
        full[:, 0] = 1.0 - raw[:, 0]
        epsilon = 1e-6
        full[:, 1] = epsilon
        denom = full.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return full / denom
    return raw


def _artifacts_exist(model_dir: Path) -> bool:
    required = [model_dir / "model.pkl", model_dir / "feature_list.json"]
    return all(path.exists() for path in required)


def _stub_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    for column in _RSI_COLUMNS:
        if column in df.columns:
            rsi = pd.to_numeric(df[column], errors="coerce").fillna(50.0)
            scaled = (rsi - 50.0) / _RSI_SIGMA
            prob_up = 1.0 / (1.0 + np.exp(-scaled))
            prob_down = 1.0 - prob_up
            data = np.vstack([prob_down, np.full_like(prob_up, 0.0), prob_up]).T
            return pd.DataFrame(data, index=df.index, columns=["prob_down", "prob_flat", "prob_up"])
    default = np.full((len(df), 3), 1 / 3)
    return pd.DataFrame(default, index=df.index, columns=["prob_down", "prob_flat", "prob_up"])


def load_artifacts(
    model_dir: Path | str = MODEL_DIR,
) -> tuple[Optional[object], Optional[list[str]], Optional[object], Optional[object], Dict[str, object]]:
    """Load XGB artifacts if available.

    Missing artifacts are tolerated – ``None`` values are returned so that the
    caller can degrade gracefully without crashing the prediction stream.
    """

    model_dir = Path(model_dir)
    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    calibrator_path = model_dir / "calibrator.pkl"
    feature_path = model_dir / "feature_list.json"
    metadata_path = model_dir / "metadata.json"

    model: Optional[object] = None
    scaler: Optional[object] = None
    calibrator: Optional[object] = None
    features: Optional[list[str]] = None
    metadata: Dict[str, object] = {}

    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load model artifact from %s: %s", model_path, exc)
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load scaler artifact from %s: %s", scaler_path, exc)
            scaler = None
    if feature_path.exists():
        try:
            with feature_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            features = list(raw)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to read feature list from %s: %s", feature_path, exc)
            features = None

    if calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load calibrator from %s: %s", calibrator_path, exc)
            calibrator = None

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load metadata from %s: %s", metadata_path, exc)

    return model, features, scaler, calibrator, metadata


def build_feature_vector(
    latest_features: Dict[str, float],
    feature_list: Iterable[str],
    scaler: Optional[object],
) -> np.ndarray:
    """Construct a model-ready feature vector.

    Missing feature values are filled with zeros and ordered strictly according
    to ``feature_list``. ``scaler`` is optional – if provided it must expose a
    ``transform`` method compatible with scikit-learn scalers.
    """

    ordered = []
    for name in feature_list:
        value = latest_features.get(name, 0.0)
        try:
            ordered.append(float(value))
        except (TypeError, ValueError):
            ordered.append(0.0)
    vector = np.array([ordered], dtype=float)
    if scaler is not None:
        vector = scaler.transform(vector)
    return vector


def predict_distribution(
    feature_row: Dict[str, float],
    model_dir: Optional[Path | str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    target_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    if not _artifacts_exist(target_dir):
        raise RuntimeError("Model artifacts are missing; unable to infer")
    model, feature_list, scaler, calibrator, metadata = load_artifacts(target_dir)
    if model is None or not feature_list:
        raise RuntimeError("Model artifacts are missing; unable to infer")
    row_scaled = build_feature_vector(feature_row, feature_list, scaler)
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(row_scaled)
    else:
        prediction = model.predict(row_scaled)
        if isinstance(prediction, np.ndarray):
            raw = prediction
        else:
            raw = np.array([[float(prediction)]])
    expanded = _expand_with_metadata(np.asarray(raw, dtype=float), metadata)
    if calibrator is not None:
        calibrated = calibrator.calibrate(expanded, features=feature_row)
    else:
        calibrated = expanded
    LOGGER.debug("Predicted distribution: %s", calibrated)
    return calibrated[0], metadata


def batch_predict(df: pd.DataFrame, model_dir: Optional[Path | str] = MODEL_DIR) -> pd.DataFrame:
    if model_dir is None:
        return _stub_probabilities(df)

    model_path = Path(model_dir)
    if not _artifacts_exist(model_path):
        return _stub_probabilities(df)

    model, feature_list, scaler, calibrator, metadata = load_artifacts(model_path)
    if model is None or not feature_list:
        return _stub_probabilities(df)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0
    ordered = df[list(feature_list)].copy()
    X_scaled = scaler.transform(ordered.values) if scaler is not None else ordered.values
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X_scaled)
    else:
        raw = np.asarray(model.predict(X_scaled), dtype=float)
    expanded = _expand_with_metadata(np.asarray(raw, dtype=float), metadata)
    if calibrator is not None:
        calibrated_rows = []
        for idx, (_, feature_row) in enumerate(ordered.iterrows()):
            calibrated_rows.append(
                calibrator.calibrate(expanded[idx : idx + 1], features=feature_row.to_dict())[0]
            )
        calibrated = np.vstack(calibrated_rows)
    else:
        calibrated = expanded
    columns = ["prob_down", "prob_flat", "prob_up"]
    return pd.DataFrame(calibrated, index=df.index, columns=columns)
