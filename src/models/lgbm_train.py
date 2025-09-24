"""Incremental LightGBM training helpers for self-train workflow."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .calibration import RegimeCalibrator

LOGGER = get_logger(__name__)

CLASS_ORDER: Tuple[int, int, int] = (-1, 0, 1)

try:  # pragma: no cover - optional dependency runtime guard
    import lightgbm as lgb
except Exception:  # pragma: no cover - handled upstream
    lgb = None  # type: ignore


@dataclass
class IncrementalTrainingResult:
    """Container describing the outcome of an incremental training step."""

    booster_path: Path
    calibrator_path: Optional[Path]
    metadata_path: Path
    metrics: Dict[str, float]
    raw_metrics: Dict[str, float]
    is_active: bool
    activation_reason: str
    predictions: pd.DataFrame
    validation_index: pd.Index
    feature_list: List[str]
    calibrator: Optional[RegimeCalibrator]


def _encode_targets(y: pd.Series) -> Tuple[np.ndarray, Dict[int, int]]:
    mapping = {value: idx for idx, value in enumerate(CLASS_ORDER)}
    encoded = y.map(mapping).fillna(mapping.get(0, 1)).astype(int)
    return encoded.to_numpy(copy=False), mapping


def _compute_sample_weights(
    encoded: np.ndarray,
    *,
    class_weights: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    weights = np.ones_like(encoded, dtype=float)
    if encoded.size == 0:
        return weights
    unique, counts = np.unique(encoded, return_counts=True)
    if class_weights:
        for class_value, weight in class_weights.items():
            weights[encoded == class_value] = float(weight)
    else:
        distinct = max(1, len(unique))
        total = float(encoded.size)
        for class_value, count in zip(unique, counts):
            if count <= 0:
                continue
            weights[encoded == class_value] = total / (distinct * float(count))
    return weights


def _expected_calibration_error(y_true: np.ndarray, probas: np.ndarray, bins: int = 15) -> float:
    if y_true.size == 0 or probas.size == 0:
        return 0.0
    max_prob = probas.max(axis=1)
    preds = probas.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(max_prob, bin_edges) - 1
    ece = 0.0
    for bucket in range(len(bin_edges) - 1):
        mask = bin_ids == bucket
        if not np.any(mask):
            continue
        bin_conf = max_prob[mask].mean()
        bin_acc = (preds[mask] == y_true[mask]).mean()
        ece += abs(bin_conf - bin_acc) * (mask.sum() / len(y_true))
    return float(ece)


def _directional_metrics(y_true: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or probas.size == 0:
        return {"mcc": 0.0, "accuracy": 0.0, "hit_rate": 0.0, "sharpe_proxy": 0.0, "ece": 0.0}

    from sklearn.metrics import matthews_corrcoef

    preds = probas.argmax(axis=1)
    accuracy = float((preds == y_true).mean()) if y_true.size else 0.0
    mcc = float(matthews_corrcoef(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0
    encoded_to_dir = np.array(CLASS_ORDER, dtype=float)
    true_dir = encoded_to_dir[y_true]
    pred_dir = encoded_to_dir[preds]
    non_flat = pred_dir != 0.0
    if np.any(non_flat):
        hit_rate = float((true_dir[non_flat] == pred_dir[non_flat]).mean())
    else:
        hit_rate = 0.0
    pnl = true_dir * pred_dir
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size and float(np.std(pnl)) > 1e-12:
        sharpe = float(np.mean(pnl) / np.std(pnl) * math.sqrt(len(pnl)))
    else:
        sharpe = 0.0
    ece = _expected_calibration_error(y_true, probas)
    return {
        "mcc": mcc,
        "accuracy": accuracy,
        "hit_rate": hit_rate,
        "sharpe_proxy": sharpe,
        "ece": ece,
    }


def _activation_gate(metrics: Dict[str, float]) -> Tuple[bool, str]:
    ece = float(metrics.get("ece", 1.0))
    mcc = float(metrics.get("mcc", 0.0))
    if ece > 0.03:
        return False, f"ece={ece:.4f}"
    if mcc <= 0.0:
        return False, f"mcc={mcc:.4f}"
    return True, "ok"


def _guess_vol_series(regimes: Optional[pd.DataFrame], features: pd.DataFrame) -> Optional[pd.Series]:
    candidates: Sequence[str] = (
        "realized_vol_60",
        "volatility",
        "realized_vol_30",
        "atr",
    )
    for column in candidates:
        if regimes is not None and column in regimes.columns:
            series = regimes[column].astype(float)
            if series.notna().any():
                return series
        if column in features.columns:
            series = features[column].astype(float)
            if series.notna().any():
                return series
    return None


def train_incremental(
    X: pd.DataFrame,
    y: pd.Series,
    regime_labels: pd.Series,
    regimes: Optional[pd.DataFrame],
    *,
    day: str,
    symbol: str,
    interval: str,
    output_dir: Path,
    prev_model_path: Optional[Path] = None,
    class_weights: Optional[Dict[int, float]] = None,
    calibration_method: str = "isotonic",
    validation_fraction: float = 0.3,
    min_validation_rows: int = 32,
    random_state: int = 1337,
) -> IncrementalTrainingResult:
    """Train or continue a LightGBM model on a single day's feature batch."""

    if lgb is None:  # pragma: no cover - runtime guard
        raise RuntimeError("lightgbm is not installed. Add lightgbm to requirements.txt")

    if X.empty:
        raise ValueError("Feature frame is empty for incremental training")

    X = X.astype(float)
    feature_list = list(X.columns)
    encoded, mapping = _encode_targets(y)
    weights = _compute_sample_weights(encoded, class_weights=class_weights)

    validation_fraction = max(0.05, min(0.5, float(validation_fraction)))
    min_validation_rows = max(1, int(min_validation_rows))
    total_rows = len(X)
    valid_rows = max(min_validation_rows, int(total_rows * validation_fraction))
    valid_rows = min(valid_rows, max(total_rows // 2, 1))
    split_index = max(total_rows - valid_rows, 1)
    if split_index >= total_rows:
        split_index = max(total_rows // 2, 1)

    train_slice = slice(0, split_index)
    valid_slice = slice(split_index, total_rows)

    X_train = X.iloc[train_slice]
    X_valid = X.iloc[valid_slice]
    y_train = encoded[train_slice]
    y_valid = encoded[valid_slice]
    w_train = weights[train_slice]
    w_valid = weights[valid_slice]

    if X_valid.empty:
        # fall back to using the tail of the training window as validation
        tail = max(1, min(64, len(X_train)))
        X_valid = X_train.tail(tail)
        y_valid = y_train[-tail:]
        w_valid = w_train[-tail:]
        X_train = X_train.iloc[:-tail]
        y_train = y_train[:-tail]
        w_train = w_train[:-tail]

    train_set = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=train_set, free_raw_data=False)

    params = {
        "objective": "multiclass",
        "num_class": len(CLASS_ORDER),
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": max(10, total_rows // 50),
        "max_depth": -1,
        "num_leaves": min(2 ** max(1, int(math.log2(max(4, len(feature_list))))), 512),
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-3,
        "verbosity": -1,
        "force_row_wise": True,
        "seed": random_state,
    }

    init_model = str(prev_model_path) if prev_model_path else None
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    callbacks = [lgb.log_evaluation(period=0)]
    if len(X_valid) >= 20:
        callbacks.append(lgb.early_stopping(30, verbose=False))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max(40, min(400, total_rows)),
        valid_sets=[valid_set],
        valid_names=["val"],
        init_model=init_model,
        keep_training_booster=True,
        callbacks=callbacks,
        evals_result=evals_result,
    )

    proba_raw = booster.predict(X.values)
    proba_valid_raw = booster.predict(X_valid.values)

    regimes_valid = regime_labels.iloc[valid_slice]
    regimes_valid_list = [str(v) if v is not None else "" for v in regimes_valid]
    vol_series = _guess_vol_series(regimes, X)

    calibrator: Optional[RegimeCalibrator]
    calibrated_all: np.ndarray
    calibrated_valid: np.ndarray
    try:
        calibrator = RegimeCalibrator(method=calibration_method, classes=CLASS_ORDER)
        calibrator.fit(y_valid, proba_valid_raw, regimes_valid_list, vol_series=vol_series.iloc[valid_slice] if vol_series is not None else None)
    except Exception as exc:  # pragma: no cover - calibration safety
        LOGGER.warning("Failed to fit calibrator for %s %s %s: %s", symbol, interval, day, exc)
        calibrator = None

    if calibrator is not None:
        calibrated_all = calibrator.calibrate(proba_raw, regime_hint=list(regime_labels.astype(str)))
        calibrated_valid = calibrator.calibrate(proba_valid_raw, regime_hint=regimes_valid_list)
    else:
        calibrated_all = proba_raw
        calibrated_valid = proba_valid_raw

    raw_metrics = _directional_metrics(y_valid, proba_valid_raw)
    calibrated_metrics = _directional_metrics(y_valid, calibrated_valid)
    is_active, reason = _activation_gate(calibrated_metrics)

    predictions = pd.DataFrame(
        {
            "prob_down": proba_raw[:, 0],
            "prob_flat": proba_raw[:, 1],
            "prob_up": proba_raw[:, 2],
            "prob_down_calibrated": calibrated_all[:, 0],
            "prob_flat_calibrated": calibrated_all[:, 1],
            "prob_up_calibrated": calibrated_all[:, 2],
        },
        index=X.index,
    )

    day_dir = output_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    booster_path = day_dir / "model.txt"
    booster.save_model(str(booster_path))

    calibrator_path: Optional[Path] = None
    if calibrator is not None:
        calibrator_path = day_dir / "calibrator.pkl"
        joblib.dump(calibrator, calibrator_path)

    metadata = {
        "symbol": symbol,
        "interval": interval,
        "day": day,
        "created_at": datetime.utcnow().isoformat(),
        "feature_list": feature_list,
        "class_order": list(CLASS_ORDER),
        "class_mapping": {int(k): int(v) for k, v in mapping.items()},
        "num_rows": int(total_rows),
        "num_train_rows": int(len(X_train)),
        "num_valid_rows": int(len(X_valid)),
        "metrics": calibrated_metrics,
        "raw_metrics": raw_metrics,
        "evals": evals_result,
        "is_active": is_active,
        "activation_reason": reason,
    }
    metadata_path = day_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return IncrementalTrainingResult(
        booster_path=booster_path,
        calibrator_path=calibrator_path,
        metadata_path=metadata_path,
        metrics=calibrated_metrics,
        raw_metrics=raw_metrics,
        is_active=is_active,
        activation_reason=reason,
        predictions=predictions,
        validation_index=X_valid.index,
        feature_list=feature_list,
        calibrator=calibrator,
    )

