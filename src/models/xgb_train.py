"""Training pipeline for XGBoost models with walk-forward validation and calibration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None
import pandas as pd
from sklearn.metrics import log_loss, matthews_corrcoef
from sklearn.preprocessing import RobustScaler

from ..config import get_settings, load_backtest_config
from ..utils.logging import get_logger
from ..utils.validation import PurgedKFold
from .model_registry import get_registry
from .calibration import RegimeCalibrator

LOGGER = get_logger(__name__)

MODEL_DIR = Path("models/xgb")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MIN_SAMPLES = 5_000
CLASS_ORDER = (-1, 0, 1)


class ConstantProbabilityModel:
    """Fallback model emitting a degenerate probability distribution."""

    def __init__(self, n_outputs: int = 1) -> None:
        self.n_outputs = max(int(n_outputs), 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        n = X.shape[0]
        proba = np.zeros((n, self.n_outputs), dtype=float)
        proba[:, 0] = 1.0
        return proba

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore


@dataclass
class TrainingArtifacts:
    model: object
    scaler: RobustScaler
    calibrator: Optional[RegimeCalibrator]
    feature_list: List[str]
    metrics: Dict[str, float]
    metadata: Dict[str, object]


def _expected_calibration_error(y_true: np.ndarray, probas: np.ndarray, bins: int = 15) -> float:
    max_prob = probas.max(axis=1)
    preds = probas.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(max_prob, bin_edges) - 1
    ece = 0.0
    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_conf = max_prob[mask].mean()
        bin_acc = (preds[mask] == y_true[mask]).mean()
        ece += abs(bin_conf - bin_acc) * (mask.sum() / len(y_true))
    return float(ece)


def _load_feature_table(path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, object]]:
    df = pd.read_parquet(path)
    if "target" not in df.columns:
        raise ValueError(f"Missing target column in {path}")
    y_raw = df.pop("target").astype(int)
    regime_labels = df.pop("regime_label") if "regime_label" in df.columns else pd.Series("med", index=df.index)
    X = df.astype(float)
    valid = y_raw.notna()
    X = X.loc[valid]
    y_raw = y_raw.loc[valid]
    regime_labels = regime_labels.loc[valid].astype(str)
    class_to_idx = {value: idx for idx, value in enumerate(CLASS_ORDER)}
    y = y_raw.map(class_to_idx).fillna(1).astype(int)
    meta_info: Dict[str, object] = {"class_to_idx": class_to_idx, "idx_to_class": {v: k for k, v in class_to_idx.items()}}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta_info.update(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            LOGGER.warning("Failed to read metadata alongside %s", path)
    return X, y, regime_labels, meta_info


def _feature_paths(root: Path, horizons: List[int]) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for horizon in horizons:
        pattern = f"**/features_h{horizon}.parquet"
        matches = list(root.glob(pattern))
        if matches:
            mapping[horizon] = matches[0]
    if not mapping:
        legacy = root / "features.parquet"
        if legacy.exists() and horizons:
            mapping[horizons[0]] = legacy
    return mapping


def _vol_reference_series(X: pd.DataFrame) -> pd.Series:
    if "realized_vol_60" in X.columns:
        return X["realized_vol_60"].astype(float)
    if "vol_30" in X.columns:
        return X["vol_30"].astype(float)
    return pd.Series(np.nan, index=X.index)


def _prepare_label_encoding(y: pd.Series) -> Tuple[pd.Series, List[int]]:
    present = sorted({int(value) for value in y.unique()})
    mapping = {original: idx for idx, original in enumerate(present)}
    encoded = y.map(mapping).astype(int)
    return encoded, present


def _expand_probabilities(compact: np.ndarray, present: List[int]) -> np.ndarray:
    if compact.ndim == 1:
        compact = compact.reshape(-1, 1)
    total_classes = len(CLASS_ORDER)
    full = np.zeros((compact.shape[0], total_classes), dtype=float)
    for compact_idx, original_idx in enumerate(present):
        if 0 <= original_idx < total_classes:
            full[:, original_idx] = compact[:, compact_idx]
    missing = [idx for idx in range(total_classes) if idx not in present]
    if missing:
        epsilon = 1e-6
        full[:, missing] = epsilon
        denom = full.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        full = full / denom
    return full


def _train_constant_model(
    X: pd.DataFrame,
    y_true: pd.Series,
    regimes: pd.Series,
    vol_series: pd.Series,
    present: List[int],
) -> TrainingArtifacts:
    compact_proba = np.ones((len(X), 1), dtype=float)
    full = _expand_probabilities(compact_proba, present)
    calibrator, calibrated = _fit_calibrator(y_true.values, full, regimes, vol_series)
    metrics = _summarise_metrics(y_true.values, calibrated)
    metrics.update(_per_regime_metrics(y_true.values, calibrated, regimes))
    metadata = {
        "params": {"model_type": "constant", "num_class": len(present)},
        "study_best_value": metrics.get("mcc", 0.0),
        "trials": 0,
        "class_order": list(CLASS_ORDER),
        "present_class_indices": present,
        "present_classes": [CLASS_ORDER[idx] for idx in present],
    }
    if calibrator is not None:
        metadata.update(
            {
                "vol_thresholds": getattr(calibrator, "vol_thresholds", {}),
                "grey_zone": getattr(calibrator, "grey_zone", {}),
                "temperatures": getattr(calibrator, "temperatures", {}),
            }
        )
    return TrainingArtifacts(
        model=ConstantProbabilityModel(n_outputs=1),
        scaler=None,
        calibrator=calibrator,
        feature_list=list(X.columns),
        metrics=metrics,
        metadata=metadata,
    )


def _passes_activation(metrics: Dict[str, float]) -> bool:
    if metrics.get("ece", 1.0) > 0.03:
        return False
    if metrics.get("mcc", 0.0) <= 0.0:
        return False
    if metrics.get("accuracy", 0.0) <= 0.5:
        return False
    for bucket in ("low", "med", "high"):
        key = f"ece_{bucket}"
        if key in metrics and metrics[key] > 0.05:
            return False
    return True


def _suggest_params(trial: optuna.trial.Trial) -> Dict[str, object]:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    return params


def _train_cross_validated(
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedKFold,
    trial: optuna.trial.Trial,
    present_classes: List[int],
) -> Tuple[float, np.ndarray]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for training")

    params = _suggest_params(trial)
    params.update({
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": len(present_classes),
        "tree_method": "hist",
        "verbosity": 0,
    })

    y_array = y.values
    n_classes = len(present_classes)
    oof_compact = np.zeros((len(X), n_classes), dtype=float)
    fold_scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_valid = scaler.transform(X.iloc[valid_idx])
        y_train = y_array[train_idx]
        y_valid = y_array[valid_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        proba = model.predict_proba(X_valid)
        if proba.ndim == 1:
            proba = proba.reshape(-1, n_classes)
        oof_compact[valid_idx] = proba
        preds = proba.argmax(axis=1)
        if len(np.unique(y_valid)) > 1:
            score = matthews_corrcoef(y_valid, preds)
        else:
            score = 0.0
        fold_scores.append(score)
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    expanded = _expand_probabilities(oof_compact, present_classes)
    return float(np.mean(fold_scores)), expanded


def _fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, object],
    num_classes: int,
    warm_start_model: Optional[object] = None,
) -> Tuple[object, RobustScaler]:
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for training")
    final_params = dict(params)
    final_params.update(
        {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": max(int(num_classes), 2),
        }
    )
    model = XGBClassifier(**final_params)
    fit_kwargs: Dict[str, object] = {}
    if warm_start_model is not None:
        try:
            booster = getattr(warm_start_model, "get_booster", lambda: warm_start_model)()
            fit_kwargs["xgb_model"] = booster
        except Exception as exc:  # pragma: no cover - defensive warm start guard
            LOGGER.warning("Failed to prepare warm start booster: %s", exc)
    model.fit(X_scaled, y.values, **fit_kwargs)
    return model, scaler


def _fit_calibrator(
    y_true: np.ndarray,
    probas: np.ndarray,
    regimes: pd.Series,
    vol_series: pd.Series,
) -> Tuple[Optional[RegimeCalibrator], np.ndarray]:
    if probas.ndim != 2:
        probas = probas.reshape(-1, len(CLASS_ORDER))
    calibrator = RegimeCalibrator()
    calibrator.fit(y_true, probas, regimes, vol_series=vol_series.to_numpy(copy=False))
    calibrated_rows = []
    regime_values = regimes.astype(str).values
    for idx, regime in enumerate(regime_values):
        calibrated_rows.append(calibrator.calibrate(probas[idx : idx + 1], regime_hint=str(regime))[0])
    calibrated = np.vstack(calibrated_rows) if calibrated_rows else probas
    return calibrator, calibrated


def _summarise_metrics(y_true: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
    preds = probas.argmax(axis=1)
    one_hot = np.zeros_like(probas)
    one_hot[np.arange(len(y_true)), y_true] = 1
    metrics = {
        "logloss": float(log_loss(y_true, probas, labels=list(range(len(CLASS_ORDER))))),
        "accuracy": float((preds == y_true).mean()),
        "mcc": float(matthews_corrcoef(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0,
        "brier": float(np.mean(np.sum((probas - one_hot) ** 2, axis=1))),
        "ece": _expected_calibration_error(y_true, probas),
    }
    return metrics


def _per_regime_metrics(y_true: np.ndarray, probas: np.ndarray, regimes: pd.Series) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    regime_values = regimes.astype(str).values
    for bucket in sorted(set(regime_values)):
        mask = regime_values == bucket
        if not np.any(mask):
            continue
        local_y = y_true[mask]
        local_proba = probas[mask]
        metrics[f"ece_{bucket}"] = _expected_calibration_error(local_y, local_proba)
        preds = local_proba.argmax(axis=1)
        metrics[f"mcc_{bucket}"] = (
            float(matthews_corrcoef(local_y, preds)) if len(np.unique(local_y)) > 1 else 0.0
        )
    return metrics


def _training_loop(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    vol_series: pd.Series,
    cfg: Dict[str, object],
    warm_start_model: Optional[object] = None,
) -> TrainingArtifacts:
    validation_cfg = cfg.get("validation", {}) if isinstance(cfg, dict) else {}
    folds = int(validation_cfg.get("folds", 5))
    embargo = int(validation_cfg.get("embargo_bars", 0))

    settings = get_settings()
    min_bars = int(validation_cfg.get("min_train_bars", settings.auto.min_bars_for_train))
    threshold = max(min_bars, folds * 5)

    y_original = y.astype(int)
    y_encoded, present = _prepare_label_encoding(y_original)
    if not present:
        raise ValueError("Training target does not contain any classes")
    if len(present) == 1:
        LOGGER.warning("Only one class present in target; falling back to constant model")
        return _train_constant_model(X, y_original, regimes, vol_series, present)

    if len(X) < threshold:
        LOGGER.warning("Not enough samples for training (%s < %s), using fallback parameters", len(X), threshold)
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
        }
        model, scaler = _fit_final_model(
            X,
            y_encoded,
            default_params,
            len(present),
            warm_start_model=warm_start_model,
        )
        compact = model.predict_proba(scaler.transform(X.values))
        raw_proba = _expand_probabilities(compact, present)
        calibrator, calibrated = _fit_calibrator(y_original.values, raw_proba, regimes, vol_series)
        metrics = _summarise_metrics(y_original.values, calibrated)
        metrics.update(_per_regime_metrics(y_original.values, calibrated, regimes))
        metadata = {
            "params": {**default_params, "objective": "multi:softprob", "num_class": len(present)},
            "study_best_value": metrics.get("mcc", 0.0),
            "trials": 0,
            "class_order": list(CLASS_ORDER),
            "present_class_indices": present,
            "present_classes": [CLASS_ORDER[idx] for idx in present],
        }
        if calibrator is not None:
            metadata.update(
                {
                    "vol_thresholds": calibrator.vol_thresholds,
                    "grey_zone": calibrator.grey_zone,
                    "temperatures": calibrator.temperatures,
                }
            )
        return TrainingArtifacts(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
            feature_list=list(X.columns),
            metrics=metrics,
            metadata=metadata,
        )

    if optuna is None:
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
        }
        model, scaler = _fit_final_model(
            X,
            y_encoded,
            default_params,
            len(present),
            warm_start_model=warm_start_model,
        )
        compact = model.predict_proba(scaler.transform(X.values))
        raw_proba = _expand_probabilities(compact, present)
        calibrator, calibrated = _fit_calibrator(y_original.values, raw_proba, regimes, vol_series)
        metrics = _summarise_metrics(y_original.values, calibrated)
        metrics.update(_per_regime_metrics(y_original.values, calibrated, regimes))
        metadata = {
            "params": {**default_params, "objective": "multi:softprob", "num_class": len(present)},
            "study_best_value": metrics.get("mcc", 0.0),
            "trials": 0,
            "class_order": list(CLASS_ORDER),
            "present_class_indices": present,
            "present_classes": [CLASS_ORDER[idx] for idx in present],
        }
        if calibrator is not None:
            metadata.update(
                {
                    "vol_thresholds": calibrator.vol_thresholds,
                    "grey_zone": calibrator.grey_zone,
                    "temperatures": calibrator.temperatures,
                }
            )
        return TrainingArtifacts(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
            feature_list=list(X.columns),
            metrics=metrics,
            metadata=metadata,
        )

    cv = PurgedKFold(n_splits=folds, embargo=embargo)
    study = optuna.create_study(direction="maximize", study_name="xgb_mcc")

    max_trials = int(cfg.get("tuning", {}).get("trials", 50)) if isinstance(cfg, dict) else 50
    adaptive_trials = min(max_trials, max(5, len(X) // 2000 + 5))

    best_oof = None

    def objective(trial: optuna.trial.Trial) -> float:
        score, oof_pred = _train_cross_validated(X, y_encoded, cv, trial, present)
        nonlocal best_oof
        if best_oof is None or score > (study.best_value if study.best_trial else -np.inf):
            best_oof = oof_pred
        return score

    study.optimize(objective, n_trials=adaptive_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "verbosity": 0,
        "n_estimators": int(best_trial.params["n_estimators"]),
        "max_depth": int(best_trial.params["max_depth"]),
        "learning_rate": float(best_trial.params["learning_rate"]),
        "subsample": float(best_trial.params["subsample"]),
        "colsample_bytree": float(best_trial.params["colsample_bytree"]),
        "min_child_weight": float(best_trial.params["min_child_weight"]),
        "gamma": float(best_trial.params["gamma"]),
        "num_class": len(present),
    }

    model, scaler = _fit_final_model(
        X,
        y_encoded,
        best_params,
        len(present),
        warm_start_model=warm_start_model,
    )
    if best_oof is None:
        base = np.full((len(X), len(present)), 1.0 / max(len(present), 1))
        oof_pred = _expand_probabilities(base, present)
    else:
        oof_pred = best_oof
    calibrator, calibrated = _fit_calibrator(y_original.values, oof_pred, regimes, vol_series)
    metrics = _summarise_metrics(y_original.values, calibrated)
    metrics.update(_per_regime_metrics(y_original.values, calibrated, regimes))
    metadata = {
        "params": best_params,
        "study_best_value": float(study.best_value),
        "trials": adaptive_trials,
        "class_order": list(CLASS_ORDER),
        "present_class_indices": present,
        "present_classes": [CLASS_ORDER[idx] for idx in present],
    }
    if calibrator is not None:
        metadata.update(
            {
                "vol_thresholds": calibrator.vol_thresholds,
                "grey_zone": calibrator.grey_zone,
                "temperatures": calibrator.temperatures,
            }
        )

    return TrainingArtifacts(
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        feature_list=list(X.columns),
        metrics=metrics,
        metadata=metadata,
    )


def _persist_artifacts(artifacts: TrainingArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, output_dir / "model.pkl")
    joblib.dump(artifacts.scaler, output_dir / "scaler.pkl")
    joblib.dump(artifacts.calibrator, output_dir / "calibrator.pkl")
    with (output_dir / "feature_list.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts.feature_list, f, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts.metrics, f, indent=2)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts.metadata, f, indent=2)


def run_training(
    features_root: Path | str = Path("data/features"),
    config_path: Path | str | None = None,
    features_path: Path | str | None = None,
    warm_start_dir: Path | str | None = None,
) -> Dict[str, object]:
    features_root = Path(features_root)
    cfg: Dict[str, object] = {}
    if config_path is not None:
        cfg = load_backtest_config(config_path)

    settings = get_settings()
    validation_cfg = cfg.get("validation", {}) if isinstance(cfg, dict) else {}
    configured_min = int(validation_cfg.get("min_train_bars", 0) or 0)
    effective_min_samples = MIN_SAMPLES
    if settings.auto.min_bars_for_train and settings.auto.min_bars_for_train < effective_min_samples:
        effective_min_samples = int(settings.auto.min_bars_for_train)
    if configured_min and configured_min < effective_min_samples:
        effective_min_samples = configured_min
    effective_min_samples = max(effective_min_samples, 1)

    symbols = cfg.get("symbols") or [settings.data.symbol]
    intervals = cfg.get("intervals") or [settings.data.interval]
    horizons = cfg.get("horizons_min") or [settings.data.horizon_min]
    horizons = [int(h) for h in horizons]
    default_horizon = int(cfg.get("default_horizon", 15))

    registry = get_registry()
    summary: Dict[str, object] = {"trained": []}

    warm_start_model = None
    if warm_start_dir is not None:
        try:
            from .xgb_predict import load_artifacts  # local import to avoid cycle

            warm_model, _, _, _, _ = load_artifacts(warm_start_dir)
            if warm_model is not None:
                warm_start_model = warm_model
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load warm start artifacts from %s: %s", warm_start_dir, exc)

    if features_path is not None:
        X, y, regimes, meta_info = _load_feature_table(Path(features_path))
        if len(X) < effective_min_samples:
            LOGGER.warning(
                "Not enough samples for training from %s (%s < %s)",
                Path(features_path).as_posix(),
                len(X),
                effective_min_samples,
            )
            summary.setdefault("skipped", []).append(
                {
                    "path": Path(features_path).as_posix(),
                    "n_samples": int(len(X)),
                    "reason": "insufficient_samples",
                }
            )
            summary["status"] = "no_models_trained"
            return summary
        try:
            vol_series = _vol_reference_series(X)
            artifacts = _training_loop(
                X,
                y,
                regimes,
                vol_series,
                cfg,
                warm_start_model=warm_start_model,
            )
        except Exception as exc:
            LOGGER.error("Training failed for %s: %s", features_path, exc)
        else:
            version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            model_dir = MODEL_DIR / version
            _persist_artifacts(artifacts, model_dir)
            activate = _passes_activation(artifacts.metrics)
            metadata = {
                "symbol": symbols[0],
                "interval": intervals[0],
                **artifacts.metadata,
                "activated": activate,
            }
            registry.register(
                "xgb",
                version,
                str(model_dir),
                horizon="default",
                metrics=artifacts.metrics,
                metadata=metadata,
                activate=activate,
            )
            summary.setdefault("trained", []).append(
                {"symbol": symbols[0], "interval": intervals[0], "version": version, "metrics": artifacts.metrics}
            )
        if not summary["trained"]:
            summary["status"] = "no_models_trained"
        else:
            summary["status"] = "ok"
        return summary

    for symbol in symbols:
        for interval in intervals:
            base_dir = features_root / f"{symbol}_{interval}"
            if not base_dir.exists():
                base_dir = features_root
            paths = _feature_paths(base_dir, horizons)
            if not paths:
                LOGGER.warning("No feature tables found under %s", base_dir)
                continue
            for horizon, path in paths.items():
                LOGGER.info("Training XGB for %s %s horizon=%s from %s", symbol, interval, horizon, path)
                X, y, regimes, meta_info = _load_feature_table(path)
                if len(X) < effective_min_samples:
                    LOGGER.warning(
                        "Not enough samples for %s %s horizon=%s (%s < %s)",
                        symbol,
                        interval,
                        horizon,
                        len(X),
                        effective_min_samples,
                    )
                    summary.setdefault("skipped", []).append(
                        {
                            "symbol": symbol,
                            "interval": interval,
                            "horizon": horizon,
                            "path": path.as_posix(),
                            "n_samples": int(len(X)),
                            "reason": "insufficient_samples",
                        }
                    )
                    continue
                try:
                    vol_series = _vol_reference_series(X)
                    artifacts = _training_loop(X, y, regimes, vol_series, cfg)
                except Exception as exc:
                    LOGGER.error("Training failed for %s horizon=%s: %s", path, horizon, exc)
                    continue

                version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                model_dir = MODEL_DIR / f"h{horizon}" / version
                _persist_artifacts(artifacts, model_dir)

                metadata = {
                    "symbol": symbol,
                    "interval": interval,
                    "horizon": horizon,
                    **artifacts.metadata,
                }
                activate = _passes_activation(artifacts.metrics)
                metadata = {"symbol": symbol, "interval": interval, "horizon": horizon, **artifacts.metadata, "activated": activate}
                registry.register(
                    "xgb",
                    version,
                    str(model_dir),
                    horizon=f"h{horizon}",
                    metrics=artifacts.metrics,
                    metadata=metadata,
                    activate=activate,
                )
                if horizon == default_horizon:
                    registry.register(
                        "xgb",
                        version,
                        str(model_dir),
                        horizon="default",
                        metrics=artifacts.metrics,
                        metadata=metadata,
                        activate=activate,
                    )

                summary.setdefault("trained", []).append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "horizon": horizon,
                        "version": version,
                        "metrics": artifacts.metrics,
                    }
                )

    if not summary["trained"]:
        summary["status"] = "no_models_trained"
    else:
        summary["status"] = "ok"

    return summary


if __name__ == "__main__":
    run_training()
