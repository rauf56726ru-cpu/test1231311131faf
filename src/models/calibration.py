"""Regime-aware probability calibration utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

_DEFAULT_GREY_ZONE = (0.45, 0.55)


class _OneVsRestCalibrator:
    """One-vs-rest probability calibrator for multi-class outputs."""

    def __init__(self, method: str, n_classes: int) -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self.n_classes = int(n_classes)
        self.models: Dict[int, Optional[object]] = {idx: None for idx in range(self.n_classes)}

    def fit(self, y_true: np.ndarray, probas: np.ndarray) -> None:
        y_true = np.asarray(y_true, dtype=int)
        probas = np.asarray(probas, dtype=float)
        if probas.ndim != 2:
            raise ValueError("Probabilities must be a 2D array")
        if probas.shape[1] != self.n_classes:
            raise ValueError("Probability array column count mismatch")

        for idx in range(self.n_classes):
            column = probas[:, idx]
            target = (y_true == idx).astype(int)
            # Skip calibration if class is missing or probabilities are constant.
            if target.sum() == 0 or target.sum() == len(target) or np.allclose(column, column[0]):
                self.models[idx] = None
                continue
            try:
                if self.method == "isotonic":
                    model = IsotonicRegression(out_of_bounds="clip")
                    model.fit(column, target)
                else:  # platt scaling via logistic regression
                    model = LogisticRegression(max_iter=200)
                    model.fit(column.reshape(-1, 1), target)
            except Exception:
                model = None
            self.models[idx] = model

    def transform(self, probas: np.ndarray) -> np.ndarray:
        probas = np.asarray(probas, dtype=float)
        if probas.ndim != 2:
            probas = probas.reshape(-1, self.n_classes)
        if probas.shape[1] != self.n_classes:
            raise ValueError("Probability array column count mismatch during transform")

        calibrated = np.array(probas, copy=True)
        for idx, model in self.models.items():
            if model is None:
                continue
            column = probas[:, idx]
            if self.method == "isotonic":
                transformed = model.predict(column)
            else:
                transformed = model.predict_proba(column.reshape(-1, 1))[:, 1]
            calibrated[:, idx] = np.clip(transformed, 1e-6, 1.0)

        row_sum = calibrated.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        return calibrated / row_sum


class RegimeCalibrator:
    """Per-regime probability calibrator with volatility-aware helpers."""

    def __init__(self, method: str = "isotonic", classes: Sequence[int] = (-1, 0, 1)) -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be either 'isotonic' or 'platt'")
        self.method = method
        self.classes = tuple(classes)
        self._n_classes = len(self.classes)
        self._global = _OneVsRestCalibrator(method, self._n_classes)
        self._per_regime: Dict[str, _OneVsRestCalibrator] = {}
        self.vol_thresholds: Dict[str, float] = {"low": 0.0, "high": 0.0}
        self.grey_zone: Dict[str, Tuple[float, float]] = {"default": _DEFAULT_GREY_ZONE}
        self.temperatures: Dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting and calibration
    # ------------------------------------------------------------------
    def fit(
        self,
        y_true: Iterable[int],
        probas: np.ndarray,
        regimes: Iterable[Union[str, int]],
        *,
        vol_series: Optional[Iterable[float]] = None,
    ) -> "RegimeCalibrator":
        y_arr = np.asarray(list(y_true), dtype=int)
        prob_arr = np.asarray(probas, dtype=float)
        reg_arr = np.asarray(list(regimes), dtype=object)
        if prob_arr.ndim != 2:
            prob_arr = prob_arr.reshape(-1, self._n_classes)
        if prob_arr.shape[1] != self._n_classes:
            raise ValueError("Probabilities must align with configured class order")
        if len(y_arr) != len(prob_arr):
            raise ValueError("y_true and probas must have the same length")

        self._global.fit(y_arr, prob_arr)

        unique_regimes = []
        for value in reg_arr:
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            unique_regimes.append(str(value).lower())
        for regime in sorted(set(unique_regimes)):
            mask = np.array([str(v).lower() == regime for v in reg_arr])
            if not mask.any():
                continue
            local = _OneVsRestCalibrator(self.method, self._n_classes)
            local.fit(y_arr[mask], prob_arr[mask])
            self._per_regime[regime] = local
            self.grey_zone[regime] = self._default_grey_zone(regime)
            self.temperatures[regime] = self._default_temperature(regime)

        if vol_series is not None:
            vol_arr = np.asarray(list(vol_series), dtype=float)
            vol_arr = vol_arr[np.isfinite(vol_arr)]
            if vol_arr.size:
                low_q = float(np.quantile(vol_arr, 0.33))
                high_q = float(np.quantile(vol_arr, 0.66))
                self.vol_thresholds = {"low": low_q, "high": high_q}

        if "med" not in self.grey_zone:
            self.grey_zone["med"] = _DEFAULT_GREY_ZONE
        if "med" not in self.temperatures:
            self.temperatures["med"] = 1.0

        self._fitted = True
        return self

    def calibrate(
        self,
        probas: np.ndarray,
        *,
        regime_hint: Optional[Union[str, Sequence[Union[str, int]]]] = None,
        features: Optional[Union[Mapping[str, float], Sequence[Mapping[str, float]]]] = None,
    ) -> np.ndarray:
        prob_arr = np.asarray(probas, dtype=float)
        if prob_arr.ndim == 1:
            prob_arr = prob_arr.reshape(1, -1)
        if prob_arr.shape[1] != self._n_classes:
            raise ValueError("Probabilities must align with configured class order")

        hints = self._expand_hints(regime_hint, len(prob_arr))
        feature_rows = self._expand_features(features, len(prob_arr))

        calibrated_rows = []
        for idx, row in enumerate(prob_arr):
            bucket = hints[idx]
            if bucket is None:
                bucket = self._infer_bucket_from_features(feature_rows[idx])
            calibrator = self._per_regime.get(bucket)
            if calibrator is None:
                calibrator = self._global
            calibrated_rows.append(calibrator.transform(row.reshape(1, -1))[0])
        return np.vstack(calibrated_rows)

    # ------------------------------------------------------------------
    # Helper utilities for policy/API integrations
    # ------------------------------------------------------------------
    def infer_bucket(self, features: Optional[Mapping[str, float]] = None) -> str:
        return self._infer_bucket_from_features(features) or "med"

    def grey_zone_bounds(self, *, features: Optional[Mapping[str, float]] = None) -> Tuple[float, float]:
        bucket = self._infer_bucket_from_features(features) or "med"
        return self.grey_zone.get(bucket, self.grey_zone.get("default", _DEFAULT_GREY_ZONE))

    def confidence_interval(
        self,
        probability: float,
        *,
        features: Optional[Mapping[str, float]] = None,
    ) -> Tuple[float, float]:
        bucket = self._infer_bucket_from_features(features) or "med"
        temperature = max(self.temperatures.get(bucket, 1.0), 1e-3)
        base_width = self._base_interval_width(bucket)
        half_width = (base_width / temperature) / 2.0
        lower = max(0.0, float(probability) - half_width)
        upper = min(1.0, float(probability) + half_width)
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _expand_hints(
        self,
        regime_hint: Optional[Union[str, Sequence[Union[str, int]]]],
        length: int,
    ) -> list[Optional[str]]:
        if regime_hint is None:
            return [None] * length
        if isinstance(regime_hint, (list, tuple, np.ndarray)):
            hints: list[Optional[str]] = []
            for value in regime_hint:
                if value is None:
                    hints.append(None)
                    continue
                hints.append(self._normalise_regime(value))
            if len(hints) != length:
                raise ValueError("regime_hint length mismatch")
            return hints
        return [self._normalise_regime(regime_hint)] * length

    def _expand_features(
        self,
        features: Optional[Union[Mapping[str, float], Sequence[Mapping[str, float]]]],
        length: int,
    ) -> list[Optional[Mapping[str, float]]]:
        if features is None:
            return [None] * length
        if isinstance(features, Mapping):
            return [features] * length
        feature_list = list(features)
        if len(feature_list) != length:
            raise ValueError("features length mismatch")
        return feature_list

    def _normalise_regime(self, value: Union[str, int]) -> Optional[str]:
        text = str(value).lower()
        if text in self._per_regime:
            return text
        if "low" in text:
            return "low" if "low" in self._per_regime else None
        if "high" in text:
            return "high" if "high" in self._per_regime else None
        if "med" in text or "mid" in text:
            return "med" if "med" in self._per_regime else None
        return text if text in self._per_regime else None

    def _infer_bucket_from_features(
        self, features: Optional[Mapping[str, float]]
    ) -> Optional[str]:
        if features is None:
            return None
        getters = features.get if hasattr(features, "get") else dict(features).get
        for key, bucket in (
            ("vol_bucket_low", "low"),
            ("vol_bucket_med", "med"),
            ("vol_bucket_high", "high"),
        ):
            value = getters(key, None)
            try:
                if value is not None and float(value) > 0.5:
                    return bucket
            except (TypeError, ValueError):
                continue
        vol_value = None
        for key in ("realized_vol_60", "realized_vol_30", "vol_30", "volatility"):
            raw = getters(key, None)
            if raw is None:
                continue
            try:
                vol_value = float(raw)
                break
            except (TypeError, ValueError):
                continue
        if vol_value is None:
            return None
        low_threshold = self.vol_thresholds.get("low")
        high_threshold = self.vol_thresholds.get("high")
        if high_threshold is not None and vol_value >= high_threshold:
            return "high"
        if low_threshold is not None and vol_value <= low_threshold:
            return "low"
        return "med"

    def _default_grey_zone(self, regime: str) -> Tuple[float, float]:
        regime = regime.lower()
        if "low" in regime:
            return (0.48, 0.52)
        if "high" in regime:
            return (0.42, 0.58)
        return _DEFAULT_GREY_ZONE

    def _default_temperature(self, regime: str) -> float:
        regime = regime.lower()
        if "low" in regime:
            return 0.8
        if "high" in regime:
            return 1.2
        return 1.0

    def _base_interval_width(self, regime: str) -> float:
        regime = regime.lower()
        if "low" in regime:
            return 0.12
        if "high" in regime:
            return 0.20
        return 0.16
