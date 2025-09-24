"""Main live trading loop (simplified)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import os

from ..config import get_settings
from ..models import tcn_predict, xgb_predict
from ..policy.runtime import PolicyRuntime
from ..utils.logging import get_logger
from .buffer import FeatureBuffer
from .execution import PaperExecution
from .monitor import Monitor

LOGGER = get_logger(__name__)


def _predict(model_kind: str, features_frame, latest_features: Dict[str, float]) -> Dict[str, object]:
    if model_kind == "xgb":
        probs, metadata = xgb_predict.predict_distribution(latest_features)
        direction = {0: "down", 1: "flat", 2: "up"}[int(probs.argmax())]
        return {
            "probs": {"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
            "pred_dir": direction,
            "pred_conf": float(probs.max()),
            "metadata": metadata,
        }
    if model_kind == "tcn":
        prob = tcn_predict.predict_proba(features_frame)
        return {
            "probs": {"down": float(1.0 - prob), "flat": 0.0, "up": float(prob)},
            "pred_dir": "up" if prob >= 0.5 else "down",
            "pred_conf": float(prob if prob >= 0.5 else 1.0 - prob),
            "metadata": {},
        }
    raise ValueError(f"Unknown model kind {model_kind}")


@dataclass
class LiveLoop:
    policy: PolicyRuntime
    model_kind: str = "xgb"

    def __post_init__(self) -> None:
        self.settings = get_settings()
        self.buffer = FeatureBuffer(window=self.settings.data.window)
        db_url = os.getenv("DB_URL", "sqlite:///autotrader.db")
        self.execution = PaperExecution(db_url=db_url)
        self.monitor = Monitor()
        self.balance = 10_000.0

    def step(self, features: Dict[str, float]) -> Dict[str, object]:
        self.buffer.add(features)
        frame = self.buffer.to_frame()
        prediction = _predict(self.model_kind, frame, features)
        decision = self.policy.apply(prediction, {"features": features, "state": {"balance": self.balance}})
        result = {**prediction, "decision": decision.side, "size": decision.size}
        self.monitor.emit("decision", result)
        if decision.side in {"long", "short"}:
            order = {
                "symbol": self.settings.data.symbol,
                "side": decision.side,
                "size": decision.size,
                "timestamp": datetime.utcnow().timestamp(),
            }
            self.execution.submit(order)
        return result
