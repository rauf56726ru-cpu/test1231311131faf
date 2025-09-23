"""Incremental fine-tuning utilities for the TCN classifier."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - handled gracefully downstream
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore

from ..config import get_settings
from ..utils.logging import get_logger
from .tcn_model import TCNClassifier
from .tcn_train import SequenceDataset, create_sequences

LOGGER = get_logger(__name__)


@dataclass
class TCNUpdateResult:
    day: str
    rows: int
    loss: float
    checkpoints_path: Path


def _load_active_state(model_dir: Path) -> Optional[Dict[str, object]]:
    active_dir = model_dir / "active"
    ckpt_path = active_dir / "ckpt.pt"
    scaler_path = active_dir / "scaler.pkl"
    feature_path = active_dir / "feature_list.pkl"
    config_path = active_dir / "config.json"
    if not ckpt_path.exists() or not scaler_path.exists() or not feature_path.exists():
        return None
    try:
        scaler = joblib.load(scaler_path)
        feature_list = joblib.load(feature_path)
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Failed to load TCN active state: %s", exc)
        return None
    return {
        "scaler": scaler,
        "feature_list": list(feature_list),
        "config": config,
        "state_dict": state,
    }


def _save_checkpoint(target: Path, *, model: TCNClassifier, scaler: StandardScaler, feature_list: list[str], config: Dict[str, object]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch not available for saving TCN checkpoint")
    torch.save(model.state_dict(), target / "ckpt.pt")
    joblib.dump(scaler, target / "scaler.pkl")
    joblib.dump(feature_list, target / "feature_list.pkl")
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _ensure_active_symlink(model_dir: Path, day_dir: Path) -> None:
    active_dir = model_dir / "active"
    if active_dir.exists():
        shutil.rmtree(active_dir)
    shutil.copytree(day_dir, active_dir)


def incremental_update(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_dir: Path,
    day: str,
    symbol: str,
    interval: str,
    window: int,
    epochs: int = 1,
    batch_size: int = 64,
) -> Optional[TCNUpdateResult]:
    """Continue TCN training with a single day's worth of features."""

    if torch is None or nn is None or DataLoader is None:  # pragma: no cover - runtime guard
        LOGGER.warning("PyTorch is required for TCN incremental update; skipping")
        return None

    model_dir = Path(model_dir)
    day_dir = model_dir / day
    settings = get_settings()
    channels = settings.model.tcn.channels
    dropout = settings.model.tcn.dropout

    state = _load_active_state(model_dir)
    if state is None:
        scaler = StandardScaler()
        feature_list = list(X.columns)
        config = {"channels": channels, "dropout": dropout, "window": int(window)}
        model = TCNClassifier(num_inputs=len(feature_list), channels=channels, dropout=dropout)
    else:
        scaler = state["scaler"]
        feature_list = list(state.get("feature_list") or list(X.columns))
        config = dict(state.get("config", {}))
        config.setdefault("channels", channels)
        config.setdefault("dropout", dropout)
        config.setdefault("window", int(window))
        model = TCNClassifier(num_inputs=len(feature_list), channels=config["channels"], dropout=config.get("dropout", 0.0))
        try:
            model.load_state_dict(state["state_dict"])
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to restore previous TCN state: %s", exc)

    if not feature_list:
        feature_list = list(X.columns)

    X_use = X[feature_list].astype(float)
    scaler.partial_fit(X_use)
    X_scaled = scaler.transform(X_use)

    frame = pd.DataFrame(X_scaled, index=X.index, columns=feature_list)
    sequences, seq_targets = create_sequences(frame, y.astype(float), window)
    if sequences.size == 0:
        LOGGER.info("TCN update skipped for %s %s %s: not enough data", symbol, interval, day)
        return None

    dataset = SequenceDataset(sequences.transpose(0, 2, 1), seq_targets.astype(np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.model.tcn.lr)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0
    for _ in range(max(1, epochs)):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_samples += len(batch_y)

    avg_loss = float(total_loss / max(1, total_samples))
    _save_checkpoint(day_dir, model=model, scaler=scaler, feature_list=feature_list, config=config)
    _ensure_active_symlink(model_dir, day_dir)

    LOGGER.info(
        "TCN checkpoint updated %s %s day=%s loss=%.4f rows=%s",
        symbol,
        interval,
        day,
        avg_loss,
        len(X),
    )

    return TCNUpdateResult(day=day, rows=len(X), loss=avg_loss, checkpoints_path=day_dir)

