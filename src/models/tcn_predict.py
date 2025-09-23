"""Inference helper for trained TCN model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import torch

from .tcn_model import TCNClassifier

MODEL_DIR = Path("models/tcn")


def load_artifacts(model_dir: Path | str = MODEL_DIR) -> Tuple[TCNClassifier, object, List[str], int]:
    model_dir = Path(model_dir)
    model_path = model_dir / "ckpt.pt"
    scaler_path = model_dir / "scaler.pkl"
    feature_path = model_dir / "feature_list.pkl"
    config_path = model_dir / "config.json"
    if not model_path.exists():
        raise FileNotFoundError("TCN checkpoint missing. Run make train_tcn")
    if not config_path.exists():
        raise FileNotFoundError("TCN config missing. Re-train the model")

    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feature_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = TCNClassifier(num_inputs=len(feature_list), channels=config["channels"], dropout=config.get("dropout", 0.0))
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    window = int(config.get("window", 10))
    return model, scaler, feature_list, window


def predict_proba(history: pd.DataFrame, model_dir: Optional[Path | str] = None) -> float:
    model, scaler, features, window = load_artifacts(model_dir or MODEL_DIR)
    if not set(features).issubset(history.columns):
        missing = set(features) - set(history.columns)
        raise ValueError(f"Missing features for TCN inference: {missing}")
    segment = history.sort_index().tail(window)
    if len(segment) < window:
        raise ValueError(f"Need at least {window} rows for TCN inference")
    arr = scaler.transform(segment[features])
    tensor = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
    return float(prob)
