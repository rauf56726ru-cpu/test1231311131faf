"""Training loop for the TCN model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import get_settings
from ..utils.logging import get_logger
from .tcn_model import TCNClassifier

LOGGER = get_logger(__name__)
MODEL_DIR = Path("models/tcn")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_table(path: Path | str = Path("data/features/features.parquet")) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    y = df.pop("target")
    return df, y


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.y)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.X[idx], self.y[idx]


def create_sequences(X: pd.DataFrame, y: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    values = X.values
    targets = y.values
    sequences = []
    seq_targets = []
    for i in range(window, len(X)):
        sequences.append(values[i - window : i, :])
        seq_targets.append(targets[i])
    return np.stack(sequences), np.array(seq_targets)


def run_training(features_path: Path | str = Path("data/features/features.parquet"), epochs: int = 5, batch_size: int = 64) -> None:
    settings = get_settings()
    X, y = load_feature_table(features_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    window = min(settings.data.window, len(X_scaled) - 1)
    if window < 2:
        raise ValueError("Not enough data for TCN training")

    X_seq, y_seq = create_sequences(X_scaled, y, window)
    dataset = SequenceDataset(X_seq.transpose(0, 2, 1), y_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TCNClassifier(num_inputs=X_seq.shape[2], channels=settings.model.tcn.channels, dropout=settings.model.tcn.dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.model.tcn.lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)
        LOGGER.info("Epoch %s loss=%.4f", epoch + 1, running_loss / len(dataset))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "ckpt.pt"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_list.pkl")
    with (MODEL_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"channels": settings.model.tcn.channels, "dropout": settings.model.tcn.dropout, "window": window}, f)
    LOGGER.info("Saved TCN checkpoint to %s", model_path)


if __name__ == "__main__":
    run_training()
