"""Cross-validation utilities including PurgedKFold for walk-forward testing."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """K-fold validator that removes leakage around the validation window."""

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.embargo = max(int(embargo), 0)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # type: ignore[override]
        return self.n_splits

    def split(self, X, y=None, groups=None):  # type: ignore[override]
        indices = np.arange(len(X))
        fold_sizes = np.full(self.n_splits, len(X) // self.n_splits, dtype=int)
        fold_sizes[: len(X) % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_mask = np.ones(len(X), dtype=bool)
            embargo_start = max(0, start - self.embargo)
            embargo_end = min(len(X), stop + self.embargo)
            train_mask[embargo_start:embargo_end] = False
            train_indices = indices[train_mask]
            current = stop
            yield train_indices, test_indices


def embargoed_train_valid_split(n_samples: int, test_size: int, embargo: int) -> tuple[np.ndarray, np.ndarray]:
    if test_size >= n_samples:
        raise ValueError("test_size must be smaller than n_samples")
    test_start = n_samples - test_size
    test_indices = np.arange(test_start, n_samples)
    embargo_start = max(0, test_start - embargo)
    train_indices = np.arange(0, embargo_start)
    return train_indices, test_indices
