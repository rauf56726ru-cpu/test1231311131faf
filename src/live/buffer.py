"""Rolling buffer for live trading."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

import pandas as pd


@dataclass
class FeatureBuffer:
    window: int

    def __post_init__(self) -> None:
        self.buffer: Deque[pd.Series] = deque(maxlen=self.window)

    def add(self, features: Dict[str, float]) -> None:
        self.buffer.append(pd.Series(features))

    def to_frame(self) -> pd.DataFrame:
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer))
