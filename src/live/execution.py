"""Order execution adapters for paper/live trading."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import pandas as pd

from ..io.store import append_sqlite


@dataclass
class PaperExecution:
    db_url: str

    def submit(self, order: Dict[str, object]) -> None:
        order.setdefault("timestamp", datetime.utcnow().timestamp())
        df = pd.DataFrame([order])
        append_sqlite(df, table="paper_orders", db_url=self.db_url)
