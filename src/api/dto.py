"""DTOs for FastAPI endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    ts: datetime
    prob_up: float
    side: str
    reasoning: Dict[str, object]


class ExplainResponse(BaseModel):
    ts: datetime
    memory_hits: list[Dict[str, object]]
    rules: Dict[str, object]


class PaperOrderRequest(BaseModel):
    symbol: str
    side: str
    size: float
    price: Optional[float] = None
    meta: Dict[str, object] = Field(default_factory=dict)


class PaperOrderResponse(BaseModel):
    status: str
    order: Dict[str, object]
