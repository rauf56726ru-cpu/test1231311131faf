"""Pydantic schema for policy rules."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, validator


class Meta(BaseModel):
    version: int
    description: Optional[str] = None


class FilterRule(BaseModel):
    name: str
    type: Literal["text_rag_any", "metric", "numeric"]
    query: Optional[str] = None
    expr: Optional[str] = None
    action: Literal["block_entries", "allow_long_only", "allow_short_only", "pass"] = "pass"

    @validator("query", always=True)
    def validate_query(cls, v, values):
        rule_type = values.get("type")
        if rule_type == "text_rag_any" and not v:
            raise ValueError("text_rag_any requires query")
        return v

    @validator("expr", always=True)
    def validate_expr(cls, v, values):
        rule_type = values.get("type")
        if rule_type in {"metric", "numeric"} and not v:
            raise ValueError("metric filters require expr")
        return v


class ThresholdRule(BaseModel):
    name: str
    long_if_prob_gt: Optional[float] = None
    short_if_prob_lt: Optional[float] = None


class SizingRule(BaseModel):
    name: str
    expr: str


class RiskGuard(BaseModel):
    name: str
    type: Literal["pnl", "exposure", "custom"] = "custom"
    expr: str
    action: Literal["disable_trading", "disable_trading_else"] = "disable_trading"


class ExitRule(BaseModel):
    name: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


class PolicyConfig(BaseModel):
    meta: Meta
    filters: List[FilterRule] = []
    thresholds: List[ThresholdRule] = []
    position_sizing: List[SizingRule] = []
    risk_guards: List[RiskGuard] = []
    exits: List[ExitRule] = []
