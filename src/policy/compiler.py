"""Compile policy rules from YAML into executable callables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..utils.logging import get_logger
from ..config import load_rules
from .schema import FilterRule, PolicyConfig, RiskGuard, SizingRule, ThresholdRule

LOGGER = get_logger(__name__)


@dataclass
class CompiledFilter:
    name: str
    action: str
    fn: Callable[[Dict[str, object]], bool]


@dataclass
class CompiledThreshold:
    name: str
    long_if_prob_gt: Optional[float]
    short_if_prob_lt: Optional[float]


@dataclass
class CompiledSizing:
    name: str
    fn: Callable[[Dict[str, object]], float]


@dataclass
class CompiledGuard:
    name: str
    action: str
    fn: Callable[[Dict[str, object]], bool]


@dataclass
class CompiledPolicy:
    filters: List[CompiledFilter]
    thresholds: List[CompiledThreshold]
    sizing: List[CompiledSizing]
    guards: List[CompiledGuard]
    exits: List[Dict[str, float]]


def _eval_expr(expr: str, context: Dict[str, object]) -> bool:
    try:
        return bool(eval(expr, {}, context))
    except Exception as exc:  # pragma: no cover - guard for runtime issues
        LOGGER.error("Failed to evaluate expr=%s error=%s", expr, exc)
        return False


def compile_filter(rule: FilterRule, retriever=None) -> CompiledFilter:
    if rule.type == "metric":
        def fn(ctx: Dict[str, object]) -> bool:
            features = ctx.get("features", {})
            return _eval_expr(rule.expr or "False", {**features})
    elif rule.type == "numeric":
        def fn(ctx: Dict[str, object]) -> bool:
            return _eval_expr(rule.expr or "False", ctx)
    elif rule.type == "text_rag_any":
        def fn(ctx: Dict[str, object]) -> bool:
            if retriever is None:
                return False
            hits = retriever.search_text(rule.query, topk=3)
            ctx.setdefault("memory_hits", {})[rule.name] = hits
            return bool(hits)
    elif rule.type == "image_rag_any":
        def fn(ctx: Dict[str, object]) -> bool:
            if retriever is None:
                return False
            hits = retriever.search_image(rule.query, topk=3)
            ctx.setdefault("memory_hits", {})[rule.name] = hits
            return bool(hits)
    else:
        raise ValueError(f"Unknown filter type {rule.type}")
    return CompiledFilter(name=rule.name, action=rule.action, fn=fn)


def compile_threshold(rule: ThresholdRule) -> CompiledThreshold:
    return CompiledThreshold(name=rule.name, long_if_prob_gt=rule.long_if_prob_gt, short_if_prob_lt=rule.short_if_prob_lt)


def compile_sizing(rule: SizingRule) -> CompiledSizing:
    def fn(ctx: Dict[str, object]) -> float:
        local_ctx = {**ctx.get("features", {}), **ctx.get("state", {}), "base_risk": ctx.get("base_risk", 1.0)}
        local_ctx.setdefault("size", 0.0)
        try:
            exec(rule.expr, {}, local_ctx)
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Failed to exec sizing expr=%s error=%s", rule.expr, exc)
        return float(local_ctx.get("size", 0.0))

    return CompiledSizing(name=rule.name, fn=fn)


def compile_guard(rule: RiskGuard) -> CompiledGuard:
    def fn(ctx: Dict[str, object]) -> bool:
        return _eval_expr(rule.expr, ctx.get("state", {}))

    return CompiledGuard(name=rule.name, action=rule.action, fn=fn)


def compile_policy(rules: Dict[str, object], retriever=None) -> CompiledPolicy:
    config = PolicyConfig.parse_obj(rules)
    filters = [compile_filter(rule, retriever=retriever) for rule in config.filters]
    thresholds = [compile_threshold(rule) for rule in config.thresholds]
    sizing = [compile_sizing(rule) for rule in config.position_sizing]
    guards = [compile_guard(rule) for rule in config.risk_guards]
    exits = [exit_rule.dict(exclude_none=True) for exit_rule in config.exits]
    return CompiledPolicy(filters=filters, thresholds=thresholds, sizing=sizing, guards=guards, exits=exits)


def load_and_compile(path: str, retriever=None) -> CompiledPolicy:
    rules = load_rules(path)
    return compile_policy(rules, retriever=retriever)
