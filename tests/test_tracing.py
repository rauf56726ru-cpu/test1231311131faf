from __future__ import annotations

import importlib
import sys

import logging

import pytest

from src.services.tracing import TraceContext

MODULE_PATH = "src.services.tracing"


def _reload_tracing(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("APP_TRACE", raising=False)
    else:
        monkeypatch.setenv("APP_TRACE", value)

    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]

    return importlib.import_module(MODULE_PATH)


def test_tracing_logger_defaults_to_info(monkeypatch: pytest.MonkeyPatch):
    tracing = _reload_tracing(monkeypatch, None)

    assert tracing.is_trace_enabled() is False
    assert tracing.LOGGER.level == logging.INFO
    assert tracing.LOGGER.propagate is True


def test_tracing_logger_promoted_to_debug(monkeypatch: pytest.MonkeyPatch, caplog):
    tracing = _reload_tracing(monkeypatch, "on")

    assert tracing.is_trace_enabled() is True
    assert tracing.LOGGER.level == logging.DEBUG

    with caplog.at_level(logging.INFO, logger=tracing.LOGGER.name):
        tracing.LOGGER.info("tracing-event", extra={"stage": "test"})

    assert any(record.message == "tracing-event" for record in caplog.records)


def test_child_inherits_identifiers_and_static_fields():
    root_ctx = TraceContext(symbol="BTCUSDT", stage="root", enabled=True)
    child_ctx = root_ctx.child(stage="pipeline", rid="r_custom")

    assert child_ctx.cid == root_ctx.cid
    assert child_ctx.rid == "r_custom"
    assert child_ctx._static["stage"] == "pipeline"
    assert child_ctx._static["symbol"] == "BTCUSDT"
    assert child_ctx._enabled is True


def test_span_generates_new_request_id():
    ctx = TraceContext(symbol="ETHUSDT")
    span_ctx = ctx.span("pipeline.start", stage="pipeline")

    assert span_ctx.cid == ctx.cid
    assert span_ctx.rid != ctx.rid
    assert span_ctx._static["stage"] == "pipeline"
