"""Shared logging configuration for all runtime entrypoints."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"
DEFAULT_CONSOLE_FLAG = "auto"
_PROGRESS_LOGGERS: Tuple[str, ...] = ("src.services.tracing", "src.services.check_all_datas")
_STANDARD_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)

_CONFIGURED = False
_CONFIGURED_PATH: Path | None = None
_LOCK = Lock()

__all__ = ["install_root_logging"]


def _coerce_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _coerce_json_value(val) for key, val in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_coerce_json_value(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


class JsonLineFormatter(logging.Formatter):
    """Formatter producing JSON lines compatible with downstream ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "ts": timestamp,
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        extras = {
            key: _coerce_json_value(getattr(record, key))
            for key in record.__dict__
            if key not in _STANDARD_RECORD_KEYS
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, ensure_ascii=True)


def _parse_level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    candidate = str(value).strip().upper()
    if candidate.isdigit():
        return int(candidate)
    numeric = getattr(logging, candidate, None)
    if isinstance(numeric, int):
        return numeric
    return logging.INFO


def _normalise_path(path_value: str | Path | None) -> Path:
    legacy_dir = os.getenv("PIPELINE_LOG_DIR")
    legacy_file = os.getenv("PIPELINE_LOG_FILE")
    if path_value:
        candidate = Path(path_value)
    elif legacy_dir or legacy_file:
        candidate = Path(legacy_dir or PROJECT_ROOT / "logs") / (legacy_file or "pipeline.log")
    else:
        candidate = Path(os.getenv("LOG_PATH", DEFAULT_LOG_PATH))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _should_enable_console() -> bool:
    flag = os.getenv("LOG_CONSOLE", DEFAULT_CONSOLE_FLAG).strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    if flag in {"1", "true", "on", "yes"}:
        return True
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    return environment not in {"prod", "production"}


def _ensure_progress_console_handlers(formatter: logging.Formatter, level: int, console_enabled: bool) -> None:
    if console_enabled:
        return
    for logger_name in _PROGRESS_LOGGERS:
        logger = logging.getLogger(logger_name)
        already_present = any(getattr(handler, "_progress_stream", False) for handler in logger.handlers)
        if already_present:
            continue
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        setattr(handler, "_progress_stream", True)
        logger.addHandler(handler)


def install_root_logging(log_path: str | Path | None = None, level: str | int = "INFO") -> Path:
    """Initialise root logging with JSON files and optional console output."""

    global _CONFIGURED, _CONFIGURED_PATH
    with _LOCK:
        resolved_path = _normalise_path(log_path)
        resolved_level = _parse_level(os.getenv("LOG_LEVEL", level))
        if _CONFIGURED:
            root_logger = logging.getLogger()
            root_logger.setLevel(resolved_level)
            for handler in root_logger.handlers:
                handler.setLevel(resolved_level)
            return _CONFIGURED_PATH or resolved_path

        formatter = JsonLineFormatter()
        file_handler = TimedRotatingFileHandler(
            resolved_path,
            when="midnight",
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "7")),
            encoding="utf-8",
            utc=True,
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)

        console_enabled = _should_enable_console()
        if console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(resolved_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        _ensure_progress_console_handlers(formatter, resolved_level, console_enabled)

        root_logger.setLevel(resolved_level)
        logging.captureWarnings(True)
        root_logger.debug("Root logging configured", extra={"log_path": str(resolved_path)})

        _CONFIGURED = True
        _CONFIGURED_PATH = resolved_path
        return resolved_path
