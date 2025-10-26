"""Compatibility wrappers for legacy logging imports."""

from __future__ import annotations

import warnings
from pathlib import Path

from src.common.logging_setup import install_root_logging

DEFAULT_LOG_FILE = "pipeline.log"

__all__ = ["DEFAULT_LOG_FILE", "ensure_pipeline_file_logging"]


def ensure_pipeline_file_logging(*, log_path: str | Path | None = None, level: str | int = "INFO") -> Path:
    """Shim around :func:`src.common.logging_setup.install_root_logging`.

    The old service layer imported ``ensure_pipeline_file_logging`` on module
    import to guarantee a file handler. New entrypoints should import and call
    :func:`install_root_logging` directly. This shim remains to avoid breaking
    external callers that still rely on the previous name.
    """

    warnings.warn(
        "ensure_pipeline_file_logging is deprecated; use install_root_logging instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return install_root_logging(log_path=log_path, level=level)
