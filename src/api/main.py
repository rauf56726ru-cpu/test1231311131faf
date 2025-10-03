"""Compatibility module to expose the FastAPI app under ``src.api.main``.

Previous tooling and deployment scripts import ``src.api.main:app`` when
starting the server. The project recently consolidated the app inside
``src.api.app`` which broke those entrypoints. Importing and re-exporting
``app`` here restores the old import path without duplicating the
application setup.
"""
from __future__ import annotations

from .app import app

__all__ = ["app"]
