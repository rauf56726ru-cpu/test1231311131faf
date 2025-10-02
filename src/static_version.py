"""Static asset cache-busting helpers."""

from datetime import datetime, timezone

from .version import APP_VERSION

STATIC_VERSION = f"{APP_VERSION}-{int(datetime.now(timezone.utc).timestamp())}"

__all__ = ["STATIC_VERSION"]
