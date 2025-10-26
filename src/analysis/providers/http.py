"""HTTP JSON adapter for analysis providers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

import httpx

from .base import AnalysisProvider

LOGGER = logging.getLogger(__name__)


def _slug_from_url(url: str) -> str:
    cleaned = url.replace("https://", "").replace("http://", "").strip("/")
    slug = cleaned.split("/")[0]
    slug = slug.replace(".", "_").replace(":", "_")
    return slug or "provider"


def _headers_for(name: str) -> Mapping[str, str]:
    env_key = f"ANALYSIS_HEADERS_{name.upper()}"
    raw = os.getenv(env_key)
    if not raw:
        raw = os.getenv("ANALYSIS_HEADERS_DEFAULT")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, Mapping):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        LOGGER.warning("analysis.provider.headers.invalid", extra={"provider": name})
    return {}


@dataclass(slots=True)
class HTTPAnalysisProvider(AnalysisProvider):
    """HTTP JSON provider that POSTs the payload to an external service."""

    name: str
    url: str
    timeout_ms: int = 30_000
    headers: Mapping[str, str] = None

    async def analyze(self, payload: dict) -> dict:
        headers = dict(self.headers or {})
        timeout = httpx.Timeout(self.timeout_ms / 1000)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Provider {self.name} returned non-JSON response") from exc


def build_http_providers(
    specs: Iterable[str],
    *,
    default_timeout_ms: int = 40_000,
) -> List[HTTPAnalysisProvider]:
    providers: List[HTTPAnalysisProvider] = []
    for spec in specs:
        if not spec:
            continue
        if "=" in spec:
            name_part, url_part = spec.split("=", 1)
            name = name_part.strip() or _slug_from_url(url_part)
            url = url_part.strip()
        else:
            url = spec.strip()
            name = _slug_from_url(url)
        if not url.startswith("http"):
            LOGGER.warning("analysis.provider.invalid_url", extra={"name": name, "url": url})
            continue
        headers = _headers_for(name)
        providers.append(
            HTTPAnalysisProvider(
                name=name,
                url=url,
                timeout_ms=default_timeout_ms,
                headers=headers,
            )
        )
    return providers


__all__ = ["HTTPAnalysisProvider", "build_http_providers"]

