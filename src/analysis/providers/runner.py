"""Scheduling helpers for running analysis providers with concurrency limits and timeouts."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from .base import AnalysisProvider

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProviderSummary:
    name: str
    status: str
    latency_ms: int | None
    result_path: str | None
    error: str | None
    zones_referenced: Sequence[str]
    metrics_referenced: Sequence[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "result_path": self.result_path,
            "error": self.error,
            "zones_referenced": list(self.zones_referenced),
            "metrics_referenced": list(self.metrics_referenced),
        }


async def run_providers(
    providers: Iterable[AnalysisProvider],
    payload: Mapping[str, object],
    *,
    output_dir: Path,
    timestamp: int,
    zone_ids: Sequence[str],
    metric_keys: Sequence[str],
    max_concurrency: int = 2,
    overall_timeout_ms: int = 40_000,
) -> List[dict]:
    provider_list = list(providers)
    if not provider_list:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    summaries: list[ProviderSummary] = []
    tasks = []
    task_map: dict[asyncio.Task, AnalysisProvider] = {}

    async def worker(provider: AnalysisProvider) -> ProviderSummary:
        started = time.perf_counter()
        status = "ok"
        result_path: str | None = None
        error_text: str | None = None
        zones_hit: list[str] = []
        metrics_hit: list[str] = []
        try:
            async with semaphore:
                result = await provider.analyze(dict(payload))
        except asyncio.TimeoutError:
            status = "timeout"
            LOGGER.warning("analysis.provider.timeout", extra={"provider": provider.name})
        except Exception as exc:  # pragma: no cover - defensive guard
            status = "error"
            error_text = str(exc)
            LOGGER.warning("analysis.provider.error", extra={"provider": provider.name, "error": error_text})
        else:
            latency_ms = int((time.perf_counter() - started) * 1000)
            result_path = str(output_dir / f"analysis_{provider.name}_{timestamp}.json")
            Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
            result_text = json.dumps(result, ensure_ascii=False)
            zones_hit = [zone_id for zone_id in zone_ids if zone_id in result_text]
            metrics_hit = [metric for metric in metric_keys if metric in result_text]
            return ProviderSummary(
                name=provider.name,
                status=status,
                latency_ms=latency_ms,
                result_path=result_path,
                error=error_text,
                zones_referenced=zones_hit,
                metrics_referenced=metrics_hit,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ProviderSummary(
            name=provider.name,
            status=status,
            latency_ms=latency_ms,
            result_path=result_path,
            error=error_text,
            zones_referenced=zones_hit,
            metrics_referenced=metrics_hit,
        )

    for provider in provider_list:
        task = asyncio.create_task(worker(provider))
        tasks.append(task)
        task_map[task] = provider

    done, pending = await asyncio.wait(
        tasks,
        timeout=max(0.001, overall_timeout_ms / 1000),
        return_when=asyncio.ALL_COMPLETED,
    )

    for task in done:
        try:
            summaries.append(task.result())
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("analysis.provider.unhandled", extra={"error": str(exc)})

    if pending:
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for task in pending:
            provider = task_map.get(task)
            provider_name = provider.name if provider is not None else "unknown"
            summaries.append(
                ProviderSummary(
                    name=provider_name,
                    status="timeout",
                    latency_ms=None,
                    result_path=None,
                    error="overall_timeout",
                    zones_referenced=[],
                    metrics_referenced=[],
                )
            )
        LOGGER.warning(
            "analysis.providers.overall_timeout",
            extra={"timeout_ms": overall_timeout_ms, "providers_pending": len(pending)},
        )

    return [summary.to_dict() for summary in summaries]


__all__ = ["run_providers"]

