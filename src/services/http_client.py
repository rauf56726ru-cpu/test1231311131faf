"""Instrumented HTTP helpers that cooperate with the shared rate limiter."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import httpx

from .rate_limiter import get_global_rate_limiter
from .tracing import TraceContext

__all__ = ["request", "RATE_LIMIT_STATUSES", "TRANSIENT_STATUSES"]

RATE_LIMIT_STATUSES = {418, 429}
TRANSIENT_STATUSES = {500, 502, 503, 504}

_RATE_LIMITER = get_global_rate_limiter()
_SCOPE_LOCKS: MutableMapping[str, asyncio.Lock] = {}


def _scope_lock(scope: str) -> asyncio.Lock:
    lock = _SCOPE_LOCKS.get(scope)
    if lock is None:
        lock = asyncio.Lock()
        _SCOPE_LOCKS[scope] = lock
    return lock


def _as_timeout(value: float | httpx.Timeout | None) -> httpx.Timeout | None:
    if value is None or isinstance(value, httpx.Timeout):
        return value
    return httpx.Timeout(float(value))


def _parse_retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _parse_used_weight(response: httpx.Response) -> int | None:
    header = response.headers.get("X-MBX-USED-WEIGHT-1m")
    if not header:
        return None
    try:
        return int(header)
    except ValueError:
        return None


def _make_window(window: Sequence[int | None] | None) -> Mapping[str, int] | None:
    if not window:
        return None
    start = window[0] if len(window) > 0 else None
    end = window[1] if len(window) > 1 else None
    payload: dict[str, int] = {}
    if isinstance(start, int):
        payload["from"] = start
    if isinstance(end, int):
        payload["to"] = end
    return payload if payload else None


async def _send(
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    scope: str,
    trace: TraceContext | None,
    params: Mapping[str, Any] | None,
    data: Any,
    json_data: Any,
    headers: Mapping[str, str] | None,
    timeout: httpx.Timeout | None,
    symbol: str | None,
    window: Sequence[int | None] | None,
    details: str | None,
    max_retries: int,
    retry_statuses: Iterable[int],
    rate_limit_statuses: Iterable[int],
    retry_on_request_error: bool,
) -> httpx.Response:
    attempt = 0
    retry_statuses = set(retry_statuses)
    rate_limit_statuses = set(rate_limit_statuses)
    while True:
        attempt += 1
        metrics: dict[str, Any] = {"req": attempt}
        window_payload = _make_window(window)
        if trace is not None:
            trace.debug(
                "fetch.batch_start",
                scope=scope,
                symbol=symbol,
                window=window_payload,
                metrics=metrics,
                details=details,
            )
        start = time.perf_counter()
        try:
            async with _scope_lock(scope):
                async with _RATE_LIMITER.limit(scope=scope, trace=trace):
                    response = await client.request(
                        method,
                        url,
                        params=params,
                        data=data,
                        json=json_data,
                        headers=headers,
                        timeout=timeout,
                    )
        except httpx.RequestError as exc:
            if trace is not None:
                trace.warn(
                    "fetch.batch_retry" if attempt <= max_retries and retry_on_request_error else "fetch.batch_failed",
                    scope=scope,
                    symbol=symbol,
                    window=window_payload,
                    metrics=metrics,
                    details=str(exc),
                )
            await _RATE_LIMITER.apply_backoff(
                None,
                scope=scope,
                trace=trace,
                reason="request_error",
            )
            if not retry_on_request_error or attempt > max_retries:
                raise
            continue

        elapsed_ms = int((time.perf_counter() - start) * 1000.0)
        metrics["ms"] = elapsed_ms
        used_weight = _parse_used_weight(response)
        if used_weight is not None:
            metrics["used_weight"] = used_weight
        await _RATE_LIMITER.note_used_weight(used_weight, scope=scope, trace=trace)
        if trace is not None:
            trace.info(
                "fetch.batch_done",
                scope=scope,
                symbol=symbol,
                window=window_payload,
                status=response.status_code,
                metrics=metrics,
                details=details,
            )

        status = response.status_code
        if status in rate_limit_statuses:
            await _RATE_LIMITER.apply_backoff(
                _parse_retry_after(response),
                scope=scope,
                trace=trace,
                reason=str(status),
            )
            if trace is not None:
                trace.warn(
                    "rate_limited",
                    scope=scope,
                    symbol=symbol,
                    window=window_payload,
                    status=status,
                    metrics=metrics,
                )
            if attempt > max_retries:
                return response
            if trace is not None:
                trace.warn(
                    "fetch.batch_retry",
                    scope=scope,
                    symbol=symbol,
                    window=window_payload,
                    status=status,
                    metrics=metrics,
                    details="rate_limit",
                )
            continue

        if status in retry_statuses and attempt <= max_retries:
            await _RATE_LIMITER.apply_backoff(
                _parse_retry_after(response),
                scope=scope,
                trace=trace,
                reason=str(status),
            )
            if trace is not None:
                trace.warn(
                    "fetch.batch_retry",
                    scope=scope,
                    symbol=symbol,
                    window=window_payload,
                    status=status,
                    metrics=metrics,
                )
            continue

        if status >= 400 and trace is not None and status not in retry_statuses and status not in rate_limit_statuses:
            trace.error(
                "fetch.batch_failed",
                scope=scope,
                symbol=symbol,
                window=window_payload,
                status=status,
                metrics=metrics,
            )
        return response


async def request(
    method: str,
    url: str,
    *,
    scope: str,
    trace: TraceContext | None = None,
    params: Mapping[str, Any] | None = None,
    data: Any = None,
    json: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    client: httpx.AsyncClient | None = None,
    symbol: str | None = None,
    window: Sequence[int | None] | None = None,
    details: str | None = None,
    max_retries: int = 0,
    retry_statuses: Iterable[int] = TRANSIENT_STATUSES,
    rate_limit_statuses: Iterable[int] = RATE_LIMIT_STATUSES,
    retry_on_request_error: bool = True,
) -> httpx.Response:
    timeout_obj = _as_timeout(timeout)

    if client is not None:
        return await _send(
            client,
            method=method,
            url=url,
            scope=scope,
            trace=trace,
            params=params,
            data=data,
            json_data=json,
            headers=headers,
            timeout=timeout_obj,
            symbol=symbol,
            window=window,
            details=details,
            max_retries=max_retries,
            retry_statuses=retry_statuses,
            rate_limit_statuses=rate_limit_statuses,
            retry_on_request_error=retry_on_request_error,
        )

    async with httpx.AsyncClient(timeout=timeout_obj or httpx.Timeout(15.0)) as owned_client:
        return await _send(
            owned_client,
            method=method,
            url=url,
            scope=scope,
            trace=trace,
            params=params,
            data=data,
            json_data=json,
            headers=headers,
            timeout=timeout_obj,
            symbol=symbol,
            window=window,
            details=details,
            max_retries=max_retries,
            retry_statuses=retry_statuses,
            rate_limit_statuses=rate_limit_statuses,
            retry_on_request_error=retry_on_request_error,
        )

