"""Shared asynchronous rate limiter for external market-data requests."""
from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from .tracing import TraceContext

__all__ = [
    "DEFAULT_TOKEN_RATE",
    "DEFAULT_TOKEN_BURST",
    "DEFAULT_MAX_CONCURRENCY",
    "RateLimiter",
    "get_global_rate_limiter",
]

DEFAULT_TOKEN_RATE = 80.0
DEFAULT_TOKEN_BURST = 160
DEFAULT_MAX_CONCURRENCY = 4
_RETRY_AFTER_FALLBACK = 1.0
_MAX_GLOBAL_BACKOFF = 5.0
_WEIGHT_LIMIT_PER_MINUTE = 1200


@dataclass(slots=True)
class _TokenBucket:
    rate: float
    capacity: int
    tokens: float
    updated: float

    @classmethod
    def build(cls, rate: float, capacity: int) -> "_TokenBucket":
        rate = max(1.0, float(rate))
        capacity = max(1, int(capacity))
        return cls(rate=rate, capacity=capacity, tokens=float(capacity), updated=time.monotonic())

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = max(0.0, now - self.updated)
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            deficit = 1.0 - self.tokens
            delay = max(deficit / self.rate, 0.02)
            await asyncio.sleep(min(delay, 0.25))


class RateLimiter:
    """Cooperative rate limiter shared by HTTP clients."""

    __slots__ = ("_bucket", "_semaphore", "_lock", "_backoff_until")

    def __init__(
        self,
        *,
        rate: float = DEFAULT_TOKEN_RATE,
        burst: int = DEFAULT_TOKEN_BURST,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self._bucket = _TokenBucket.build(rate, burst)
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._lock = asyncio.Lock()
        self._backoff_until = 0.0

    async def _wait_for_global_backoff(self, scope: str, trace: TraceContext | None) -> None:
        while True:
            async with self._lock:
                delay = self._backoff_until - time.monotonic()
            if delay <= 0:
                return
            if trace is not None:
                trace.debug("rate.sleep", scope=scope, delay_ms=int(delay * 1000))
            await asyncio.sleep(min(delay, 0.5))

    @asynccontextmanager
    async def limit(
        self,
        *,
        scope: str,
        trace: TraceContext | None = None,
    ) -> AsyncIterator[None]:
        await self._semaphore.acquire()
        try:
            await self._wait_for_global_backoff(scope, trace)
            await self._bucket.acquire()
            yield
        finally:
            self._semaphore.release()

    async def apply_backoff(
        self,
        delay_seconds: float | None,
        *,
        scope: str,
        trace: TraceContext | None = None,
        reason: str | None = None,
    ) -> None:
        if delay_seconds is None:
            delay_seconds = _RETRY_AFTER_FALLBACK
        delay_seconds = max(0.05, min(float(delay_seconds), _MAX_GLOBAL_BACKOFF))
        async with self._lock:
            target = time.monotonic() + delay_seconds
            if target > self._backoff_until:
                self._backoff_until = target
        if trace is not None:
            trace.warn(
                "rate_limited",
                scope=scope,
                reason=reason,
                delay_ms=int(delay_seconds * 1000),
            )

    async def note_used_weight(
        self,
        used_weight: Optional[int],
        *,
        scope: str,
        trace: TraceContext | None = None,
    ) -> None:
        if used_weight is None or used_weight <= 0:
            return
        if used_weight < int(_WEIGHT_LIMIT_PER_MINUTE * 0.9):
            return
        ratio = min(1.0, used_weight / float(_WEIGHT_LIMIT_PER_MINUTE))
        jitter = random.uniform(0.1, 0.3)
        delay = jitter + (ratio - 0.9) * 2.0
        await self.apply_backoff(delay, scope=scope, trace=trace, reason="weight_guard")


_GLOBAL_RATE_LIMITER: RateLimiter | None = None


def get_global_rate_limiter() -> RateLimiter:
    global _GLOBAL_RATE_LIMITER
    if _GLOBAL_RATE_LIMITER is None:
        _GLOBAL_RATE_LIMITER = RateLimiter()
    return _GLOBAL_RATE_LIMITER

