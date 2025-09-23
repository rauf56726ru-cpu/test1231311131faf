"""Asynchronous job manager with deduplication and cooldowns."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


JobHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class Job:
    id: str
    job_type: str
    key: str
    params: Dict[str, Any]
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "type": self.job_type,
            "key": self.key,
            "status": self.status,
            "params": self.params,
            "created_at": self.created_at,
        }
        if self.started_at is not None:
            payload["started_at"] = self.started_at
        if self.finished_at is not None:
            payload["finished_at"] = self.finished_at
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


class JobManager:
    def __init__(self, cooldown_minutes: int = 15) -> None:
        self._handlers: Dict[str, JobHandler] = {}
        self._jobs: Dict[str, Job] = {}
        self._per_key: Dict[str, List[Job]] = {}
        self._active: Dict[Tuple[str, str], Job] = {}
        self._cooldowns: Dict[Tuple[str, str], float] = {}
        self._lock = asyncio.Lock()
        self._cooldown_seconds = max(0, cooldown_minutes * 60)

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    async def enqueue(self, job_type: str, key: str, params: Dict[str, Any]) -> str:
        handler = self._handlers.get(job_type)
        if handler is None:
            raise ValueError(f"No handler registered for job '{job_type}'")

        async with self._lock:
            existing = self._active.get((job_type, key))
            if existing is not None and existing.status in {"queued", "in_progress"}:
                return existing.id

            now = time.time()
            if job_type.startswith("train_"):
                finished = self._cooldowns.get((job_type, key))
                if finished and now - finished < self._cooldown_seconds:
                    cooldown_remaining = max(0.0, self._cooldown_seconds - (now - finished))
                    job = Job(
                        id=uuid.uuid4().hex,
                        job_type=job_type,
                        key=key,
                        params=dict(params),
                        status="done",
                        result={
                            "status": "cooldown",
                            "cooldown_remaining": cooldown_remaining,
                        },
                    )
                    job.event.set()
                    self._jobs[job.id] = job
                    self._per_key.setdefault(key, []).append(job)
                    return job.id

            job = Job(id=uuid.uuid4().hex, job_type=job_type, key=key, params=dict(params))
            self._jobs[job.id] = job
            self._per_key.setdefault(key, []).append(job)
            self._active[(job_type, key)] = job

        asyncio.create_task(self._run(job, handler))
        return job.id

    async def _run(self, job: Job, handler: JobHandler) -> None:
        job.status = "in_progress"
        job.started_at = time.time()
        try:
            result = await handler(job.params)
            job.result = result
            job.status = "done"
        except Exception as exc:  # pragma: no cover - defensive logging
            job.error = str(exc)
            job.status = "error"
        finally:
            job.finished_at = time.time()
            job.event.set()
            async with self._lock:
                active = self._active.get((job.job_type, job.key))
                if active and active.id == job.id:
                    self._active.pop((job.job_type, job.key), None)
                    if job.job_type.startswith("train_") and job.status == "done":
                        self._cooldowns[(job.job_type, job.key)] = job.finished_at or time.time()

    async def wait(self, job_id: str) -> Job:
        job = self._jobs[job_id]
        await job.event.wait()
        return job

    def status_for_key(self, key: str) -> List[Dict[str, Any]]:
        jobs = self._per_key.get(key, [])
        return [job.to_dict() for job in jobs]

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
