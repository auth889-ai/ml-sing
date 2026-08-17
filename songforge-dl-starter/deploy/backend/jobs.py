"""Single-worker job queue for GPU generation.

One GPU means one job at a time, so this is a queue rather than a thread pool.
Requests return immediately with a job id and the client polls; nothing blocks
an HTTP worker for the length of a render.

The queue is in-process and non-durable: a restart loses queued work. That is
the right trade for a single-GPU demo, and it is stated rather than hidden — if
this ever needs to survive restarts, the replacement is Redis or a database,
not a bigger dict.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class Job:
    id: str
    client: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    audio_path: Path | None = None
    error: str | None = None
    #: Controls the model could not honour. Surfaced to the client so the UI can
    #: say "BPM was applied, instruments were a suggestion" rather than implying
    #: every knob was obeyed.
    control_warnings: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def queue_seconds(self) -> float:
        start = self.started_at or time.time()
        return round(start - self.created_at, 2)

    @property
    def generate_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "queue_seconds": self.queue_seconds,
            "generate_seconds": self.generate_seconds,
            "error": self.error,
            "control_warnings": self.control_warnings,
            "audio_ready": self.status is JobStatus.DONE and bool(self.audio_path),
            **({"result": self.result} if self.status is JobStatus.DONE else {}),
        }


class JobQueue:
    """FIFO queue drained by a fixed number of workers (normally one)."""

    def __init__(self, runner: Callable[[Job], None], concurrency: int = 1,
                 max_depth: int = 20, max_stored: int = 200) -> None:
        self._runner = runner
        self._max_depth = max_depth
        self._max_stored = max_stored
        self._pending: deque[Job] = deque()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._workers = [
            threading.Thread(target=self._work, name=f"songforge-worker-{i}", daemon=True)
            for i in range(max(1, concurrency))
        ]
        for worker in self._workers:
            worker.start()

    # --- submission ------------------------------------------------------

    def submit(self, client: str, payload: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:16], client=client, payload=payload)
        with self._lock:
            if len(self._pending) >= self._max_depth:
                job.status = JobStatus.REJECTED
                job.error = (
                    f"Queue is full ({self._max_depth} waiting). "
                    "This is a single-GPU deployment; try again shortly."
                )
                self._store(job)
                return job
            self._pending.append(job)
            self._store(job)
            self._wake.notify()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def position(self, job_id: str) -> int | None:
        with self._lock:
            for index, job in enumerate(self._pending):
                if job.id == job_id:
                    return index + 1
        return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status.value] = counts.get(job.status.value, 0) + 1
            return {"queue_depth": len(self._pending), "jobs": counts}

    # --- internals -------------------------------------------------------

    def _store(self, job: Job) -> None:
        self._jobs[job.id] = job
        while len(self._jobs) > self._max_stored:
            _, evicted = self._jobs.popitem(last=False)
            if evicted.audio_path and evicted.audio_path.exists():
                evicted.audio_path.unlink(missing_ok=True)

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._pending:
                    self._wake.wait()
                job = self._pending.popleft()
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
            try:
                self._runner(job)
                if job.status is JobStatus.RUNNING:
                    job.status = JobStatus.DONE
            except TimeoutError as exc:
                job.status = JobStatus.TIMEOUT
                job.error = str(exc)
            except Exception as exc:  # noqa: BLE001 - a failed job is a result, not a crash
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = time.time()


class RateLimiter:
    """Per-client requests-per-minute plus a daily quota.

    Keyed on whatever the caller passes as the client id. Behind a proxy that
    is a forwarded IP, which is trivially spoofable — this discourages casual
    over-use, and is not a security control.
    """

    def __init__(self, per_minute: int, daily_quota: int) -> None:
        self._per_minute = per_minute
        self._daily = daily_quota
        self._recent: dict[str, deque[float]] = {}
        self._today: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, str | None]:
        now = time.time()
        with self._lock:
            window = self._recent.setdefault(client, deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self._per_minute:
                wait = int(60.0 - (now - window[0])) + 1
                return False, f"Rate limit: {self._per_minute} requests per minute. Try again in {wait}s."

            day_start, used = self._today.get(client, (now, 0))
            if now - day_start > 86400.0:
                day_start, used = now, 0
            if used >= self._daily:
                return False, f"Daily quota reached ({self._daily} songs). Resets 24h after your first request."

            window.append(now)
            self._today[client] = (day_start, used + 1)
            return True, None


def cleanup_expired(output_dir: Path, retain_hours: float) -> int:
    """Delete generated audio older than the retention window."""
    if not output_dir.exists():
        return 0
    cutoff = time.time() - retain_hours * 3600.0
    removed = 0
    for path in output_dir.glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed
