"""SongForge public generation API.

A thin, honest shell over `songforge.generation`. The important design choice:
the API returns the *control resolution* with every job, so the UI can tell a
user that BPM was a real conditioning input while instruments were only a
suggestion. The rule that no control may be presented as stronger than it is
does not stop at the library boundary.

    uvicorn deploy.backend.app:app --host 0.0.0.0 --port 8000

Endpoints
    POST /api/generate        submit a job, returns 202 + job_id
    GET  /api/jobs/{id}       poll status, queue position, control warnings
    GET  /api/jobs/{id}/audio download the finished WAV
    GET  /api/capabilities    what this deployment can actually honour
    GET  /health              liveness, model state, queue depth
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from songforge.generation import (
    Section,
    SongRequest,
    VocalSpec,
    build,
    resolve_controls,
)

from .config import SETTINGS
from .jobs import Job, JobQueue, JobStatus, RateLimiter, cleanup_expired

app = FastAPI(title="SongForge API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_adapter: Any = None
_adapter_lock = threading.Lock()
_adapter_error: str | None = None
_started_at = time.time()


# --- request schema -------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=SETTINGS.max_prompt_chars)
    lyrics: str | None = Field(None, max_length=SETTINGS.max_lyrics_chars)
    duration_seconds: float = SETTINGS.default_duration_seconds
    bpm: int | None = None
    key: str | None = None
    time_signature: str | None = None
    vocal_language: str = "en"
    vocal_gender: str | None = None
    vocal_style: str | None = None
    instruments: list[str] = []
    genre: list[str] = []
    mood: list[str] = []
    structure: list[str] = []
    seed: int = 0

    def to_song_request(self) -> SongRequest:
        vocal = None
        if self.lyrics or self.vocal_gender or self.vocal_style:
            vocal = VocalSpec(
                present=True,
                gender=self.vocal_gender,
                style=self.vocal_style,
                language=self.vocal_language,
            )
        return SongRequest(
            prompt=self.prompt.strip(),
            lyrics=self.lyrics,
            genre=tuple(self.genre),
            mood=tuple(self.mood),
            instruments=tuple(self.instruments),
            vocal=vocal,
            bpm=self.bpm,
            key=self.key,
            duration_seconds=self.duration_seconds,
            structure=tuple(Section(kind=k) for k in self.structure),
            seed=self.seed,
            extra={"time_signature": self.time_signature} if self.time_signature else {},
        )


# --- model lifecycle ------------------------------------------------------


def get_adapter() -> Any:
    global _adapter, _adapter_error
    with _adapter_lock:
        if _adapter is None and _adapter_error is None:
            try:
                adapter = build(
                    SETTINGS.adapter,
                    checkpoint=SETTINGS.checkpoint,
                    device=SETTINGS.device,
                    dtype=SETTINGS.dtype,
                )
                adapter.load()
                _adapter = adapter
            except Exception as exc:  # noqa: BLE001
                _adapter_error = f"{type(exc).__name__}: {exc}"
        if _adapter_error:
            raise RuntimeError(f"model unavailable: {_adapter_error}")
        return _adapter


def run_job(job: Job) -> None:
    """Executed on the queue worker. One job at a time on the GPU."""
    adapter = get_adapter()
    request = SongRequest.from_dict(job.payload)
    deadline = time.time() + SETTINGS.job_timeout_seconds

    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    target = SETTINGS.output_dir / f"{job.id}.wav"
    result = adapter.generate(request, target)

    if time.time() > deadline:
        raise TimeoutError(
            f"generation exceeded {SETTINGS.job_timeout_seconds:.0f}s and was discarded"
        )
    if result.error:
        raise RuntimeError(result.error)

    job.audio_path = target
    job.control_warnings = list(result.resolution.warnings)
    job.result = {
        "duration_seconds": result.duration_seconds,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "model": result.model_id,
        "checkpoint": result.checkpoint,
        "seed": request.seed,
        "real_time_factor": result.real_time_factor,
        "controls_applied": result.resolution.applied,
    }


queue = JobQueue(
    runner=run_job,
    concurrency=SETTINGS.concurrency,
    max_depth=SETTINGS.max_queue_depth,
    max_stored=SETTINGS.max_stored_jobs,
)
limiter = RateLimiter(SETTINGS.requests_per_minute, SETTINGS.daily_quota_per_client)


@app.on_event("startup")
def _startup() -> None:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    if SETTINGS.warmup_on_start:
        # Load weights now so the first visitor does not pay the cold start.
        threading.Thread(target=lambda: _safe_warmup(), daemon=True).start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()


def _safe_warmup() -> None:
    try:
        get_adapter()
    except Exception:  # noqa: BLE001 - /health reports it; startup must not die
        pass


def _cleanup_loop() -> None:
    while True:
        time.sleep(900)
        cleanup_expired(SETTINGS.output_dir, SETTINGS.retain_hours)


def client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return (forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown"))


# --- endpoints ------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _adapter is not None else ("loading" if _adapter_error is None else "degraded"),
        "model_loaded": _adapter is not None,
        "model_error": _adapter_error,
        "uptime_seconds": round(time.time() - _started_at, 1),
        **queue.stats(),
        "limits": {
            "max_duration_seconds": SETTINGS.max_duration_seconds,
            "concurrency": SETTINGS.concurrency,
            "requests_per_minute": SETTINGS.requests_per_minute,
            "daily_quota": SETTINGS.daily_quota_per_client,
        },
    }


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    """What this deployment genuinely honours — the UI builds its form from this."""
    try:
        adapter = get_adapter()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "capabilities": adapter.capabilities.to_dict(),
        "license": adapter.license.to_dict(),
        "limits": {
            "min_duration_seconds": SETTINGS.min_duration_seconds,
            "max_duration_seconds": SETTINGS.max_duration_seconds,
            "max_prompt_chars": SETTINGS.max_prompt_chars,
            "max_lyrics_chars": SETTINGS.max_lyrics_chars,
        },
    }


@app.post("/api/generate", status_code=202)
def generate(body: GenerateRequest, request: Request) -> JSONResponse:
    allowed, why = limiter.check(client_id(request))
    if not allowed:
        raise HTTPException(status_code=429, detail=why)

    lowered = body.prompt.lower()
    hit = next((t for t in SETTINGS.blocked_prompt_terms if t and t in lowered), None)
    if hit:
        raise HTTPException(status_code=400, detail=f"Prompt contains a blocked term: {hit!r}")

    try:
        SETTINGS.validate_duration(body.duration_seconds)
        song = body.to_song_request()
        song.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Tell the caller up front what will and will not be honoured, so the UI can
    # show it before the render rather than after.
    warnings: list[str] = []
    applied: dict[str, str] = {}
    if _adapter is not None:
        resolution = resolve_controls(song, _adapter.capabilities)
        warnings = resolution.warnings
        applied = resolution.applied

    job = queue.submit(client_id(request), song.to_dict())
    if job.status is JobStatus.REJECTED:
        raise HTTPException(status_code=503, detail=job.error)

    return JSONResponse(
        status_code=202,
        content={
            **job.to_dict(),
            "queue_position": queue.position(job.id),
            "control_warnings": warnings,
            "controls_applied": applied,
            "poll": f"/api/jobs/{job.id}",
        },
    )


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job. Jobs are dropped after a while.")
    payload = job.to_dict()
    if job.status is JobStatus.QUEUED:
        payload["queue_position"] = queue.position(job_id)
    if job.status is JobStatus.DONE:
        payload["audio_url"] = f"/api/jobs/{job_id}/audio"
    return payload


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str) -> Response:
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if job.status is not JobStatus.DONE or not job.audio_path:
        raise HTTPException(status_code=409, detail=f"Job is {job.status.value}, not ready")
    if not job.audio_path.exists():
        raise HTTPException(status_code=410, detail="Audio expired and was cleaned up")
    return FileResponse(
        job.audio_path,
        media_type="audio/wav",
        filename=f"songforge_{job_id}.wav",
    )
