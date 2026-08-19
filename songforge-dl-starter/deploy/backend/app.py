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
    #: "fast" = one take; "best" = up to SONGFORGE_BEST_N takes, objectively
    #: ranked, top take returned. Ranking removes provably broken takes; it
    #: does not certify quality, and the response says which seeds ran.
    quality: str = Field("fast", pattern="^(fast|best)$")

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
    """Executed on the queue worker. One job at a time on the GPU.

    Pipeline: generate (1 take, or up to N for quality=best) → rank (best
    mode only) → finish (loudness/peaks/fades, WAV + MP3). Phase is written
    to the job so the UI can show real progress rather than a spinner.
    """
    adapter = get_adapter()
    quality = job.payload.pop("quality", "fast") if isinstance(job.payload, dict) else "fast"
    request = SongRequest.from_dict(job.payload)
    deadline = time.time() + SETTINGS.job_timeout_seconds

    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    workdir = SETTINGS.output_dir / job.id
    workdir.mkdir(parents=True, exist_ok=True)

    # --- generate ---------------------------------------------------------
    job.phase = "generating"
    n_takes = SETTINGS.best_candidates if quality == "best" else 1
    takes: list[tuple[int, Path, Any]] = []
    for offset in range(n_takes):
        seed = request.seed + offset
        take_request = SongRequest.from_dict({**job.payload, "seed": seed})
        path = workdir / f"take_{seed}.wav"
        result = adapter.generate(take_request, path)
        if result.error:
            if not takes:
                raise RuntimeError(result.error)
            break  # keep the takes we have rather than failing the job
        takes.append((seed, path, result))
        # Never start a take the deadline cannot afford: assume the next take
        # costs what the slowest one so far cost.
        slowest = max((r.real_time_factor or 1.0) for _, _, r in takes)
        if time.time() + slowest * request.duration_seconds > deadline:
            break
    if not takes:
        raise RuntimeError("no takes were generated")
    if time.time() > deadline:
        raise TimeoutError(
            f"generation exceeded {SETTINGS.job_timeout_seconds:.0f}s and was discarded"
        )

    # --- rank -------------------------------------------------------------
    ranking_summary: list[dict[str, Any]] = []
    best_seed, best_path, best_result = takes[0]
    if len(takes) > 1:
        job.phase = "ranking"
        from songforge.evaluation.song import analyze_song
        from songforge.inference.ranking import FATAL_FLAGS, score_candidate

        scored = []
        for seed, path, result in takes:
            report = analyze_song(path)
            entry = {
                "seed": seed,
                "score": round(score_candidate(report), 2),
                "flags": report.get("flags", []),
                "fatal": any(f in FATAL_FLAGS for f in report.get("flags", [])),
            }
            scored.append((entry, path, result))
            ranking_summary.append(entry)
        usable = [s for s in scored if not s[0]["fatal"]] or scored
        winner = max(usable, key=lambda s: s[0]["score"])
        best_seed, best_path, best_result = winner[0]["seed"], winner[1], winner[2]

    # --- finish -----------------------------------------------------------
    final_wav = SETTINGS.output_dir / f"{job.id}.wav"
    final_mp3 = SETTINGS.output_dir / f"{job.id}.mp3"
    finishing_report: dict[str, Any] | None = None
    if SETTINGS.finishing_enabled:
        job.phase = "finishing"
        from songforge.inference.finishing import finish

        finishing_report = finish(best_path, final_wav, final_mp3).to_dict()
    else:
        best_path.replace(final_wav)

    # workdir holds the losing takes; they served their purpose
    for _, path, _ in takes:
        path.unlink(missing_ok=True)
    try:
        workdir.rmdir()
    except OSError:
        pass

    job.audio_path = final_wav
    job.mp3_path = final_mp3 if final_mp3.exists() else None
    job.control_warnings = list(best_result.resolution.warnings)
    job.result = {
        "duration_seconds": best_result.duration_seconds,
        "sample_rate": best_result.sample_rate,
        "channels": best_result.channels,
        "model": best_result.model_id,
        "checkpoint": best_result.checkpoint,
        "seed": best_seed,
        "quality": quality,
        "candidates_evaluated": len(takes) if len(takes) > 1 else None,
        "ranking": ranking_summary or None,
        "finishing": finishing_report,
        "mp3_available": job.mp3_path is not None,
        "real_time_factor": best_result.real_time_factor,
        "controls_applied": best_result.resolution.applied,
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

    job = queue.submit(client_id(request), {**song.to_dict(), "quality": body.quality})
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


@app.get("/api/jobs/{job_id}/audio.mp3")
def job_audio_mp3(job_id: str) -> Response:
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if job.status is not JobStatus.DONE or not job.mp3_path:
        raise HTTPException(status_code=409, detail="MP3 not available for this job")
    if not job.mp3_path.exists():
        raise HTTPException(status_code=410, detail="Audio expired and was cleaned up")
    return FileResponse(
        job.mp3_path,
        media_type="audio/mpeg",
        filename=f"songforge_{job_id}.mp3",
    )
