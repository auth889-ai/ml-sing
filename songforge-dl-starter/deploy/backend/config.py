"""Operational limits for the public SongForge API.

Every value here exists to stop one specific failure: a GPU serialised behind an
unbounded queue, a user asking for a ten-minute song on a shared L4, a job that
hangs forever holding the only worker, or a disk that fills with abandoned
output. Defaults are deliberately conservative — this is a single-GPU service,
not a scaled one, and pretending otherwise is how the first public link dies.

Override any of these with environment variables in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # --- model ---------------------------------------------------------
    adapter: str = os.environ.get("SONGFORGE_ADAPTER", "acestep")
    checkpoint: str = os.environ.get("SONGFORGE_CHECKPOINT", "xl-turbo")
    lora_path: str | None = os.environ.get("SONGFORGE_LORA") or None
    device: str = os.environ.get("SONGFORGE_DEVICE", "cuda")
    dtype: str = os.environ.get("SONGFORGE_DTYPE", "bfloat16")

    #: Load weights at startup rather than on the first request, so the first
    #: visitor does not pay a 60 s cold start and time out.
    warmup_on_start: bool = os.environ.get("SONGFORGE_WARMUP", "1") == "1"

    # --- generation limits ---------------------------------------------
    #: The model can do 600 s. A public single-GPU demo cannot afford to.
    max_duration_seconds: float = _float("SONGFORGE_MAX_DURATION", 120.0)
    min_duration_seconds: float = _float("SONGFORGE_MIN_DURATION", 10.0)
    default_duration_seconds: float = _float("SONGFORGE_DEFAULT_DURATION", 60.0)
    max_prompt_chars: int = _int("SONGFORGE_MAX_PROMPT", 1000)
    max_lyrics_chars: int = _int("SONGFORGE_MAX_LYRICS", 4000)

    # --- queue ----------------------------------------------------------
    #: One GPU, one job. Raising this without a second GPU just adds latency
    #: and risks OOM mid-render.
    concurrency: int = _int("SONGFORGE_CONCURRENCY", 1)
    max_queue_depth: int = _int("SONGFORGE_MAX_QUEUE", 20)
    job_timeout_seconds: float = _float("SONGFORGE_JOB_TIMEOUT", 300.0)

    # --- quality mode -----------------------------------------------------
    #: "best" renders up to this many seeds and returns the top-ranked take.
    #: Rendering stops early if the job deadline would be blown.
    best_candidates: int = _int("SONGFORGE_BEST_N", 3)
    #: Conservative finishing (loudness/peaks/fades) + MP3 alongside the WAV.
    finishing_enabled: bool = os.environ.get("SONGFORGE_FINISHING", "1") == "1"

    # --- rate limiting and quota ----------------------------------------
    requests_per_minute: int = _int("SONGFORGE_RPM", 4)
    daily_quota_per_client: int = _int("SONGFORGE_DAILY_QUOTA", 30)

    # --- storage ---------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("SONGFORGE_OUTPUT", "/tmp/songforge_jobs")))
    #: Generated audio is deleted after this long. A public demo that keeps
    #: every render forever fills its disk within days.
    retain_hours: float = _float("SONGFORGE_RETAIN_HOURS", 6.0)
    max_stored_jobs: int = _int("SONGFORGE_MAX_STORED", 200)

    # --- abuse prevention -------------------------------------------------
    #: Cheap, honest guardrails only. This blocks obvious attempts to make the
    #: service impersonate a named artist; it is not content moderation and is
    #: not claimed to be.
    blocked_prompt_terms: tuple[str, ...] = tuple(
        term.strip().lower()
        for term in os.environ.get("SONGFORGE_BLOCKED_TERMS", "").split(",")
        if term.strip()
    )

    def validate_duration(self, seconds: float) -> float:
        if seconds < self.min_duration_seconds:
            raise ValueError(f"duration must be at least {self.min_duration_seconds:.0f}s")
        if seconds > self.max_duration_seconds:
            raise ValueError(
                f"duration must be at most {self.max_duration_seconds:.0f}s on this deployment"
            )
        return seconds


SETTINGS = Settings()
