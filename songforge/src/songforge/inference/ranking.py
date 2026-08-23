"""Best-of-N candidate selection for SongForge's quality mode.

Fast mode generates one seed and returns it. Best mode generates several,
scores each with the objective measures we can actually defend
(`songforge.evaluation.song`), rejects broken candidates, and returns the
strongest survivor. The scoring is transparent: every candidate's report,
flags, penalty arithmetic, and the final ordering are all returned so the
API can expose *why* a take was chosen.

These measures prove problems, not quality — the module docstring of
`evaluation.song` says so and it stays true here. Ranking removes the worst
takes; it does not certify the best one as good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..evaluation.song import analyze_song

#: Flags that disqualify a candidate outright (unless every candidate is
#: disqualified, in which case the least-bad one is returned, still flagged).
FATAL_FLAGS = ("mostly silent",)
FLAG_PENALTY = 25.0


@dataclass
class Candidate:
    seed: int
    path: Path
    report: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    score: float = 0.0
    fatal: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "path": str(self.path),
            "score": round(self.score, 2),
            "flags": self.flags,
            "fatal": self.fatal,
            "error": self.error,
        }


def score_candidate(report: dict) -> float:
    """Higher is better. Starts at 100, subtracts provable defects.

    Keys are the flat report of `analyze_song`; missing keys read as benign,
    matching that module's own convention.
    """
    score = 100.0
    score -= FLAG_PENALTY * len(report.get("flags", []))

    score -= 40.0 * float(report.get("silent_ratio", 0.0))
    longest_gap = float(report.get("longest_silence_seconds", 0.0))
    if longest_gap > 2.0:
        score -= 5.0 * (longest_gap - 2.0)

    score -= 100.0 * float(report.get("clipped_sample_ratio", 0.0))

    # repetition_score near 1.0 = one bar on repeat; low energy_variation on
    # top of that = static texture rather than a song that develops.
    repetition = float(report.get("repetition_score", 0.0))
    if repetition > 0.7:
        score -= 30.0 * (repetition - 0.7) / 0.3
    if float(report.get("energy_variation", 1.0)) < 0.05:
        score -= 10.0

    return score


def rank_candidates(
    generate: Callable[[int, Path], None],
    seeds: list[int],
    workdir: str | Path,
) -> tuple[Candidate | None, list[Candidate]]:
    """Generate one take per seed, score them, return (best, all).

    `generate(seed, path)` renders one candidate; exceptions are recorded on
    the candidate rather than aborting the batch, so one bad seed cannot
    sink the request.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    for seed in seeds:
        cand = Candidate(seed=seed, path=workdir / f"candidate_{seed}.wav")
        try:
            generate(seed, cand.path)
            cand.report = analyze_song(cand.path)
            cand.flags = list(cand.report.get("flags", []))
            cand.fatal = any(flag in FATAL_FLAGS for flag in cand.flags)
            cand.score = score_candidate(cand.report)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            cand.error = f"{type(exc).__name__}: {exc}"
            cand.fatal = True
            cand.score = float("-inf")
        candidates.append(cand)

    usable = [c for c in candidates if not c.fatal and c.error is None]
    pool = usable or [c for c in candidates if c.error is None]
    best = max(pool, key=lambda c: c.score, default=None)
    return best, candidates
