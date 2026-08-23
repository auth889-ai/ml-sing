"""Originality-risk screening for generated audio.

Two layers, in increasing cost:

1. **Exact duplicate** — SHA256 over decoded PCM (container-independent).
   Catches verbatim regurgitation of a training item.
2. **Chroma similarity** — a coarse pitch-class-profile sequence, compared
   against a fingerprint database built over the training corpus. High
   cosine similarity across aligned windows marks a candidate *suspicious*
   so the pipeline can regenerate with a different seed.

This is a screen, not a proof. It cannot certify that output is free of
copyright similarity, and nothing downstream may claim that it does. What
it can do is catch the two failure modes an adapter actually produces —
verbatim copies and close melodic/harmonic tracings of training items —
cheaply enough to run on every Best-mode candidate.

Fingerprint DB format: JSONL, one line per corpus item:
    {"id": ..., "pcm_sha256": ..., "chroma": [[12 floats] * n_windows]}
Built once per corpus by `build_fingerprint` over each file (run where the
audio lives); the DB rides with the deployment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

WINDOW_S = 5.0        # chroma summarised per 5 s window
FRAME = 4096
HOP = 2048
A4 = 440.0

#: cosine similarity at or above this, sustained over MIN_MATCH_WINDOWS
#: consecutive windows, marks the candidate suspicious.
SUSPICIOUS_COSINE = 0.985
MIN_MATCH_WINDOWS = 3  # 15 s of near-identical harmony


def _load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def pcm_sha256(path: str | Path) -> str:
    """Hash the decoded samples, not the container bytes."""
    mono, _ = _load_mono(path)
    # 16-bit quantisation so a lossless re-encode of the same audio matches
    q = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    return hashlib.sha256(q.tobytes()).hexdigest()


def chroma_sequence(path: str | Path) -> np.ndarray:
    """(n_windows, 12) L2-normalised pitch-class profiles, one per 5 s."""
    mono, sample_rate = _load_mono(path)
    if mono.size < FRAME:
        return np.zeros((0, 12), dtype=np.float32)

    freqs = np.fft.rfftfreq(FRAME, 1.0 / sample_rate)
    # map each bin (80 Hz–5 kHz, where harmony lives) to a pitch class
    voiced = (freqs >= 80.0) & (freqs <= 5000.0)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-6) / A4)
    pitch_class = np.mod(np.round(midi), 12).astype(int)

    window = np.hanning(FRAME).astype(np.float32)
    frames_per_window = max(int(WINDOW_S * sample_rate / HOP), 1)

    profiles: list[np.ndarray] = []
    accumulator = np.zeros(12, dtype=np.float64)
    count = 0
    for start in range(0, mono.size - FRAME, HOP):
        spectrum = np.abs(np.fft.rfft(mono[start:start + FRAME] * window))
        chroma = np.zeros(12)
        np.add.at(chroma, pitch_class[voiced], spectrum[voiced])
        accumulator += chroma
        count += 1
        if count == frames_per_window:
            norm = np.linalg.norm(accumulator)
            profiles.append((accumulator / norm if norm > 0 else accumulator).astype(np.float32))
            accumulator = np.zeros(12, dtype=np.float64)
            count = 0
    return np.stack(profiles) if profiles else np.zeros((0, 12), dtype=np.float32)


def build_fingerprint(item_id: str, path: str | Path) -> dict:
    return {
        "id": item_id,
        "pcm_sha256": pcm_sha256(path),
        "chroma": chroma_sequence(path).tolist(),
    }


@dataclass
class OriginalityVerdict:
    suspicious: bool
    exact_duplicate_of: str | None
    closest_item: str | None
    max_sustained_cosine: float
    matched_windows: int

    def to_dict(self) -> dict:
        return {
            "suspicious": self.suspicious,
            "exact_duplicate_of": self.exact_duplicate_of,
            "closest_item": self.closest_item,
            "max_sustained_cosine": round(self.max_sustained_cosine, 4),
            "matched_windows": self.matched_windows,
            "caveat": "similarity screen only; not a proof of originality or infringement",
        }


def screen(candidate_path: str | Path, db_path: str | Path) -> OriginalityVerdict:
    """Compare one candidate against the corpus fingerprint DB."""
    cand_hash = pcm_sha256(candidate_path)
    cand_chroma = chroma_sequence(candidate_path)

    exact: str | None = None
    closest: str | None = None
    best_sustained = 0.0
    best_windows = 0

    with Path(db_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("pcm_sha256") == cand_hash:
                exact = entry.get("id")
                break
            ref = np.asarray(entry.get("chroma") or [], dtype=np.float32)
            if not len(ref) or not len(cand_chroma):
                continue
            # slide the shorter sequence over the longer; track the longest
            # run of consecutive windows above the threshold
            short, long_ = (cand_chroma, ref) if len(cand_chroma) <= len(ref) else (ref, cand_chroma)
            for offset in range(len(long_) - len(short) + 1):
                sims = np.sum(short * long_[offset:offset + len(short)], axis=1)
                run = best_run = 0
                for value in sims:
                    run = run + 1 if value >= SUSPICIOUS_COSINE else 0
                    best_run = max(best_run, run)
                sustained = float(np.max(sims)) if len(sims) else 0.0
                if best_run > best_windows or (best_run == best_windows and sustained > best_sustained):
                    best_windows = best_run
                    best_sustained = sustained
                    closest = entry.get("id")

    return OriginalityVerdict(
        suspicious=exact is not None or best_windows >= MIN_MATCH_WINDOWS,
        exact_duplicate_of=exact,
        closest_item=closest if exact is None else exact,
        max_sustained_cosine=best_sustained if exact is None else 1.0,
        matched_windows=best_windows,
    )
