"""Objective measures for a generated song.

These are the checks that need no extra model, so they run anywhere and cost
nothing: level, clipping, dead air, spectral content, stereo behaviour and how
much the material actually changes over time.

What they can and cannot tell you matters. They can prove a file is clipped,
muffled, mono, silent, or a four-bar loop repeated for a minute. They cannot
tell you whether a violin sounds like a violin, whether the lyrics are
intelligible, or whether the song is any good. Those need a tagger, a
transcriber, and ears — see `docs/EVALUATION.md`. Nothing here should ever be
reported as a quality score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12

#: Per-frame spectral flatness above this reads as noise rather than music.
#: White noise sits near 1.0 and tonal material well below 0.2. PROVISIONAL:
#: recalibrate against real generated songs before trusting borderline cases.
NOISE_LIKE_FLATNESS = 0.5


def _to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a file as float32 with shape (frames,) or (frames, channels)."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def level_measures(audio: np.ndarray) -> dict[str, float]:
    """Peak, RMS, crest factor and clipping."""
    mono = _to_mono(audio)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    # A sample within ~0.05 dB of full scale is treated as clipped.
    clipped = int(np.sum(np.abs(mono) >= 0.9994)) if mono.size else 0
    return {
        "peak": peak,
        "peak_dbfs": _db(peak),
        "rms_dbfs": _db(rms),
        "crest_factor_db": _db(peak) - _db(rms),
        "clipped_sample_ratio": clipped / max(mono.size, 1),
    }


def silence_measures(audio: np.ndarray, sample_rate: int, threshold_db: float = -60.0) -> dict[str, float]:
    """How much of the file is effectively silent, and the longest dead span.

    A model that returns 50 s of near-silence after a 10 s intro has failed in a
    way that average level alone would hide.
    """
    mono = np.abs(_to_mono(audio))
    if mono.size == 0:
        return {"silent_ratio": 1.0, "longest_silence_seconds": 0.0, "leading_silence_seconds": 0.0}

    window = max(int(sample_rate * 0.05), 1)
    trimmed = mono[: (mono.size // window) * window].reshape(-1, window)
    frame_rms = np.sqrt(np.mean(trimmed**2, axis=1)) if trimmed.size else np.zeros(1)
    quiet = frame_rms < (10 ** (threshold_db / 20.0))

    longest = current = 0
    for is_quiet in quiet:
        current = current + 1 if is_quiet else 0
        longest = max(longest, current)
    leading = 0
    for is_quiet in quiet:
        if not is_quiet:
            break
        leading += 1

    seconds_per_frame = window / sample_rate
    return {
        "silent_ratio": float(np.mean(quiet)),
        "longest_silence_seconds": longest * seconds_per_frame,
        "leading_silence_seconds": leading * seconds_per_frame,
    }


def spectral_measures(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Where the energy sits. Catches muffled, band-limited or noise-like output."""
    mono = _to_mono(audio)
    if mono.size < 2048:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff95_hz": 0.0,
                "spectral_flatness": 0.0, "high_frequency_ratio": 0.0}

    frame, hop = 2048, 512
    window = np.hanning(frame).astype(np.float32)
    frames = 1 + (mono.size - frame) // hop
    frames = min(frames, 2000)  # bounded: a scorecard must not cost minutes
    stride = max(1, (1 + (mono.size - frame) // hop) // frames)

    spectra = []
    for index in range(frames):
        start = index * hop * stride
        chunk = mono[start : start + frame]
        if chunk.size < frame:
            break
        spectra.append(np.abs(np.fft.rfft(chunk * window)))
    if not spectra:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff95_hz": 0.0,
                "spectral_flatness": 0.0, "high_frequency_ratio": 0.0}

    stack = np.stack(spectra)
    magnitude = stack.mean(axis=0)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    total = float(magnitude.sum()) + EPS

    centroid = float((freqs * magnitude).sum() / total)
    cumulative = np.cumsum(magnitude) / total
    rolloff = float(freqs[int(np.searchsorted(cumulative, 0.95))]) if cumulative[-1] >= 0.95 else float(freqs[-1])
    high = float(magnitude[freqs >= 8000].sum() / total)

    # Flatness must be computed per frame and then averaged. Averaging the
    # spectra first smooths away the peaks that make tonal music tonal, which
    # inflates flatness and makes ordinary music look like noise.
    log_mean = np.mean(np.log(stack + EPS), axis=1)
    linear_mean = np.mean(stack, axis=1) + EPS
    flatness = float(np.mean(np.exp(log_mean) / linear_mean))

    return {
        "spectral_centroid_hz": centroid,
        "spectral_rolloff95_hz": rolloff,
        "spectral_flatness": flatness,
        "high_frequency_ratio": high,
    }


def stereo_measures(audio: np.ndarray) -> dict[str, Any]:
    """Real stereo, or a mono file in a stereo container?"""
    if audio.ndim != 2 or audio.shape[1] < 2:
        return {"channels": 1 if audio.ndim == 1 else int(audio.shape[1]),
                "stereo_correlation": 1.0, "effectively_mono": True}
    left, right = audio[:, 0], audio[:, 1]
    if np.std(left) < EPS or np.std(right) < EPS:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])
    return {
        "channels": int(audio.shape[1]),
        "stereo_correlation": correlation,
        "effectively_mono": bool(correlation > 0.999),
    }


def structure_measures(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Does the material actually develop, or is it one bar on repeat?

    Coarse log-mel-ish frames are compared to their own mean; low variation over
    time is the signature of a loop, which is a real failure mode for a model
    asked to produce a song with sections.
    """
    mono = _to_mono(audio)
    if mono.size < sample_rate:
        return {"section_novelty": 0.0, "repetition_score": 1.0, "energy_variation": 0.0}

    seconds = max(int(mono.size / sample_rate), 1)
    per_second = np.array_split(mono, seconds)
    energies = np.array([float(np.sqrt(np.mean(chunk**2) + EPS)) for chunk in per_second])

    bands = []
    for chunk in per_second:
        if chunk.size < 512:
            continue
        magnitude = np.abs(np.fft.rfft(chunk[: 1 << int(math.log2(chunk.size))]))
        edges = np.array_split(magnitude, 12)
        bands.append(np.array([float(np.log(b.sum() + EPS)) for b in edges]))
    if len(bands) < 2:
        return {"section_novelty": 0.0, "repetition_score": 1.0, "energy_variation": 0.0}

    matrix = np.stack(bands)
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + EPS)
    # Mean cosine similarity between all second-pairs: 1.0 means nothing changes.
    normed = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + EPS)
    similarity = normed @ normed.T
    off_diagonal = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    repetition = float(np.mean(off_diagonal))

    return {
        "section_novelty": float(1.0 - repetition),
        "repetition_score": repetition,
        "energy_variation": float(np.std(energies) / (np.mean(energies) + EPS)),
    }


def analyze_song(path: str | Path) -> dict[str, Any]:
    """Full objective report for one generated file."""
    audio, sample_rate = load_audio(path)
    mono_frames = audio.shape[0] if audio.ndim else 0

    report: dict[str, Any] = {
        "path": str(path),
        "sample_rate": sample_rate,
        "duration_seconds": mono_frames / sample_rate if sample_rate else 0.0,
    }
    report.update(level_measures(audio))
    report.update(silence_measures(audio, sample_rate))
    report.update(spectral_measures(audio, sample_rate))
    report.update(stereo_measures(audio))
    report.update(structure_measures(audio, sample_rate))
    report["flags"] = objective_flags(report)
    return report


#: Defaults used when a measure is absent. Every one is deliberately benign: a
#: measure that was not taken must never invent a defect.
_BENIGN = {
    "clipped_sample_ratio": 0.0,
    "peak_dbfs": 0.0,
    "silent_ratio": 0.0,
    "longest_silence_seconds": 0.0,
    "spectral_rolloff95_hz": float("inf"),
    "spectral_flatness": 0.0,
    "repetition_score": 0.0,
    "channels": 1,
    "effectively_mono": False,
}


def objective_flags(report: dict[str, Any]) -> list[str]:
    """Problems these measures can genuinely prove. Absence of flags is not quality."""

    def value(key: str) -> Any:
        return report.get(key, _BENIGN[key])

    flags: list[str] = []
    if value("clipped_sample_ratio") > 1e-4:
        flags.append("clipping")
    if value("peak_dbfs") < -30:
        flags.append("very quiet")
    if value("silent_ratio") > 0.35:
        flags.append("mostly silent")
    if value("longest_silence_seconds") > 5.0:
        flags.append("long dead span")
    if value("spectral_rolloff95_hz") < 6000:
        flags.append("band-limited / muffled")
    if value("spectral_flatness") > NOISE_LIKE_FLATNESS:
        flags.append("noise-like")
    if value("effectively_mono") and value("channels") > 1:
        flags.append("dual-mono (no stereo image)")
    if value("repetition_score") > 0.85:
        flags.append("highly repetitive / loop-like")
    return flags
