"""Conservative audio finishing for generated songs.

The chain deliberately does the minimum a release needs and nothing a
mastering engineer would argue about: sanitize → DC removal → peak control →
loudness normalization → gentle limiter → edge fades → WAV (+ optional MP3).
Dynamics are preserved: no compression, no EQ, no stereo tricks. Every stage
reports what it actually changed so the job metadata can say so.

MP3 encoding shells out to ffmpeg when available; the WAV is always the
authoritative artifact.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

#: Streaming-standard integrated loudness target and true-peak ceiling.
TARGET_LUFS = -14.0
PEAK_CEILING_DB = -1.0
FADE_IN_S = 0.01     # click protection only, inaudible as a "fade"
FADE_OUT_S = 0.05
MAX_GAIN_DB = 12.0   # never amplify more than this; quiet ≠ broken


@dataclass
class FinishingReport:
    sample_rate: int
    channels: int
    duration_seconds: float
    nan_inf_samples_repaired: int = 0
    dc_offset_removed: float = 0.0
    gain_applied_db: float = 0.0
    limiter_engaged: bool = False
    limiter_max_reduction_db: float = 0.0
    input_peak_db: float = -np.inf
    output_peak_db: float = -np.inf
    approx_input_lufs: float = -np.inf
    approx_output_lufs: float = -np.inf
    mp3_written: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (float, np.floating)):
                out[key] = round(float(value), 3) if np.isfinite(value) else None
            elif isinstance(value, np.integer):
                out[key] = int(value)
            else:
                out[key] = value
        return out


def _db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def _k_weighted_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Approximate integrated loudness (LUFS-like, K-weighting via biquads).

    Close enough to steer normalization; not a certified BS.1770 meter, and
    the report labels it "approx" for that reason.
    """
    from scipy.signal import lfilter  # local import; scipy already a dep

    # BS.1770 pre-filter (shelf) and RLB high-pass, coefficients for 48 kHz;
    # for other rates this stays an approximation, which the caller knows.
    b0 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a0 = [1.0, -1.69065929318241, 0.73248077421585]
    b1 = [1.0, -2.0, 1.0]
    a1 = [1.0, -1.99004745483398, 0.99007225036621]
    mono_channels = audio if audio.ndim == 2 else audio[:, None]
    power = 0.0
    for ch in range(mono_channels.shape[1]):
        x = lfilter(b0, a0, mono_channels[:, ch])
        x = lfilter(b1, a1, x)
        power += float(np.mean(np.square(x)))
    return -0.691 + 10.0 * np.log10(max(power, 1e-12))


def _soft_limiter(audio: np.ndarray, ceiling: float) -> tuple[np.ndarray, float]:
    """tanh-knee limiter above the ceiling; below it the signal is untouched."""
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling:
        return audio, 0.0
    over = np.abs(audio) > ceiling * 0.98
    limited = audio.copy()
    knee = ceiling * 0.98
    excess = np.abs(audio[over]) - knee
    limited[over] = np.sign(audio[over]) * (knee + (ceiling - knee) * np.tanh(excess / (ceiling - knee)))
    return limited, _db(peak) - _db(float(np.max(np.abs(limited))))


def finish(
    input_wav: str | Path,
    output_wav: str | Path,
    output_mp3: str | Path | None = None,
    target_lufs: float = TARGET_LUFS,
    peak_ceiling_db: float = PEAK_CEILING_DB,
) -> FinishingReport:
    """Run the finishing chain. Returns a report of every change made."""
    audio, sample_rate = sf.read(str(input_wav), dtype="float64", always_2d=True)
    report = FinishingReport(
        sample_rate=sample_rate,
        channels=audio.shape[1],
        duration_seconds=audio.shape[0] / sample_rate,
    )

    # 1. sanitize
    bad = ~np.isfinite(audio)
    if bad.any():
        report.nan_inf_samples_repaired = int(bad.sum())
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        report.notes.append("non-finite samples replaced with silence")

    report.input_peak_db = _db(float(np.max(np.abs(audio))) or 1e-12)

    # 2. DC removal
    dc = float(np.mean(audio))
    if abs(dc) > 1e-4:
        audio = audio - np.mean(audio, axis=0, keepdims=True)
        report.dc_offset_removed = dc

    # 3. loudness normalization toward target, bounded gain
    try:
        loudness = _k_weighted_loudness(audio, sample_rate)
    except Exception:  # scipy missing or filter blowup — peak-only fallback
        loudness = float("-inf")
        report.notes.append("loudness meter unavailable; peak-normalized only")
    report.approx_input_lufs = loudness
    if np.isfinite(loudness):
        gain_db = float(np.clip(target_lufs - loudness, -MAX_GAIN_DB, MAX_GAIN_DB))
    else:
        gain_db = float(np.clip(-1.0 - report.input_peak_db, 0.0, MAX_GAIN_DB))
    audio = audio * (10.0 ** (gain_db / 20.0))
    report.gain_applied_db = gain_db

    # 4. peak control + gentle limiter at the ceiling
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    audio, reduction = _soft_limiter(audio, ceiling)
    if reduction > 0.01:
        report.limiter_engaged = True
        report.limiter_max_reduction_db = reduction

    # 5. edge fades (click protection, not an artistic fade)
    n_in = min(int(FADE_IN_S * sample_rate), audio.shape[0] // 4)
    n_out = min(int(FADE_OUT_S * sample_rate), audio.shape[0] // 4)
    if n_in > 0:
        audio[:n_in] *= np.linspace(0.0, 1.0, n_in)[:, None]
    if n_out > 0:
        audio[-n_out:] *= np.linspace(1.0, 0.0, n_out)[:, None]

    report.output_peak_db = _db(float(np.max(np.abs(audio))) or 1e-12)
    try:
        report.approx_output_lufs = _k_weighted_loudness(audio, sample_rate)
    except Exception:
        pass

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio.astype(np.float32), sample_rate, subtype="PCM_16")

    # 6. MP3 via ffmpeg, best effort — WAV remains authoritative
    if output_mp3 is not None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            proc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(output_wav),
                 "-codec:a", "libmp3lame", "-qscale:a", "1", str(output_mp3)],
                capture_output=True, text=True,
            )
            report.mp3_written = proc.returncode == 0
            if proc.returncode != 0:
                report.notes.append(f"mp3 encode failed: {proc.stderr.strip()[:200]}")
        else:
            report.notes.append("ffmpeg not found; mp3 skipped")

    return report
