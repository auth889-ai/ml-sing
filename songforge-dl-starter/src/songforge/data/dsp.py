"""M02 signal processing: resampling, channel policy, loudness, segmentation.

Everything here is pure torch and deterministic. The resampler is implemented
locally rather than delegated to torchaudio so a manifest built on a workstation
and a manifest built on a Colab runtime contain identical numbers. Preprocessing
output is part of the experiment record, so it must not vary with which optional
audio library happens to be installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

SILENCE_FLOOR_DBFS = -160.0


def _dbfs(value: float) -> float:
    return SILENCE_FLOOR_DBFS if value <= 0 else max(20.0 * math.log10(value), SILENCE_FLOOR_DBFS)


@dataclass(frozen=True)
class AudioStats:
    """Amplitude description of one waveform."""

    peak: float
    rms: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    dc_offset: float

    def to_dict(self) -> dict:
        return {
            "peak": self.peak,
            "rms": self.rms,
            "peak_dbfs": self.peak_dbfs,
            "rms_dbfs": self.rms_dbfs,
            "clipping_ratio": self.clipping_ratio,
            "dc_offset": self.dc_offset,
        }


def compute_stats(waveform: torch.Tensor, clipping_threshold: float = 0.999) -> AudioStats:
    """Amplitude statistics for a ``[channels, samples]`` waveform."""
    if waveform.numel() == 0:
        return AudioStats(0.0, 0.0, SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS, 0.0, 0.0)
    audio = waveform.detach().float()
    peak = float(audio.abs().max().item())
    rms = float(audio.pow(2).mean().sqrt().item())
    clipped = float((audio.abs() >= clipping_threshold).float().mean().item())
    return AudioStats(
        peak=peak,
        rms=rms,
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        clipping_ratio=clipped,
        dc_offset=float(audio.mean().item()),
    )


def is_clipped(waveform: torch.Tensor, threshold: float = 0.999, max_ratio: float = 0.01) -> bool:
    return compute_stats(waveform, threshold).clipping_ratio > max_ratio


def is_silent(waveform: torch.Tensor, threshold_dbfs: float = -60.0) -> bool:
    return compute_stats(waveform).rms_dbfs < threshold_dbfs


def to_channels(waveform: torch.Tensor, channels: int, policy: str = "mean") -> torch.Tensor:
    """Force a ``[channels, samples]`` layout.

    policy "mean" downmixes to mono by averaging; "first" keeps channel 0.
    Upmixing repeats the available channels.
    """
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(waveform.shape)}")
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    current = waveform.size(0)
    if current == channels:
        return waveform
    if channels == 1:
        if policy == "first":
            return waveform[:1]
        if policy == "mean":
            return waveform.mean(dim=0, keepdim=True)
        raise ValueError(f"Unknown channel policy {policy!r}")
    if current == 1:
        return waveform.repeat(channels, 1)
    if current > channels:
        return waveform[:channels]
    padding = waveform[-1:].repeat(channels - current, 1)
    return torch.cat([waveform, padding], dim=0)


@lru_cache(maxsize=32)
def _resample_kernel(
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int,
    rolloff: float,
) -> tuple[torch.Tensor, int]:
    """Windowed-sinc polyphase kernel, cached per rate pair."""
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / min(orig_freq, new_freq))
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)

    kernels = []
    for i in range(new_freq):
        t = (-i / new_freq + idx / orig_freq) * base_freq
        t = t.clamp(-lowpass_filter_width, lowpass_filter_width)
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
        sinc = torch.where(t == 0, torch.ones_like(t), torch.sin(math.pi * t) / (math.pi * t))
        kernels.append(sinc * window)

    kernel = torch.stack(kernels).unsqueeze(1).mul_(base_freq / orig_freq)
    return kernel.to(torch.float32), width


def resample_waveform(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> torch.Tensor:
    """Band-limited rational resampling of a ``[channels, samples]`` waveform."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(waveform.shape)}")
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError(f"Sample rates must be positive, got {orig_freq} -> {new_freq}")
    if orig_freq == new_freq:
        return waveform
    if waveform.numel() == 0:
        return waveform

    divisor = math.gcd(int(orig_freq), int(new_freq))
    reduced_orig = int(orig_freq) // divisor
    reduced_new = int(new_freq) // divisor

    kernel, width = _resample_kernel(reduced_orig, reduced_new, lowpass_filter_width, rolloff)
    kernel = kernel.to(waveform.dtype)

    length = waveform.size(-1)
    padded = F.pad(waveform.unsqueeze(1), (width, width + reduced_orig))
    convolved = F.conv1d(padded, kernel, stride=reduced_orig)
    resampled = convolved.transpose(1, 2).reshape(waveform.size(0), -1)
    target_length = math.ceil(new_freq * length / orig_freq)
    return resampled[..., :target_length]


def normalize_amplitude(
    waveform: torch.Tensor,
    mode: str = "peak",
    target_dbfs: float = -1.0,
    max_gain_db: float = 30.0,
) -> tuple[torch.Tensor, float]:
    """Scale toward a target level. Returns ``(waveform, applied_gain_db)``.

    ``max_gain_db`` stops near-silent input from being amplified into noise.
    """
    if mode == "none":
        return waveform, 0.0
    stats = compute_stats(waveform)
    current = stats.peak_dbfs if mode == "peak" else stats.rms_dbfs if mode == "rms" else None
    if current is None:
        raise ValueError(f"Unknown normalization mode {mode!r}")
    if current <= SILENCE_FLOOR_DBFS:
        return waveform, 0.0

    gain_db = min(target_dbfs - current, max_gain_db)
    gain = 10.0 ** (gain_db / 20.0)
    return (waveform * gain).clamp(-1.0, 1.0), gain_db


def segment_bounds(
    num_samples: int,
    segment_samples: int,
    hop_samples: int | None = None,
    pad_final_partial: bool = False,
) -> list[tuple[int, int]]:
    """Deterministic segment boundaries as ``[(start, end), ...]``.

    Depends only on the arguments, never on file order or wall-clock time, so a
    rerun reproduces the manifest exactly.
    """
    if segment_samples <= 0:
        raise ValueError(f"segment_samples must be positive, got {segment_samples}")
    hop = segment_samples if hop_samples is None else hop_samples
    if hop <= 0:
        raise ValueError(f"hop_samples must be positive, got {hop}")

    bounds: list[tuple[int, int]] = []
    start = 0
    while start + segment_samples <= num_samples:
        bounds.append((start, start + segment_samples))
        start += hop

    if pad_final_partial:
        tail_start = len(bounds) * hop
        if tail_start < num_samples and (not bounds or tail_start + segment_samples > num_samples):
            bounds.append((tail_start, tail_start + segment_samples))
    return bounds


def extract_segment(waveform: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice ``[start, end)``, zero-padding when the request runs past the end."""
    segment = waveform[..., start:end]
    missing = (end - start) - segment.size(-1)
    if missing > 0:
        segment = F.pad(segment, (0, missing))
    return segment
