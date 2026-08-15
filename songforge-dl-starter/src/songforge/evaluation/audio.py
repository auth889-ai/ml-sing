from __future__ import annotations

import time

import torch

from songforge.losses.audio import (
    multi_resolution_stft_loss,
    spectral_convergence,
    waveform_l1,
)


@torch.no_grad()
def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    noise = target - pred
    snr = 10.0 * torch.log10(target.pow(2).mean().clamp_min(1e-8) / noise.pow(2).mean().clamp_min(1e-8))
    return {
        "waveform_l1": float(waveform_l1(pred, target).item()),
        "mrstft": float(multi_resolution_stft_loss(pred, target).item()),
        "spectral_convergence": float(spectral_convergence(pred, target).item()),
        "snr_db": float(snr.item()),
    }


#: Spectral character buckets used to pick varied held-out listening examples.
#: These are signal heuristics, not instrument ground truth: BabySlakh stem names
#: are not reliably recoverable from a segment path, so the label describes what
#: the audio looks like spectrally.
AUDIO_CHARACTERS = ("percussive", "harmonic", "bass_heavy", "mixed")


@torch.no_grad()
def audio_character_features(audio: torch.Tensor, sample_rate: int, n_fft: int = 1024) -> dict[str, float]:
    """Cheap spectral descriptors used to bucket a segment for listening tests."""
    signal = audio.detach().float()
    if signal.ndim == 3:
        signal = signal[0]
    if signal.ndim == 2:
        signal = signal.mean(dim=0)
    if signal.numel() < n_fft:
        return {
            "low_band_ratio": 0.0, "high_band_ratio": 0.0, "spectral_flatness": 0.0,
            "spectral_flux": 0.0, "zero_crossing_rate": 0.0, "rms": float(signal.pow(2).mean().sqrt()),
        }

    window = torch.hann_window(n_fft, device=signal.device)
    spec = torch.stft(
        signal, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True
    ).abs()

    freqs = torch.linspace(0, sample_rate / 2, spec.size(0), device=signal.device)
    total = spec.sum().clamp_min(1e-8)
    low_band = spec[freqs < 250.0].sum() / total
    high_band = spec[freqs > 4000.0].sum() / total

    magnitude = spec.clamp_min(1e-10)
    flatness = torch.exp(magnitude.log().mean(dim=0)) / magnitude.mean(dim=0).clamp_min(1e-10)
    # Frame-normalized positive spectral flux, bounded in [0, 1]. An unnormalized
    # sum scales with signal magnitude and would swamp every other feature.
    rise = (spec[:, 1:] - spec[:, :-1]).clamp_min(0).sum(dim=0)
    frame_energy = spec[:, 1:].sum(dim=0).clamp_min(1e-8)
    flux = (rise / frame_energy).mean()
    zero_crossings = (torch.sign(signal[1:]) != torch.sign(signal[:-1])).float().mean()

    return {
        "low_band_ratio": float(low_band),
        "high_band_ratio": float(high_band),
        "spectral_flatness": float(flatness.mean()),
        "spectral_flux": float(flux),
        "zero_crossing_rate": float(zero_crossings),
        "rms": float(signal.pow(2).mean().sqrt()),
    }


def character_scores(features: dict[str, float]) -> dict[str, float]:
    """Score a segment against each listening-example bucket. Highest wins.

    All inputs are bounded roughly in [0, 1]. `harmonic` subtracts low-band
    dominance because a solo bass line is also maximally tonal and would
    otherwise win both the harmonic and the bass bucket.
    """
    low = features["low_band_ratio"]
    high = features["high_band_ratio"]
    flatness = features["spectral_flatness"]
    flux = features["spectral_flux"]
    zcr = features["zero_crossing_rate"]
    return {
        "percussive": flux + flatness + zcr,
        "harmonic": (1.0 - flatness) + (1.0 - zcr) - 1.5 * low - flux,
        "bass_heavy": 2.0 * low - high - flux,
        "mixed": 1.0
        - abs(low - 0.30)
        - abs(high - 0.20)
        - abs(flatness - 0.40)
        - abs(flux - 0.30),
    }


def classify_audio_character(features: dict[str, float]) -> str:
    scores = character_scores(features)
    return max(scores, key=lambda name: scores[name])


def select_character_examples(
    candidates: list[dict],
    characters: tuple[str, ...] = AUDIO_CHARACTERS,
) -> dict[str, dict]:
    """Pick the strongest held-out segment for each character bucket.

    ``candidates`` is a list of ``{"index": int, "features": {...}}``. A segment is
    used at most once, so four distinct examples come back when the validation set
    is large enough; buckets with no candidate left are simply absent.
    """
    remaining = {entry["index"]: entry for entry in candidates}
    chosen: dict[str, dict] = {}
    for character in characters:
        if not remaining:
            break
        best_index = max(
            remaining,
            key=lambda index: character_scores(remaining[index]["features"])[character],
        )
        entry = remaining.pop(best_index)
        chosen[character] = {
            "index": best_index,
            "features": entry["features"],
            "score": character_scores(entry["features"])[character],
        }
    return chosen


@torch.no_grad()
def codec_timing_metrics(model, audio: torch.Tensor, repeats: int = 3) -> dict[str, float]:
    device = next(model.parameters()).device
    audio = audio.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(repeats, 1)):
        encoded = model.encode(audio)
        _ = model.decode(encoded["quantized"])[..., : audio.shape[-1]]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / max(repeats, 1)
    audio_seconds = audio.shape[-1] / float(model.sample_rate)
    return {
        "encode_decode_seconds": float(elapsed),
        "audio_seconds": float(audio_seconds),
        "real_time_factor": float(elapsed / max(audio_seconds, 1e-8)),
    }
