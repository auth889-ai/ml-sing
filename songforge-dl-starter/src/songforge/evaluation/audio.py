from __future__ import annotations

import time

import torch

from songforge.losses.audio import (
    multi_resolution_stft_loss,
    spectral_convergence,
    waveform_l1,
)


@torch.no_grad()
def si_sdr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Scale-invariant SDR in dB.

    Unlike plain SNR this is immune to a global gain difference, so a codec that
    reconstructs the waveform shape but not its level is not unfairly punished.
    """
    pred_flat = pred.reshape(-1).float()
    target_flat = target.reshape(-1).float()
    target_energy = target_flat.pow(2).sum().clamp_min(1e-12)
    alpha = torch.dot(pred_flat, target_flat) / target_energy
    projection = alpha * target_flat
    noise = pred_flat - projection
    ratio = projection.pow(2).sum().clamp_min(1e-12) / noise.pow(2).sum().clamp_min(1e-12)
    return float(10.0 * torch.log10(ratio))


def _magnitude(signal: torch.Tensor, n_fft: int) -> torch.Tensor:
    flat = signal.reshape(-1).float()
    if flat.numel() < n_fft:
        flat = torch.nn.functional.pad(flat, (0, n_fft - flat.numel()))
    window = torch.hann_window(n_fft, device=flat.device)
    return torch.stft(flat, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()


@torch.no_grad()
def log_spectral_distance(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024) -> float:
    """RMS difference of log-magnitude spectra, in dB. Lower is closer."""
    p = _magnitude(pred, n_fft).clamp_min(1e-8)
    y = _magnitude(target, n_fft).clamp_min(1e-8)
    diff = 20.0 * (torch.log10(p) - torch.log10(y))
    return float(diff.pow(2).mean().sqrt())


@torch.no_grad()
def transient_preservation(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024) -> float:
    """Correlation of onset envelopes, in [-1, 1]. 1 means transients line up.

    Percussive detail is the first thing an aggressively downsampled codec loses,
    and waveform L1 barely registers it, so M04 measures it directly.
    """
    p = _magnitude(pred, n_fft)
    y = _magnitude(target, n_fft)
    frames = min(p.size(-1), y.size(-1))
    if frames < 3:
        return 0.0
    p_env = (p[:, 1:frames] - p[:, : frames - 1]).clamp_min(0).sum(dim=0)
    y_env = (y[:, 1:frames] - y[:, : frames - 1]).clamp_min(0).sum(dim=0)
    p_env = p_env - p_env.mean()
    y_env = y_env - y_env.mean()
    denominator = (p_env.norm() * y_env.norm()).clamp_min(1e-8)
    return float(torch.dot(p_env, y_env) / denominator)


@torch.no_grad()
def high_frequency_preservation(
    pred: torch.Tensor, target: torch.Tensor, sample_rate: int = 24000, cutoff_hz: float = 4000.0,
    n_fft: int = 1024,
) -> float:
    """Ratio of reconstructed to original energy above `cutoff_hz`, in dB.

    0 dB is perfect; negative means the codec dulled the top end, which is the
    characteristic failure of a low latent rate.
    """
    p = _magnitude(pred, n_fft)
    y = _magnitude(target, n_fft)
    freqs = torch.linspace(0, sample_rate / 2, p.size(0), device=p.device)
    band = freqs > cutoff_hz
    if not bool(band.any()):
        return 0.0
    p_energy = p[band].pow(2).sum().clamp_min(1e-12)
    y_energy = y[band].pow(2).sum().clamp_min(1e-12)
    return float(10.0 * torch.log10(p_energy / y_energy))


@torch.no_grad()
def reconstruction_metrics(
    pred: torch.Tensor, target: torch.Tensor, sample_rate: int = 24000
) -> dict[str, float]:
    """Objective reconstruction quality.

    The extra measures beyond L1/MR-STFT exist because M04 compares codecs at
    different latent rates, where a single distortion number hides which kind of
    detail was traded away.
    """
    noise = target - pred
    snr = 10.0 * torch.log10(target.pow(2).mean().clamp_min(1e-8) / noise.pow(2).mean().clamp_min(1e-8))
    return {
        "waveform_l1": float(waveform_l1(pred, target).item()),
        "mrstft": float(multi_resolution_stft_loss(pred, target).item()),
        "spectral_convergence": float(spectral_convergence(pred, target).item()),
        "snr_db": float(snr.item()),
        "si_sdr_db": si_sdr(pred, target),
        "log_spectral_distance_db": log_spectral_distance(pred, target),
        "transient_preservation": transient_preservation(pred, target),
        "high_frequency_preservation_db": high_frequency_preservation(pred, target, sample_rate),
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
