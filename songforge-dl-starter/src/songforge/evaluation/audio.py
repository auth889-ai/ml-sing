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
