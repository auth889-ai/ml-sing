from __future__ import annotations

import torch
import torch.nn.functional as F


def waveform_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def multi_resolution_stft_loss(pred: torch.Tensor, target: torch.Tensor, fft_sizes=(256, 512, 1024)) -> torch.Tensor:
    total = pred.new_zeros(())
    pred_1d = pred.squeeze(1)
    target_1d = target.squeeze(1)
    for n_fft in fft_sizes:
        if pred_1d.shape[-1] < n_fft or target_1d.shape[-1] < n_fft:
            continue
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        p = torch.stft(pred_1d, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        y = torch.stft(target_1d, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        total = total + F.l1_loss(torch.log1p(p.abs()), torch.log1p(y.abs()))
    used = sum(1 for n_fft in fft_sizes if pred_1d.shape[-1] >= n_fft and target_1d.shape[-1] >= n_fft)
    if used == 0:
        return waveform_l1(pred, target)
    return total / used


def spectral_convergence(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    pred_1d = pred.squeeze(1)
    target_1d = target.squeeze(1)
    if pred_1d.shape[-1] < n_fft:
        return waveform_l1(pred, target)
    window = torch.hann_window(n_fft, device=pred.device)
    p = torch.stft(pred_1d, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
    y = torch.stft(target_1d, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
    return torch.linalg.norm(y - p) / torch.linalg.norm(y).clamp_min(1e-8)


def codec_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vq_loss: torch.Tensor,
    waveform_weight: float = 1.0,
    spectral_weight: float = 1.0,
    vq_weight: float = 1.0,
    fft_sizes=(256, 512, 1024),
) -> dict[str, torch.Tensor]:
    waveform = waveform_l1(pred, target)
    spectral = multi_resolution_stft_loss(pred, target, fft_sizes=fft_sizes)
    total = waveform_weight * waveform + spectral_weight * spectral + vq_weight * vq_loss
    return {
        "loss": total,
        "waveform_l1": waveform,
        "mrstft": spectral,
        "vq_loss": vq_loss,
    }
