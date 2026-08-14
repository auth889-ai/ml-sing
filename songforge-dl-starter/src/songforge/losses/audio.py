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
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        p = torch.stft(pred_1d, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        y = torch.stft(target_1d, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        total = total + F.l1_loss(torch.log1p(p.abs()), torch.log1p(y.abs()))
    return total / len(fft_sizes)
