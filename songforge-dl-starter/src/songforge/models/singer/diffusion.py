from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class SingerDenoiser(nn.Module):
    """Small conditional noise predictor for mel diffusion experiments."""

    def __init__(self, mel_bins: int = 80, hidden_dim: int = 256, phoneme_vocab: int = 256, note_vocab: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phoneme_emb = nn.Embedding(phoneme_vocab, hidden_dim)
        self.note_emb = nn.Embedding(note_vocab, hidden_dim)
        self.in_proj = nn.Conv1d(mel_bins, hidden_dim, 1)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden_dim, mel_bins, 1),
        )

    def forward(self, noisy_mel: torch.Tensor, t: torch.Tensor, phonemes: torch.Tensor, notes: torch.Tensor) -> torch.Tensor:
        # mel [B, M, T], token conditions [B, T]
        cond = self.phoneme_emb(phonemes) + self.note_emb(notes)
        cond = cond.transpose(1, 2)
        time = self.time_mlp(sinusoidal_timestep_embedding(t, self.hidden_dim)).unsqueeze(-1)
        x = self.in_proj(noisy_mel) + cond + time
        return self.net(x)
