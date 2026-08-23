from __future__ import annotations

import torch
from torch import nn


class AccompanimentTransformer(nn.Module):
    def __init__(self, vocab_size: int = 2048, d_model: int = 256, nhead: int = 8, num_layers: int = 6, max_seq_len: int = 2048):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True, norm_first=True)
        self.core = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t = x.shape
        p = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.token(x) + self.pos(p)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), 1)
        return self.head(self.core(h, mask=mask))
