from __future__ import annotations

import math
import torch
from torch import nn


class SongPlannerTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        if t > self.max_seq_len:
            raise ValueError("Sequence longer than max_seq_len")
        positions = torch.arange(t, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) * math.sqrt(self.token_embedding.embedding_dim)
        x = x + self.position_embedding(positions)
        causal_mask = torch.full((t, t), float("-inf"), device=tokens.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        x = self.transformer(x, mask=causal_mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 50):
        self.eval()
        out = prefix
        for _ in range(max_new_tokens):
            ctx = out[:, -self.max_seq_len :]
            logits = self(ctx)[:, -1] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            out = torch.cat([out, next_token], dim=1)
        return out
