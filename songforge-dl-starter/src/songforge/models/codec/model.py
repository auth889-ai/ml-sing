from __future__ import annotations

import torch
from torch import nn

from .quantizer import ResidualVectorQuantizer


class NeuralCodec(nn.Module):
    """Small V1 codec. Codex should upgrade blocks/losses after smoke PASS."""

    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 64,
        codebook_size: int = 256,
        num_quantizers: int = 4,
    ):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c, 7, padding=3), nn.GELU(),
            nn.Conv1d(c, c * 2, 8, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(c * 2, c * 4, 10, stride=5, padding=3), nn.GELU(),
            nn.Conv1d(c * 4, latent_dim, 10, stride=5, padding=3),
        )
        self.quantizer = ResidualVectorQuantizer(latent_dim, codebook_size, num_quantizers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, c * 4, 10, stride=5, padding=3, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(c * 4, c * 2, 10, stride=5, padding=3, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(c * 2, c, 8, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(c, 1, 7, padding=3), nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.quantizer(z)

    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        return self.decoder(zq)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        zq, indices, vq_loss = self.encode(x)
        reconstruction = self.decode(zq)
        # Decoder strides are chosen to preserve lengths divisible by 100.
        reconstruction = reconstruction[..., : x.shape[-1]]
        return {"reconstruction": reconstruction, "indices": indices, "vq_loss": vq_loss}
