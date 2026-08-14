from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Simple straight-through VQ layer for smoke/research baselines."""

    def __init__(self, dim: int, codebook_size: int, commitment_weight: float = 0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, D, T]
        if z.ndim != 3 or z.size(1) != self.dim:
            raise ValueError(f"Expected [B,{self.dim},T], got {tuple(z.shape)}")
        flat = z.transpose(1, 2).contiguous().view(-1, self.dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(z.size(0), z.size(2), self.dim).transpose(1, 2)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices.view(z.size(0), z.size(2)), loss


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, num_quantizers: int, commitment_weight: float = 0.25):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size, commitment_weight) for _ in range(num_quantizers)]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices: list[torch.Tensor] = []
        total_loss = z.new_zeros(())
        for layer in self.layers:
            q, idx, loss = layer(residual)
            quantized_sum = quantized_sum + q
            residual = residual - q.detach()
            all_indices.append(idx)
            total_loss = total_loss + loss
        return quantized_sum, all_indices, total_loss
