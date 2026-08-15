from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    """Straight-through VQ layer with data-dependent init and dead-code restart.

    A fixed-scale random codebook collapses: if the codebook is initialized far
    from the encoder output distribution, every frame maps to the same entry and
    only that entry ever receives gradient. The codebook is therefore seeded from
    real encoder outputs on the first training batch, and entries that stop being
    selected are re-seeded from current encoder outputs.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        usage_decay: float = 0.95,
        dead_code_fraction: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.usage_decay = usage_decay
        self.dead_code_fraction = dead_code_fraction
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
        # Buffers live in state_dict so resume keeps the learned codebook state.
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("cluster_usage", torch.full((codebook_size,), 1.0 / codebook_size))

    @torch.no_grad()
    def _sample_vectors(self, flat: torch.Tensor, count: int) -> torch.Tensor:
        total = flat.size(0)
        if total >= count:
            indices = torch.randperm(total, device=flat.device)[:count]
            return flat[indices]
        indices = torch.randint(0, total, (count,), device=flat.device)
        chosen = flat[indices]
        return chosen + 0.01 * torch.randn_like(chosen)

    @torch.no_grad()
    def _data_init(self, flat: torch.Tensor) -> None:
        chosen = self._sample_vectors(flat, self.codebook_size)
        self.codebook.weight.data.copy_(chosen.to(self.codebook.weight.dtype))
        self.cluster_usage.fill_(1.0 / self.codebook_size)
        self.initted.fill_(True)

    @torch.no_grad()
    def _update_usage_and_restart(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        histogram = torch.bincount(indices, minlength=self.codebook_size).to(self.cluster_usage.dtype)
        probs = histogram / histogram.sum().clamp_min(1.0)
        self.cluster_usage.mul_(self.usage_decay).add_(probs, alpha=1.0 - self.usage_decay)
        dead = self.cluster_usage < (self.dead_code_fraction / self.codebook_size)
        dead_count = int(dead.sum().item())
        if dead_count == 0:
            return
        replacements = self._sample_vectors(flat, dead_count)
        self.codebook.weight.data[dead] = replacements.to(self.codebook.weight.dtype)
        self.cluster_usage[dead] = 1.0 / self.codebook_size

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # z: [B, D, T]
        if z.ndim != 3 or z.size(1) != self.dim:
            raise ValueError(f"Expected [B,{self.dim},T], got {tuple(z.shape)}")
        flat = z.transpose(1, 2).contiguous().view(-1, self.dim)
        if self.training and not bool(self.initted):
            self._data_init(flat.detach().float())
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
        # Restart after the lookup so this batch stays consistent with `indices`.
        if self.training:
            self._update_usage_and_restart(flat.detach().float(), indices.detach())
        return {
            "quantized": quantized_st,
            "codes": indices.view(z.size(0), z.size(2)),
            "loss": loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2:
            raise ValueError(f"Expected [B,T] codes, got {tuple(codes.shape)}")
        return self.codebook(codes).transpose(1, 2)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, num_quantizers: int, commitment_weight: float = 0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size, commitment_weight) for _ in range(num_quantizers)]
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_codes: list[torch.Tensor] = []
        total_loss = z.new_zeros(())
        total_codebook_loss = z.new_zeros(())
        total_commitment_loss = z.new_zeros(())
        for layer in self.layers:
            out = layer(residual)
            q = out["quantized"]
            quantized_sum = quantized_sum + q
            residual = residual - q.detach()
            all_codes.append(out["codes"])
            total_loss = total_loss + out["loss"]
            total_codebook_loss = total_codebook_loss + out["codebook_loss"]
            total_commitment_loss = total_commitment_loss + out["commitment_loss"]
        codes = torch.stack(all_codes, dim=1)
        return {
            "quantized": quantized_sum,
            "codes": codes,
            "loss": total_loss,
            "codebook_loss": total_codebook_loss,
            "commitment_loss": total_commitment_loss,
        }

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(f"Expected [B,Q,T] codes, got {tuple(codes.shape)}")
        if codes.size(1) > len(self.layers):
            raise ValueError(f"Got {codes.size(1)} codebooks, model has {len(self.layers)}")
        quantized = None
        for i in range(codes.size(1)):
            decoded = self.layers[i].decode_codes(codes[:, i])
            quantized = decoded if quantized is None else quantized + decoded
        if quantized is None:
            raise ValueError("At least one quantizer code stream is required")
        return quantized

    @torch.no_grad()
    def codebook_usage(self, codes: torch.Tensor) -> dict[str, float]:
        if codes.ndim != 3:
            raise ValueError(f"Expected [B,Q,T] codes, got {tuple(codes.shape)}")
        unique_counts = []
        perplexities = []
        entropies = []
        dead_counts = []
        utilizations = []
        for i in range(codes.size(1)):
            hist = torch.bincount(codes[:, i].reshape(-1), minlength=self.codebook_size).float()
            probs = hist / hist.sum().clamp_min(1.0)
            entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
            unique = float((hist > 0).sum().item())
            unique_counts.append(unique)
            perplexities.append(float(entropy.exp().item()))
            entropies.append(float(entropy.item()))
            dead_counts.append(float((hist == 0).sum().item()))
            utilizations.append(unique / float(self.codebook_size))
        # Per-codebook detail as well as aggregates: an RVQ can look healthy on
        # average while its deepest quantizer is dead, and acceptance needs to
        # see that. Aggregate key names are unchanged.
        per_codebook = [
            {
                "index": index,
                "unique": unique_counts[index],
                "utilization": utilizations[index],
                "dead_codes": dead_counts[index],
                "entropy": entropies[index],
                "perplexity": perplexities[index],
                "collapsed": bool(utilizations[index] < 0.05 or perplexities[index] < 2.0),
            }
            for index in range(len(unique_counts))
        ]
        return {
            "codebook_unique_avg": sum(unique_counts) / max(len(unique_counts), 1),
            "codebook_perplexity_avg": sum(perplexities) / max(len(perplexities), 1),
            "codebook_entropy_avg": sum(entropies) / max(len(entropies), 1),
            "codebook_dead_codes_avg": sum(dead_counts) / max(len(dead_counts), 1),
            "codebook_utilization_avg": sum(utilizations) / max(len(utilizations), 1),
            "codebook_unique_min": min(unique_counts) if unique_counts else 0.0,
            "codebook_perplexity_min": min(perplexities) if perplexities else 0.0,
            "codebook_utilization_min": min(utilizations) if utilizations else 0.0,
            "codebook_entropy_min": min(entropies) if entropies else 0.0,
            "codebook_dead_codes_max": max(dead_counts) if dead_counts else 0.0,
            "codebook_count": float(len(unique_counts)),
            "codebook_size": float(self.codebook_size),
            "per_codebook": per_codebook,
            "rvq_collapse_suspected": float(any(u < 0.05 or p < 2.0 for u, p in zip(utilizations, perplexities))),
        }
