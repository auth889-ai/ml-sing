from __future__ import annotations

import torch

from songforge.losses.audio import waveform_l1
from songforge.models.codec.model import NeuralCodec
from songforge.training.seed import seed_everything


def main() -> None:
    seed_everything(42)
    model = NeuralCodec(base_channels=8, latent_dim=16, codebook_size=32, num_quantizers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # length divisible by codec downsampling factor=100
    x = torch.randn(2, 1, 4000).clamp(-1, 1)
    for step in range(3):
        out = model(x)
        loss = waveform_l1(out["reconstruction"], x) + out["vq_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
