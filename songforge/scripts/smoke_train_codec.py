from __future__ import annotations

import torch

from songforge.losses.audio import codec_reconstruction_loss
from songforge.models.codec.model import NeuralCodec
from songforge.training.seed import seed_everything


def main() -> None:
    seed_everything(42)
    model = NeuralCodec(base_channels=8, latent_dim=16, codebook_size=32, num_quantizers=2, strides=(2, 4, 5))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 4097).clamp(-1, 1)
    for step in range(3):
        out = model(x)
        losses = codec_reconstruction_loss(out["reconstruction"], x, out["vq_loss"], spectral_weight=0.25)
        loss = losses["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"step={step} loss={loss.item():.4f} frame_rate={model.compression_stats(x.shape[-1])['latent_frame_rate_hz']:.1f}")


if __name__ == "__main__":
    main()
