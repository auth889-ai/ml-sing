import torch
from songforge.models.codec.model import NeuralCodec


def test_codec_shape_and_backward():
    model = NeuralCodec(base_channels=8, latent_dim=16, codebook_size=32, num_quantizers=2)
    x = torch.randn(2, 1, 4000)
    out = model(x)
    assert out["reconstruction"].shape == x.shape
    loss = (out["reconstruction"] - x).abs().mean() + out["vq_loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
