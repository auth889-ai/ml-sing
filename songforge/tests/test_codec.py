import math
import wave

import pytest
import torch

from songforge.data.audio import write_wav
from songforge.evaluation.audio import reconstruction_metrics
from songforge.losses.audio import codec_reconstruction_loss
from songforge.models.codec.model import NeuralCodec
from songforge.training.checkpoint import load_checkpoint, save_checkpoint


def tiny_codec() -> NeuralCodec:
    return NeuralCodec(
        sample_rate=24000,
        channels=1,
        base_channels=8,
        latent_dim=16,
        codebook_size=32,
        num_quantizers=2,
        strides=(2, 4, 5),
    )


@pytest.mark.parametrize("length", [3999, 4000, 4097])
def test_codec_shape_arbitrary_supported_lengths(length):
    model = tiny_codec()
    x = torch.randn(2, 1, length)
    out = model(x)
    assert out["reconstruction"].shape == x.shape
    assert out["codes"].shape[0] == x.shape[0]
    assert out["codes"].shape[1] == model.num_quantizers
    assert out["codes"].shape[2] == math.ceil(length / model.downsample_factor)


def test_codec_backward_and_finite_losses():
    model = tiny_codec()
    x = torch.randn(2, 1, 4097).clamp(-1, 1)
    out = model(x)
    losses = codec_reconstruction_loss(out["reconstruction"], x, out["vq_loss"], spectral_weight=0.25)
    assert all(torch.isfinite(v).item() for v in losses.values())
    losses["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_codec_checkpoint_round_trip(tmp_path):
    model = tiny_codec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(1, 1, 2048)
    out = model(x)
    loss = (out["reconstruction"] - x).abs().mean() + out["vq_loss"]
    loss.backward()
    optimizer.step()
    path = tmp_path / "codec.pt"
    save_checkpoint(path, model, optimizer, step=3, config={"model": "tiny"}, metrics={"loss": float(loss.item())})

    restored = tiny_codec()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(path, restored, restored_optimizer)
    assert checkpoint["step"] == 3
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(left, right)


def test_codec_eval_is_deterministic():
    model = tiny_codec().eval()
    x = torch.randn(1, 1, 2048)
    with torch.no_grad():
        first = model(x)["reconstruction"]
        second = model(x)["reconstruction"]
    assert torch.equal(first, second)


def test_codec_metrics_and_wav_write(tmp_path):
    model = tiny_codec()
    x = torch.randn(1, 1, 2400).clamp(-1, 1)
    out = model(x)
    metrics = reconstruction_metrics(out["reconstruction"], x)
    stats = model.compression_stats(x.shape[-1])
    assert metrics["waveform_l1"] >= 0
    assert stats["latent_frame_rate_hz"] > 0
    assert stats["pcm16_compression_ratio"] > 1

    wav_path = tmp_path / "reconstructed.wav"
    write_wav(wav_path, out["reconstruction"], model.sample_rate)
    with wave.open(str(wav_path), "rb") as f:
        assert f.getframerate() == model.sample_rate
        assert f.getnchannels() == 1
        assert f.getnframes() == x.shape[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_codec_cuda_smoke():
    model = tiny_codec().cuda()
    x = torch.randn(1, 1, 2048, device="cuda")
    out = model(x)
    loss = (out["reconstruction"] - x).abs().mean() + out["vq_loss"]
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_codec_cuda_amp_smoke():
    model = tiny_codec().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(1, 1, 4097, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda"):
        out = model(x)
        loss = (out["reconstruction"] - x).abs().mean() + out["vq_loss"]
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss)
