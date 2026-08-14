import torch
from songforge.models.singer.diffusion import SingerDenoiser


def test_singer_denoiser_shape_and_backward():
    model = SingerDenoiser(mel_bins=16, hidden_dim=32, phoneme_vocab=32, note_vocab=128)
    mel = torch.randn(2, 16, 20)
    t = torch.randint(0, 1000, (2,))
    ph = torch.randint(0, 32, (2, 20))
    notes = torch.randint(0, 128, (2, 20))
    pred = model(mel, t, ph, notes)
    assert pred.shape == mel.shape
    pred.mean().backward()
