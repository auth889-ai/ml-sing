import math

import torch

from songforge.evaluation.audio import (
    AUDIO_CHARACTERS,
    audio_character_features,
    character_scores,
    classify_audio_character,
    reconstruction_metrics,
    select_character_examples,
)

SAMPLE_RATE = 24000


def bass(seconds: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(seconds * SAMPLE_RATE), dtype=torch.float32) / SAMPLE_RATE
    return (0.5 * torch.sin(2 * math.pi * 55 * t)).unsqueeze(0)


def harmonic(seconds: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(seconds * SAMPLE_RATE), dtype=torch.float32) / SAMPLE_RATE
    tone = sum(torch.sin(2 * math.pi * 440 * k * t) / k for k in range(1, 6))
    return (0.3 * tone).unsqueeze(0)


def percussive(seconds: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    t = torch.arange(int(seconds * SAMPLE_RATE), dtype=torch.float32) / SAMPLE_RATE
    envelope = torch.sin(2 * math.pi * 8 * t).clamp_min(0) ** 8
    noise = torch.randn(t.numel(), generator=generator)
    return (0.5 * noise * envelope).unsqueeze(0)


def mixed(seconds: float = 1.0) -> torch.Tensor:
    return (0.3 * bass(seconds) + 0.3 * harmonic(seconds) + 0.3 * percussive(seconds)).clamp(-1, 1)


def test_features_are_bounded():
    """Unbounded features would swamp the others when scoring."""
    for signal in (bass(), harmonic(), percussive(), mixed()):
        features = audio_character_features(signal, SAMPLE_RATE)
        for name in ("low_band_ratio", "high_band_ratio", "spectral_flatness", "spectral_flux"):
            assert 0.0 <= features[name] <= 1.0, f"{name}={features[name]}"


def test_features_are_deterministic():
    signal = mixed()
    assert audio_character_features(signal, SAMPLE_RATE) == audio_character_features(signal, SAMPLE_RATE)


def test_short_signal_does_not_crash():
    features = audio_character_features(torch.zeros(1, 16), SAMPLE_RATE)
    assert features["spectral_flux"] == 0.0


def test_bass_is_low_band_dominated():
    assert audio_character_features(bass(), SAMPLE_RATE)["low_band_ratio"] > 0.8


def test_percussion_is_noisy_and_flat():
    features = audio_character_features(percussive(), SAMPLE_RATE)
    tonal = audio_character_features(harmonic(), SAMPLE_RATE)
    assert features["spectral_flatness"] > tonal["spectral_flatness"]
    assert features["spectral_flux"] > tonal["spectral_flux"]


def test_bass_does_not_win_the_harmonic_bucket():
    """A solo bass line is maximally tonal; it must not outrank real harmonic content."""
    bass_scores = character_scores(audio_character_features(bass(), SAMPLE_RATE))
    harmonic_scores = character_scores(audio_character_features(harmonic(), SAMPLE_RATE))
    assert harmonic_scores["harmonic"] > bass_scores["harmonic"]
    assert bass_scores["bass_heavy"] > harmonic_scores["bass_heavy"]


def test_classification_matches_signal_type():
    assert classify_audio_character(audio_character_features(bass(), SAMPLE_RATE)) == "bass_heavy"
    assert classify_audio_character(audio_character_features(percussive(), SAMPLE_RATE)) == "percussive"


def test_selection_picks_one_distinct_segment_per_character():
    signals = [bass(), harmonic(), percussive(), mixed()]
    candidates = [
        {"index": index, "features": audio_character_features(signal, SAMPLE_RATE)}
        for index, signal in enumerate(signals)
    ]
    chosen = select_character_examples(candidates)

    assert set(chosen) == set(AUDIO_CHARACTERS)
    assert len({entry["index"] for entry in chosen.values()}) == 4
    assert chosen["bass_heavy"]["index"] == 0
    assert chosen["harmonic"]["index"] == 1
    assert chosen["percussive"]["index"] == 2
    assert chosen["mixed"]["index"] == 3


def test_selection_never_reuses_a_segment():
    candidates = [{"index": 0, "features": audio_character_features(mixed(), SAMPLE_RATE)}]
    chosen = select_character_examples(candidates)
    assert len(chosen) == 1


def test_selection_handles_empty_candidates():
    assert select_character_examples([]) == {}


def test_reconstruction_metrics_are_finite():
    target = mixed()
    metrics = reconstruction_metrics(target * 0.9, target)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["waveform_l1"] > 0
