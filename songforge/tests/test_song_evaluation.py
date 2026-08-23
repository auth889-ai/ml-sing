"""Objective song measures must detect the failure modes they claim to detect.

Each test builds a signal with a known defect and asserts the measure finds it.
These checks prove a file is broken; they never prove it is good.
"""

import numpy as np
import pytest
import soundfile as sf

from songforge.evaluation.song import analyze_song, objective_flags

SR = 22050


def write(tmp_path, audio, name="x.wav", sample_rate=SR):
    path = tmp_path / name
    sf.write(str(path), audio, sample_rate)
    return path


def tone(seconds=4.0, freq=440.0, amp=0.3, sample_rate=SR):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_clipping_is_detected(tmp_path):
    report = analyze_song(write(tmp_path, np.clip(tone(amp=2.0), -1, 1)))
    assert report["clipped_sample_ratio"] > 1e-4
    assert "clipping" in report["flags"]


def test_clean_signal_is_not_flagged_as_clipped(tmp_path):
    report = analyze_song(write(tmp_path, tone(amp=0.3)))
    assert "clipping" not in report["flags"]


def test_mostly_silent_output_is_detected(tmp_path):
    audio = np.concatenate([tone(seconds=1.0), np.zeros(int(SR * 5), dtype=np.float32)])
    report = analyze_song(write(tmp_path, audio))
    assert report["silent_ratio"] > 0.35
    assert "mostly silent" in report["flags"]
    assert report["longest_silence_seconds"] > 4.0


def test_band_limited_output_is_detected(tmp_path):
    """A muffled 200 Hz-only render should not pass as full-band audio."""
    report = analyze_song(write(tmp_path, tone(freq=200.0)))
    assert report["spectral_rolloff95_hz"] < 6000
    assert "band-limited / muffled" in report["flags"]


def test_noise_is_flagged_as_noise_like(tmp_path):
    rng = np.random.default_rng(0)
    noise = (0.2 * rng.standard_normal(int(SR * 4))).astype(np.float32)
    report = analyze_song(write(tmp_path, noise))
    assert report["spectral_flatness"] > 0.35
    assert "noise-like" in report["flags"]


def test_dual_mono_is_distinguished_from_real_stereo(tmp_path):
    mono = tone()
    dual = np.stack([mono, mono], axis=1)
    assert analyze_song(write(tmp_path, dual, "dual.wav"))["effectively_mono"]

    rng = np.random.default_rng(1)
    wide = np.stack([mono, mono * 0.4 + 0.2 * rng.standard_normal(mono.size)], axis=1).astype(np.float32)
    assert not analyze_song(write(tmp_path, wide, "wide.wav"))["effectively_mono"]


def test_looped_material_scores_more_repetitive_than_developing_material(tmp_path):
    loop = np.tile(tone(seconds=1.0), 8)
    developing = np.concatenate([tone(seconds=2.0, freq=f) for f in (220, 440, 660, 880)])
    looped = analyze_song(write(tmp_path, loop, "loop.wav"))
    varied = analyze_song(write(tmp_path, developing, "varied.wav"))
    assert looped["repetition_score"] > varied["repetition_score"]
    assert varied["section_novelty"] > looped["section_novelty"]


def test_duration_is_reported_accurately(tmp_path):
    report = analyze_song(write(tmp_path, tone(seconds=3.0)))
    assert report["duration_seconds"] == pytest.approx(3.0, abs=0.02)


def harmonic(seconds=1.5, root=180.0, amp=0.22, sample_rate=SR):
    """A tonal note with a full harmonic series, plus light percussive ticks.

    Real music is tonal (low spectral flatness) but still carries energy well
    above 6 kHz. A handful of sines has neither property, which is why the
    earlier fixture tripped the muffled and noise-like checks correctly.
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    partials = [n for n in range(1, 46) if root * n < sample_rate / 2]
    signal = sum((amp / n) * np.sin(2 * np.pi * root * n * t) for n in partials)
    signal = signal * np.exp(-1.5 * (t % 0.5))
    # brief broadband ticks: high-frequency energy without a noise-like average
    rng = np.random.default_rng(int(root))
    for onset in range(0, len(t), int(0.25 * sample_rate)):
        width = int(0.01 * sample_rate)
        tick = 0.25 * rng.standard_normal(width) * np.exp(-np.linspace(0, 6, width))
        signal[onset : onset + width] += tick[: max(0, min(width, len(signal) - onset))]
    return signal.astype(np.float32)


def test_flags_are_empty_for_a_plausible_signal(tmp_path):
    """Tonal, moving, stereo, well-levelled audio should raise nothing."""
    mono = np.concatenate([harmonic(root=f) for f in (180, 240, 320, 200)])
    stereo = np.stack([mono, np.roll(mono, 128) * 0.85], axis=1).astype(np.float32)
    report = analyze_song(write(tmp_path, stereo, "ok.wav"))
    assert report["flags"] == [], report["flags"]


def test_a_missing_measure_never_invents_a_defect():
    """Partial reports must stay silent rather than flag what was not measured."""
    assert objective_flags({"clipped_sample_ratio": 0.5}) == ["clipping"]
    assert objective_flags({}) == []
