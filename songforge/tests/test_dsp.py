import math

import pytest
import torch

from songforge.data.dsp import (
    compute_stats,
    extract_segment,
    is_clipped,
    is_silent,
    normalize_amplitude,
    resample_waveform,
    segment_bounds,
    to_channels,
)


def sine(seconds: float, sample_rate: int, frequency: float = 1000.0, amplitude: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(seconds * sample_rate), dtype=torch.float32) / sample_rate
    return (amplitude * torch.sin(2 * math.pi * frequency * t)).unsqueeze(0)


def dominant_frequency(waveform: torch.Tensor, sample_rate: int) -> float:
    signal = waveform[0] * torch.hann_window(waveform.size(-1))
    spectrum = torch.fft.rfft(signal).abs()
    return float(spectrum.argmax().item()) * sample_rate / waveform.size(-1)


# --- resampling ---------------------------------------------------------


@pytest.mark.parametrize(
    "orig_rate,target_rate",
    [(48000, 24000), (16000, 24000), (44100, 24000), (22050, 24000), (24000, 24000)],
)
def test_resample_preserves_length_and_tone(orig_rate, target_rate):
    waveform = sine(1.0, orig_rate, frequency=1000.0)
    resampled = resample_waveform(waveform, orig_rate, target_rate)

    expected_length = math.ceil(target_rate * waveform.size(-1) / orig_rate)
    assert resampled.size(-1) == expected_length
    assert abs(dominant_frequency(resampled, target_rate) - 1000.0) < 15.0


def test_resample_preserves_energy():
    waveform = sine(1.0, 48000, frequency=440.0, amplitude=0.5)
    resampled = resample_waveform(waveform, 48000, 24000)
    original_rms = float(waveform.pow(2).mean().sqrt())
    resampled_rms = float(resampled.pow(2).mean().sqrt())
    assert abs(original_rms - resampled_rms) < 0.01


def test_resample_round_trip_is_close_to_original():
    waveform = sine(0.5, 24000, frequency=440.0, amplitude=0.5)
    round_tripped = resample_waveform(resample_waveform(waveform, 24000, 48000), 48000, 24000)
    usable = min(round_tripped.size(-1), waveform.size(-1))
    error = (round_tripped[..., 200 : usable - 200] - waveform[..., 200 : usable - 200]).abs().max()
    assert float(error) < 0.01


def test_resample_is_deterministic():
    waveform = sine(0.3, 16000, frequency=300.0)
    first = resample_waveform(waveform, 16000, 24000)
    second = resample_waveform(waveform, 16000, 24000)
    assert torch.equal(first, second)


def test_resample_rejects_bad_rates():
    with pytest.raises(ValueError):
        resample_waveform(sine(0.1, 8000), 0, 24000)


# --- segmentation -------------------------------------------------------


def test_segment_bounds_non_overlapping():
    assert segment_bounds(10, 4, 4) == [(0, 4), (4, 8)]


def test_segment_bounds_overlapping_hop():
    assert segment_bounds(10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_segment_bounds_drops_short_tail_by_default():
    assert segment_bounds(7, 4, 4) == [(0, 4)]


def test_segment_bounds_can_pad_final_partial():
    assert segment_bounds(7, 4, 4, pad_final_partial=True) == [(0, 4), (4, 8)]


def test_segment_bounds_shorter_than_one_segment():
    assert segment_bounds(3, 4, 4) == []
    assert segment_bounds(3, 4, 4, pad_final_partial=True) == [(0, 4)]


def test_segment_bounds_is_deterministic():
    assert segment_bounds(100, 7, 3) == segment_bounds(100, 7, 3)


def test_extract_segment_zero_pads_past_end():
    waveform = torch.ones(1, 5)
    segment = extract_segment(waveform, 3, 9)
    assert segment.shape == (1, 6)
    assert torch.equal(segment[0, 2:], torch.zeros(4))


# --- channels -----------------------------------------------------------


def test_to_channels_downmixes_to_mono_by_mean():
    stereo = torch.stack([torch.ones(4), torch.zeros(4)])
    assert torch.allclose(to_channels(stereo, 1, "mean")[0], torch.full((4,), 0.5))


def test_to_channels_first_policy_keeps_channel_zero():
    stereo = torch.stack([torch.ones(4), torch.zeros(4)])
    assert torch.equal(to_channels(stereo, 1, "first")[0], torch.ones(4))


def test_to_channels_upmixes_mono():
    assert to_channels(torch.ones(1, 4), 2).shape == (2, 4)


def test_to_channels_rejects_bad_shape():
    with pytest.raises(ValueError):
        to_channels(torch.ones(4), 1)


# --- silence, clipping, loudness ---------------------------------------


def test_silence_is_detected():
    assert is_silent(torch.zeros(1, 1000))
    assert not is_silent(sine(0.1, 8000, amplitude=0.5))


def test_near_silence_is_detected_against_threshold():
    quiet = sine(0.1, 8000, amplitude=1e-4)
    assert is_silent(quiet, threshold_dbfs=-60.0)
    assert not is_silent(quiet, threshold_dbfs=-100.0)


def test_clipping_is_detected():
    clipped = (6.0 * sine(0.1, 8000)).clamp(-1.0, 1.0)
    assert is_clipped(clipped)
    assert compute_stats(clipped).clipping_ratio > 0.5


def test_clean_signal_is_not_clipped():
    assert not is_clipped(sine(0.1, 8000, amplitude=0.5))


def test_compute_stats_reports_peak_and_rms():
    stats = compute_stats(sine(0.5, 8000, amplitude=0.5))
    assert abs(stats.peak - 0.5) < 0.01
    assert abs(stats.rms - 0.5 / math.sqrt(2)) < 0.01
    assert stats.peak_dbfs < 0.0


def test_normalize_peak_reaches_target():
    quiet = sine(0.2, 8000, amplitude=0.05)
    normalized, gain_db = normalize_amplitude(quiet, "peak", target_dbfs=-1.0)
    assert gain_db > 0
    assert abs(compute_stats(normalized).peak_dbfs - (-1.0)) < 0.1


def test_normalize_none_is_a_passthrough():
    waveform = sine(0.2, 8000, amplitude=0.3)
    normalized, gain_db = normalize_amplitude(waveform, "none")
    assert gain_db == 0.0
    assert torch.equal(normalized, waveform)


def test_normalize_does_not_amplify_silence_beyond_max_gain():
    normalized, gain_db = normalize_amplitude(torch.zeros(1, 100), "peak", target_dbfs=-1.0)
    assert gain_db == 0.0
    assert float(normalized.abs().max()) == 0.0
