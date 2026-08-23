import pytest
import torch

from songforge.data.fixtures import (
    write_clipped_wav,
    write_corrupt_wav,
    write_empty_file,
    write_silent_wav,
    write_tone_wav,
)
from songforge.data.media import (
    AudioValidationError,
    decode_audio,
    probe_media,
    validate_audio_file,
)


def test_probe_reports_stream_layout(tmp_path):
    path = write_tone_wav(tmp_path / "tone.wav", seconds=1.0, sample_rate=16000)
    info = probe_media(path)

    assert info.exists and info.decodable
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.duration_seconds == pytest.approx(1.0, abs=0.05)
    assert info.prober in ("ffprobe", "soundfile", "wave")


def test_probe_missing_file_does_not_raise(tmp_path):
    info = probe_media(tmp_path / "absent.wav")
    assert not info.exists
    assert info.error


def test_decode_returns_channels_first_float(tmp_path):
    path = write_tone_wav(tmp_path / "tone.wav", seconds=0.5, sample_rate=16000, channels=2)
    audio, sample_rate = decode_audio(path)

    assert sample_rate == 16000
    assert audio.ndim == 2 and audio.size(0) == 2
    assert audio.dtype == torch.float32
    assert audio.size(-1) == pytest.approx(8000, abs=2)
    assert float(audio.abs().max()) <= 1.0


def test_decode_is_deterministic(tmp_path):
    path = write_tone_wav(tmp_path / "tone.wav", seconds=0.3, sample_rate=16000)
    first, _ = decode_audio(path)
    second, _ = decode_audio(path)
    assert torch.equal(first, second)


# --- corrupt / unusable input ------------------------------------------


def test_corrupt_file_is_rejected(tmp_path):
    path = write_corrupt_wav(tmp_path / "corrupt.wav")
    report = validate_audio_file(path)

    assert not report.ok
    assert report.reasons


def test_corrupt_file_decode_raises_typed_error(tmp_path):
    path = write_corrupt_wav(tmp_path / "corrupt.wav")
    with pytest.raises(AudioValidationError):
        decode_audio(path)


def test_empty_file_is_rejected(tmp_path):
    report = validate_audio_file(write_empty_file(tmp_path / "empty.wav"))
    assert not report.ok
    assert any("empty" in reason for reason in report.reasons)


def test_missing_file_is_rejected(tmp_path):
    report = validate_audio_file(tmp_path / "absent.wav")
    assert not report.ok


def test_unsupported_suffix_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio", encoding="utf-8")
    report = validate_audio_file(path)
    assert not report.ok
    assert any("suffix" in reason for reason in report.reasons)


def test_too_short_file_is_rejected(tmp_path):
    path = write_tone_wav(tmp_path / "blip.wav", seconds=0.05, sample_rate=16000)
    report = validate_audio_file(path, min_duration_seconds=1.0)
    assert not report.ok
    assert any("below minimum" in reason for reason in report.reasons)


def test_valid_file_passes_with_decode_check(tmp_path):
    path = write_tone_wav(tmp_path / "tone.wav", seconds=2.0, sample_rate=16000)
    report = validate_audio_file(path, min_duration_seconds=1.0, require_decode=True)

    assert report.ok
    assert report.reasons == []
    assert report.info.decodable
    assert report.to_dict()["ok"] is True


def test_silent_and_clipped_files_still_decode(tmp_path):
    """Silence and clipping are content problems, not media corruption."""
    silent = validate_audio_file(write_silent_wav(tmp_path / "silent.wav", seconds=1.0, sample_rate=16000))
    clipped = validate_audio_file(write_clipped_wav(tmp_path / "clipped.wav", seconds=1.0, sample_rate=16000))
    assert silent.ok
    assert clipped.ok
