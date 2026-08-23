"""M02 media probing and validation.

Decoding and probing degrade across backends so the same code runs on a Colab
runtime (soundfile/torchaudio present) and on a bare workstation (ffmpeg only,
or nothing but the standard library).

Probe order : ffprobe -> soundfile -> stdlib wave
Decode order: soundfile -> torchaudio -> ffmpeg -> stdlib wave
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif")


class AudioValidationError(RuntimeError):
    """Raised when an audio file cannot be used for training."""


@dataclass(frozen=True)
class MediaInfo:
    """What a prober could learn about a media file."""

    path: str
    exists: bool = False
    decodable: bool = False
    prober: str = "none"
    codec_name: str | None = None
    format_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    bit_rate: int | None = None
    size_bytes: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Per-file verdict produced before any preprocessing work happens."""

    path: str
    ok: bool
    info: MediaInfo
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "ok": self.ok, "reasons": list(self.reasons), "info": self.info.to_dict()}


@lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _probe_with_ffprobe(path: Path) -> MediaInfo:
    binary = ffprobe_path()
    if binary is None:
        raise RuntimeError("ffprobe not installed")
    result = subprocess.run(
        [
            binary,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate:format=format_name,duration,size",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("no audio stream found")
    stream = streams[0]
    fmt = payload.get("format") or {}

    def _int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return MediaInfo(
        path=str(path),
        exists=True,
        decodable=True,
        prober="ffprobe",
        codec_name=stream.get("codec_name"),
        format_name=fmt.get("format_name"),
        sample_rate=_int(stream.get("sample_rate")),
        channels=_int(stream.get("channels")),
        duration_seconds=_float(fmt.get("duration")),
        bit_rate=_int(stream.get("bit_rate")) or _int(fmt.get("bit_rate")),
        size_bytes=_int(fmt.get("size")) or path.stat().st_size,
    )


def _probe_with_soundfile(path: Path) -> MediaInfo:
    import soundfile as sf

    with sf.SoundFile(str(path)) as handle:
        frames = len(handle)
        sample_rate = int(handle.samplerate)
        return MediaInfo(
            path=str(path),
            exists=True,
            decodable=True,
            prober="soundfile",
            codec_name=handle.subtype,
            format_name=handle.format,
            sample_rate=sample_rate,
            channels=int(handle.channels),
            duration_seconds=frames / sample_rate if sample_rate else None,
            size_bytes=path.stat().st_size,
        )


def _probe_with_wave(path: Path) -> MediaInfo:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        return MediaInfo(
            path=str(path),
            exists=True,
            decodable=True,
            prober="wave",
            codec_name=f"pcm_s{handle.getsampwidth() * 8}le",
            format_name="wav",
            sample_rate=int(sample_rate),
            channels=int(handle.getnchannels()),
            duration_seconds=frames / sample_rate if sample_rate else None,
            size_bytes=path.stat().st_size,
        )


def probe_media(path: str | Path) -> MediaInfo:
    """Describe a media file without fully decoding it. Never raises."""
    path = Path(path)
    if not path.exists():
        return MediaInfo(path=str(path), exists=False, error="file does not exist")
    if path.stat().st_size == 0:
        return MediaInfo(path=str(path), exists=True, size_bytes=0, error="file is empty")

    errors: list[str] = []
    for probe in (_probe_with_ffprobe, _probe_with_soundfile, _probe_with_wave):
        try:
            return probe(path)
        except Exception as exc:  # noqa: BLE001 - any backend failure falls through
            errors.append(f"{probe.__name__}: {exc}")
    return MediaInfo(
        path=str(path),
        exists=True,
        size_bytes=path.stat().st_size,
        error="; ".join(errors) or "no usable prober",
    )


def _decode_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return torch.from_numpy(audio).t().contiguous(), int(sample_rate)


def _decode_with_torchaudio(path: Path) -> tuple[torch.Tensor, int]:
    import torchaudio

    audio, sample_rate = torchaudio.load(str(path))
    return audio.float(), int(sample_rate)


def _decode_with_ffmpeg(path: Path) -> tuple[torch.Tensor, int]:
    binary = ffmpeg_path()
    if binary is None:
        raise RuntimeError("ffmpeg not installed")
    info = probe_media(path)
    sample_rate = info.sample_rate
    channels = info.channels
    if not sample_rate or not channels:
        raise RuntimeError("ffprobe could not determine stream layout")
    result = subprocess.run(
        [binary, "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError((result.stderr or b"").decode("utf-8", "replace").strip() or "ffmpeg decode failed")
    pcm = np.frombuffer(result.stdout, dtype="<f4")
    usable = (pcm.size // channels) * channels
    audio = pcm[:usable].reshape(-1, channels).T.copy()
    return torch.from_numpy(audio), int(sample_rate)


def _decode_with_wave(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise RuntimeError("stdlib WAV fallback supports 16-bit PCM only")
    pcm = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    usable = (pcm.size // channels) * channels
    audio = pcm[:usable].reshape(-1, channels).T.copy()
    return torch.from_numpy(audio), int(sample_rate)


def decode_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode to float32 ``[channels, samples]`` at the file's native rate.

    Raises AudioValidationError when every backend fails, so corrupt files are a
    typed failure rather than a surprise exception deep inside preprocessing.
    """
    path = Path(path)
    if not path.exists():
        raise AudioValidationError(f"{path}: file does not exist")

    errors: list[str] = []
    for decoder in (_decode_with_soundfile, _decode_with_torchaudio, _decode_with_ffmpeg, _decode_with_wave):
        try:
            audio, sample_rate = decoder(path)
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            errors.append(f"{decoder.__name__}: {exc}")
            continue
        if audio.numel() == 0:
            errors.append(f"{decoder.__name__}: decoded zero samples")
            continue
        if not torch.isfinite(audio).all():
            raise AudioValidationError(f"{path}: decoded audio contains NaN or Inf")
        return audio.float(), sample_rate
    raise AudioValidationError(f"{path}: could not decode. Tried: " + "; ".join(errors))


def validate_audio_file(
    path: str | Path,
    min_duration_seconds: float = 0.0,
    max_duration_seconds: float | None = None,
    allowed_suffixes: tuple[str, ...] = AUDIO_SUFFIXES,
    require_decode: bool = True,
) -> ValidationReport:
    """Check one file. Returns a report instead of raising so callers can skip and log."""
    path = Path(path)
    reasons: list[str] = []
    info = probe_media(path)

    if not info.exists:
        return ValidationReport(str(path), False, info, ["file does not exist"])
    if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
        reasons.append(f"unsupported suffix {path.suffix!r}")
    if info.error:
        reasons.append(f"probe failed: {info.error}")
    if info.size_bytes == 0:
        reasons.append("file is empty")

    duration = info.duration_seconds
    if duration is not None:
        if duration < min_duration_seconds:
            reasons.append(f"duration {duration:.3f}s below minimum {min_duration_seconds:.3f}s")
        if max_duration_seconds is not None and duration > max_duration_seconds:
            reasons.append(f"duration {duration:.3f}s above maximum {max_duration_seconds:.3f}s")

    if require_decode and not reasons:
        try:
            audio, sample_rate = decode_audio(path)
        except AudioValidationError as exc:
            reasons.append(str(exc))
        else:
            decoded_duration = audio.shape[-1] / sample_rate if sample_rate else 0.0
            if decoded_duration < min_duration_seconds:
                reasons.append(f"decoded duration {decoded_duration:.3f}s below minimum {min_duration_seconds:.3f}s")
            info = MediaInfo(
                path=str(path),
                exists=True,
                decodable=True,
                prober=info.prober,
                codec_name=info.codec_name,
                format_name=info.format_name,
                sample_rate=sample_rate,
                channels=int(audio.shape[0]),
                duration_seconds=decoded_duration,
                bit_rate=info.bit_rate,
                size_bytes=info.size_bytes,
            )

    return ValidationReport(str(path), not reasons, info, reasons)
