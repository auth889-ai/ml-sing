"""Synthetic audio fixtures.

M02 tests must not download BabySlakh, so the corpora here reproduce the
structural properties preprocessing cares about: Slakh-style ``TrackNNNNN``
directories, GTSinger-style ``<language>/<singer>/<song>`` nesting, plus the
pathological files (silent, clipped, corrupt, empty) that validation must reject.

Everything is generated from a fixed seed and written as 16-bit PCM WAV, which
the standard library can decode, so fixtures work with no optional audio
dependency and no ffmpeg installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .audio import write_wav

DEFAULT_SAMPLE_RATE = 16000


def tone(
    seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frequency: float = 220.0,
    amplitude: float = 0.4,
    channels: int = 1,
    noise: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """A deterministic test signal shaped ``[channels, samples]``."""
    num_samples = max(round(seconds * sample_rate), 1)
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    wave = amplitude * torch.sin(2 * math.pi * frequency * t)
    if noise > 0:
        generator = torch.Generator().manual_seed(seed)
        wave = wave + noise * torch.randn(num_samples, generator=generator)
    wave = wave.unsqueeze(0)
    if channels > 1:
        harmonics = [wave * (1.0 - 0.1 * index) for index in range(channels)]
        wave = torch.cat(harmonics, dim=0)
    return wave.clamp(-1.0, 1.0)


def write_tone_wav(path: str | Path, seconds: float = 2.0, sample_rate: int = DEFAULT_SAMPLE_RATE, **kwargs) -> Path:
    path = Path(path)
    write_wav(path, tone(seconds, sample_rate, **kwargs), sample_rate)
    return path


def write_silent_wav(path: str | Path, seconds: float = 2.0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Digital silence: must be detected and dropped, not trained on."""
    path = Path(path)
    num_samples = max(round(seconds * sample_rate), 1)
    write_wav(path, torch.zeros(1, num_samples), sample_rate)
    return path


def write_clipped_wav(
    path: str | Path,
    seconds: float = 2.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frequency: float = 220.0,
) -> Path:
    """A heavily overdriven tone: most samples sit at full scale."""
    path = Path(path)
    num_samples = max(round(seconds * sample_rate), 1)
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    wave = (6.0 * torch.sin(2 * math.pi * frequency * t)).clamp(-1.0, 1.0)
    write_wav(path, wave.unsqueeze(0), sample_rate)
    return path


def write_corrupt_wav(path: str | Path) -> Path:
    """A .wav suffix over bytes that are not a RIFF file at all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00NOTAWAVE" + bytes(range(64)))
    return path


def write_empty_file(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@dataclass
class SyntheticCorpus:
    """Where fixtures were written and what the caller should expect back."""

    root: Path
    audio_paths: list[Path] = field(default_factory=list)
    track_ids: list[str] = field(default_factory=list)
    singer_ids: list[str] = field(default_factory=list)
    broken_paths: list[Path] = field(default_factory=list)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    seconds_per_file: float = 4.0


def build_slakh_like_corpus(
    root: str | Path,
    tracks: int = 4,
    stems_per_track: int = 2,
    seconds: float = 4.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    include_broken: bool = False,
) -> SyntheticCorpus:
    """A BabySlakh-shaped corpus: ``TrackNNNNN/mix.wav`` plus ``TrackNNNNN/stems/*.wav``.

    Every file under one ``TrackNNNNN`` directory shares a track id, which is what
    song-disjoint splitting relies on.
    """
    root = Path(root)
    corpus = SyntheticCorpus(root=root, sample_rate=sample_rate, seconds_per_file=seconds)

    for index in range(tracks):
        track_id = f"Track{index + 1:05d}"
        track_dir = root / track_id
        corpus.track_ids.append(track_id)

        corpus.audio_paths.append(
            write_tone_wav(
                track_dir / "mix.wav",
                seconds=seconds,
                sample_rate=sample_rate,
                frequency=110.0 + 40.0 * index,
                amplitude=0.35,
                noise=0.01,
                seed=index,
            )
        )
        for stem in range(stems_per_track):
            corpus.audio_paths.append(
                write_tone_wav(
                    track_dir / "stems" / f"S{stem:02d}.wav",
                    seconds=seconds,
                    sample_rate=sample_rate,
                    frequency=220.0 + 55.0 * index + 15.0 * stem,
                    amplitude=0.25,
                    noise=0.005,
                    seed=100 * index + stem,
                )
            )

    if include_broken:
        broken_dir = root / "Track99999"
        corpus.broken_paths = [
            write_corrupt_wav(broken_dir / "corrupt.wav"),
            write_empty_file(broken_dir / "empty.wav"),
            write_silent_wav(broken_dir / "silent.wav", seconds=seconds, sample_rate=sample_rate),
        ]
    return corpus


def build_singer_corpus(
    root: str | Path,
    singers: int = 4,
    songs_per_singer: int = 2,
    seconds: float = 4.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> SyntheticCorpus:
    """A GTSinger-shaped corpus: ``<language>/<singer>/<song>/vocal.wav``."""
    root = Path(root)
    corpus = SyntheticCorpus(root=root, sample_rate=sample_rate, seconds_per_file=seconds)

    for index in range(singers):
        language = "english" if index % 2 == 0 else "chinese"
        singer = f"Singer{index + 1:02d}"
        corpus.singer_ids.append(f"{language}__{singer}")
        for song in range(songs_per_singer):
            song_id = f"Song{song + 1:02d}"
            # Matches derive_track_id(path, root=root): root-relative, suffix dropped.
            corpus.track_ids.append(f"{language}__{singer}__{song_id}__vocal")
            corpus.audio_paths.append(
                write_tone_wav(
                    root / language / singer / song_id / "vocal.wav",
                    seconds=seconds,
                    sample_rate=sample_rate,
                    frequency=180.0 + 30.0 * index + 10.0 * song,
                    amplitude=0.3,
                    noise=0.004,
                    seed=1000 * index + song,
                )
            )
    return corpus
