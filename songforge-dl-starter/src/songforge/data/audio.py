from __future__ import annotations

import glob
import json
import wave
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dsp import resample_waveform, to_channels
from .media import decode_audio


def load_audio(path: str | Path, sample_rate: int, channels: int = 1) -> torch.Tensor:
    """Decode to ``[channels, samples]`` at ``sample_rate``, clamped to [-1, 1].

    Decoding and resampling are delegated to `songforge.data.media` and
    `songforge.data.dsp` so training and M02 preprocessing share one
    implementation. The resampler is pure torch, so this no longer requires
    torchaudio to be installed when the source rate differs from the target.
    """
    audio, source_rate = decode_audio(path)
    audio = to_channels(audio, channels)
    if source_rate != sample_rate:
        audio = resample_waveform(audio, source_rate, sample_rate)
    return audio.clamp(-1.0, 1.0)


def write_wav(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().float()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio.t().contiguous() * 32767.0).round().to(torch.int16).numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(audio.size(0))
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_audio_paths(manifest: str | Path | None = None, audio_glob: str | None = None) -> list[Path]:
    paths: list[Path] = []
    if manifest:
        with Path(manifest).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    paths.append(Path(json.loads(line)["path"]))
    if audio_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(audio_glob, recursive=True)))
    if not paths:
        raise ValueError("Provide --manifest or --audio-glob with at least one audio file")
    return paths


class AudioSegmentDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        sample_rate: int,
        channels: int,
        segment_samples: int,
    ):
        self.paths = paths
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_samples = segment_samples

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio = load_audio(self.paths[idx], self.sample_rate, self.channels)
        if audio.size(-1) >= self.segment_samples:
            start = max((audio.size(-1) - self.segment_samples) // 2, 0)
            audio = audio[..., start : start + self.segment_samples]
        else:
            audio = F.pad(audio, (0, self.segment_samples - audio.size(-1)))
        return audio
