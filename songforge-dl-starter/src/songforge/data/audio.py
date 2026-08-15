from __future__ import annotations

import glob
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _read_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    tensor = torch.from_numpy(audio).t().contiguous()
    return tensor, int(sr)


def _read_with_torchaudio(path: Path) -> tuple[torch.Tensor, int]:
    import torchaudio

    audio, sr = torchaudio.load(str(path))
    return audio.float(), int(sr)


def _read_with_wave(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sr = f.getframerate()
        frames = f.readframes(f.getnframes())
    if sample_width != 2:
        raise RuntimeError("stdlib WAV fallback supports 16-bit PCM only")
    pcm = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
    audio = torch.from_numpy(pcm.reshape(-1, channels).T.copy())
    return audio, int(sr)


def load_audio(path: str | Path, sample_rate: int, channels: int = 1) -> torch.Tensor:
    path = Path(path)
    try:
        audio, sr = _read_with_soundfile(path)
    except (ImportError, OSError, RuntimeError, ValueError):
        try:
            audio, sr = _read_with_torchaudio(path)
        except (ImportError, OSError, RuntimeError, ValueError):
            audio, sr = _read_with_wave(path)

    if channels == 1 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    elif channels > 1 and audio.size(0) == 1:
        audio = audio.repeat(channels, 1)
    elif audio.size(0) != channels:
        audio = audio[:channels]

    if sr != sample_rate:
        try:
            import torchaudio.functional as AF

            audio = AF.resample(audio, sr, sample_rate)
        except Exception as exc:
            raise RuntimeError("Resampling requires torchaudio when source sample rate differs") from exc
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
