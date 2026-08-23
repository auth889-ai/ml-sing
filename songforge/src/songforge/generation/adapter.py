"""The seam between SongForge and whichever pretrained foundation we select.

SongForge owns the request format, the control semantics, the evaluation and the
experiment record. A foundation is a replaceable component behind this
interface, so selecting a different one later is an adapter swap rather than a
rewrite — and so the benchmark can drive several of them with identical inputs.

Every adapter must also declare its licence position, separately for code and
weights, because those routinely differ and the difference decides whether a
model can be the product foundation or only a research baseline.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .capabilities import Capabilities, ControlResolution, resolve_controls
from .request import SongRequest


@dataclass(frozen=True)
class LicensePosition:
    """Code and weights licences are recorded separately, never conflated."""

    code_license: str
    weights_license: str
    commercial_use: str            # "allowed" | "prohibited" | "unclear"
    redistribution: str
    attribution: str
    training_data_notes: str
    usable_as_research_baseline: bool
    usable_for_finetuning: bool
    usable_as_product_foundation: bool
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = list(self.sources)
        return payload


@dataclass
class SongResult:
    """One generated song plus everything needed to reproduce and judge it."""

    audio_path: Path
    request: SongRequest
    capabilities: Capabilities
    resolution: ControlResolution
    sample_rate: int
    channels: int
    duration_seconds: float
    model_id: str
    checkpoint: str
    inference_settings: dict[str, Any] = field(default_factory=dict)
    gpu_name: str = ""
    peak_vram_gb: float = 0.0
    generation_seconds: float = 0.0
    error: str | None = None

    @property
    def real_time_factor(self) -> float:
        """Compute seconds per audio second. Below 1.0 is faster than realtime."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.generation_seconds / self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": str(self.audio_path),
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "request": self.request.to_dict(),
            "request_fingerprint": self.request.fingerprint(),
            "capabilities": self.capabilities.to_dict(),
            "control_resolution": self.resolution.to_dict(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "inference_settings": self.inference_settings,
            "gpu": self.gpu_name,
            "peak_vram_gb": self.peak_vram_gb,
            "generation_seconds": self.generation_seconds,
            "real_time_factor": self.real_time_factor,
            "error": self.error,
        }

    def write_metadata(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


class FoundationAdapter(ABC):
    """Base class for a pretrained music foundation."""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """What this model genuinely supports. Asserted against reality by tests."""

    @property
    @abstractmethod
    def license(self) -> LicensePosition:
        """Code and weights licence position, with sources."""

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the device. Separate from __init__ so the benchmark
        can report licence and capability information without downloading GBs."""

    @abstractmethod
    def _generate(self, request: SongRequest, output_path: Path) -> dict[str, Any]:
        """Do the generation. Return inference settings actually used.

        Implementations must pass only controls their capabilities declare, and
        must not silently substitute a default for something they cannot honour.
        """

    def generate(self, request: SongRequest, output_path: str | Path) -> SongResult:
        """Resolve controls honestly, generate, and record the full provenance."""
        request.validate()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolution = resolve_controls(request, self.capabilities)

        gpu_name, peak_before = _gpu_state()
        started = time.perf_counter()
        error: str | None = None
        settings: dict[str, Any] = {}
        try:
            settings = self._generate(request, output_path) or {}
        except Exception as exc:  # noqa: BLE001 - a failed candidate is a result, not a crash
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        _, peak_after = _gpu_state()

        duration, sample_rate, channels = _probe_audio(output_path) if error is None else (0.0, 0, 0)

        return SongResult(
            audio_path=output_path,
            request=request,
            capabilities=self.capabilities,
            resolution=resolution,
            sample_rate=sample_rate or self.capabilities.sample_rate,
            channels=channels or self.capabilities.channels,
            duration_seconds=duration,
            model_id=self.capabilities.model,
            checkpoint=settings.get("checkpoint", ""),
            inference_settings=settings,
            gpu_name=gpu_name,
            peak_vram_gb=max(peak_after, peak_before),
            generation_seconds=elapsed,
            error=error,
        )


def _gpu_state() -> tuple[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu", 0.0
        return torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001 - reporting hardware must never break a run
        return "unknown", 0.0


def _probe_audio(path: Path) -> tuple[float, int, int]:
    if not path.exists():
        return 0.0, 0, 0
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception:  # noqa: BLE001
        return 0.0, 0, 0
