"""A test double, not a model.

`NullAdapter` writes silence. It exists so the benchmark harness, the control
resolver and the evaluation stack can be tested without a GPU or multi-gigabyte
weights. It declares every control IGNORED and `produces_vocals=False`, which is
the literal truth about silence, and which means any request run through it
comes back covered in warnings — exactly as it should.

Never report NullAdapter output as a generation result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..adapter import FoundationAdapter, LicensePosition
from ..capabilities import CONTROLS, Capabilities, ControlSupport
from ..registry import register
from ..request import SongRequest


class NullAdapter(FoundationAdapter):
    """Writes silence of the requested length. For pipeline tests only."""

    def __init__(self, sample_rate: int = 44100, channels: int = 2) -> None:
        self._sample_rate = sample_rate
        self._channels = channels

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            model="null",
            version="0",
            controls=dict.fromkeys(CONTROLS, ControlSupport.IGNORED),
            max_duration_seconds=600.0,
            sample_rate=self._sample_rate,
            channels=self._channels,
            produces_vocals=False,
            notes="Test double. Produces silence. Not a music model.",
        )

    @property
    def license(self) -> LicensePosition:
        return LicensePosition(
            code_license="MIT (this repository)",
            weights_license="not applicable - no weights",
            commercial_use="allowed",
            redistribution="allowed",
            attribution="none",
            training_data_notes="no training data; emits silence",
            usable_as_research_baseline=False,
            usable_for_finetuning=False,
            usable_as_product_foundation=False,
        )

    def load(self) -> None:
        return None

    def _generate(self, request: SongRequest, output_path: Path) -> dict[str, Any]:
        import soundfile as sf

        frames = int(request.duration_seconds * self._sample_rate)
        audio = np.zeros((frames, self._channels), dtype=np.float32)
        sf.write(str(output_path), audio, self._sample_rate)
        return {"checkpoint": "none", "note": "silence written by the null test double"}


@register("null")
def _build(**kwargs: Any) -> NullAdapter:
    return NullAdapter(**kwargs)
