"""Which controls a foundation actually obeys.

The rule this module exists to enforce: **do not create fake controls**. A UI
knob that quietly does nothing is worse than a missing knob, because it makes
the evaluation dishonest — a "BPM 90" song that ignored BPM would be scored as
though BPM had been requested and met.

Every adapter declares, per control, whether it is:

``NATIVE``
    a real conditioning input of the model (dedicated argument, embedding,
    or token stream). The model is genuinely being told.
``PROMPT``
    folded into the free-text prompt. The model may or may not comply; this is
    a soft request and is reported as such.
``IGNORED``
    not supported at all. Setting it changes nothing.

`resolve_controls` turns a request plus a capability table into a report that
records exactly how each requested control was applied, including warnings for
anything the caller asked for that the model cannot honour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .request import SongRequest

#: Every control SongForge exposes. Adapters must have an opinion about each.
CONTROLS = (
    "prompt", "lyrics", "genre", "mood", "instruments", "vocal_style",
    "bpm", "key", "duration", "structure", "seed",
)


class ControlSupport(str, Enum):
    NATIVE = "native"
    PROMPT = "prompt"
    IGNORED = "ignored"

    @property
    def is_honoured(self) -> bool:
        """True if the model is told at all. PROMPT counts, but only softly."""
        return self is not ControlSupport.IGNORED


@dataclass(frozen=True)
class Capabilities:
    """What a foundation can do, declared per adapter and asserted by tests."""

    model: str
    version: str
    controls: dict[str, ControlSupport]
    max_duration_seconds: float
    sample_rate: int
    channels: int = 2
    produces_vocals: bool = False
    languages: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        missing = [name for name in CONTROLS if name not in self.controls]
        if missing:
            raise ValueError(
                f"{self.model}: capabilities must declare every control; missing {', '.join(missing)}. "
                "An undeclared control is exactly the silent no-op this class exists to prevent."
            )
        unknown = [name for name in self.controls if name not in CONTROLS]
        if unknown:
            raise ValueError(f"{self.model}: unknown controls declared: {', '.join(unknown)}")

    def support(self, control: str) -> ControlSupport:
        return self.controls[control]

    def native_controls(self) -> list[str]:
        return sorted(n for n, s in self.controls.items() if s is ControlSupport.NATIVE)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["controls"] = {name: support.value for name, support in self.controls.items()}
        payload["languages"] = list(self.languages)
        return payload


@dataclass
class ControlResolution:
    """How one request was actually mapped onto one model."""

    applied: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def honest(self) -> bool:
        """True when nothing the caller asked for was silently dropped."""
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {"applied": dict(self.applied), "warnings": list(self.warnings), "honest": self.honest}


def resolve_controls(request: SongRequest, capabilities: Capabilities) -> ControlResolution:
    """Report how each requested control maps onto this model.

    Only controls the caller actually set produce warnings: a model that ignores
    key is not a problem for a request that never specified one.
    """
    resolution = ControlResolution()
    requested = request.requested_controls()

    for control in sorted(requested):
        support = capabilities.support(control)
        resolution.applied[control] = support.value
        if support is ControlSupport.IGNORED:
            resolution.warnings.append(
                f"{control}: requested but {capabilities.model} has no support for it; the value was dropped"
            )
        elif support is ControlSupport.PROMPT:
            resolution.applied[control] = "prompt (soft: folded into text, compliance not guaranteed)"

    if request.wants_vocals and not capabilities.produces_vocals:
        resolution.warnings.append(
            f"vocals: requested but {capabilities.model} does not sing; the output will be instrumental"
        )

    if request.duration_seconds > capabilities.max_duration_seconds:
        resolution.warnings.append(
            f"duration: requested {request.duration_seconds:.0f}s but "
            f"{capabilities.model} generates at most {capabilities.max_duration_seconds:.0f}s"
        )

    language = request.vocal.language if request.vocal else None
    if language and capabilities.languages and language not in capabilities.languages:
        resolution.warnings.append(
            f"vocal language {language!r} is not in this model's supported set "
            f"({', '.join(capabilities.languages)}); intelligibility is not expected"
        )

    return resolution
