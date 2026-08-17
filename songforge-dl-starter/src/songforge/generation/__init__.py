"""SongForge generation layer: our control surface over a pretrained foundation.

The foundation is a replaceable component. SongForge owns the request format,
the honesty rules about which controls are real, the evaluation, and the
experiment record.
"""

from .adapter import FoundationAdapter, LicensePosition, SongResult
from .capabilities import (
    CONTROLS,
    Capabilities,
    ControlResolution,
    ControlSupport,
    resolve_controls,
)
from .registry import available, build, load_prompts, register
from .request import (
    SECTION_KINDS,
    Section,
    SongRequest,
    VocalSpec,
    parse_lyric_sections,
)

__all__ = [
    "CONTROLS",
    "SECTION_KINDS",
    "Capabilities",
    "ControlResolution",
    "ControlSupport",
    "FoundationAdapter",
    "LicensePosition",
    "Section",
    "SongRequest",
    "SongResult",
    "VocalSpec",
    "available",
    "build",
    "load_prompts",
    "parse_lyric_sections",
    "register",
    "resolve_controls",
]
