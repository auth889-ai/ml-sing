"""What the user asks for, independent of which model answers.

`SongRequest` is SongForge's own control surface. It is deliberately decoupled
from any one foundation model: adapters translate it, and declare honestly which
parts they can actually honour (see `capabilities`).

Nothing here selects instruments or genres by rule. Instrument and style names
are *conditioning text and metadata* carried to a learned model; they never
switch on a code path that synthesises a particular instrument.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

#: Canonical section names understood by the structure control. Foundations that
#: support structure tags usually recognise this vocabulary or a subset.
SECTION_KINDS = (
    "intro", "verse", "pre_chorus", "chorus", "post_chorus",
    "bridge", "solo", "breakdown", "drop", "outro", "instrumental",
)

_LYRIC_TAG = re.compile(r"^\s*\[([a-zA-Z _-]+)\]\s*$")


@dataclass(frozen=True)
class Section:
    """One structural span of the song."""

    kind: str
    seconds: float | None = None
    lyrics: str | None = None
    note: str | None = None

    def validate(self) -> None:
        if self.kind not in SECTION_KINDS:
            raise ValueError(f"unknown section kind {self.kind!r}; known: {', '.join(SECTION_KINDS)}")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError(f"section {self.kind}: seconds must be positive")


@dataclass(frozen=True)
class VocalSpec:
    """Requested vocal character. All fields optional; None means 'unspecified'."""

    present: bool = True
    gender: str | None = None       # e.g. "female", "male", "androgynous"
    register: str | None = None     # e.g. "alto", "tenor", "soprano"
    style: str | None = None        # e.g. "belted", "breathy", "operatic", "rap"
    language: str | None = None     # BCP-47-ish, e.g. "en"

    def descriptors(self) -> list[str]:
        return [v for v in (self.gender, self.register, self.style) if v]


@dataclass(frozen=True)
class SongRequest:
    """A complete generation request."""

    prompt: str
    lyrics: str | None = None
    genre: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    vocal: VocalSpec | None = None
    bpm: int | None = None
    key: str | None = None
    duration_seconds: float = 60.0
    structure: tuple[Section, ...] = ()
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.bpm is not None and not (20 <= self.bpm <= 300):
            raise ValueError(f"bpm {self.bpm} is outside a musically plausible 20-300")
        for section in self.structure:
            section.validate()

    @property
    def wants_vocals(self) -> bool:
        return bool(self.lyrics) or bool(self.vocal and self.vocal.present)

    def requested_controls(self) -> set[str]:
        """Which controls the caller actually set.

        Only these matter when reporting unsupported controls: a model that
        ignores BPM is irrelevant to a request that never asked for a BPM.
        """
        requested = {"prompt", "seed", "duration"}
        if self.lyrics:
            requested.add("lyrics")
        if self.genre:
            requested.add("genre")
        if self.mood:
            requested.add("mood")
        if self.instruments:
            requested.add("instruments")
        if self.vocal is not None:
            requested.add("vocal_style")
        if self.bpm is not None:
            requested.add("bpm")
        if self.key is not None:
            requested.add("key")
        if self.structure:
            requested.add("structure")
        return requested

    def fingerprint(self) -> str:
        """Stable id for this exact request, for experiment tracking."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["genre"] = list(self.genre)
        payload["mood"] = list(self.mood)
        payload["instruments"] = list(self.instruments)
        payload["structure"] = [asdict(section) for section in self.structure]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SongRequest:
        payload = dict(payload)
        vocal = payload.get("vocal")
        structure = payload.get("structure") or ()
        return cls(
            prompt=payload["prompt"],
            lyrics=payload.get("lyrics"),
            genre=tuple(payload.get("genre") or ()),
            mood=tuple(payload.get("mood") or ()),
            instruments=tuple(payload.get("instruments") or ()),
            vocal=VocalSpec(**vocal) if isinstance(vocal, dict) else None,
            bpm=payload.get("bpm"),
            key=payload.get("key"),
            duration_seconds=float(payload.get("duration_seconds", 60.0)),
            structure=tuple(Section(**section) for section in structure),
            seed=int(payload.get("seed", 0)),
            extra=dict(payload.get("extra") or {}),
        )


def parse_lyric_sections(lyrics: str) -> tuple[Section, ...]:
    """Split ``[Verse] ... [Chorus] ...`` lyrics into sections.

    Unknown tags are kept as text inside the current section rather than
    guessed at, so a tag we do not model cannot silently become a wrong one.
    """
    sections: list[Section] = []
    current_kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_kind is None:
            return
        text = "\n".join(buffer).strip()
        sections.append(Section(kind=current_kind, lyrics=text or None))

    for line in lyrics.splitlines():
        match = _LYRIC_TAG.match(line)
        kind = match.group(1).strip().lower().replace(" ", "_").replace("-", "_") if match else None
        if kind in SECTION_KINDS:
            flush()
            current_kind = kind
            buffer = []
            continue
        buffer.append(line)

    flush()
    return tuple(sections)
