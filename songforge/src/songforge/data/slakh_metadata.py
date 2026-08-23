"""BabySlakh / Slakh instrument metadata adapter.

Slakh ships a `metadata.yaml` beside every ``TrackNNNNN`` directory describing
what each stem actually is. Reading it gives real instrument labels instead of
guesses: spectral heuristics can tell "bass-heavy" from "percussive", but they
cannot tell a piano from a guitar, and inventing that distinction would put
fiction into the evaluation record.

This module only *reads* labels. Nothing here may drive generation behaviour -
instruments must be learned from data and conditioning, never branched on.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

METADATA_NAME = "metadata.yaml"
_TRACK_DIR = re.compile(r"^Track\d+$", re.IGNORECASE)

#: Canonical family for the rendered full mix, which has no stem entry.
MIX_FAMILY = "Mixture"


@dataclass(frozen=True)
class StemMetadata:
    """Canonical instrument description for one audio file."""

    source_track_id: str
    stem_id: str | None
    instrument_name: str | None
    instrument_family: str | None
    midi_program: int | None
    is_drum: bool | None
    plugin_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_track_dir(path: str | Path) -> Path | None:
    """The ``TrackNNNNN`` directory an audio file belongs to, if any."""
    path = Path(path)
    for parent in path.parents:
        if _TRACK_DIR.match(parent.name):
            return parent
    return None


@lru_cache(maxsize=256)
def _load_metadata(track_dir: str) -> dict[str, Any]:
    path = Path(track_dir) / METADATA_NAME
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}


def stem_metadata(path: str | Path) -> StemMetadata | None:
    """Canonical instrument metadata for one BabySlakh audio file.

    Returns None when the file is not inside a Slakh-style track directory or
    the track carries no metadata; callers must handle that rather than
    substituting a guess.
    """
    path = Path(path)
    track_dir = find_track_dir(path)
    if track_dir is None:
        return None

    metadata = _load_metadata(str(track_dir))
    track_id = track_dir.name
    stem_id = path.stem

    # The rendered full mix has no stem entry of its own.
    if stem_id in {"mix", "mixture"}:
        return StemMetadata(
            source_track_id=track_id,
            stem_id=None,
            instrument_name="mix",
            instrument_family=MIX_FAMILY,
            midi_program=None,
            is_drum=False,
        )

    stems = metadata.get("stems") or {}
    entry = stems.get(stem_id)
    if not isinstance(entry, dict):
        return StemMetadata(
            source_track_id=track_id,
            stem_id=stem_id,
            instrument_name=None,
            instrument_family=None,
            midi_program=None,
            is_drum=None,
        )

    program = entry.get("program_num")
    return StemMetadata(
        source_track_id=track_id,
        stem_id=stem_id,
        instrument_name=entry.get("midi_program_name"),
        instrument_family=entry.get("inst_class"),
        midi_program=int(program) if isinstance(program, (int, float)) else None,
        is_drum=bool(entry["is_drum"]) if "is_drum" in entry else None,
        plugin_name=entry.get("plugin_name"),
    )


def instrument_lookup(path: str | Path) -> dict[str, Any]:
    """Adapter shaped for `songforge.data.preprocess`. Empty dict when unknown."""
    metadata = stem_metadata(path)
    return metadata.to_dict() if metadata else {}


def family_counts(paths: list[Path]) -> dict[str, int]:
    """How many files fall in each instrument family. Useful for dataset reports."""
    counts: dict[str, int] = {}
    for path in paths:
        metadata = stem_metadata(path)
        family = (metadata.instrument_family if metadata else None) or "unknown"
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
