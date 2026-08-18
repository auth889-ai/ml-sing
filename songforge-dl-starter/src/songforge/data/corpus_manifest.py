"""The canonical multi-corpus manifest: one record shape for every dataset.

SongForge is multi-corpus (Slakh, FMA, VocalSet, E-GMD, GuitarSet, ...) and
corpora are never concatenated blindly: each dataset's adapter emits records
in THIS shape, and only records that validate enter a training mix. The
record carries everything needed to trace any trained behaviour back to its
audio, its licence, and its source — and to exclude a record later without
rebuilding anything.

Unknown is always represented as None/empty, never guessed: a missing BPM
stays missing, an unknown licence fails validation rather than defaulting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SPLITS = ("train", "val", "test")

#: Licence classes a record may carry. `permissive` is the only class allowed
#: into the deployable adapter line; `research-only` trains clearly-labelled
#: research adapters that are never merged.
LICENCE_CLASSES = ("permissive", "research-only")

#: Exact licence spellings accepted as permissive. Anything else claiming
#: `permissive` fails validation — a new licence must be added here on
#: purpose, with the audit that justifies it.
PERMISSIVE_LICENCES = frozenset({
    "CC0-1.0", "CC-BY-3.0", "CC-BY-4.0", "MIT", "public-domain",
})


@dataclass(frozen=True)
class CorpusRecord:
    """One training example in the unified SongForge corpus."""

    dataset: str                #: registry id, e.g. "slakh100", "fma_ccby"
    track_id: str               #: stable id within the dataset
    audio_path: str             #: path relative to the corpus root
    licence: str                #: exact licence spelling, e.g. "CC-BY-4.0"
    licence_class: str          #: "permissive" | "research-only"
    source_url: str             #: where this audio provably came from
    split: str                  #: "train" | "val" | "test"
    duplicate_hash: str         #: content hash used for cross-corpus dedup

    caption: str = ""           #: text conditioning; derived from metadata only
    lyrics: str = ""            #: empty means explicitly no lyrics
    instrument_tags: tuple[str, ...] = ()
    genre: tuple[str, ...] = ()
    bpm: float | None = None
    key: str | None = None
    artist: str | None = None
    language: str | None = None
    quality_score: float | None = None  #: 0-1; None until a scorer has run
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("dataset", "track_id", "audio_path", "licence",
                     "source_url", "duplicate_hash"):
            if not getattr(self, name):
                raise ValueError(f"{self.dataset or '?'}/{self.track_id or '?'}: "
                                 f"missing required field {name!r}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.dataset}/{self.track_id}: bad split {self.split!r}")
        if self.licence_class not in LICENCE_CLASSES:
            raise ValueError(
                f"{self.dataset}/{self.track_id}: bad licence_class {self.licence_class!r}")
        if self.licence_class == "permissive" and self.licence not in PERMISSIVE_LICENCES:
            raise ValueError(
                f"{self.dataset}/{self.track_id}: licence {self.licence!r} is not on the "
                "permissive allowlist; audit it and add it deliberately, or mark the "
                "record research-only")
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                f"{self.dataset}/{self.track_id}: quality_score outside [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["instrument_tags"] = list(self.instrument_tags)
        payload["genre"] = list(self.genre)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusRecord":
        data = dict(payload)
        data["instrument_tags"] = tuple(data.get("instrument_tags") or ())
        data["genre"] = tuple(data.get("genre") or ())
        record = cls(**data)
        record.validate()
        return record


def write_corpus_manifest(path: str | Path, records: Iterable[CorpusRecord]) -> int:
    """Validated JSONL write; returns the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            record.validate()
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_corpus_manifest(path: str | Path) -> Iterator[CorpusRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield CorpusRecord.from_dict(json.loads(line))


def assert_deployable(records: Iterable[CorpusRecord]) -> None:
    """Refuse a deployable-line training mix containing any non-permissive record."""
    offenders = [f"{r.dataset}/{r.track_id} ({r.licence})"
                 for r in records if r.licence_class != "permissive"]
    if offenders:
        raise ValueError(
            "non-permissive records in a deployable training mix: "
            + ", ".join(offenders[:10])
            + (" ..." if len(offenders) > 10 else ""))


def assert_no_cross_corpus_duplicates(records: Iterable[CorpusRecord]) -> None:
    """The same audio content must not enter twice via different corpora."""
    seen: dict[str, str] = {}
    for record in records:
        origin = f"{record.dataset}/{record.track_id}"
        prior = seen.get(record.duplicate_hash)
        if prior is not None and not prior.startswith(f"{record.dataset}/"):
            raise ValueError(
                f"duplicate audio across corpora: {origin} repeats {prior}")
        seen.setdefault(record.duplicate_hash, origin)
