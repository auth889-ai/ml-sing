"""The canonical SongForge training manifest.

`AudioRecord` is the single manifest schema for the whole project. Every
milestone that needs audio reads and writes this format; there is deliberately no
second "processed" or "segment" record type. M02 extended the original M01 record
with preprocessing, provenance, and integrity fields, all optional and defaulted,
so records written before M02 still load.

Serialization is JSON Lines: one record per line, sorted-key JSON, UTF-8.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

MANIFEST_SCHEMA = "songforge.audio.v1"

#: Fields that must be present and non-empty in every manifest record.
#: Mirrors `global_split_policy.manifest_required_fields` in the dataset registry.
REQUIRED_FIELDS = ("id", "path", "split", "source", "license", "track_id")

EVAL_SPLITS = frozenset({"val", "valid", "validation", "test"})


@dataclass(frozen=True)
class AudioRecord:
    """One training example: a single deterministic segment of one source file."""

    # --- identity and licensing (M01, required) ---
    id: str
    path: str
    split: str
    source: str
    license: str
    track_id: str
    singer_id: str | None = None
    tags: tuple[str, ...] = ()

    # --- M02 preprocessing description ---
    schema: str = MANIFEST_SCHEMA
    source_path: str | None = None
    segment_index: int = 0
    start_sample: int = 0
    num_samples: int = 0
    sample_rate: int = 0
    channels: int = 0
    duration_seconds: float = 0.0

    # --- M02 amplitude description ---
    peak: float = 0.0
    rms: float = 0.0
    peak_dbfs: float = 0.0
    rms_dbfs: float = 0.0
    clipping_ratio: float = 0.0
    silent: bool = False

    # --- M02 integrity and duplicate-detection hooks ---
    audio_sha256: str | None = None
    source_sha256: str | None = None

    # --- M02 provenance, license propagation, and reproducibility ---
    provenance: dict = field(default_factory=dict)
    preprocessing: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> AudioRecord:
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in known}
        if "tags" in kwargs and kwargs["tags"] is not None:
            kwargs["tags"] = tuple(kwargs["tags"])
        unknown = {key: value for key, value in payload.items() if key not in known}
        if unknown:
            merged = dict(kwargs.get("extra") or {})
            merged.update(unknown)
            kwargs["extra"] = merged
        return cls(**kwargs)


def stable_id(*parts: str) -> str:
    """Deterministic short id. Same inputs always produce the same value.

    Uses a unit-separator join so ("a", "bc") and ("ab", "c") differ.
    """
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def segment_id(dataset_id: str, track_id: str, source_path: str, segment_index: int, start_sample: int) -> str:
    """Stable id for one segment, independent of processing order and output path."""
    return stable_id(dataset_id, track_id, source_path, str(segment_index), str(start_sample))


def write_jsonl(records: Iterable[AudioRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[AudioRecord]:
    path = Path(path)
    records: list[AudioRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(AudioRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed manifest record: {exc}") from exc
    return records


def validate_records(records: Iterable[AudioRecord]) -> list[str]:
    """Return human-readable problems. Empty list means the manifest is usable."""
    errors: list[str] = []
    seen_ids: dict[str, str] = {}

    for record in records:
        for name in REQUIRED_FIELDS:
            if not getattr(record, name, None):
                errors.append(f"{record.id or '<no id>'}: required field {name!r} is empty")
        previous = seen_ids.setdefault(record.id, record.path)
        if previous != record.path:
            errors.append(f"duplicate record id {record.id} used for {previous} and {record.path}")
        if record.num_samples and record.sample_rate:
            expected = record.num_samples / record.sample_rate
            if abs(expected - record.duration_seconds) > 1e-3:
                errors.append(f"{record.id}: duration_seconds does not match num_samples/sample_rate")
    return errors


def assert_provenance_complete(records: Iterable[AudioRecord]) -> None:
    """Every record must carry the licence and source it came from."""
    for record in records:
        if not record.license:
            raise ValueError(f"{record.id}: license was not propagated")
        provenance = record.provenance or {}
        missing = [key for key in ("dataset_id", "source_url", "license_name") if not provenance.get(key)]
        if missing:
            raise ValueError(f"{record.id}: provenance missing {', '.join(missing)}")


def assert_no_track_leakage(records: Iterable[AudioRecord]) -> None:
    """A source song must live in exactly one split."""
    seen: dict[str, str] = {}
    for record in records:
        old = seen.setdefault(record.track_id, record.split)
        if old != record.split:
            raise ValueError(f"Track leakage: {record.track_id} appears in {old} and {record.split}")


def assert_singer_disjoint(records: Iterable[AudioRecord], eval_splits: set[str] | None = None) -> None:
    """Require singers in eval splits to be absent from training records."""
    eval_splits = eval_splits or set(EVAL_SPLITS)
    train_singers: set[str] = set()
    eval_singers: dict[str, str] = {}

    for record in records:
        if record.singer_id is None:
            continue
        if record.split == "train":
            train_singers.add(record.singer_id)
        elif record.split in eval_splits:
            eval_singers.setdefault(record.singer_id, record.split)

    overlap = train_singers.intersection(eval_singers)
    if overlap:
        singer = min(overlap)
        raise ValueError(f"Singer leakage: {singer} appears in train and {eval_singers[singer]}")


def manifest_summary(records: Iterable[AudioRecord]) -> dict:
    """Counts and durations per split, for acceptance reports and experiment logs."""
    records = list(records)
    summary: dict = {
        "schema": MANIFEST_SCHEMA,
        "segments": len(records),
        "tracks": len({record.track_id for record in records}),
        "singers": len({record.singer_id for record in records if record.singer_id}),
        "sources": sorted({record.source for record in records}),
        "licenses": sorted({record.license for record in records}),
        "total_seconds": round(sum(record.duration_seconds for record in records), 6),
        "silent_segments": sum(1 for record in records if record.silent),
        "splits": {},
    }
    for split in sorted({record.split for record in records}):
        in_split = [record for record in records if record.split == split]
        summary["splits"][split] = {
            "segments": len(in_split),
            "tracks": len({record.track_id for record in in_split}),
            "singers": len({record.singer_id for record in in_split if record.singer_id}),
            "seconds": round(sum(record.duration_seconds for record in in_split), 6),
        }
    return summary


def write_split_manifests(records: Iterable[AudioRecord], output_dir: str | Path) -> dict[str, Path]:
    """Write ``all.jsonl`` plus one file per split. Returns the written paths."""
    records = list(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {"all": write_jsonl(records, output_dir / "all.jsonl")}
    for split in sorted({record.split for record in records}):
        in_split = [record for record in records if record.split == split]
        written[split] = write_jsonl(in_split, output_dir / f"{split}.jsonl")

    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(manifest_summary(records), indent=2, sort_keys=True), encoding="utf-8")
    written["summary"] = summary_path
    return written
