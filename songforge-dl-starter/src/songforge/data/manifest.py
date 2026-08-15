from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioRecord:
    id: str
    path: str
    split: str
    source: str
    license: str
    track_id: str
    singer_id: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def write_jsonl(records: Iterable[AudioRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def assert_no_track_leakage(records: Iterable[AudioRecord]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        old = seen.setdefault(record.track_id, record.split)
        if old != record.split:
            raise ValueError(f"Track leakage: {record.track_id} appears in {old} and {record.split}")


def assert_singer_disjoint(records: Iterable[AudioRecord], eval_splits: set[str] | None = None) -> None:
    """Require singers in eval splits to be absent from training records."""

    eval_splits = eval_splits or {"val", "valid", "validation", "test"}
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
