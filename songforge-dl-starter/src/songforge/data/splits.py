"""Deterministic, leakage-free train/val/test assignment.

Splits are assigned to *groups*, never to individual segments. In ``song`` mode
the group is the source song; in ``singer`` mode it is the performer, which keeps
songs disjoint too because a song belongs to one performer. Segments therefore
cannot straddle a split boundary by construction.

Assignment is quota-based over a hash ordering: deterministic for a given seed,
independent of input order, and it still fills every split when there are enough
groups to go around. Pure hash bucketing is available via ``strategy="hash"`` but
happily puts all four of your debug tracks in train.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .manifest import AudioRecord

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    mode: str = "song"  # "song" (song-disjoint) or "singer" (singer-disjoint)
    strategy: str = "quota"  # "quota" (balanced) or "hash" (pure hash bucketing)

    def validate(self) -> None:
        if self.mode not in ("song", "singer"):
            raise ValueError(f"mode must be 'song' or 'singer', got {self.mode!r}")
        if self.strategy not in ("quota", "hash"):
            raise ValueError(f"strategy must be 'quota' or 'hash', got {self.strategy!r}")
        for name in SPLIT_NAMES:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} ratio must be >= 0")
        if sum(getattr(self, name) for name in SPLIT_NAMES) <= 0:
            raise ValueError("at least one split ratio must be positive")

    @property
    def active(self) -> list[tuple[str, float]]:
        return [(name, getattr(self, name)) for name in SPLIT_NAMES if getattr(self, name) > 0]

    @classmethod
    def from_dict(cls, payload: dict) -> SplitConfig:
        known = {"train", "val", "test", "seed", "mode", "strategy"}
        return cls(**{key: value for key, value in (payload or {}).items() if key in known})


def group_key(record: AudioRecord, mode: str) -> str:
    """The unit that must not cross splits."""
    if mode == "singer":
        return record.singer_id or record.track_id
    return record.track_id


def _hash_position(seed: int, mode: str, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{mode}:{key}".encode()).hexdigest()[:16]
    return int(digest, 16) / float(1 << 64)


def plan_group_splits(keys: Iterable[str], config: SplitConfig) -> dict[str, str]:
    """Map each group key to a split name. Deterministic for a given seed."""
    config.validate()
    unique = sorted(set(keys))
    if not unique:
        return {}

    active = config.active
    if config.strategy == "hash":
        total = sum(ratio for _, ratio in active)
        plan: dict[str, str] = {}
        for key in unique:
            position = _hash_position(config.seed, config.mode, key) * total
            cumulative = 0.0
            chosen = active[-1][0]
            for name, ratio in active:
                cumulative += ratio
                if position < cumulative:
                    chosen = name
                    break
            plan[key] = chosen
        return plan

    ordered = sorted(unique, key=lambda key: (_hash_position(config.seed, config.mode, key), key))
    total_groups = len(ordered)
    total_ratio = sum(ratio for _, ratio in active)

    counts: dict[str, int] = {}
    assigned = 0
    for index, (name, ratio) in enumerate(active):
        if index == len(active) - 1:
            counts[name] = total_groups - assigned
        else:
            count = round(total_groups * ratio / total_ratio)
            counts[name] = count
            assigned += count

    # Give every requested split at least one group when there are enough to share.
    if total_groups >= len(active):
        for name, _ in active:
            if counts[name] < 1:
                donor = max(counts, key=lambda candidate: counts[candidate])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] = 1
    counts = {name: max(count, 0) for name, count in counts.items()}

    plan = {}
    cursor = 0
    for name, _ in active:
        for key in ordered[cursor : cursor + counts[name]]:
            plan[key] = name
        cursor += counts[name]
    for key in ordered[cursor:]:  # rounding remainder
        plan[key] = active[-1][0]
    return plan


def assign_splits(records: Iterable[AudioRecord], config: SplitConfig | None = None) -> list[AudioRecord]:
    """Return copies of ``records`` with ``split`` set group-disjointly."""
    config = config or SplitConfig()
    records = list(records)
    plan = plan_group_splits((group_key(record, config.mode) for record in records), config)
    return [replace(record, split=plan[group_key(record, config.mode)]) for record in records]


def assert_group_disjoint(records: Iterable[AudioRecord], mode: str = "song") -> None:
    """Fail if any group appears in more than one split."""
    seen: dict[str, str] = {}
    for record in records:
        key = group_key(record, mode)
        if key is None:
            continue
        previous = seen.setdefault(key, record.split)
        if previous != record.split:
            label = "Song" if mode == "song" else "Singer"
            raise ValueError(f"{label} leakage: {key} appears in {previous} and {record.split}")


def split_report(records: Iterable[AudioRecord], mode: str = "song") -> dict:
    """Per-split group and segment counts, plus an explicit leakage verdict."""
    records = list(records)
    report: dict = {"mode": mode, "splits": {}, "leakage": None, "ok": True}

    for split in sorted({record.split for record in records}):
        in_split = [record for record in records if record.split == split]
        report["splits"][split] = {
            "segments": len(in_split),
            "groups": len({group_key(record, mode) for record in in_split}),
            "tracks": len({record.track_id for record in in_split}),
            "singers": len({record.singer_id for record in in_split if record.singer_id}),
        }

    try:
        assert_group_disjoint(records, mode)
    except ValueError as exc:
        report["ok"] = False
        report["leakage"] = str(exc)
    return report
