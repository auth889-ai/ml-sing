"""Duplicate detection hooks.

Two levels are provided. `file_sha256` fingerprints bytes on disk and catches the
same file downloaded twice. `waveform_sha256` fingerprints decoded audio content
after quantization, so it still matches when the same recording is redistributed
in a different container or bit depth.

Near-duplicate detection (audio fingerprinting, chroma matching) is deliberately
out of scope for M02. `duplicate_report` is the hook a later milestone extends.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import torch

from .manifest import AudioRecord


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of raw file bytes, streamed so large audio never loads into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def waveform_sha256(waveform: torch.Tensor, sample_rate: int) -> str:
    """Content hash of decoded audio, stable across container and float noise.

    Quantizing to int16 before hashing means a WAV and a FLAC of the same master
    agree, while genuinely different audio still differs.
    """
    audio = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    pcm = (audio * 32767.0).round().to(torch.int16).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{tuple(audio.shape)}".encode())
    digest.update(pcm)
    return digest.hexdigest()


def find_duplicate_groups(records: Iterable[AudioRecord], key: str = "audio_sha256") -> dict[str, list[AudioRecord]]:
    """Group records that share a fingerprint. Only groups of 2+ are returned."""
    buckets: dict[str, list[AudioRecord]] = {}
    for record in records:
        fingerprint = getattr(record, key, None)
        if not fingerprint:
            continue
        buckets.setdefault(fingerprint, []).append(record)
    return {fingerprint: group for fingerprint, group in buckets.items() if len(group) > 1}


def assert_no_cross_split_duplicates(records: Iterable[AudioRecord], key: str = "audio_sha256") -> None:
    """Identical audio in two different splits is leakage even with disjoint ids."""
    for fingerprint, group in find_duplicate_groups(records, key).items():
        splits = {record.split for record in group}
        if len(splits) > 1:
            ids = ", ".join(sorted(record.id for record in group)[:4])
            raise ValueError(
                f"Duplicate audio {fingerprint[:12]} spans splits {sorted(splits)} (records: {ids})"
            )


def deduplicate(
    records: Iterable[AudioRecord], key: str = "audio_sha256"
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    """Keep the first record per fingerprint. Returns ``(kept, dropped)``."""
    seen: set[str] = set()
    kept: list[AudioRecord] = []
    dropped: list[AudioRecord] = []
    for record in records:
        fingerprint = getattr(record, key, None)
        if fingerprint and fingerprint in seen:
            dropped.append(record)
            continue
        if fingerprint:
            seen.add(fingerprint)
        kept.append(record)
    return kept, dropped


def duplicate_report(records: Iterable[AudioRecord], key: str = "audio_sha256") -> dict:
    """Summary for acceptance reports. Extend here for near-duplicate detection."""
    records = list(records)
    groups = find_duplicate_groups(records, key)
    cross_split = {
        fingerprint: sorted({record.split for record in group})
        for fingerprint, group in groups.items()
        if len({record.split for record in group}) > 1
    }
    return {
        "key": key,
        "records": len(records),
        "fingerprinted": sum(1 for record in records if getattr(record, key, None)),
        "duplicate_groups": len(groups),
        "duplicate_records": sum(len(group) for group in groups.values()),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_examples": dict(list(cross_split.items())[:5]),
        "ok": not cross_split,
    }
