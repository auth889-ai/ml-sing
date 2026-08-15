"""M02 canonical audio preprocessing pipeline.

    raw file
      -> media validation (corrupt / empty / unreadable are skipped, not fatal)
      -> decode
      -> channel policy
      -> resample
      -> amplitude normalization
      -> deterministic segmentation
      -> silence and clipping filtering
      -> canonical AudioRecord per segment

Rerunning with the same config and the same inputs reproduces byte-identical
manifests: ids derive from content coordinates rather than iteration order, and
files are always processed in sorted path order.

Splits are NOT assigned here. Preprocessing emits records with
``split="unassigned"``; `songforge.data.splits.assign_splits` assigns them
group-disjointly afterwards. Keeping those steps apart is what makes it possible
to re-split a corpus without re-decoding it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .audio import write_wav
from .dedup import file_sha256, waveform_sha256
from .dsp import (
    compute_stats,
    extract_segment,
    normalize_amplitude,
    resample_waveform,
    segment_bounds,
    to_channels,
)
from .manifest import AudioRecord, segment_id
from .media import AudioValidationError, decode_audio, validate_audio_file

PREPROCESS_VERSION = "m02.v1"
UNASSIGNED_SPLIT = "unassigned"

_SLAKH_TRACK = re.compile(r"^Track\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class PreprocessConfig:
    """Every knob that changes preprocessing output. Serialized into each record."""

    sample_rate: int = 24000
    channels: int = 1
    channel_policy: str = "mean"

    segment_seconds: float = 2.0
    hop_seconds: float | None = None
    pad_final_partial: bool = False

    normalize: str = "peak"  # peak | rms | none
    target_dbfs: float = -1.0
    max_gain_db: float = 30.0

    silence_threshold_dbfs: float = -60.0
    drop_silent: bool = True

    clipping_threshold: float = 0.999
    max_clipping_ratio: float = 0.01
    drop_clipped: bool = False

    min_source_seconds: float = 0.0
    max_source_seconds: float | None = None

    resample_filter_width: int = 6
    resample_rolloff: float = 0.99

    write_audio: bool = True
    compute_source_hash: bool = True

    @property
    def segment_samples(self) -> int:
        return max(round(self.segment_seconds * self.sample_rate), 1)

    @property
    def hop_samples(self) -> int:
        if self.hop_seconds is None:
            return self.segment_samples
        return max(round(self.hop_seconds * self.sample_rate), 1)

    def to_dict(self) -> dict:
        payload = {
            "version": PREPROCESS_VERSION,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "channel_policy": self.channel_policy,
            "segment_seconds": self.segment_seconds,
            "hop_seconds": self.hop_seconds,
            "pad_final_partial": self.pad_final_partial,
            "normalize": self.normalize,
            "target_dbfs": self.target_dbfs,
            "silence_threshold_dbfs": self.silence_threshold_dbfs,
            "drop_silent": self.drop_silent,
            "clipping_threshold": self.clipping_threshold,
            "max_clipping_ratio": self.max_clipping_ratio,
            "resample_filter_width": self.resample_filter_width,
            "resample_rolloff": self.resample_rolloff,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict | None) -> PreprocessConfig:
        known = {
            "sample_rate", "channels", "channel_policy", "segment_seconds", "hop_seconds",
            "pad_final_partial", "normalize", "target_dbfs", "max_gain_db",
            "silence_threshold_dbfs", "drop_silent", "clipping_threshold", "max_clipping_ratio",
            "drop_clipped", "min_source_seconds", "max_source_seconds",
            "resample_filter_width", "resample_rolloff", "write_audio", "compute_source_hash",
        }
        return cls(**{key: value for key, value in (payload or {}).items() if key in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw.get("preprocess", raw))


@dataclass
class PreprocessResult:
    records: list[AudioRecord] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"records": len(self.records), "skipped": self.skipped, "stats": self.stats}


def provenance_from_registry(registry: Any, dataset_id: str) -> dict:
    """Build the provenance block that every record carries.

    Accepts a DatasetRegistry, a raw ``{"datasets": {...}}`` mapping, or a bare
    dataset spec, so callers are not forced to load the full registry in tests.
    """
    spec: dict
    datasets = getattr(registry, "datasets", None)
    if isinstance(datasets, dict):
        spec = datasets.get(dataset_id, {})
    elif isinstance(registry, dict) and "datasets" in registry:
        spec = registry["datasets"].get(dataset_id, {})
    elif isinstance(registry, dict):
        spec = registry
    else:
        spec = {}

    license_spec = spec.get("license", {}) or {}
    access = spec.get("access", {}) or {}
    return {
        "dataset_id": dataset_id,
        "dataset_name": spec.get("name", dataset_id),
        "role": spec.get("role"),
        "source_url": spec.get("source_url"),
        "citation_url": spec.get("citation_url"),
        "license_name": license_spec.get("name"),
        "license_url": license_spec.get("url"),
        "commercial_allowed": license_spec.get("commercial_allowed"),
        "requires_user_acceptance": bool(
            license_spec.get("requires_user_acceptance") or access.get("gated")
        ),
        "access_method": access.get("method"),
    }


def derive_track_id(path: str | Path, dataset_id: str = "", root: str | Path | None = None) -> str:
    """The source-song id. Segments of one song must share it.

    Slakh-family layouts key on the ``TrackXXXXX`` directory so that every stem
    and the mix of one song land in the same split.
    """
    path = Path(path)
    if dataset_id in ("babyslakh", "slakh2100"):
        for parent in path.parents:
            if _SLAKH_TRACK.match(parent.name):
                return parent.name
    if path.stem in ("mix", "audio", "song") and path.parent.name:
        return path.parent.name
    if root is not None:
        try:
            relative = path.relative_to(Path(root))
        except ValueError:
            relative = path
        return relative.with_suffix("").as_posix().replace("/", "__")
    return path.stem


def derive_singer_id(path: str | Path, dataset_id: str = "") -> str | None:
    """Performer id where the layout encodes one. None when unknown.

    GTSinger nests as ``<language>/<singer>/<song>/...``; other corpora in the
    registry are instrumental or unlabelled, so they return None and singer-
    disjoint splitting falls back to song-disjoint.
    """
    path = Path(path)
    if dataset_id == "gtsinger":
        parts = path.parts
        if len(parts) >= 4:
            return f"{parts[-4]}__{parts[-3]}"
        if len(parts) >= 3:
            return parts[-3]
    return None


def normalize_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    """Lowercase, strip, drop empties, de-duplicate, sort. Metadata normalization."""
    if not tags:
        return ()
    cleaned = {str(tag).strip().lower().replace(" ", "_") for tag in tags}
    return tuple(sorted(tag for tag in cleaned if tag))


def preprocess_file(
    path: str | Path,
    config: PreprocessConfig,
    provenance: dict,
    output_dir: str | Path,
    dataset_id: str = "",
    track_id: str | None = None,
    singer_id: str | None = None,
    tags: Iterable[str] | None = None,
    source_root: str | Path | None = None,
) -> tuple[list[AudioRecord], list[dict]]:
    """Preprocess one file into canonical segment records.

    Returns ``(records, skipped)``. A corrupt or unusable file yields no records
    and one skip entry rather than raising, so one bad file cannot abort a corpus.
    """
    path = Path(path)
    output_dir = Path(output_dir)
    skipped: list[dict] = []

    report = validate_audio_file(
        path,
        min_duration_seconds=max(config.min_source_seconds, 0.0),
        max_duration_seconds=config.max_source_seconds,
        require_decode=False,
    )
    if not report.ok:
        return [], [{"path": str(path), "stage": "validation", "reasons": report.reasons}]

    try:
        waveform, source_rate = decode_audio(path)
    except AudioValidationError as exc:
        return [], [{"path": str(path), "stage": "decode", "reasons": [str(exc)]}]

    waveform = to_channels(waveform, config.channels, config.channel_policy)
    if source_rate != config.sample_rate:
        waveform = resample_waveform(
            waveform,
            source_rate,
            config.sample_rate,
            lowpass_filter_width=config.resample_filter_width,
            rolloff=config.resample_rolloff,
        )

    source_stats = compute_stats(waveform, config.clipping_threshold)
    if config.drop_clipped and source_stats.clipping_ratio > config.max_clipping_ratio:
        return [], [
            {
                "path": str(path),
                "stage": "clipping",
                "reasons": [f"clipping ratio {source_stats.clipping_ratio:.4f} exceeds {config.max_clipping_ratio}"],
            }
        ]

    waveform, gain_db = normalize_amplitude(
        waveform, config.normalize, config.target_dbfs, config.max_gain_db
    )

    duration = waveform.size(-1) / config.sample_rate
    if duration < config.min_source_seconds:
        return [], [
            {"path": str(path), "stage": "duration", "reasons": [f"decoded duration {duration:.3f}s too short"]}
        ]

    resolved_track = track_id or derive_track_id(path, dataset_id, source_root)
    resolved_singer = singer_id if singer_id is not None else derive_singer_id(path, dataset_id)
    normalized_tags = normalize_tags(tags)
    source_hash = file_sha256(path) if config.compute_source_hash else None

    preprocessing = config.to_dict()
    preprocessing.update(
        {
            "source_sample_rate": source_rate,
            "source_channels": int(report.info.channels or 0),
            "source_codec": report.info.codec_name,
            "prober": report.info.prober,
            "applied_gain_db": round(gain_db, 6),
            "source_peak_dbfs": round(source_stats.peak_dbfs, 6),
            "source_clipping_ratio": round(source_stats.clipping_ratio, 8),
        }
    )

    bounds = segment_bounds(
        waveform.size(-1), config.segment_samples, config.hop_samples, config.pad_final_partial
    )
    if not bounds:
        return [], [
            {
                "path": str(path),
                "stage": "segmentation",
                "reasons": [f"{duration:.3f}s shorter than one {config.segment_seconds}s segment"],
            }
        ]

    records: list[AudioRecord] = []
    for index, (start, end) in enumerate(bounds):
        segment = extract_segment(waveform, start, end)
        stats = compute_stats(segment, config.clipping_threshold)
        silent = stats.rms_dbfs < config.silence_threshold_dbfs

        if silent and config.drop_silent:
            skipped.append(
                {
                    "path": str(path),
                    "stage": "silence",
                    "segment_index": index,
                    "reasons": [f"segment rms {stats.rms_dbfs:.1f} dBFS below {config.silence_threshold_dbfs} dBFS"],
                }
            )
            continue

        record_id = segment_id(dataset_id, resolved_track, str(path), index, start)
        destination = output_dir / "audio" / resolved_track / f"{record_id}.wav"
        if config.write_audio:
            write_wav(destination, segment, config.sample_rate)

        records.append(
            AudioRecord(
                id=record_id,
                path=str(destination),
                split=UNASSIGNED_SPLIT,
                source=dataset_id or provenance.get("dataset_id", ""),
                license=provenance.get("license_name") or "unknown",
                track_id=resolved_track,
                singer_id=resolved_singer,
                tags=normalized_tags,
                source_path=str(path),
                segment_index=index,
                start_sample=start,
                num_samples=int(segment.size(-1)),
                sample_rate=config.sample_rate,
                channels=int(segment.size(0)),
                duration_seconds=round(segment.size(-1) / config.sample_rate, 9),
                peak=round(stats.peak, 8),
                rms=round(stats.rms, 8),
                peak_dbfs=round(stats.peak_dbfs, 6),
                rms_dbfs=round(stats.rms_dbfs, 6),
                clipping_ratio=round(stats.clipping_ratio, 8),
                silent=silent,
                audio_sha256=waveform_sha256(segment, config.sample_rate),
                source_sha256=source_hash,
                provenance=dict(provenance),
                preprocessing=preprocessing,
            )
        )

    return records, skipped


def preprocess_paths(
    paths: Sequence[str | Path],
    config: PreprocessConfig,
    provenance: dict,
    output_dir: str | Path,
    dataset_id: str = "",
    source_root: str | Path | None = None,
    tags: Iterable[str] | None = None,
) -> PreprocessResult:
    """Preprocess many files in sorted order so output is order-independent."""
    output_dir = Path(output_dir)
    ordered = sorted({Path(path) for path in paths}, key=lambda item: item.as_posix())

    result = PreprocessResult()
    for path in ordered:
        records, skipped = preprocess_file(
            path,
            config=config,
            provenance=provenance,
            output_dir=output_dir,
            dataset_id=dataset_id,
            source_root=source_root,
            tags=tags,
        )
        result.records.extend(records)
        result.skipped.extend(skipped)

    result.stats = {
        "dataset_id": dataset_id,
        "input_files": len(ordered),
        "skipped_files": len({entry["path"] for entry in result.skipped if entry["stage"] != "silence"}),
        "skipped_entries": len(result.skipped),
        "segments": len(result.records),
        "tracks": len({record.track_id for record in result.records}),
        "singers": len({record.singer_id for record in result.records if record.singer_id}),
        "total_seconds": round(sum(record.duration_seconds for record in result.records), 6),
        "sample_rate": config.sample_rate,
        "channels": config.channels,
        "preprocess_version": PREPROCESS_VERSION,
    }
    return result


def find_audio_files(root: str | Path, patterns: Sequence[str] = ("*.wav", "*.flac")) -> list[Path]:
    """Recursively collect audio files under ``root`` in deterministic order."""
    root = Path(root)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(root.rglob(pattern))
    return sorted(found, key=lambda item: item.as_posix())


def write_preprocess_report(result: PreprocessResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "PREPROCESS_VERSION",
    "UNASSIGNED_SPLIT",
    "PreprocessConfig",
    "PreprocessResult",
    "derive_singer_id",
    "derive_track_id",
    "find_audio_files",
    "normalize_tags",
    "preprocess_file",
    "preprocess_paths",
    "provenance_from_registry",
    "write_preprocess_report",
]
