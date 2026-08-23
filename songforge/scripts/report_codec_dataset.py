"""M04 — High-Quality Codec Optimization & Latent-Rate Selection: dataset report.

Describes an expanded M02-format corpus and independently re-checks the
invariants that make it usable for a codec comparison: whole songs on one side
of every split, no duplicate audio across splits, licence and provenance intact.

    python scripts/report_codec_dataset.py \
        --manifests "$SONGFORGE_DATA/processed/babyslakh_m04_expanded/manifests"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from songforge.data.dedup import duplicate_report
from songforge.data.manifest import (
    assert_no_track_leakage,
    assert_provenance_complete,
    read_jsonl,
)
from songforge.data.splits import assert_group_disjoint
from songforge.milestones import milestone


def describe(records: list[Any]) -> dict[str, Any]:
    tracks = {record.track_id for record in records}
    sources = {record.source_path for record in records if record.source_path}
    seconds = sum(record.duration_seconds for record in records)
    families = Counter(record.instrument_family or "unlabelled" for record in records)
    return {
        "tracks": len(tracks),
        "source_wavs": len(sources),
        "segments": len(records),
        "seconds": round(seconds, 2),
        "minutes": round(seconds / 60.0, 2),
        "instrument_families": dict(families.most_common()),
        "labelled_fraction": round(
            sum(1 for r in records if r.instrument_family) / max(len(records), 1), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{milestone('M04')} dataset report.")
    parser.add_argument("--manifests", required=True, help="Directory holding all.jsonl and split files.")
    parser.add_argument("--output", default=None, help="Where to write the JSON report.")
    args = parser.parse_args()

    manifests = Path(args.manifests)
    every = read_jsonl(manifests / "all.jsonl")
    splits = {
        name: read_jsonl(manifests / f"{name}.jsonl")
        for name in ("train", "val", "test")
        if (manifests / f"{name}.jsonl").exists()
    }

    overall = describe(every)
    total_seconds = overall["seconds"] or 1.0

    leakage: dict[str, Any] = {"song": None, "duplicates": None, "provenance": None}
    try:
        assert_no_track_leakage(every)
        assert_group_disjoint(every, "song")
    except ValueError as exc:
        leakage["song"] = str(exc)
    duplicates = duplicate_report(every)
    if not duplicates["ok"]:
        leakage["duplicates"] = f"{duplicates['cross_split_duplicate_groups']} cross-split duplicate groups"
    try:
        assert_provenance_complete(every)
    except ValueError as exc:
        leakage["provenance"] = str(exc)

    per_split = {}
    for name, records in splits.items():
        stats = describe(records)
        stats["share_of_seconds"] = round(100.0 * stats["seconds"] / total_seconds, 2)
        stats["share_of_segments"] = round(100.0 * stats["segments"] / max(overall["segments"], 1), 2)
        per_split[name] = stats

    report = {
        "manifests": str(manifests),
        "overall": overall,
        "splits": per_split,
        "leakage": leakage,
        "duplicates": duplicates,
        "clean": all(value is None for value in leakage.values()),
    }

    print(f"{milestone('M04')}: expanded dataset\n")
    print(
        f"tracks {overall['tracks']} | source wavs {overall['source_wavs']} | "
        f"segments {overall['segments']} | {overall['minutes']} min | "
        f"labelled {overall['labelled_fraction'] * 100:.1f}%"
    )
    print(f"\n{'split':>6}{'tracks':>8}{'wavs':>7}{'segments':>10}{'minutes':>10}{'seg %':>8}{'time %':>8}")
    for name in ("train", "val", "test"):
        if name not in per_split:
            continue
        s = per_split[name]
        print(
            f"{name:>6}{s['tracks']:>8}{s['source_wavs']:>7}{s['segments']:>10}"
            f"{s['minutes']:>10.2f}{s['share_of_segments']:>8.2f}{s['share_of_seconds']:>8.2f}"
        )
    print(f"\ninstrument families: {json.dumps(overall['instrument_families'])}")
    print(f"song leakage      : {leakage['song'] or 'none'}")
    print(f"duplicate leakage : {leakage['duplicates'] or 'none'}")
    print(f"provenance        : {leakage['provenance'] or 'complete'}")
    print(f"clean             : {report['clean']}")

    output = Path(args.output) if args.output else manifests.parent / "m04_dataset_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nreport: {output}")


if __name__ == "__main__":
    main()
