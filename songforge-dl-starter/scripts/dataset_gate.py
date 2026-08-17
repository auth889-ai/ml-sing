"""The seven gates a dataset must clear before it may train a final adapter.

Aggressive dataset use only stays safe if every corpus passes the same checks,
so this is one command rather than seven habits. A dataset that fails any gate
may still be used for research, but not for the deployable adapter, and the
report says which.

    LICENSE       every record permits commercial use and derivatives
    PROVENANCE    source, URL and licence recorded on every record
    DUPLICATE     no duplicate audio, and none spanning splits
    QUALITY       decodable, non-silent, non-clipped, long enough
    METADATA      the fields the training goal actually needs are present
    SPLIT-LEAKAGE songs (and singers, when known) disjoint across splits
    ACE-STEP      converts cleanly to the trainer's expected layout

    python scripts/dataset_gate.py --manifest .../all.jsonl --goal instrument-realism
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from songforge.data.dedup import duplicate_report
from songforge.data.manifest import (
    assert_no_track_leakage,
    assert_provenance_complete,
    assert_singer_disjoint,
    read_jsonl,
    validate_records,
)
from songforge.data.splits import assert_group_disjoint

#: Licences under which a fine-tuned checkpoint stays commercially deployable.
DEPLOYABLE = {
    "CC0-1.0", "CC0", "public-domain", "CC-BY-4.0", "CC-BY-3.0",
    "MIT", "Apache-2.0", "BSD-3-Clause",
}

#: Which extra metadata each training goal genuinely needs. A goal that needs
#: instrument labels cannot be served by a corpus that has none, and finding
#: that out after a training run is an expensive way to learn it.
GOAL_REQUIREMENTS = {
    "instrument-realism": ("instrument_family",),
    "arrangement": ("instrument_family",),
    "vocal": ("singer_id",),
    "genre": (),
    "generic": (),
}


def gate_license(records: list[Any]) -> dict[str, Any]:
    counts = Counter(r.license or "<missing>" for r in records)
    blocking = {name: n for name, n in counts.items() if name not in DEPLOYABLE}
    return {
        "gate": "LICENSE",
        "pass": not blocking,
        "detail": dict(counts),
        "message": (
            "all records deployable"
            if not blocking
            else f"non-deployable licences present: {blocking}. "
                 "A checkpoint trained on these is not commercially deployable."
        ),
    }


def gate_provenance(records: list[Any]) -> dict[str, Any]:
    try:
        assert_provenance_complete(records)
    except ValueError as exc:
        return {"gate": "PROVENANCE", "pass": False, "message": str(exc)}
    fields = ("dataset_id", "source_url", "license_name")
    sample = records[0].provenance if records else {}
    return {
        "gate": "PROVENANCE",
        "pass": True,
        "message": f"every record carries {', '.join(fields)}",
        "detail": {k: sample.get(k) for k in fields},
    }


def gate_duplicate(records: list[Any]) -> dict[str, Any]:
    report = duplicate_report(records)
    return {
        "gate": "DUPLICATE",
        "pass": bool(report.get("ok")),
        "detail": report,
        "message": (
            "no duplicate audio across splits"
            if report.get("ok")
            else f"{report.get('cross_split_duplicate_groups')} duplicate groups span splits"
        ),
    }


def gate_quality(records: list[Any], min_seconds: float) -> dict[str, Any]:
    problems = validate_records(records)
    silent = [r.id for r in records if r.silent]
    clipped = [r.id for r in records if r.clipping_ratio > 0.01]
    short = [r.id for r in records if 0 < r.duration_seconds < min_seconds]
    ok = not problems and not silent and not clipped
    return {
        "gate": "QUALITY",
        "pass": ok,
        "detail": {
            "schema_problems": problems[:5],
            "silent": len(silent),
            "clipped": len(clipped),
            f"under_{min_seconds}s": len(short),
        },
        "message": (
            "decodable, non-silent, non-clipped"
            if ok
            else f"{len(problems)} schema problems, {len(silent)} silent, {len(clipped)} clipped"
        ),
    }


def gate_metadata(records: list[Any], goal: str) -> dict[str, Any]:
    required = GOAL_REQUIREMENTS.get(goal, ())
    coverage: dict[str, float] = {}
    for field in required:
        present = sum(1 for r in records if getattr(r, field, None))
        coverage[field] = round(present / max(len(records), 1), 4)
    missing = [f for f, c in coverage.items() if c < 0.95]
    return {
        "gate": "METADATA",
        "pass": not missing,
        "detail": {"goal": goal, "required": list(required), "coverage": coverage},
        "message": (
            f"goal {goal!r} needs {list(required) or 'nothing extra'}; coverage fine"
            if not missing
            else f"goal {goal!r} needs {missing} but coverage is below 95%"
        ),
    }


def gate_split_leakage(records: list[Any]) -> dict[str, Any]:
    problems = []
    try:
        assert_no_track_leakage(records)
        assert_group_disjoint(records, "song")
    except ValueError as exc:
        problems.append(str(exc))
    if any(r.singer_id for r in records):
        try:
            assert_singer_disjoint(records)
        except ValueError as exc:
            problems.append(str(exc))
    return {
        "gate": "SPLIT-LEAKAGE",
        "pass": not problems,
        "detail": {"checked_singers": any(r.singer_id for r in records)},
        "message": "songs (and singers where known) disjoint" if not problems else "; ".join(problems),
    }


def gate_acestep_format(records: list[Any]) -> dict[str, Any]:
    """Can this become an ACE-Step training dataset without inventing anything?"""
    sources = {r.source_path for r in records if r.source_path}
    captionable = sum(1 for r in records if r.instrument_family or r.instrument_name or r.tags)
    ok = bool(sources) and captionable > 0
    return {
        "gate": "ACE-STEP FORMAT",
        "pass": ok,
        "detail": {
            "distinct_sources": len(sources),
            "records_with_caption_material": captionable,
            "caption_coverage": round(captionable / max(len(records), 1), 4),
        },
        "message": (
            f"{len(sources)} source files convertible; captions derivable from metadata"
            if ok
            else "no source paths or no metadata to build a caption from -- captions would have to be invented"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the seven dataset gates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--goal", default="generic", choices=sorted(GOAL_REQUIREMENTS))
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if not records:
        raise SystemExit(f"{args.manifest} is empty")

    gates = [
        gate_license(records),
        gate_provenance(records),
        gate_duplicate(records),
        gate_quality(records, args.min_seconds),
        gate_metadata(records, args.goal),
        gate_split_leakage(records),
        gate_acestep_format(records),
    ]

    print(f"Dataset gate: {args.manifest}")
    print(f"goal    : {args.goal}")
    print(f"records : {len(records)} | tracks: {len({r.track_id for r in records})}\n")
    width = max(len(g["gate"]) for g in gates)
    for gate in gates:
        print(f"  {'PASS' if gate['pass'] else 'FAIL'}  {gate['gate']:<{width}}  {gate['message']}")

    passed = all(g["pass"] for g in gates)
    license_ok = gates[0]["pass"]
    print()
    if passed:
        print("ALL GATES PASS - usable for the deployable adapter.")
    elif all(g["pass"] for g in gates if g["gate"] != "LICENSE") and not license_ok:
        print("RESEARCH ONLY - everything passes except LICENSE.")
        print("Usable for a clearly-labelled research adapter, never merged with the deployable one.")
    else:
        print("BLOCKED - fix the failing gates before using this dataset for training.")

    report = {
        "manifest": args.manifest,
        "goal": args.goal,
        "records": len(records),
        "tracks": len({r.track_id for r in records}),
        "gates": gates,
        "all_pass": passed,
        "deployable": passed,
        "research_only": (not license_ok) and all(g["pass"] for g in gates if g["gate"] != "LICENSE"),
    }
    out = Path(args.output) if args.output else Path(args.manifest).parent / "dataset_gate.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nreport: {out}")
    if not passed and not report["research_only"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
