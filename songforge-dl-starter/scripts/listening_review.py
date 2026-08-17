"""Ingest the human listening review and rank ACE-Step's weaknesses.

The listening scores are the only evidence we will have for the dimensions no
measurement can reach — realism, phrasing, emotion, whether a violin actually
sounds like a violin. This script refuses to invent them: an unfilled sheet is
an error, not a zero, because a fabricated weakness ranking would send the whole
fine-tuning phase after the wrong problem.

Objective measures from the benchmark are folded in only as corroboration, and
are reported alongside the human scores rather than blended into them.

    python scripts/listening_review.py --sheet benchmarks/listening_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Every dimension is 1-10, 10 best. `artifacts` is phrased as freedom from
#: artifacts so that "higher is better" holds everywhere without exception.
GENERAL_DIMENSIONS = (
    "overall", "realism", "instrument_presence", "arrangement",
    "prompt_adherence", "artifacts", "coherence",
)
VOCAL_DIMENSIONS = (
    "vocal_realism", "lyric_intelligibility", "phrasing", "pitch_stability", "emotion",
)
ALL_DIMENSIONS = GENERAL_DIMENSIONS + VOCAL_DIMENSIONS
VOCAL_TRACKS = ("vocal", "rich_mix")

#: Human-readable weakness names, so the ranking reads as a to-do list rather
#: than a column dump.
WEAKNESS_LABEL = {
    "overall": "overall quality low",
    "realism": "sounds synthetic / not like real instruments",
    "instrument_presence": "requested instruments not audible",
    "arrangement": "arrangement too simple / thin",
    "prompt_adherence": "output does not follow the prompt",
    "artifacts": "audible artifacts, noise or distortion",
    "coherence": "musically incoherent / does not hold together",
    "vocal_realism": "vocals sound synthetic",
    "lyric_intelligibility": "lyrics hard to make out",
    "phrasing": "vocal phrasing mechanical / segmented",
    "pitch_stability": "pitch unstable or drifting",
    "emotion": "performance flat, no emotional delivery",
}


def read_sheet(path: Path) -> dict[str, dict[str, float | None]]:
    """Read the review CSV. Blank cells stay None; they are not zeros."""
    rows: dict[str, dict[str, float | None]] = {}
    notes: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            track = (raw.get("id") or "").strip()
            if not track or track.startswith("_"):
                continue  # the direction/legend line
            scores: dict[str, float | None] = {}
            for dimension in ALL_DIMENSIONS:
                value = (raw.get(dimension) or "").strip()
                if not value:
                    scores[dimension] = None
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise SystemExit(f"{track}.{dimension}: {value!r} is not a number") from exc
                if not 1.0 <= number <= 10.0:
                    raise SystemExit(f"{track}.{dimension}: {number} is outside 1-10")
                scores[dimension] = number
            rows[track] = scores
            notes[track] = (raw.get("notes") or "").strip()
    for track, note in notes.items():
        rows[track]["_note"] = note  # type: ignore[assignment]
    return rows


def completeness(rows: dict[str, dict[str, Any]]) -> tuple[int, int, list[str]]:
    """How much of the sheet is filled, and what is still missing."""
    expected = 0
    filled = 0
    missing: list[str] = []
    for track, scores in rows.items():
        dimensions = ALL_DIMENSIONS if track in VOCAL_TRACKS else GENERAL_DIMENSIONS
        for dimension in dimensions:
            expected += 1
            if scores.get(dimension) is None:
                missing.append(f"{track}.{dimension}")
            else:
                filled += 1
    return filled, expected, missing


def rank_weaknesses(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank dimensions by how badly they scored, worst first.

    Ranked on the mean of the tracks that were actually scored for that
    dimension, with the worst single track carried along so a dimension that is
    fine everywhere except one track is not hidden by averaging.
    """
    ranked: list[dict[str, Any]] = []
    for dimension in ALL_DIMENSIONS:
        scored = {
            track: scores[dimension]
            for track, scores in rows.items()
            if scores.get(dimension) is not None
        }
        if not scored:
            continue
        worst_track = min(scored, key=lambda t: scored[t])
        ranked.append({
            "dimension": dimension,
            "label": WEAKNESS_LABEL[dimension],
            "mean": round(sum(scored.values()) / len(scored), 2),
            "worst_track": worst_track,
            "worst_score": scored[worst_track],
            "tracks_scored": len(scored),
            "per_track": {t: scored[t] for t in sorted(scored)},
        })
    ranked.sort(key=lambda row: (row["mean"], row["worst_score"]))
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the listening review.")
    parser.add_argument("--sheet", default=str(ROOT / "benchmarks" / "listening_review.csv"))
    parser.add_argument("--objective", default=None, help="objective_analysis.json from the benchmark")
    parser.add_argument("--output", default=None)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    rows = read_sheet(Path(args.sheet))
    filled, expected, missing = completeness(rows)

    if filled == 0:
        raise SystemExit(
            f"{args.sheet} has no scores yet.\n"
            "This script will not guess them: a fabricated weakness ranking would aim the "
            "entire fine-tuning phase at the wrong problem.\n"
            "Fill the sheet after listening, then re-run."
        )
    if missing:
        print(f"WARNING: {len(missing)}/{expected} cells unscored; ranking uses what exists.")
        print(f"  missing: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}\n")

    print(f"Listening review: {filled}/{expected} cells scored\n")

    print(f"{'track':<12}" + "".join(f"{d[:9]:>10}" for d in GENERAL_DIMENSIONS))
    for track in rows:
        cells = "".join(
            f"{rows[track][d]:>10.0f}" if rows[track].get(d) is not None else f"{'-':>10}"
            for d in GENERAL_DIMENSIONS
        )
        print(f"{track:<12}{cells}")

    print(f"\n{'track':<12}" + "".join(f"{d[:9]:>10}" for d in VOCAL_DIMENSIONS))
    for track in VOCAL_TRACKS:
        if track not in rows:
            continue
        cells = "".join(
            f"{rows[track][d]:>10.0f}" if rows[track].get(d) is not None else f"{'-':>10}"
            for d in VOCAL_DIMENSIONS
        )
        print(f"{track:<12}{cells}")

    ranked = rank_weaknesses(rows)
    print(f"\nTOP {args.top} WEAKNESSES (worst mean first)")
    for index, row in enumerate(ranked[: args.top], start=1):
        print(f"  {index}. {row['label']}")
        print(f"     mean {row['mean']}/10 across {row['tracks_scored']} tracks; "
              f"worst: {row['worst_track']} at {row['worst_score']:.0f}")

    if args.objective:
        objective = json.loads(Path(args.objective).read_text(encoding="utf-8"))
        flagged = {t: r.get("flags", []) for t, r in objective.items() if r.get("flags")}
        print("\nObjective corroboration (measured, not scored):")
        print(f"  {flagged or 'no objective flags raised'}")

    notes = {t: rows[t].get("_note") for t in rows if rows[t].get("_note")}
    if notes:
        print("\nListener notes:")
        for track, note in notes.items():
            print(f"  {track}: {note}")

    report = {
        "scored_cells": filled,
        "expected_cells": expected,
        "missing": missing,
        "ranked_weaknesses": ranked,
        "top_weaknesses": [r["dimension"] for r in ranked[: args.top]],
        "notes": notes,
    }
    out = Path(args.output) if args.output else Path(args.sheet).with_suffix(".summary.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
