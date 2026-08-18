"""Ingest the human listening review, rank the five weakness areas, attribute
each to a likely cause, and choose exactly ONE first intervention.

The listening scores are the only evidence we will have for the dimensions no
measurement can reach — realism, phrasing, emotion, whether a violin actually
sounds like a violin. This script refuses to invent them: an unfilled sheet is
an error, not a zero, because a fabricated weakness ranking would send the whole
fine-tuning phase after the wrong problem.

Sheet conventions (benchmarks/listening_review.csv):
    1-10, 10 best, every dimension phrased so higher is better
    N/A   the dimension does not apply to this track (e.g. vocals on piano)
    blank not scored yet — never a zero

Objective measures from the benchmark are folded in only as corroboration, and
are reported alongside the human scores rather than blended into them.

    python scripts/listening_review.py --sheet benchmarks/listening_review.csv \
        --objective .../objective_analysis.json
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

#: The eight required dimensions, exactly as named in the sheet header.
REQUIRED_DIMENSIONS = (
    "overall_realism", "instrument_realism", "vocal_realism",
    "lyrics_intelligibility", "prompt_adherence", "structure_coherence",
    "spectral_clarity", "artifact_freedom",
)
#: Optional diagnostic dimensions — scored only when something stands out.
#: They sharpen the Case A/B selection but never gate completeness.
OPTIONAL_DIMENSIONS = (
    "instrument_presence", "arrangement", "phrasing", "pitch_stability", "emotion",
)
ALL_DIMENSIONS = REQUIRED_DIMENSIONS + OPTIONAL_DIMENSIONS
NOTES_COLUMN = "free_text_notes"

VOCAL_TRACKS = ("vocal", "rich_mix")
#: Sparse-acoustic prompts — the ones the objective rolloff flags implicated.
SPARSE_TRACKS = ("piano", "violin", "guitar")

NOT_APPLICABLE = "N/A"

#: The five weakness areas the fine-tuning phase chooses between, mapped to the
#: dimensions that evidence them. Area score = mean of every numeric cell of its
#: dimensions (area D restricts to the sparse tracks — muffling on a dense mix
#: is a different problem than muffling on a solo piano).
WEAKNESS_AREAS: dict[str, dict[str, Any]] = {
    "A": {
        "name": "instrument realism",
        "dimensions": ("instrument_realism", "instrument_presence", "arrangement"),
        "tracks": None,
        "case": "caseA_slakh_instrument",
    },
    "B": {
        "name": "vocal realism",
        "dimensions": ("vocal_realism", "phrasing", "pitch_stability", "emotion"),
        "tracks": VOCAL_TRACKS,
        "case": "caseB_vocal",
    },
    "C": {
        "name": "lyrics / structure / prompt controllability",
        "dimensions": ("lyrics_intelligibility", "prompt_adherence", "structure_coherence"),
        "tracks": None,
        "case": "caseC_control_experiments",
    },
    "D": {
        "name": "sparse-acoustic muffling / frequency loss",
        "dimensions": ("spectral_clarity",),
        "tracks": SPARSE_TRACKS,
        "case": "caseD_sparse_acoustic",
    },
    "E": {
        "name": "artifacts / general audio quality",
        "dimensions": ("artifact_freedom", "overall_realism"),
        "tracks": None,
        "case": None,  # no prepared training case: see attribution
    },
}

#: An area mean below this is treated as a real weakness worth intervening on.
WEAK_THRESHOLD = 7.0
#: Below this, a deficit is so uniform that a small adapter may not close it.
SEVERE_THRESHOLD = 4.0
#: Minimum numeric cells before an area ranking is trusted.
MIN_EVIDENCE_CELLS = 3


def read_sheet(path: Path) -> dict[str, dict[str, Any]]:
    """Read the review CSV. Blank stays None (unscored); N/A stays NOT_APPLICABLE."""
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            track = (raw.get("id") or "").strip()
            if not track or track.startswith("_"):
                continue  # the direction/legend line
            scores: dict[str, Any] = {}
            for dimension in ALL_DIMENSIONS:
                value = (raw.get(dimension) or "").strip()
                if not value:
                    scores[dimension] = None
                    continue
                if value.upper().replace(" ", "") in ("N/A", "NA"):
                    scores[dimension] = NOT_APPLICABLE
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise SystemExit(f"{track}.{dimension}: {value!r} is not a number or N/A") from exc
                if not 1.0 <= number <= 10.0:
                    raise SystemExit(f"{track}.{dimension}: {number} is outside 1-10")
                scores[dimension] = number
            scores["_note"] = (raw.get(NOTES_COLUMN) or "").strip()
            rows[track] = scores
    return rows


def completeness(rows: dict[str, dict[str, Any]]) -> tuple[int, int, list[str]]:
    """Scored / applicable-required cells, and what is still missing.

    N/A cells are resolved (the listener made a decision); blanks are missing.
    Optional dimensions never count against completeness.
    """
    expected = 0
    filled = 0
    missing: list[str] = []
    for track, scores in rows.items():
        for dimension in REQUIRED_DIMENSIONS:
            value = scores.get(dimension)
            if value == NOT_APPLICABLE:
                continue
            expected += 1
            if value is None:
                missing.append(f"{track}.{dimension}")
            else:
                filled += 1
    return filled, expected, missing


def numeric(rows: dict[str, dict[str, Any]], dimension: str,
            tracks: tuple[str, ...] | None = None) -> dict[str, float]:
    """The numeric scores for a dimension, optionally restricted to tracks."""
    return {
        track: scores[dimension]
        for track, scores in rows.items()
        if isinstance(scores.get(dimension), float) and (tracks is None or track in tracks)
    }


def rank_dimensions(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-dimension ranking, worst mean first, worst single track carried along."""
    ranked: list[dict[str, Any]] = []
    for dimension in ALL_DIMENSIONS:
        scored = numeric(rows, dimension)
        if not scored:
            continue
        worst_track = min(scored, key=lambda t: scored[t])
        ranked.append({
            "dimension": dimension,
            "mean": round(sum(scored.values()) / len(scored), 2),
            "worst_track": worst_track,
            "worst_score": scored[worst_track],
            "tracks_scored": len(scored),
            "per_track": {t: scored[t] for t in sorted(scored)},
        })
    ranked.sort(key=lambda row: (row["mean"], row["worst_score"]))
    return ranked


def rank_areas(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Score the five weakness areas, worst mean first."""
    areas: list[dict[str, Any]] = []
    for key, spec in WEAKNESS_AREAS.items():
        cells: dict[str, float] = {}
        for dimension in spec["dimensions"]:
            for track, value in numeric(rows, dimension, spec["tracks"]).items():
                cells[f"{track}.{dimension}"] = value
        area: dict[str, Any] = {
            "area": key,
            "name": spec["name"],
            "cells_scored": len(cells),
            "sufficient": len(cells) >= MIN_EVIDENCE_CELLS,
            "mean": round(sum(cells.values()) / len(cells), 2) if cells else None,
            "worst_cell": min(cells, key=lambda c: cells[c]) if cells else None,
            "per_cell": {c: cells[c] for c in sorted(cells)},
        }
        areas.append(area)
    # Insufficient-evidence areas sink to the bottom rather than ranking on noise.
    areas.sort(key=lambda a: (not a["sufficient"], a["mean"] if a["mean"] is not None else 99.0))
    return areas


def attribute(area: dict[str, Any], rows: dict[str, dict[str, Any]],
              objective: dict[str, Any] | None) -> dict[str, Any]:
    """Attribute one weakness area to its likely cause bucket, with the reason.

    Buckets: prompt/control-layer problem, pretrained-model limitation,
    LoRA-addressable, insufficient evidence. Rules are deliberately simple and
    printed with their evidence — an unexplained classification would be a
    fabricated one.
    """
    key = area["area"]
    mean = area["mean"]
    if not area["sufficient"]:
        return {"bucket": "insufficient evidence",
                "reason": f"only {area['cells_scored']} scored cells (need {MIN_EVIDENCE_CELLS})"}
    if mean is not None and mean >= WEAK_THRESHOLD:
        return {"bucket": "not a weakness",
                "reason": f"mean {mean} is at or above the {WEAK_THRESHOLD} threshold"}

    if key == "A":
        presence = numeric(rows, "instrument_presence")
        realism = numeric(rows, "instrument_realism")
        if presence and realism:
            gap = (sum(realism.values()) / len(realism)) - (sum(presence.values()) / len(presence))
            if gap > 1.5:
                return {"bucket": "prompt/control-layer problem",
                        "reason": ("requested instruments are missing (presence "
                                   f"{gap:.1f} below realism) — caption phrasing first, then Case A")}
        return {"bucket": "LoRA-addressable",
                "reason": "instrument rendering weakness; Case A (Slakh-100 stems + captions)"}

    if key == "B":
        if mean is not None and mean < SEVERE_THRESHOLD:
            return {"bucket": "pretrained-model limitation (suspected)",
                    "reason": (f"uniformly severe vocal deficit (mean {mean} < {SEVERE_THRESHOLD}); "
                               "a small adapter may not close it — Case B is still the test")}
        return {"bucket": "LoRA-addressable",
                "reason": "vocal timbre/expressiveness weakness; Case B (VocalSet stack)"}

    if key == "C":
        note = ""
        recall = _whisper_recall(objective)
        intelligibility = numeric(rows, "lyrics_intelligibility")
        if recall and intelligibility and min(intelligibility.values()) < WEAK_THRESHOLD:
            low = min(recall.values())
            if low >= 0.90:
                note = (f" (Whisper already recovers {low:.0%} of words — words are present "
                        "but poorly articulated; if Case C cannot move this, it escalates to Case B)")
        return {"bucket": "prompt/control-layer problem",
                "reason": "controllability weaknesses get the free Case C prompt-side ablation "
                          "before any GPU hour is spent" + note}

    if key == "D":
        dense = numeric(rows, "spectral_clarity",
                        tuple(t for t in rows if t not in SPARSE_TRACKS))
        corroborated = _rolloff_flagged(objective)
        if dense and (sum(dense.values()) / len(dense)) < WEAK_THRESHOLD:
            return {"bucket": "pretrained-model limitation (suspected)",
                    "reason": "clarity is low on dense mixes too — not the sparse-material "
                              "band-limit; Case D would target the wrong thing"}
        reason = "muffling confined to sparse acoustic material; Case D (narrow stem subset)"
        if corroborated:
            reason += " — corroborated by the measured 95% rolloff flags"
        return {"bucket": "LoRA-addressable", "reason": reason}

    # E — artifacts / general audio quality. There is no prepared training case
    # on purpose: artifacts usually trace to inference settings or the decoder,
    # not to anything a small adapter learns away.
    return {"bucket": "prompt/control-layer problem",
            "reason": "sweep inference settings first (steps/shift/dtype); if artifacts "
                      "survive that sweep, they are a pretrained/decoder limitation"}


def _whisper_recall(objective: dict[str, Any] | None) -> dict[str, float]:
    if not objective:
        return {}
    return {
        track: report["lyric_recall"]
        for track, report in objective.items()
        if isinstance(report, dict) and isinstance(report.get("lyric_recall"), (int, float))
    }


def _rolloff_flagged(objective: dict[str, Any] | None) -> bool:
    if not objective:
        return False
    return any(
        any("rolloff" in str(flag) or "band" in str(flag) for flag in report.get("flags", []))
        for report in objective.values()
        if isinstance(report, dict)
    )


def choose_intervention(areas: list[dict[str, Any]],
                        attributions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Exactly ONE first intervention, by the rule fixed before scores existed.

    1. Rank areas worst-first among those with sufficient evidence and a mean
       below the weakness threshold.
    2. If the worst is C — or C is weak at all — the first intervention is the
       free Case C prompt-side ablation. GPU hours are never spent on a problem
       a prompt change might fix.
    3. Otherwise the worst area's case runs (A/B/D). Area E maps to an
       inference-settings sweep, not training.
    """
    weak = [a for a in areas
            if a["sufficient"] and a["mean"] is not None and a["mean"] < WEAK_THRESHOLD]
    if not weak:
        return {"intervention": "none",
                "reason": "no area scored below the weakness threshold with sufficient "
                          "evidence — no training is justified by this review"}

    worst = weak[0]
    c_weak = next((a for a in weak if a["area"] == "C"), None)
    if c_weak is not None:
        follow = next((a for a in weak if a["area"] != "C"), None)
        return {
            "intervention": "caseC_control_experiments",
            "reason": (f"area C is weak (mean {c_weak['mean']}); prompt-side experiments are "
                       "near-free and run before any training"),
            "training_candidate_after": (
                {"area": follow["area"], "name": follow["name"], "mean": follow["mean"],
                 "case": WEAKNESS_AREAS[follow["area"]]["case"]}
                if follow else None),
        }

    case = WEAKNESS_AREAS[worst["area"]]["case"]
    if case is None:  # area E
        return {"intervention": "inference_settings_sweep",
                "reason": (f"worst area is E (mean {worst['mean']}); artifacts get a settings "
                           "sweep, not a LoRA — see attribution"),
                "training_candidate_after": None}
    return {"intervention": case,
            "reason": (f"worst area with sufficient evidence is {worst['area']} — "
                       f"{worst['name']} (mean {worst['mean']}); "
                       f"attribution: {attributions[worst['area']]['bucket']}"),
            "training_candidate_after": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the listening review.")
    parser.add_argument("--sheet", default=str(ROOT / "benchmarks" / "listening_review.csv"))
    parser.add_argument("--objective", default=None, help="objective_analysis.json from the benchmark")
    parser.add_argument("--output", default=None)
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
        print(f"WARNING: {len(missing)}/{expected} required cells unscored; using what exists.")
        print(f"  missing: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}\n")

    print(f"Listening review: {filled}/{expected} applicable required cells scored\n")

    def cell(track: str, dimension: str) -> str:
        value = rows[track].get(dimension)
        if value is None:
            return f"{'-':>10}"
        if value == NOT_APPLICABLE:
            return f"{'n/a':>10}"
        return f"{value:>10.0f}"

    print(f"{'track':<12}" + "".join(f"{d[:9]:>10}" for d in REQUIRED_DIMENSIONS))
    for track in rows:
        print(f"{track:<12}" + "".join(cell(track, d) for d in REQUIRED_DIMENSIONS))

    scored_optional = [d for d in OPTIONAL_DIMENSIONS if numeric(rows, d)]
    if scored_optional:
        print(f"\n{'track':<12}" + "".join(f"{d[:9]:>10}" for d in scored_optional))
        for track in rows:
            print(f"{track:<12}" + "".join(cell(track, d) for d in scored_optional))

    objective = None
    if args.objective:
        objective = json.loads(Path(args.objective).read_text(encoding="utf-8"))

    areas = rank_areas(rows)
    attributions = {a["area"]: attribute(a, rows, objective) for a in areas}

    print("\nWEAKNESS AREAS (worst mean first)")
    for area in areas:
        mean = f"{area['mean']}/10" if area["mean"] is not None else "unscored"
        print(f"  {area['area']}. {area['name']}: {mean} over {area['cells_scored']} cells"
              + ("" if area["sufficient"] else "  [insufficient evidence]"))
        verdict = attributions[area["area"]]
        print(f"     -> {verdict['bucket']}: {verdict['reason']}")

    decision = choose_intervention(areas, attributions)
    print(f"\nFIRST INTERVENTION: {decision['intervention']}")
    print(f"  {decision['reason']}")
    if decision.get("training_candidate_after"):
        candidate = decision["training_candidate_after"]
        print(f"  training candidate after Case C: {candidate['case']} "
              f"(area {candidate['area']}, mean {candidate['mean']})")

    if objective is not None:
        flagged = {t: r.get("flags", []) for t, r in objective.items()
                   if isinstance(r, dict) and r.get("flags")}
        print("\nObjective corroboration (measured, not scored):")
        print(f"  {flagged or 'no objective flags raised'}")

    notes = {t: rows[t]["_note"] for t in rows if rows[t].get("_note")}
    if notes:
        print("\nListener notes:")
        for track, note in notes.items():
            print(f"  {track}: {note}")

    report = {
        "scored_cells": filled,
        "expected_cells": expected,
        "missing": missing,
        "ranked_dimensions": rank_dimensions(rows),
        "weakness_areas": areas,
        "attribution": attributions,
        "first_intervention": decision,
        "notes": notes,
    }
    out = Path(args.output) if args.output else Path(args.sheet).with_suffix(".summary.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nreport: {out}")
    print("Next step: fill benchmarks/EXPERIMENT_CARD.md from this report. "
          "Nothing is downloaded or trained until that card is complete.")


if __name__ == "__main__":
    main()
