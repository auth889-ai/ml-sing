"""Select the highest-value FMA tracks for the SongForge V2 corpus.

Phase B rule: do NOT train on all 106k FMA tracks. Take the deployable
(CC0 + CC BY) census population and pick the best few thousand by explicit,
reproducible criteria — genre diversity, metadata richness, audio quality
proxies — so FMA adds breadth without drowning the rest of the mix.

    python scripts/select_fma_by_quality.py \
        --tracks-csv data_local/fma_metadata/tracks.csv \
        --genres-csv data_local/fma_metadata/genres.csv \
        --target 4000 \
        --output benchmarks/fma_v2_selection.json

Selection is deterministic: same inputs → same output (ties broken by
track_id). The output records every criterion per track so the corpus
manifest can carry provenance, and flags CC0 rows for re-verification
(the known ~14% misfiling issue).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from fma_license_report import classify, read_tracks  # same directory

DEPLOYABLE = ("cc0_public_domain", "cc_by")

#: genre_top buckets excluded outright: not music, or not trainable audio.
EXCLUDED_GENRES = {"Spoken"}

#: Cap on the share of the final selection that untagged-genre tracks may
#: take. Untagged tracks cannot contribute genre words to captions, so they
#: are worth less per training minute than tagged ones.
UNTAGGED_CAP = 0.15

MIN_DURATION_S = 30.0
MAX_DURATION_S = 600.0
MIN_BITRATE = 128_000  # below this the artefacts outweigh the diversity win


def load_genre_names(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    names: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gid = (row.get("genre_id") or "").strip()
            title = (row.get("title") or "").strip()
            if gid and title:
                names[gid] = title
    return names


def parse_genre_ids(raw: str) -> list[str]:
    """FMA lists fine genres like "[21, 76]"; tolerate any bracketed ints."""
    return [tok for tok in
            (piece.strip() for piece in raw.strip("[] ").split(","))
            if tok.isdigit()]


def score_track(row: dict, fine_genres: list[str]) -> tuple[float, dict]:
    """Value score plus the per-criterion breakdown we persist."""
    breakdown: dict[str, float] = {}

    genre_top = (row.get("track.genre_top") or "").strip()
    breakdown["has_genre_top"] = 2.0 if genre_top else 0.0
    breakdown["fine_genres"] = 0.5 * min(len(fine_genres), 5)

    language = (row.get("track.language_code") or "").strip()
    breakdown["has_language"] = 1.0 if language else 0.0  # weak vocals proxy

    try:
        bitrate = float(row.get("track.bit_rate") or 0)
    except ValueError:
        bitrate = 0.0
    breakdown["bitrate"] = 2.0 if bitrate >= 256_000 else (1.0 if bitrate >= 192_000 else 0.0)

    try:
        duration = float(row.get("track.duration") or 0)
    except ValueError:
        duration = 0.0
    breakdown["duration_sweet_spot"] = 1.0 if 60.0 <= duration <= 420.0 else 0.0

    try:
        listens = float(row.get("track.listens") or 0)
    except ValueError:
        listens = 0.0
    breakdown["popularity"] = min(math.log10(listens + 1.0), 2.0)

    return sum(breakdown.values()), breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="FMA V2 value-ranked selection.")
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--genres-csv", default=None)
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--output", default="benchmarks/fma_v2_selection.json")
    args = parser.parse_args()

    genre_names = load_genre_names(Path(args.genres_csv) if args.genres_csv else None)

    candidates: list[dict] = []
    rejected = {"license": 0, "duration": 0, "bitrate": 0, "genre_excluded": 0}
    for row in read_tracks(Path(args.tracks_csv)):
        bucket = classify(row.get("track.license", ""))
        if bucket not in DEPLOYABLE:
            rejected["license"] += 1
            continue
        try:
            duration = float(row.get("track.duration") or 0)
        except ValueError:
            duration = 0.0
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            rejected["duration"] += 1
            continue
        try:
            bitrate = float(row.get("track.bit_rate") or 0)
        except ValueError:
            bitrate = 0.0
        # missing bitrate metadata is unknown, not disqualifying; a stated
        # low bitrate is disqualifying.
        if 0 < bitrate < MIN_BITRATE:
            rejected["bitrate"] += 1
            continue
        genre_top = (row.get("track.genre_top") or "").strip()
        if genre_top in EXCLUDED_GENRES:
            rejected["genre_excluded"] += 1
            continue

        fine_ids = parse_genre_ids(row.get("track.genres_all") or "")
        score, breakdown = score_track(row, fine_ids)
        candidates.append({
            "track_id": (row.get("track_id") or "").strip(),
            "title": (row.get("track.title") or "").strip(),
            "artist": (row.get("artist.name") or "").strip(),
            "license": (row.get("track.license") or "").strip(),
            "license_bucket": bucket,
            "needs_license_reverify": bucket == "cc0_public_domain",
            "duration_seconds": duration,
            "bit_rate": bitrate or None,
            "genre_top": genre_top or None,
            "genres_all": [genre_names.get(g, g) for g in fine_ids],
            "language_code": (row.get("track.language_code") or "").strip() or None,
            "listens": row.get("track.listens") or None,
            "score": round(score, 3),
            "score_breakdown": breakdown,
        })

    # --- genre-balanced quota selection ----------------------------------
    # Bucket by genre_top; quota per bucket ∝ sqrt(bucket size) flattens the
    # Pop/Rock/Electronic dominance while leaving them the largest shares.
    buckets: dict[str, list[dict]] = {}
    for cand in candidates:
        buckets.setdefault(cand["genre_top"] or "(untagged)", []).append(cand)
    for members in buckets.values():
        members.sort(key=lambda c: (-c["score"], c["track_id"]))

    weights = {name: math.sqrt(len(members)) for name, members in buckets.items()}
    total_weight = sum(weights.values()) or 1.0
    quotas = {name: max(1, round(args.target * weight / total_weight))
              for name, weight in weights.items()}
    untagged_cap = int(args.target * UNTAGGED_CAP)
    if quotas.get("(untagged)", 0) > untagged_cap:
        quotas["(untagged)"] = untagged_cap

    selection: list[dict] = []
    for name, members in sorted(buckets.items()):
        selection.extend(members[: quotas[name]])
    # trim any rounding overshoot lowest-score-first, keep deterministic order
    selection.sort(key=lambda c: (-c["score"], c["track_id"]))
    selection = selection[: args.target]

    hours = sum(c["duration_seconds"] for c in selection) / 3600
    est_gb = sum(c["duration_seconds"] * (c["bit_rate"] or 320_000) / 8
                 for c in selection) / 1e9
    per_genre = {}
    for cand in selection:
        key = cand["genre_top"] or "(untagged)"
        per_genre[key] = per_genre.get(key, 0) + 1

    report = {
        "candidates_after_gates": len(candidates),
        "rejected": rejected,
        "selected": len(selection),
        "hours": round(hours, 1),
        "estimated_gb": round(est_gb, 1),
        "per_genre": dict(sorted(per_genre.items(), key=lambda kv: -kv[1])),
        "cc0_needing_reverify": sum(1 for c in selection if c["needs_license_reverify"]),
        "criteria": {
            "licenses": list(DEPLOYABLE),
            "duration_s": [MIN_DURATION_S, MAX_DURATION_S],
            "min_stated_bitrate": MIN_BITRATE,
            "excluded_genres": sorted(EXCLUDED_GENRES),
            "untagged_cap": UNTAGGED_CAP,
            "quota_rule": "per-genre quota proportional to sqrt(bucket size)",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "selection": selection}, indent=1),
                   encoding="utf-8")

    print(f"candidates after gates: {len(candidates)}  (rejected: {rejected})")
    print(f"selected: {len(selection)}  {report['hours']} h  ~{report['estimated_gb']} GB")
    for genre, count in report["per_genre"].items():
        print(f"  {genre:<24}{count:>6}")
    print(f"CC0 rows needing re-verification: {report['cc0_needing_reverify']}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
