"""FMA license census — exact per-license counts BEFORE any audio download.

Reads the official ``tracks.csv`` from ``fma_metadata.zip`` (metadata only,
~342 MB) and reports, per license bucket: track count, hours, and estimated
audio GB — so the deployable-corpus decision is made on numbers, not vibes.

Deployable default: CC0/public-domain and plain CC BY. CC BY-SA is counted
and flagged separately (share-alike scope needs a decision, not a default).
Anything NC or ND is never in the deployable corpus.

Honesty notes baked into the output:
  - hours/GB come from per-track duration and bit_rate metadata; both are
    artist-supplied and imperfect.
  - a known audit found ~14% of CC0-labelled FMA items were later relicensed
    NC; high-stakes tracks must be re-verified against freemusicarchive.org
    before deployable training.
  - tracks.csv has NO instrument tags. Instrument coverage cannot be computed
    from metadata; genre_top and language_code are the only proxies. Real
    instrument/vocal coverage is measured after audio-side tagging.

    python scripts/fma_license_report.py --tracks-csv fma_metadata/tracks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BUCKETS = ("cc0_public_domain", "cc_by", "cc_by_sa", "cc_by_nc", "cc_by_nd",
           "other_unknown")

#: Default deployable buckets. BY-SA is deliberately NOT here.
DEPLOYABLE = ("cc0_public_domain", "cc_by")

FALLBACK_BITRATE = 320_000  # bits/s, conservative upper bound when missing


def classify(license_text: str) -> str:
    """Bucket one FMA license string (name or URL). NC/ND checked first."""
    text = (license_text or "").strip().lower()
    if not text:
        return "other_unknown"
    compact = text.replace("_", "-")
    if "by-nc" in compact or "noncommercial" in compact or "non-commercial" in compact:
        return "cc_by_nc"
    if "by-nd" in compact or "noderiv" in compact or "no derivatives" in compact:
        return "cc_by_nd"
    if "by-sa" in compact or "sharealike" in compact or "share alike" in compact:
        return "cc_by_sa"
    if "zero" in compact or "cc0" in compact or "public domain" in compact or "publicdomain" in compact:
        return "cc0_public_domain"
    if "/by/" in compact or "attribution" in compact or compact in ("cc by", "cc-by"):
        return "cc_by"
    return "other_unknown"


def read_tracks(path: Path):
    """Yield dicts from FMA's three-row-header tracks.csv, stdlib only."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        groups = next(reader)
        fields = next(reader)
        columns = [
            f"{group}.{field}" if field else (group or "track_id")
            for group, field in zip(groups, fields)
        ]
        columns[0] = "track_id"
        third = next(reader)  # index-label row ("track_id,,,...") in real files
        if any(cell.strip() for cell in third[1:]):
            yield dict(zip(columns, third))  # synthetic files may omit it
        for row in reader:
            if row:
                yield dict(zip(columns, row))


def main() -> None:
    parser = argparse.ArgumentParser(description="FMA per-license census.")
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--output", default="fma_license_report.json")
    parser.add_argument("--top-genres", type=int, default=20)
    args = parser.parse_args()

    stats = {bucket: {"tracks": 0, "seconds": 0.0, "bytes": 0.0} for bucket in BUCKETS}
    genres: dict[str, dict[str, int]] = {bucket: {} for bucket in BUCKETS}
    languages: dict[str, dict[str, int]] = {bucket: {} for bucket in BUCKETS}
    total = 0

    for row in read_tracks(Path(args.tracks_csv)):
        total += 1
        bucket = classify(row.get("track.license", ""))
        try:
            seconds = float(row.get("track.duration") or 0)
        except ValueError:
            seconds = 0.0
        try:
            bitrate = float(row.get("track.bit_rate") or 0) or FALLBACK_BITRATE
        except ValueError:
            bitrate = FALLBACK_BITRATE
        stats[bucket]["tracks"] += 1
        stats[bucket]["seconds"] += seconds
        stats[bucket]["bytes"] += seconds * bitrate / 8
        genre = (row.get("track.genre_top") or "").strip() or "(untagged)"
        genres[bucket][genre] = genres[bucket].get(genre, 0) + 1
        language = (row.get("track.language_code") or "").strip() or "(none)"
        languages[bucket][language] = languages[bucket].get(language, 0) + 1

    def rollup(buckets) -> dict:
        tracks = sum(stats[b]["tracks"] for b in buckets)
        seconds = sum(stats[b]["seconds"] for b in buckets)
        size = sum(stats[b]["bytes"] for b in buckets)
        merged_genres: dict[str, int] = {}
        merged_languages: dict[str, int] = {}
        for b in buckets:
            for genre, count in genres[b].items():
                merged_genres[genre] = merged_genres.get(genre, 0) + count
            for language, count in languages[b].items():
                merged_languages[language] = merged_languages.get(language, 0) + count
        top = dict(sorted(merged_genres.items(), key=lambda kv: -kv[1])[: args.top_genres])
        langs = dict(sorted(merged_languages.items(), key=lambda kv: -kv[1])[:12])
        return {"tracks": tracks, "hours": round(seconds / 3600, 1),
                "estimated_gb": round(size / 1e9, 1),
                "genre_top_distribution": top, "language_code_distribution": langs}

    report = {
        "total_tracks": total,
        "per_bucket": {
            bucket: {"tracks": stats[bucket]["tracks"],
                     "hours": round(stats[bucket]["seconds"] / 3600, 1),
                     "estimated_gb": round(stats[bucket]["bytes"] / 1e9, 1)}
            for bucket in BUCKETS
        },
        "deployable_default": rollup(DEPLOYABLE),
        "flagged_cc_by_sa": rollup(("cc_by_sa",)),
        "caveats": [
            "duration/bit_rate are artist-supplied metadata; sizes are estimates",
            "~14% of CC0 labels historically misfiled — re-verify high-stakes tracks",
            "no instrument tags exist in tracks.csv; instrument coverage requires audio-side tagging",
            "language_code is a weak vocal/multilingual proxy, often blank",
        ],
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"FMA tracks scanned: {total}")
    print(f"{'bucket':<20}{'tracks':>10}{'hours':>10}{'~GB':>8}")
    for bucket in BUCKETS:
        entry = report["per_bucket"][bucket]
        print(f"{bucket:<20}{entry['tracks']:>10}{entry['hours']:>10}{entry['estimated_gb']:>8}")
    accepted = report["deployable_default"]
    print(f"\nDEPLOYABLE (CC0 + CC BY): {accepted['tracks']} tracks, "
          f"{accepted['hours']} h, ~{accepted['estimated_gb']} GB")
    flagged = report["flagged_cc_by_sa"]
    print(f"FLAGGED CC BY-SA (decision needed): {flagged['tracks']} tracks, "
          f"{flagged['hours']} h, ~{flagged['estimated_gb']} GB")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
