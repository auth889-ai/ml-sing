"""Intersect the licence-cleared FMA selection with the audio actually on disk.

``fma_license_report.py`` censuses the whole 106,574-track FMA catalogue, but a
sprint only downloads a subset (``fma_medium``: 25,000 tracks x 30 s, 23.8 GB,
versus ``fma_full`` at ~100 GB). A selection naming tracks we do not hold would
shrink the corpus silently at preprocessing time, so the intersection of
"licence-cleared" and "present on disk" is computed explicitly and reported.

Licence rule matches ``fma_license_report.py``'s buckets exactly: a track is
deployable when it is CC0/public-domain, or carries an Attribution licence with
no NonCommercial / NoDerivatives / ShareAlike qualifier. Version numbers are
deliberately NOT enumerated — CC BY 1.0/2.0/2.5/3.0/4.0 are all CC BY, and an
earlier version of this script that listed only 3.0 and 4.0 silently discarded
1,546 usable tracks (876 kept instead of 2,422).

    python scripts/select_fma_available_on_disk.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

AUDIO = Path("/content/v2_raw/audio/fma_medium")
TRACKS = Path("/content/v2_raw/meta/fma_metadata/tracks.csv")
OUT = Path("/content/drive/MyDrive/songforge-dl/v2_fma_selected.json")

# Any of these qualifiers makes a track non-deployable for the released model.
EXCLUDE = ("noncommercial", "no derivative", "noderiv", "share-alike",
           "sharealike", "-nc", "nc-", " nd")


def deployable(licence: str) -> bool:
    low = licence.lower()
    if any(x in low for x in EXCLUDE):
        return False
    return ("attribution" in low or "public domain" in low
            or "cc0" in low or "zero" in low)


def main() -> int:
    # tracks.csv carries a three-row header; track_id is the unnamed column 0.
    rows = list(csv.reader(TRACKS.open(encoding="utf-8")))
    cols = {(a.strip(), b.strip()): i
            for i, (a, b) in enumerate(zip(rows[0], rows[1]))}
    lic_col = cols.get(("track", "license"))
    gen_col = cols.get(("track", "genre_top"))
    ttl_col = cols.get(("track", "title"))
    art_col = cols.get(("artist", "name"))

    present = {}
    for root, _, files in os.walk(AUDIO):
        for name in files:
            if name.endswith(".mp3"):
                present[int(name[:-4])] = os.path.join(root, name)
    print("mp3 files on disk :", len(present))

    def field(row, idx):
        return row[idx] if idx is not None and idx < len(row) else ""

    selected, genres = [], {}
    for row in rows[3:]:
        if not row or not row[0].strip().isdigit():
            continue
        tid = int(row[0])
        if tid not in present:
            continue
        licence = field(row, lic_col)
        if not deployable(licence):
            continue
        genre = field(row, gen_col) or "Unknown"
        genres[genre] = genres.get(genre, 0) + 1
        selected.append({
            "track_id": tid,
            "path": present[tid],
            "license": licence,
            "genre": genre,
            "title": field(row, ttl_col),
            "artist": field(row, art_col),
        })

    print("DEPLOYABLE and ON DISK:", len(selected))
    print("distinct genres      :", len(genres))
    for genre, count in sorted(genres.items(), key=lambda kv: -kv[1])[:12]:
        print("   %-22s %d" % (genre, count))

    OUT.write_text(json.dumps(
        {"count": len(selected), "genres": genres, "tracks": selected},
        indent=1))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
