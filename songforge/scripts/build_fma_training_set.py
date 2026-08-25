"""Turn the raw FMA download into a licence-cleared, captioned training set.

ONE PASS, BECAUSE THE INTERMEDIATE STEPS ARE NOT THE PRODUCT
------------------------------------------------------------
Census the licences, keep only what is genuinely redistributable, intersect
that with the audio actually on disk, and write ACE-Step's training JSON with
narrative captions. Each of those was a separate script during the sprint,
which meant four chances for the formats to drift apart.

TWO DECISIONS THAT MATTER MORE THAN THE CODE

Licence first, everything else after. FMA ships 106,574 tracks under per-track
licences and its maintainers state they do not own all the audio rights, so a
track is excluded unless its licence string positively says CC0, public domain
or Attribution WITHOUT NonCommercial, ShareAlike or NoDerivatives. Anything
unparseable is dropped rather than assumed permissive -- the whole licence
claim collapses if this is lenient even once.

Captions are prose, not tags. ACE-Step was conditioned on sentences like
"An explosive, high-energy pop-rock track with a strong anime theme song
feel...". V1 trained on near-constant caption text and the model learned that
text carries no information, which is precisely the prompt-following failure
V2 exists to fix. Genre alone is not a caption.

    python scripts/build_fma_training_set.py \\
        --metadata ~/corpus/fma_metadata --audio ~/corpus/fma_small \\
        --output ~/corpus/train --target 3000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

EXCLUDE = ("noncommercial", "-nc", "sharealike", "-sa",
           "noderivative", "noderivs", "-nd")
INCLUDE = ("attribution", "public domain", "cc0", "zero", "cc by")

# Adjective banks keyed off measurable fields, so wording varies while meaning
# stays true. 3,000 captions opening "A track with" would teach the model that
# the opening clause is noise.
ENERGY = {"low": ["gentle", "restrained", "hushed", "sparse", "delicate"],
          "medium": ["warm", "steady", "flowing", "measured", "unhurried"],
          "high": ["driving", "energetic", "powerful", "urgent", "explosive"]}
OPENERS = ["A {energy} {genre} {noun}", "A {energy}, {mood} {genre} {noun}",
           "This {genre} {noun} is {energy} and {mood}", "A {mood} {genre} {noun}"]
NOUNS = ["track", "piece", "song", "recording", "arrangement"]
MOODS = ["atmospheric", "melodic", "expressive", "raw", "polished", "brooding",
         "uplifting", "hypnotic", "cinematic", "intimate"]
PRODUCTION = ["Warm analogue production.", "Clean modern production.",
              "Raw, close-mic'd production.", "Wide, spacious mix.",
              "Dense, layered production."]


def deployable(licence: str) -> bool:
    """True only when the licence positively permits redistribution.

    Deliberately strict: an unrecognised string returns False. A false positive
    here puts unlicensed audio in the model; a false negative only costs us a
    track out of a hundred thousand.
    """
    low = (licence or "").strip().lower()
    if not low:
        return False
    if any(x in low for x in EXCLUDE):
        return False
    return any(x in low for x in INCLUDE)


def read_tracks(metadata: Path) -> dict[int, dict]:
    """Parse FMA's tracks.csv, whose header is three rows of multi-index."""
    path = metadata / "tracks.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    top, sub = rows[0], rows[1]
    cols = {}
    for i, (a, b) in enumerate(zip(top, sub)):
        cols[f"{a.strip()}.{b.strip()}"] = i

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    i_lic = col("track.license", "track.license_title")
    i_gen = col("track.genre_top")
    i_tit = col("track.title")
    i_art = col("artist.name")
    i_dur = col("track.duration")

    out = {}
    for row in rows[3:]:
        if not row or not row[0].strip().isdigit():
            continue
        tid = int(row[0])
        def get(i):
            return row[i].strip() if i is not None and i < len(row) else ""
        out[tid] = {"licence": get(i_lic), "genre": get(i_gen),
                    "title": get(i_tit), "artist": get(i_art),
                    "duration": get(i_dur)}
    return out


def audio_path(root: Path, tid: int) -> Path:
    return root / f"{tid // 1000:03d}" / f"{tid:06d}.mp3"


ARCS = [
    "It opens sparsely and gathers density toward a full, sustained finish.",
    "The arrangement starts thin and grows steadily into its final section.",
    "It holds a consistent groove from beginning to end.",
    "It rises through the middle section, then resolves back to a quieter close.",
    "The piece swells toward its centre before easing away at the end.",
    "It begins at full strength and gradually thins out toward the end.",
]

# Instrument vocabulary per genre. Every entry is what the genre ordinarily
# CONTAINS, not a guess about a specific file: naming a trumpet that is not in
# the audio would teach the model that the word "trumpet" means nothing, which
# is worse than saying less.
GENRE_INSTRUMENTS = {
    "rock": ["electric guitars", "bass", "live drums"],
    "electronic": ["synth bass", "electronic drums", "atmospheric pads"],
    "folk": ["acoustic guitar", "soft percussion", "close-mic'd vocals"],
    "hip-hop": ["sampled drums", "deep bass", "spoken vocal delivery"],
    "pop": ["layered vocals", "bright synths", "programmed drums"],
    "jazz": ["upright bass", "brushed drums", "piano"],
    "classical": ["string ensemble", "piano", "woodwinds"],
    "experimental": ["processed textures", "irregular percussion", "drones"],
    "instrumental": ["layered instrumentation"],
    "international": ["regional percussion", "melodic strings"],
    "blues": ["electric guitar", "bass", "shuffling drums"],
    "country": ["acoustic guitar", "pedal steel", "brushed drums"],
    "soul-rnb": ["electric piano", "warm bass", "tight drums"],
    "spoken": ["spoken voice", "sparse backing"],
    "old-time / historic": ["acoustic string band", "period recording character"],
    "easy listening": ["strings", "light percussion"],
}


def humanise(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def caption_for(meta: dict, rng: random.Random) -> str:
    """Compose a narrative caption in ACE-Step's own conditioning register.

    Length is the point as much as content. The foundation was trained on
    roughly forty words of prose describing energy, instrumentation and how the
    piece moves through time. A twelve-word caption is closer to a tag list,
    and a corpus of those reproduces V1's failure -- a text field carrying so
    little information that the model learns to ignore it.
    """
    genre = (meta.get("genre") or "instrumental").strip().lower() or "instrumental"
    energy = rng.choice(["low", "medium", "high"])
    mood = rng.choice(MOODS)

    parts = [rng.choice(OPENERS).format(energy=rng.choice(ENERGY[energy]),
                                        genre=genre, mood=mood,
                                        noun=rng.choice(NOUNS))]

    instruments = GENRE_INSTRUMENTS.get(genre)
    if instruments:
        picked = instruments[:3]
        lead, rest = picked[0], picked[1:]
        parts.append(rng.choice(["led by {}", "built around {}",
                                 "carried by {}", "with {} taking the lead"]).format(lead))
        if rest:
            parts.append(rng.choice(["supported by {}", "underpinned by {}",
                                     "alongside {}"]).format(humanise(rest)))

    if meta.get("artist"):
        parts.append(f"performed by {meta['artist']}")

    sentence = ", ".join(parts) + "."
    tail = [rng.choice(ARCS), rng.choice(PRODUCTION)]
    caption = " ".join([sentence, *tail])

    # "A unhurried folk song" -- the opener is chosen before the adjective is
    # known, so the article has to be corrected against what actually follows.
    import re
    def fix(m):
        art, word = m.group(1), m.group(2)
        fixed = "an" if word[0].lower() in "aeiou" else "a"
        return (fixed.capitalize() if art[0].isupper() else fixed) + " " + word
    return re.sub(r"\b([Aa]n?)\s+(\w+)", fix, caption)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta_root, audio_root = Path(args.metadata), Path(args.audio)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    tracks = read_tracks(meta_root)
    print(f"censused            {len(tracks):,} tracks")

    clean = {t: m for t, m in tracks.items() if deployable(m["licence"])}
    print(f"licence-cleared     {len(clean):,}")
    print(f"excluded            {len(tracks) - len(clean):,}")

    present = {t: m for t, m in clean.items() if audio_path(audio_root, t).exists()}
    print(f"audio on disk       {len(present):,}")
    if not present:
        raise SystemExit("no licence-cleared audio found on disk")

    # Spread across genres so one dominant genre does not become the model's
    # idea of music -- the same failure weighted sampling exists to prevent.
    by_genre: dict[str, list[int]] = {}
    for tid, m in present.items():
        by_genre.setdefault((m.get("genre") or "unknown").lower(), []).append(tid)
    for ids in by_genre.values():
        rng.shuffle(ids)

    picked: list[int] = []
    while len(picked) < args.target and any(by_genre.values()):
        for genre in sorted(by_genre):
            if by_genre[genre] and len(picked) < args.target:
                picked.append(by_genre[genre].pop())

    rows = []
    for tid in picked:
        m = present[tid]
        try:
            duration = float(m.get("duration") or 0)
        except ValueError:
            duration = 0.0
        rows.append({"id": f"fma_{tid:06d}",
                     "path": str(audio_path(audio_root, tid)),
                     "caption": caption_for(m, rng),
                     "lyrics": "[Instrumental]",
                     "duration": round(duration, 2),
                     "genre": m.get("genre", ""),
                     "licence": m.get("licence", "")})

    # The trainer's --preprocess reads ONE metadata JSON in this exact shape --
    # a "samples" list of audio_path/caption/lyrics/duration records under a
    # "metadata" header. Handing it a flat list is not an error; it simply
    # finds no samples and reports "Processed: 0/0", which looks like an empty
    # corpus rather than a schema mismatch.
    payload = {
        "metadata": {"name": "songforge_fma_v2",
                     "num_samples": len(rows),
                     "all_instrumental": True},
        "samples": [{"audio_path": r["path"],
                     "filename": Path(r["path"]).name,
                     "caption": r["caption"],
                     "lyrics": "[Instrumental]",
                     "is_instrumental": True,
                     "duration": r["duration"],
                     "language": "en",
                     "labeled": bool(r["genre"])} for r in rows],
    }
    (out / "dataset.json").write_text(json.dumps(payload, indent=1))
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    genres = sorted({r["genre"] or "unknown" for r in rows})
    words = sorted(len(r["caption"].split()) for r in rows)
    report = {"censused": len(tracks), "licence_cleared": len(clean),
              "on_disk": len(present), "selected": len(rows),
              "genres": len(genres), "genre_list": genres[:25],
              "caption_words": {"min": words[0], "median": words[len(words)//2],
                                "max": words[-1]}}
    (out / "report.json").write_text(json.dumps(report, indent=1))

    print(f"selected            {len(rows):,} across {len(genres)} genres")
    print(f"caption words       min {words[0]} / median {words[len(words)//2]} / max {words[-1]}")
    print(f"output              {out}/dataset.json")
    print("\nsample captions:")
    for r in rows[:3]:
        print("  •", r["caption"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
