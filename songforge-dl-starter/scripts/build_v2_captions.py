"""Write V2 training captions in ACE-Step's native narrative format.

WHY THIS FILE DECIDES V2's QUALITY
----------------------------------
Reading ACE-Step's own `examples/text2music/*.json` showed how the foundation
was conditioned during pretraining:

    "An explosive, high-energy pop-rock track with a strong anime theme song
     feel. The song kicks off with a catchy, synthesized brass fanfare..."

Prose. Instruments named inside a sentence, a stated energy level, and an
account of how the piece moves through time. Not `"rock"`, and not a tag list.

V1 trained with `lyrics = "[Instrumental]"` on all 3,626 rows and a caption
built from corpus metadata fields. A model shown a near-constant text field
learns that text carries no information, and then ignores the prompt at
inference — which is exactly the "every song sounds the same" failure.

The published benchmarks say the same thing from the other side: ACE-Step 1.5
beats Suno v5 on overall SongEval quality (8.12 vs 7.87) but trails it on
*style alignment* (39.1 vs 46.8) and *lyric alignment* (26.3 vs 34.2). Both are
prompt-following measures. Closing them is a conditioning problem before it is
a capacity problem, and conditioning is free — it costs no extra GPU time.

So captions are generated to match the pretraining distribution rather than to
describe the file. Every clause is grounded in a manifest field; nothing is
invented, because a caption asserting a trumpet that is not in the audio
teaches the model that the word "trumpet" means nothing.

    python scripts/build_v2_captions.py \\
        --manifest processed/v2/manifests/train.jsonl \\
        --output   processed/v2/captions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------- vocabulary
#
# Adjective banks are keyed off measurable manifest fields so the wording
# varies while the meaning stays true to the audio. Variety matters: 8,000
# captions opening "A track with..." would teach the model that the opening
# clause is noise.

ENERGY_WORDS = {
    "low": ["gentle", "restrained", "hushed", "sparse", "delicate"],
    "medium": ["warm", "steady", "flowing", "measured", "unhurried"],
    "high": ["driving", "energetic", "powerful", "urgent", "explosive"],
}

OPENERS = [
    "A {energy} {genre} {noun}",
    "A {energy}, {mood} {genre} {noun}",
    "This {genre} {noun} is {energy} and {mood}",
    "A {mood} {genre} {noun}",
]

NOUNS = ["track", "piece", "song", "recording", "arrangement"]

LEAD_PHRASES = [
    "led by {lead}",
    "built around {lead}",
    "carried by {lead}",
    "with {lead} taking the lead",
]

SUPPORT_PHRASES = [
    "supported by {rest}",
    "underpinned by {rest}",
    "with {rest} filling out the arrangement",
    "alongside {rest}",
]

ARC_PHRASES = {
    "build": [
        "It opens sparsely and gathers density toward a full, sustained finish.",
        "The arrangement starts thin and grows steadily into its final section.",
        "It begins quietly before building to a broad, full-band close.",
    ],
    "steady": [
        "It holds a consistent groove from beginning to end.",
        "The energy stays level throughout, with the texture largely unchanged.",
        "It settles into a steady feel and stays there.",
    ],
    "arc": [
        "It rises through the middle section, then resolves back to a quieter close.",
        "The piece swells toward its centre before easing away at the end.",
        "It grows, peaks, and then settles.",
    ],
    "decay": [
        "It begins at full strength and gradually thins out toward the end.",
        "The opening is dense, and the arrangement clears as it goes.",
    ],
}

VOCAL_PHRASES = {
    "female": "a female lead vocal",
    "male": "a male lead vocal",
    "choir": "layered choral voices",
    "duet": "male and female voices trading lines",
}


def fix_articles(text: str) -> str:
    """Correct a/an against the word that actually follows.

    The opener templates are chosen before the adjective is known, so "A" can
    land in front of a vowel — "A unhurried folk song". Small, but a corpus of
    8,000 captions with broken articles is teaching the model bad English
    alongside the music.
    """
    def swap(m: re.Match) -> str:
        article, word = m.group(1), m.group(2)
        vowel = word[0].lower() in "aeiou"
        fixed = ("an" if vowel else "a")
        if article[0].isupper():
            fixed = fixed.capitalize()
        return f"{fixed} {word}"

    return re.sub(r"\b([Aa]n?)\s+(\w+)", swap, text)


def humanise(items: list[str]) -> str:
    """Join instrument names the way a person writing prose would."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def energy_band(value: float | None) -> str:
    """Map measured RMS energy onto the adjective bank."""
    if value is None:
        return "medium"
    if value < 0.08:
        return "low"
    if value > 0.22:
        return "high"
    return "medium"


def caption_for(record: dict, rng: random.Random) -> str:
    """Compose one narrative caption from a manifest row.

    Only fields the manifest actually carries are used. A missing field drops
    its clause rather than being filled with a plausible guess — the gate
    chain's whole purpose is that captions describe the audio.
    """
    instruments = list(record.get("instruments") or [])
    genre = (record.get("genre") or "instrumental").strip().lower()
    mood = (record.get("mood") or "").strip().lower()
    energy = energy_band(record.get("rms_energy"))
    arc = record.get("energy_arc") or ("build" if rng.random() < 0.4 else "steady")

    parts: list[str] = []

    opener = rng.choice(OPENERS).format(
        energy=rng.choice(ENERGY_WORDS[energy]),
        genre=genre,
        mood=mood or rng.choice(["expressive", "atmospheric", "melodic"]),
        noun=rng.choice(NOUNS),
    )
    parts.append(opener)

    # Instrumentation: the first listed instrument is treated as the lead,
    # because the corpus manifests order stems by prominence.
    if instruments:
        lead, rest = instruments[0], instruments[1:]
        parts.append(rng.choice(LEAD_PHRASES).format(lead=lead))
        if rest:
            parts.append(rng.choice(SUPPORT_PHRASES).format(rest=humanise(rest[:5])))

    vocal = record.get("vocal_type")
    if vocal and vocal in VOCAL_PHRASES:
        parts.append(f"featuring {VOCAL_PHRASES[vocal]}")
    elif not vocal:
        parts.append("performed instrumentally")

    sentence = ", ".join(parts) + "."

    tail: list[str] = []
    tail.append(rng.choice(ARC_PHRASES.get(arc, ARC_PHRASES["steady"])))

    bpm, key = record.get("bpm"), record.get("key")
    if bpm and key:
        tail.append(f"Around {int(bpm)} BPM in {key}.")
    elif bpm:
        tail.append(f"Around {int(bpm)} BPM.")
    elif key:
        tail.append(f"In {key}.")

    if record.get("production"):
        tail.append(f"{record['production'].capitalize()} production.")

    return fix_articles(" ".join([sentence, *tail]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="corpus manifest .jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260824,
                    help="fixed so a rerun reproduces the same corpus")
    ap.add_argument("--sample", type=int, default=3,
                    help="captions to print for eyeballing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src, dst = Path(args.manifest), Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    lengths: list[int] = []
    with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Gate 9 (rich caption): a row with neither instruments nor genre
            # cannot produce a caption that describes the audio, so it is
            # excluded and counted rather than given a generic one.
            if not record.get("instruments") and not record.get("genre"):
                skipped += 1
                continue
            caption = caption_for(record, rng)
            lengths.append(len(caption.split()))
            record["caption"] = caption
            record.setdefault("lyrics", "[Instrumental]")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"captions written : {written}")
    print(f"skipped (gate 9) : {skipped}")
    if lengths:
        lengths.sort()
        print(f"words: min {lengths[0]}  median {lengths[len(lengths)//2]}  max {lengths[-1]}")
    print(f"output: {dst}")

    if written:
        print("\nsamples:")
        with dst.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= args.sample:
                    break
                print(" •", json.loads(line)["caption"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
