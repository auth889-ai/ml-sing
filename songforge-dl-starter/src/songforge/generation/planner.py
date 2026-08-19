"""Derive structured song controls from a free-form prompt.

The product promise is arbitrary prompts, so this planner never constrains
what a user may ask for. It reads the prompt and *derives* whatever typed
controls it can recognize — instruments, vocal character, language, BPM,
key, duration, section structure, energy shape — and leaves the full raw
prompt in place as the primary conditioning. Extraction is additive: a word
the lexicons don't know is not dropped, it simply rides along in the prompt
text the model reads anyway.

Everything here is deterministic and rule-based on purpose: the same prompt
plans the same way in every experiment, and the plan reports *which words*
produced each control, so nothing is presented as smarter than it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .request import Section, SongRequest, VocalSpec

# --- lexicons -------------------------------------------------------------
# Extraction vocab only. This does not limit what the model can be asked for;
# unknown words stay in the prompt untouched.

INSTRUMENT_WORDS = (
    "piano", "violin", "cello", "viola", "guitar", "bass", "drum",
    "percussion", "strings", "synth", "synthesizer", "brass", "trumpet",
    "trombone", "horn", "saxophone", "sax", "flute", "clarinet", "oboe",
    "organ", "keyboard", "keys", "harp", "banjo", "mandolin", "ukulele",
    "sitar", "tabla", "accordion", "pad", "arpeggio", "choir", "bells",
    "vibraphone", "marimba", "timpani", "orchestra", "orchestral",
)
#: words that, immediately before an instrument, describe it and belong with it
INSTRUMENT_MODIFIERS = (
    "grand", "electric", "acoustic", "upright", "warm", "deep", "distorted",
    "clean", "muted", "plucked", "bowed", "live", "heavy", "soft", "subtle",
    "expressive", "solo", "lead", "rhythm", "slap", "fretless", "twelve-string",
    "nylon", "steel", "analog", "ambient", "lush", "powerful", "driving",
)

GENRE_WORDS = (
    "rock", "pop", "jazz", "blues", "folk", "country", "classical", "cinematic",
    "orchestral", "electronic", "edm", "house", "techno", "trance", "ambient",
    "hip-hop", "hip hop", "rap", "trap", "r&b", "rnb", "soul", "funk", "gospel",
    "metal", "punk", "indie", "alternative", "lofi", "lo-fi", "reggae", "ska",
    "latin", "salsa", "bossa nova", "flamenco", "afrobeat", "k-pop", "j-pop",
    "synthwave", "drum and bass", "dubstep", "disco", "swing", "ballad",
    "singer-songwriter", "chillout", "downtempo", "world", "fusion",
)

MOOD_WORDS = (
    "emotional", "happy", "sad", "melancholic", "melancholy", "uplifting",
    "dark", "bright", "epic", "intimate", "dreamy", "nostalgic", "romantic",
    "angry", "aggressive", "peaceful", "calm", "tense", "hopeful", "triumphant",
    "mysterious", "playful", "sombre", "somber", "joyful", "bittersweet",
    "haunting", "euphoric", "gritty", "warm", "cold", "wistful", "yearning",
)

#: language name → code the vocal spec carries. Names only; codes typed by the
#: caller (e.g. the API's language field) take precedence over derivation.
LANGUAGE_NAMES = {
    "english": "en", "bengali": "bn", "bangla": "bn", "hindi": "hi",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "russian": "ru", "japanese": "ja", "korean": "ko",
    "chinese": "zh", "mandarin": "zh", "arabic": "ar", "turkish": "tr",
    "urdu": "ur", "tamil": "ta", "punjabi": "pa",
}

#: phrase → section kinds, in the order phrases appear in the prompt.
SECTION_PHRASES = (
    (r"\bintro\b", "intro"),
    (r"\bverse\b", "verse"),
    (r"\bpre[- ]?chorus\b", "pre_chorus"),
    (r"\bchorus\b", "chorus"),
    (r"\bbridge\b", "bridge"),
    (r"\bdrop\b", "drop"),
    (r"\bbreakdown\b", "breakdown"),
    (r"\bsolo\b", "solo"),
    (r"\boutro\b", "outro"),
    (r"\bfinale?\b", "outro"),
)

ENERGY_PHRASES = (
    (r"\b(huge|massive|powerful|epic)\s+(final\s+)?(chorus|climax|ending|finale)\b", "climactic ending"),
    (r"\bbuild(s|ing)?\s+(up|to)\b", "gradual build"),
    (r"\bsparse\b.*\bhuge\b|\bquiet\b.*\bloud\b|\bintimate\b.*\bepic\b", "quiet-to-loud arc"),
    (r"\bfade[s]?\s+out\b", "fade-out ending"),
    (r"\bslow\s+start\b", "slow start"),
)


@dataclass
class Plan:
    """A planned request plus the evidence for every derived control."""

    request: SongRequest
    derived: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"request": self.request.to_dict(), "derived": self.derived}


def _find_instruments(text: str) -> tuple[list[str], list[str]]:
    found: list[str] = []
    evidence: list[str] = []
    claimed: set[int] = set()  # text offsets already owned by a longer word
    for word in sorted(INSTRUMENT_WORDS, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(word)}s?\b", text):
            if match.start() in claimed:
                continue
            claimed.update(range(match.start(), match.end()))
            start = match.start()
            prefix = text[max(0, start - 40):start].rstrip()
            mods = []
            for mod in INSTRUMENT_MODIFIERS:
                if prefix.endswith(mod):
                    mods.append(mod)
                    prefix = prefix[: -len(mod)].rstrip()
            # keep the surface form (plural stays plural: "live drums")
            surface = match.group(0)
            phrase = " ".join(reversed(mods)) + (" " if mods else "") + surface
            if phrase not in found:
                found.append(phrase)
                evidence.append(surface)
            break  # one hit per instrument word is enough
    return found, evidence


def plan(
    prompt: str,
    lyrics: str | None = None,
    duration_seconds: float | None = None,
    language: str | None = None,
    bpm: int | None = None,
    key: str | None = None,
    seed: int = 0,
) -> Plan:
    """Plan a SongRequest from a free-form prompt.

    Explicit arguments always beat derived values: a typed BPM wins over one
    read out of the prompt, and a typed language wins over a language name
    the prompt mentions.
    """
    text = prompt.lower()
    derived: dict[str, list[str]] = {}

    instruments, inst_evidence = _find_instruments(text)
    if instruments:
        derived["instruments"] = inst_evidence

    genres = [g for g in GENRE_WORDS if re.search(rf"\b{re.escape(g)}\b", text)]
    # drop shorter aliases shadowed by longer hits ("hip hop" vs "hip-hop")
    genres = [g for g in genres if not any(g != o and g in o for o in genres)]
    if genres:
        derived["genre"] = genres

    moods = [m for m in MOOD_WORDS if re.search(rf"\b{re.escape(m)}\b", text)]
    if moods:
        derived["mood"] = moods

    # --- vocals ----------------------------------------------------------
    vocal: VocalSpec | None = None
    wants_instrumental = bool(re.search(r"\binstrumental\b|\bno vocals?\b", text))
    gender = None
    if re.search(r"\b(female|woman|girl)\b", text):
        gender = "female"
    elif re.search(r"\b(male|man|boy)\b", text):
        gender = "male"
    style = None
    for candidate in ("breathy", "belted", "operatic", "rap", "whispered",
                      "soulful", "powerful", "intimate", "raspy", "smooth"):
        if re.search(rf"\b{candidate}\b\s+(voice|vocals?|singing)?", text) and \
           re.search(rf"\b{candidate}\b(?:\s+\w+){{0,2}}\s+(voice|vocal|singing|singer)", text):
            style = candidate
            break
    derived_language = next(
        (code for name, code in LANGUAGE_NAMES.items() if re.search(rf"\b{name}\b", text)),
        None,
    )
    mentions_vocal = bool(re.search(r"\bvocal|voice|singer|singing|sung|choir\b", text))
    if wants_instrumental:
        vocal = VocalSpec(present=False)
        derived["vocal"] = ["instrumental/no vocals"]
    elif lyrics or mentions_vocal or gender or derived_language:
        vocal = VocalSpec(
            present=True,
            gender=gender,
            style=style,
            language=language or derived_language,
        )
        derived["vocal"] = [w for w in (gender, style, derived_language) if w] or ["vocal mention"]

    # --- numeric controls -------------------------------------------------
    derived_bpm = None
    bpm_match = re.search(r"\b(\d{2,3})\s*bpm\b", text)
    if bpm_match and 20 <= int(bpm_match.group(1)) <= 300:
        derived_bpm = int(bpm_match.group(1))
        derived["bpm"] = [bpm_match.group(0)]

    derived_key = None
    key_match = re.search(r"\bin\s+([a-g][#b♯♭]?)\s*(major|minor|min|maj)\b", text)
    if key_match:
        quality = {"maj": "major", "min": "minor"}.get(key_match.group(2), key_match.group(2))
        derived_key = f"{key_match.group(1).upper()} {quality}"
        derived["key"] = [key_match.group(0)]

    derived_duration = None
    duration_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?)\b", text)
    if duration_match:
        value = float(duration_match.group(1))
        if duration_match.group(2).startswith("min"):
            value *= 60.0
        if 10.0 <= value <= 600.0:
            derived_duration = value
            derived["duration"] = [duration_match.group(0)]

    # --- structure and energy --------------------------------------------
    sections: list[tuple[int, Section]] = []
    for pattern, kind in SECTION_PHRASES:
        for match in re.finditer(pattern, text):
            sections.append((match.start(), Section(kind=kind)))
    sections.sort(key=lambda pair: pair[0])
    structure = tuple(dict.fromkeys(section for _, section in sections))
    if structure:
        derived["structure"] = [section.kind for section in structure]

    energy_notes = [label for pattern, label in ENERGY_PHRASES if re.search(pattern, text)]
    extra = {"energy": energy_notes} if energy_notes else {}
    if energy_notes:
        derived["energy"] = energy_notes

    request = SongRequest(
        prompt=prompt.strip(),
        lyrics=lyrics,
        genre=tuple(genres),
        mood=tuple(moods),
        instruments=tuple(instruments),
        vocal=vocal,
        bpm=bpm if bpm is not None else derived_bpm,
        key=key if key is not None else derived_key,
        duration_seconds=duration_seconds or derived_duration or 60.0,
        structure=structure,
        seed=seed,
        extra=extra,
    )
    request.validate()
    return Plan(request=request, derived=derived)
