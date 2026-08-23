"""Official SongForge milestone names.

Milestones are always written as number + full descriptive title. A bare "M03"
hides what the gate actually covers, so every generated artifact - terminal
reports, experiment logs, acceptance reports, notebook headings - resolves the
name through here rather than hardcoding a string.
"""

from __future__ import annotations

MILESTONES: dict[str, str] = {
    "M00": "Repository & Project Bootstrap",
    "M01": "Dataset Registry, Licensing & Provenance",
    "M02": "Audio Preprocessing & Dataset Pipeline",
    "M03": "Neural Audio Codec & Discrete Audio Representation",
    "M04": "High-Quality Codec Optimization & Latent-Rate Selection",
    "M05": "Musical Representation & Tokenization",
    "M06": "Song Planning & Semantic Music Generation",
    "M07": "Lyrics, Phoneme & Vocal Alignment Pipeline",
    "M08": "Singing Voice Generation Model",
    "M09": "High-Quality Audio Decoder / Vocoder",
    "M10": "Instrumental & Full-Band Music Generation",
    "M11": "Full Song Generation Integration",
    "M12": "Objective & Human Evaluation",
    "M13": "Long-Form Song Generation",
    "M14": "Genre, Style & Instrument Conditioning Expansion",
    "M15": "Preference & Quality Optimization",
    "M16": "Song Generation Inference UI / Product",
}


def milestone(number: str) -> str:
    """Full milestone label, e.g. ``M03 - Neural Audio Codec & ...``.

    Uses an em dash to match the project's written convention.
    """
    key = number.upper()
    if key not in MILESTONES:
        raise KeyError(f"Unknown milestone {number!r}. Known: {', '.join(sorted(MILESTONES))}")
    return f"{key} — {MILESTONES[key]}"
