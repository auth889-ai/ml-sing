"""M04 — High-Quality Codec Optimization & Latent-Rate Selection: listening matrix.

Collects the per-instrument held-out examples from every candidate into one
folder laid out for A/B/C listening:

    listening/piano__source.wav
    listening/piano__120hz.wav
    listening/piano__75hz.wav
    listening/piano__50hz.wav

Instrument identity comes from the BabySlakh metadata, never from a spectral
heuristic. The script refuses to build a row whose candidates reconstructed
different source audio, because that row would look like a codec difference
while actually being a different song.

    python scripts/m04_listening_matrix.py \
        --output-root "$DRIVE/outputs/m04_stage1_authoritative_expanded" \
        --candidates m04_baseline_120hz_q2 m04_a_75hz_q2 m04_b_50hz_q2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from songforge.milestones import milestone

CATEGORIES = ("piano", "guitar", "bass", "percussion", "strings", "full_mix")


def candidate_rate(run_dir: Path) -> str:
    """Short label for a candidate, taken from its own config."""
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        hz = (config.get("m04") or {}).get("latent_frame_rate_hz")
        if hz:
            return f"{hz:g}hz"
    return run_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{milestone('M04')} listening matrix.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    listening = root / "listening"
    listening.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    for name in args.candidates:
        run_dir = root / name
        payload = json.loads((run_dir / "instrument_examples.json").read_text(encoding="utf-8"))
        loaded[name] = payload.get("exported") or {}
        labels[name] = candidate_rate(run_dir)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for category in CATEGORIES:
        entries = {name: loaded[name].get(category) for name in args.candidates}
        if any(entry is None for entry in entries.values()):
            absent = [name for name, entry in entries.items() if entry is None]
            skipped.append(f"{category}: not exported by {', '.join(absent)}")
            continue

        sources = {entry["source_path"] for entry in entries.values()}
        if len(sources) != 1:
            skipped.append(f"{category}: candidates used different source audio, row omitted")
            continue

        first = entries[args.candidates[0]]
        source_target = listening / f"{category}__source.wav"
        shutil.copyfile(first["original"], source_target)

        row: dict[str, Any] = {
            "category": category,
            "source_path": first["source_path"],
            "instrument_family": first.get("instrument_family"),
            "source": str(source_target),
            "reconstructions": {},
        }
        for name in args.candidates:
            target = listening / f"{category}__{labels[name]}.wav"
            shutil.copyfile(entries[name]["reconstructed"], target)
            row["reconstructions"][labels[name]] = {
                "path": str(target),
                "candidate": name,
                "metrics": entries[name].get("metrics") or {},
            }
        rows.append(row)

    order = [labels[name] for name in args.candidates]
    lines = [
        f"# {milestone('M04')}: listening matrix",
        "",
        "Instrument identity is read from the BabySlakh `metadata.yaml`, not inferred",
        "from the audio. Every candidate reconstructs the *same* held-out source file",
        "per category, so differences between the files below are codec differences.",
        "",
        "| category | family | source | " + " | ".join(order) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in order) + " |",
    ]
    for row in rows:
        cells = " | ".join(
            f"`{Path(row['reconstructions'][label]['path']).name}`" for label in order
        )
        lines.append(
            f"| {row['category']} | {row['instrument_family']} | "
            f"`{Path(row['source']).name}` | {cells} |"
        )

    lines += ["", "## Per-category objective metrics", ""]
    for row in rows:
        lines += [f"### {row['category']} — {row['instrument_family']}", "",
                  f"Source: `{row['source_path']}`", "",
                  "| candidate | L1 | MR-STFT | SNR dB | SI-SDR dB | LSD dB | transient | HF dB |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for label in order:
            metrics = row["reconstructions"][label]["metrics"]

            def fmt(key: str, digits: int = 4, metrics: dict = metrics) -> str:
                value = metrics.get(key)
                return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

            lines.append(
                f"| {label} | {fmt('l1')} | {fmt('mrstft')} | {fmt('snr_db', 2)} | "
                f"{fmt('si_sdr_db', 2)} | {fmt('log_spectral_distance_db', 2)} | "
                f"{fmt('transient_preservation', 3)} | {fmt('high_frequency_preservation_db', 2)} |"
            )
        lines.append("")

    if skipped:
        lines += ["## Categories not covered", ""]
        lines += [f"- {note}" for note in skipped]
        lines += ["",
                  "These are reported rather than filled in with a similar-sounding",
                  "substitute; the held-out split simply does not contain them.", ""]

    index = listening / "README.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (listening / "listening_matrix.json").write_text(
        json.dumps({"rows": rows, "skipped": skipped}, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"{milestone('M04')}: listening matrix\n")
    for row in rows:
        print(f"  {row['category']:<12} {row['instrument_family']:<22} {row['source_path']}")
    for note in skipped:
        print(f"  SKIPPED {note}")
    print(f"\n{len(rows)} categories x {len(order) + 1} files -> {listening}")
    print(f"index: {index}")


if __name__ == "__main__":
    main()
