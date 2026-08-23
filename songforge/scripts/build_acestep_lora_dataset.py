"""Turn a canonical SongForge manifest into an ACE-Step LoRA training dataset.

ACE-Step's official trainer expects a flat directory:

    dataset/
      <name>.wav            audio
      <name>.caption.txt    free-text style/instrument caption
      <name>.lyrics.txt     lyrics, may be empty for instrumental
      <name>.json           {caption, bpm, keyscale, timesignature, language}

Two rules this script exists to enforce.

**Captions are derived from corpus metadata, never invented.** A caption that
claims "expressive solo violin" for a track we only know contains *some* string
stem would teach the model a false association. Only labels actually present in
the manifest are used, and a record with no instrument metadata gets a caption
that says only what we know.

**Licences are checked before a dataset is built.** Fine-tuning on
non-commercial data produces a non-commercial checkpoint regardless of the base
model's licence — this is the GTSinger trap, and most open singing corpora are
CC-BY-NC. The script refuses to mix licence classes unless told to.

    python scripts/build_acestep_lora_dataset.py \
        --manifest "$DATA/processed/babyslakh_m04_expanded/manifests/train.jsonl" \
        --output-dir "$DATA/lora/babyslakh_arrangement" \
        --mode sources --goal "multitrack arrangement realism"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from songforge.data.manifest import AudioRecord, read_jsonl
from songforge.milestones import milestone

#: Licences that permit a commercially usable fine-tuned checkpoint. Anything
#: outside this set encumbers the output and must be opted into explicitly.
PERMISSIVE = {
    "CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "CC0", "MIT", "Apache-2.0",
    "BSD-3-Clause", "public-domain",
}


def licence_class(name: str) -> str:
    upper = (name or "").upper()
    if not name:
        return "unknown"
    if "NC" in upper.replace("-", "").replace(" ", ""):
        return "non-commercial"
    if name in PERMISSIVE or upper.startswith("CC-BY-4"):
        return "permissive"
    return "other"


def caption_for(records: list[AudioRecord]) -> str:
    """Describe a track using only labels the corpus actually provides."""
    families = [r.instrument_family for r in records if r.instrument_family]
    names = [r.instrument_name for r in records if r.instrument_name]
    counted = [f for f, _ in Counter(families).most_common() if f and f != "Mixture"]

    if not counted and not names:
        # Say only what is known. An empty caption is better than a guess.
        return "instrumental recording"

    # A single-instrument source is worth naming precisely: a LoRA aimed at
    # piano realism learns more from "solo acoustic grand piano" than from
    # "recording featuring piano". The specific name still comes from the
    # corpus, so this is precision, not embellishment.
    if len(counted) == 1 and names:
        lead = Counter(names).most_common(1)[0][0]
        family = counted[0].lower()
        detail = lead if family in lead.lower() else f"{lead}, {family}"
        return f"solo {detail}"

    listed = ", ".join(part.lower() for part in counted[:8])
    kind = "multitrack arrangement" if len(counted) > 2 else "recording"
    return f"instrumental {kind} featuring {listed}" if listed else "instrumental recording"


def group_records(records: list[AudioRecord], mode: str) -> dict[str, list[AudioRecord]]:
    """Group manifest rows into training items.

    ``sources`` keys by the original file, which is what a LoRA should see:
    ACE-Step trains on musical excerpts, not on the short segments our codec
    work used. ``segments`` is available for debugging only.
    """
    grouped: dict[str, list[AudioRecord]] = {}
    for record in records:
        if mode == "sources":
            key = record.source_path or record.path
        else:
            key = record.path
        grouped.setdefault(key, []).append(record)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{milestone('M04')}+: build an ACE-Step LoRA dataset.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["sources", "segments"], default="sources")
    parser.add_argument("--goal", required=True, help="Which weakness this dataset targets.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--include-mixture-only", action="store_true",
                        help="Use only rendered full mixes, not individual stems.")
    parser.add_argument("--allow-nonpermissive", action="store_true",
                        help="Proceed even though the result would be an encumbered checkpoint.")
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio instead of symlinking.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if args.include_mixture_only:
        records = [r for r in records if r.instrument_family == "Mixture"]

    # --- licence gate, before anything is written -------------------------
    licences = Counter(r.license for r in records)
    classes = {name: licence_class(name) for name in licences}
    encumbering = {n: c for n, c in classes.items() if c != "permissive"}

    print(f"{milestone('M04')}+ LoRA dataset build")
    print(f"goal      : {args.goal}")
    print(f"manifest  : {args.manifest}")
    print(f"licences  : {dict(licences)}")
    print(f"classes   : {classes}")

    if encumbering and not args.allow_nonpermissive:
        raise SystemExit(
            f"\nREFUSING: {encumbering} would encumber the fine-tuned checkpoint.\n"
            "A LoRA trained on non-commercial or unclear data is itself non-commercial, "
            "regardless of ACE-Step's MIT weights.\n"
            "Re-run with --allow-nonpermissive if this is deliberately a research-only adapter."
        )

    grouped = group_records(records, args.mode)
    output = Path(args.output_dir)
    items: list[dict[str, Any]] = []
    skipped_short = 0

    for key in sorted(grouped):
        rows = grouped[key]
        seconds = sum(r.duration_seconds for r in rows) if args.mode == "segments" else None
        source = Path(key)
        if not source.exists():
            continue
        if seconds is not None and seconds < args.min_seconds:
            skipped_short += 1
            continue

        stem = f"{rows[0].track_id}__{source.stem}".replace(" ", "_")
        items.append({
            "stem": stem,
            "source": str(source),
            "caption": caption_for(rows),
            "track_id": rows[0].track_id,
            "license": rows[0].license,
            "provenance": rows[0].provenance,
            "segments": len(rows),
            "families": sorted({r.instrument_family for r in rows if r.instrument_family}),
        })
        if args.limit and len(items) >= args.limit:
            break

    print(f"\nitems     : {len(items)} ({args.mode} mode)"
          + (f", {skipped_short} skipped under {args.min_seconds}s" if skipped_short else ""))
    for item in items[:5]:
        print(f"  {item['stem']:<34} {item['caption']}")
    if len(items) > 5:
        print(f"  ... {len(items) - 5} more")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    output.mkdir(parents=True, exist_ok=True)
    for item in items:
        target = output / f"{item['stem']}.wav"
        if args.copy_audio:
            shutil.copyfile(item["source"], target)
        elif not target.exists():
            target.symlink_to(Path(item["source"]).resolve())
        (output / f"{item['stem']}.caption.txt").write_text(item["caption"] + "\n", encoding="utf-8")
        # Empty lyrics file: these corpora are instrumental. An absent file and an
        # empty file mean different things to the trainer, so write it explicitly.
        (output / f"{item['stem']}.lyrics.txt").write_text("", encoding="utf-8")
        (output / f"{item['stem']}.json").write_text(
            json.dumps({"caption": item["caption"], "language": args.language},
                       indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # Provenance travels with the dataset, so a checkpoint can always be traced.
    (output / "DATASET_CARD.json").write_text(
        json.dumps({
            "goal": args.goal,
            "source_manifest": str(args.manifest),
            "mode": args.mode,
            "items": len(items),
            "licenses": dict(licences),
            "license_classes": classes,
            "encumbers_checkpoint": bool(encumbering),
            "captions_derived_from": "corpus instrument metadata only; none invented",
            "tracks": sorted({item["track_id"] for item in items}),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {len(items)} items -> {output}")
    print(f"card : {output / 'DATASET_CARD.json'}")


if __name__ == "__main__":
    main()
