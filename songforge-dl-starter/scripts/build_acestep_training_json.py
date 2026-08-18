"""Emit the ACE-Step training_v2 dataset JSON from our canonical manifests.

The headless trainer (`train.py fixed --preprocess`) consumes raw audio plus
ONE metadata JSON — not the sidecar .caption.txt layout the Gradio dataset
builder uses. This script bridges our manifests to that format. Captions are
derived from corpus instrument metadata only (reusing the same caption logic
as the sidecar converter); instrumental samples carry the literal lyrics
string "[Instrumental]" the trainer expects.

    python scripts/build_acestep_training_json.py \
        --manifest  .../processed/slakh100_44k_lora/manifests \
        --audio-root .../processed/slakh100_44k_lora \
        --output .../acestep_lora/slakh100/dataset.json \
        --split train
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from songforge.data.manifest import read_jsonl  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sidecar_builder", ROOT / "scripts" / "build_acestep_lora_dataset.py"
)
_sidecar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sidecar)
caption_for = _sidecar.caption_for
licence_class = getattr(_sidecar, "licence_class", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACE-Step trainer dataset JSON.")
    parser.add_argument("--manifest", required=True,
                        help="manifests dir or a single manifest .jsonl")
    parser.add_argument("--audio-root", required=True,
                        help="directory the manifest audio paths are relative to")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--allow-nonpermissive", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    files = sorted(manifest_path.glob("*.jsonl")) if manifest_path.is_dir() else [manifest_path]
    records = [r for f in files for r in read_jsonl(str(f))]
    if args.split != "all":
        records = [r for r in records if r.split == args.split]

    audio_root = Path(args.audio_root)
    samples = []
    skipped_licence = 0
    skipped_short = 0
    missing_audio = 0
    for record in records:
        licence = (record.license or "").upper()
        if not (licence.startswith("CC-BY-4") or licence in ("CC0-1.0", "MIT", "PUBLIC-DOMAIN")):
            if not args.allow_nonpermissive:
                skipped_licence += 1
                continue
        duration = float(record.duration_seconds or 0)
        if duration < args.min_seconds:
            skipped_short += 1
            continue
        rel = record.path
        audio_path = audio_root / rel
        if not audio_path.exists():
            missing_audio += 1
            continue
        samples.append({
            "audio_path": str(audio_path),
            "filename": Path(rel).name,
            "caption": caption_for([record]),
            "lyrics": "[Instrumental]",
            "is_instrumental": True,
            "duration": round(duration, 2),
            "language": args.language,
            "labeled": bool(record.instrument_family),
        })

    if skipped_licence and not args.allow_nonpermissive:
        print(f"excluded {skipped_licence} non-permissive records (deployable line)")
    if not samples:
        raise SystemExit("no samples survived filtering — refusing to write an empty dataset")

    payload = {
        "metadata": {
            "name": "songforge_slakh100_v1",
            "num_samples": len(samples),
            "all_instrumental": True,
        },
        "samples": samples,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{len(samples)} samples -> {out}")
    print(f"skipped: licence {skipped_licence}, short {skipped_short}, missing audio {missing_audio}")


if __name__ == "__main__":
    main()
