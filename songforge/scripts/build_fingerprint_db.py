"""Build the originality-screen fingerprint DB over a training corpus.

Run wherever the corpus audio lives (Colab for V1/V2). One JSONL line per
item: id, decoded-PCM SHA256, and the chroma window sequence used by
`songforge.inference.originality.screen`. Resumable: existing ids in the
output are skipped, so a dropped runtime just reruns the same command.

    python scripts/build_fingerprint_db.py \
        --manifest <manifests dir or .jsonl with audio_path/id fields> \
        --audio-root <corpus root> \
        --output fingerprints.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from songforge.inference.originality import build_fingerprint  # noqa: E402


def iter_items(manifest: Path, audio_root: Path):
    files = sorted(manifest.glob("*.jsonl")) if manifest.is_dir() else [manifest]
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rel = row.get("audio_path") or row.get("path")
                if not rel:
                    continue
                item_id = row.get("id") or row.get("segment_id") or rel
                path = (audio_root / rel) if not Path(rel).is_absolute() else Path(rel)
                yield str(item_id), path


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus fingerprint DB builder.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    done: set[str] = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"resuming: {len(done)} fingerprints already present")

    written = failed = 0
    with output.open("a", encoding="utf-8") as out:
        for item_id, path in iter_items(Path(args.manifest), Path(args.audio_root)):
            if item_id in done:
                continue
            try:
                entry = build_fingerprint(item_id, path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the corpus
                print(f"FAIL {item_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed += 1
                continue
            out.write(json.dumps(entry) + "\n")
            written += 1
            if written % 200 == 0:
                out.flush()
                print(f"{written} fingerprints written")

    print(f"done: {written} written, {failed} failed, output {output}")


if __name__ == "__main__":
    main()
