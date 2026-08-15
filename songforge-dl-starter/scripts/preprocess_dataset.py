"""M02 CLI: raw audio -> validated, segmented, canonical manifests.

Example:

    python scripts/preprocess_dataset.py \
        --dataset-id babyslakh \
        --input-dir "$SONGFORGE_DATA/raw/babyslakh" \
        --output-dir "$SONGFORGE_DATA/processed/babyslakh" \
        --config configs/data/preprocess_m02.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from songforge.data.dedup import assert_no_cross_split_duplicates, duplicate_report
from songforge.data.manifest import (
    assert_no_track_leakage,
    assert_provenance_complete,
    assert_singer_disjoint,
    manifest_summary,
    validate_records,
    write_split_manifests,
)
from songforge.data.preprocess import (
    PreprocessConfig,
    find_audio_files,
    preprocess_paths,
    provenance_from_registry,
    write_preprocess_report,
)
from songforge.data.registry import load_dataset_registry
from songforge.data.splits import (
    SplitConfig,
    assert_group_disjoint,
    assign_splits,
    split_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M02 canonical audio preprocessing.")
    parser.add_argument("--dataset-id", default="babyslakh", help="Key in configs/data/datasets.yaml.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw audio.")
    parser.add_argument("--output-dir", required=True, help="Where processed audio and manifests are written.")
    parser.add_argument("--config", default="configs/data/preprocess_m02.yaml")
    parser.add_argument("--registry", default="configs/data/datasets.yaml")
    parser.add_argument("--patterns", nargs="+", default=["*.wav", "*.flac"])
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files (debug).")
    parser.add_argument("--split-mode", default=None, choices=["song", "singer"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-audio", action="store_true", help="Write manifests only, skip WAV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    preprocess_config = PreprocessConfig.from_dict(raw_config.get("preprocess", {}))
    if args.no_audio:
        preprocess_config = PreprocessConfig.from_dict(
            {**raw_config.get("preprocess", {}), "write_audio": False}
        )

    split_payload = dict(raw_config.get("splits", {}))
    if args.split_mode:
        split_payload["mode"] = args.split_mode
    if args.seed is not None:
        split_payload["seed"] = args.seed
    split_config = SplitConfig.from_dict(split_payload)
    split_config.validate()

    registry = load_dataset_registry(args.registry)
    if args.dataset_id not in registry.datasets:
        raise SystemExit(f"Unknown dataset id {args.dataset_id!r}. Known: {', '.join(sorted(registry.datasets))}")
    provenance = provenance_from_registry(registry, args.dataset_id)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    paths = find_audio_files(input_dir, tuple(args.patterns))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No audio matching {args.patterns} under {input_dir}")

    output_dir = Path(args.output_dir)
    print(f"dataset   : {args.dataset_id} ({provenance.get('license_name')})")
    print(f"input     : {input_dir} ({len(paths)} files)")
    print(f"output    : {output_dir}")
    print(f"target    : {preprocess_config.sample_rate} Hz, {preprocess_config.channels} ch, "
          f"{preprocess_config.segment_seconds}s segments")

    result = preprocess_paths(
        paths,
        config=preprocess_config,
        provenance=provenance,
        output_dir=output_dir,
        dataset_id=args.dataset_id,
        source_root=input_dir,
    )
    if not result.records:
        write_preprocess_report(result, output_dir / "preprocess_report.json")
        raise SystemExit("No usable segments were produced. See preprocess_report.json.")

    records = assign_splits(result.records, split_config)

    errors = validate_records(records)
    if errors:
        raise SystemExit("Manifest validation failed:\n  " + "\n  ".join(errors[:20]))

    assert_provenance_complete(records)
    assert_no_track_leakage(records)
    assert_group_disjoint(records, split_config.mode)
    assert_singer_disjoint(records)
    assert_no_cross_split_duplicates(records)

    manifest_dir = output_dir / "manifests"
    written = write_split_manifests(records, manifest_dir)
    write_preprocess_report(result, output_dir / "preprocess_report.json")

    summary = manifest_summary(records)
    report = {
        "dataset_id": args.dataset_id,
        "preprocess": result.stats,
        "manifest": summary,
        "splits": split_report(records, split_config.mode),
        "duplicates": duplicate_report(records),
        "split_config": {
            "train": split_config.train,
            "val": split_config.val,
            "test": split_config.test,
            "seed": split_config.seed,
            "mode": split_config.mode,
            "strategy": split_config.strategy,
        },
        "manifests": {name: str(path) for name, path in written.items()},
    }
    (output_dir / "m02_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nM02 preprocessing OK: {summary['segments']} segments from {summary['tracks']} tracks")
    print(f"manifests: {manifest_dir}")


if __name__ == "__main__":
    main()
