"""M02 — Audio Preprocessing & Dataset Pipeline: Colab acceptance runner.

Drives the full gate end to end:

    raw audio -> validation -> preprocessing -> segmentation
              -> canonical manifest -> train/val/test split -> Drive persistence

M02 PASS requires a real approved audio subset (BabySlakh by default). Use
``--synthetic`` for a dependency-free rehearsal of the same pipeline; that mode
reports ``synthetic: true`` and is explicitly NOT an M02 pass.

    python scripts/colab_m02_acceptance.py \
        --audio-dir "$SONGFORGE_DATA/raw/babyslakh" \
        --output-dir "$DRIVE_ROOT/processed/babyslakh_m02" \
        --limit-files 12
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from songforge.data.dedup import assert_no_cross_split_duplicates, duplicate_report
from songforge.data.manifest import (
    assert_no_track_leakage,
    assert_provenance_complete,
    assert_singer_disjoint,
    manifest_summary,
    read_jsonl,
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
from songforge.milestones import milestone

ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> dict[str, Any]:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-4000:]}


def environment_report() -> dict[str, Any]:
    import torch

    from songforge.data.media import ffmpeg_path, ffprobe_path

    report = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "ffprobe": ffprobe_path(),
        "ffmpeg": ffmpeg_path(),
    }
    for module in ("soundfile", "torchaudio"):
        try:
            __import__(module)
            report[module] = True
        except ImportError:
            report[module] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_synthetic_corpus(target: Path, seconds: float) -> Path:
    from songforge.data.fixtures import build_slakh_like_corpus

    print(f"Building synthetic corpus at {target} (NOT an M02 pass)")
    build_slakh_like_corpus(target, tracks=5, stems_per_track=2, seconds=seconds, include_broken=True)
    return target


def verify_persistence(paths: dict[str, Path], audio_samples: list[Path]) -> dict[str, Any]:
    """Confirm manifests and processed audio really landed on the target volume."""
    missing = [name for name, path in paths.items() if not Path(path).exists()]
    empty = [name for name, path in paths.items() if Path(path).exists() and Path(path).stat().st_size == 0]
    missing_audio = [str(path) for path in audio_samples if not Path(path).exists()]
    return {
        "manifest_files": {name: str(path) for name, path in paths.items()},
        "missing_manifests": missing,
        "empty_manifests": empty,
        "checked_audio_files": len(audio_samples),
        "missing_audio_files": missing_audio[:10],
        "ok": not missing and not empty and not missing_audio,
    }


def write_experiment_log(path: Path, report: dict[str, Any]) -> None:
    status = "PASS" if report["acceptance_pass"] else "FAIL"
    summary = report["manifest"]
    stats = report["preprocess"]
    splits = report["splits"]["splits"]

    lines = [
        f"# Experiment Log — {milestone('M02')}",
        "",
        f"## {milestone('M02')}: Acceptance",
        "",
        f"Status: **{milestone('M02')} = {status}**" + ("  (synthetic rehearsal, not a real pass)" if report["synthetic"] else ""),
        "",
        "### Environment",
        "",
        "```json",
        json.dumps(report["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "### Pipeline",
        "",
        f"- Dataset: `{report['dataset_id']}` ({summary.get('licenses')})",
        f"- Input files discovered: `{stats['input_files']}`",
        f"- Files skipped (corrupt/empty/short): `{stats['skipped_files']}`",
        f"- Segments produced: `{stats['segments']}`",
        f"- Source songs: `{stats['tracks']}`",
        f"- Singers: `{stats['singers']}`",
        f"- Total audio: `{stats['total_seconds']}` s",
        f"- Target format: `{stats['sample_rate']} Hz`, `{stats['channels']}` ch",
        "",
        "### Splits",
        "",
        "| Split | Segments | Songs | Singers |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(splits):
        entry = splits[name]
        lines.append(f"| {name} | {entry['segments']} | {entry['tracks']} | {entry['singers']} |")

    lines.extend(
        [
            "",
            f"- Split mode: `{report['split_config']['mode']}`, seed `{report['split_config']['seed']}`",
            f"- Song leakage: `{report['leakage']['song']}`",
            f"- Singer leakage: `{report['leakage']['singer']}`",
            f"- Cross-split duplicate audio: `{report['duplicates']['cross_split_duplicate_groups']}`",
            "",
            "### Persistence",
            "",
            f"- Output directory: `{report['output_dir']}`",
            f"- Manifest round-trip verified: `{report['round_trip_ok']}`",
            f"- All artifacts present: `{report['persistence']['ok']}`",
            "",
            "### Canonical Manifest",
            "",
            f"Schema: `{summary['schema']}`. One record per segment; see `docs/MANIFEST_SCHEMA.md`.",
            "",
            "### Milestone Note",
            "",
            "M03 codec work predates M02 and remains an experimental spike. It is not accepted by",
            "this run. M02 only certifies that dataset preprocessing is production-ready.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M02 acceptance on Google Colab.")
    parser.add_argument("--dataset-id", default="babyslakh")
    parser.add_argument("--audio-dir", default=None, help="Directory of approved raw audio.")
    parser.add_argument("--output-dir", required=True, help="Drive path for processed audio and manifests.")
    parser.add_argument("--config", default="configs/data/preprocess_m02.yaml")
    parser.add_argument("--registry", default="configs/data/datasets.yaml")
    parser.add_argument("--patterns", nargs="+", default=["*.wav", "*.flac"])
    parser.add_argument("--limit-files", type=int, default=None, help="Keep the acceptance run small.")
    parser.add_argument("--split-mode", default=None, choices=["song", "singer"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Rehearse with fixtures; never an M02 pass.")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--log-path", default="docs/EXPERIMENT_LOG_M02.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = environment_report()

    tests = {"command": ["skipped"], "returncode": 0, "stdout": ""}
    if not args.skip_tests:
        tests = run_command(
            [sys.executable, "-m", "pytest", "-q",
             "tests/test_dsp.py", "tests/test_media.py",
             "tests/test_preprocess.py", "tests/test_splits.py", "tests/test_manifest.py"]
        )
        if tests["returncode"] != 0:
            raise SystemExit("M02 tests failed; not running acceptance.")

    if args.synthetic:
        audio_dir = build_synthetic_corpus(output_dir / "synthetic_raw", seconds=6.0)
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
    else:
        raise SystemExit("Provide --audio-dir with approved audio, or --synthetic to rehearse.")

    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    import yaml

    with Path(args.config).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    preprocess_config = PreprocessConfig.from_dict(raw_config.get("preprocess", {}))

    split_payload = dict(raw_config.get("splits", {}))
    if args.split_mode:
        split_payload["mode"] = args.split_mode
    if args.seed is not None:
        split_payload["seed"] = args.seed
    split_config = SplitConfig.from_dict(split_payload)
    split_config.validate()

    registry = load_dataset_registry(args.registry)
    provenance = provenance_from_registry(registry, args.dataset_id)

    paths = find_audio_files(audio_dir, tuple(args.patterns))
    if args.limit_files:
        paths = paths[: args.limit_files]
    if not paths:
        raise SystemExit(f"No audio matching {args.patterns} under {audio_dir}")
    print(f"\nPreprocessing {len(paths)} files from {audio_dir}")

    result = preprocess_paths(
        paths,
        config=preprocess_config,
        provenance=provenance,
        output_dir=output_dir,
        dataset_id=args.dataset_id,
        source_root=audio_dir,
    )
    if not result.records:
        write_preprocess_report(result, output_dir / "preprocess_report.json")
        raise SystemExit("No usable segments produced. See preprocess_report.json.")

    records = assign_splits(result.records, split_config)

    validation_errors = validate_records(records)
    leakage = {"song": None, "singer": None, "duplicates": None}
    try:
        assert_no_track_leakage(records)
        assert_group_disjoint(records, split_config.mode)
    except ValueError as exc:
        leakage["song"] = str(exc)
    try:
        assert_singer_disjoint(records)
    except ValueError as exc:
        leakage["singer"] = str(exc)
    try:
        assert_no_cross_split_duplicates(records)
    except ValueError as exc:
        leakage["duplicates"] = str(exc)

    provenance_error = None
    try:
        assert_provenance_complete(records)
    except ValueError as exc:
        provenance_error = str(exc)

    manifest_dir = output_dir / "manifests"
    written = write_split_manifests(records, manifest_dir)
    write_preprocess_report(result, output_dir / "preprocess_report.json")

    round_trip_ok = read_jsonl(written["all"]) == records
    persistence = verify_persistence(written, [Path(record.path) for record in records[:25]])

    summary = manifest_summary(records)
    duplicates = duplicate_report(records)
    acceptance_pass = bool(
        not args.synthetic
        and not validation_errors
        and provenance_error is None
        and all(value is None for value in leakage.values())
        and round_trip_ok
        and persistence["ok"]
        and summary["segments"] > 0
        and len(summary["splits"]) >= 2
    )

    report = {
        "dataset_id": args.dataset_id,
        "synthetic": bool(args.synthetic),
        "environment": environment,
        "tests": tests,
        "audio_dir": str(audio_dir),
        "output_dir": str(output_dir),
        "preprocess": result.stats,
        "skipped": result.skipped[:50],
        "manifest": summary,
        "splits": split_report(records, split_config.mode),
        "split_config": {
            "train": split_config.train, "val": split_config.val, "test": split_config.test,
            "seed": split_config.seed, "mode": split_config.mode, "strategy": split_config.strategy,
        },
        "validation_errors": validation_errors[:20],
        "provenance_error": provenance_error,
        "leakage": leakage,
        "duplicates": duplicates,
        "round_trip_ok": round_trip_ok,
        "persistence": persistence,
        "acceptance_pass": acceptance_pass,
    }

    (output_dir / "m02_acceptance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    log_path = ROOT / args.log_path
    write_experiment_log(log_path, report)

    print("\n" + json.dumps(
        {
            "segments": summary["segments"],
            "tracks": summary["tracks"],
            "splits": {name: entry["segments"] for name, entry in summary["splits"].items()},
            "leakage": leakage,
            "round_trip_ok": round_trip_ok,
            "persistence_ok": persistence["ok"],
            "acceptance_pass": acceptance_pass,
        },
        indent=2,
        sort_keys=True,
    ))
    print(f"\nmanifests   : {manifest_dir}")
    print(f"report      : {output_dir / 'm02_acceptance_report.json'}")
    print(f"log         : {log_path}")

    if args.synthetic:
        print("\nSynthetic rehearsal complete. Rerun with --audio-dir on real approved audio for M02 PASS.")
        return
    if not acceptance_pass:
        raise SystemExit(f"{milestone('M02')} = FAIL. Inspect m02_acceptance_report.json.")
    print(f"\n{milestone('M02')} = PASS")


if __name__ == "__main__":
    main()
