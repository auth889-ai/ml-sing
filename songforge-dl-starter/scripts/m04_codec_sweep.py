"""M04 — High-Quality Codec Optimization & Latent-Rate Selection: candidate sweep.

Trains each codec candidate in a completely isolated run directory, on the same
M02 data, with the same training budget, the same deterministic held-out probe,
and the same evaluation procedure, then aggregates one comparison table.

Staged on purpose. Stage 1 establishes the quality-versus-temporal-rate curve at
constant depth (120 / 75 / 50 Hz, Q=2). Stage 2 is only worth running if the
lower rates give up too much quality, and asks whether extra codebook depth buys
it back at comparable bitrate.

    python scripts/m04_codec_sweep.py --stage 1 \
        --train-manifest "$SONGFORGE_DATA/processed/babyslakh_m02/manifests/train.jsonl" \
        --val-manifest   "$SONGFORGE_DATA/processed/babyslakh_m02/manifests/val.jsonl" \
        --output-root    "$DRIVE_ROOT/outputs/m04_codec_sweep" \
        --steps 4000

Every candidate reuses the M03 reliability machinery: atomic periodic
checkpoints, strict RNG/optimizer/scaler restoration, one run_id across resumes,
crash-tail reconciliation. Re-running the same command resumes whatever was
interrupted instead of starting a second experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from songforge.milestones import milestone
from songforge.training.run import curve_run_ids, read_run_manifest

ROOT = Path(__file__).resolve().parents[1]

STAGE_1 = ["m04_baseline_120hz_q2", "m04_a_75hz_q2", "m04_b_50hz_q2"]
STAGE_2 = ["m04_a2_75hz_q4", "m04_b2_50hz_q4", "m04_c_25hz_q8"]


def candidate_config(name: str) -> Path:
    path = ROOT / "configs" / "codec" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"Unknown candidate {name!r} (no {path})")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_curve(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "time":
                    continue
                if key == "run_id":
                    row["run_id"] = value
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def train_candidate(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Train one candidate in its own directory, resuming if it was interrupted."""
    config = candidate_config(name)
    run_dir = Path(args.output_root) / name
    latest = run_dir / "checkpoint_latest.pt"
    resuming = bool(read_run_manifest(run_dir) and latest.exists())

    command = [
        sys.executable, "scripts/train_codec.py",
        "--config", str(config),
        "--train-manifest", args.train_manifest,
        "--val-manifest", args.val_manifest,
        "--output-dir", str(run_dir),
        "--steps", str(args.steps),
        "--rvq-log-every", str(args.rvq_log_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--eval-max-segments", str(args.eval_max_segments),
        "--validation-probe", str(args.validation_probe),
        "--export-examples", str(args.export_examples),
        "--run-label", name,
        "--device", args.device,
    ]
    if resuming:
        command += ["--resume", str(latest)]
    if args.device == "cuda":
        command.append("--amp")

    print(f"\n{'=' * 78}\n{name}  ({'resuming' if resuming else 'fresh'})\n{'=' * 78}", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, check=False)
    return {"candidate": name, "returncode": result.returncode, "run_dir": str(run_dir)}


def collect_candidate(name: str, output_root: Path) -> dict[str, Any]:
    """Read one candidate's artifacts into a comparable row."""
    run_dir = output_root / name
    config = yaml.safe_load(candidate_config(name).read_text(encoding="utf-8"))
    spec = config.get("m04", {})

    train_metrics = read_json(run_dir / "train_metrics.json")
    val_metrics = read_json(run_dir / "validation_metrics.json")
    probe = read_json(run_dir / "validation_probe.json")
    metadata = read_json(run_dir / "experiment_metadata.json")
    examples = read_json(run_dir / "listening_examples.json")
    manifest = read_run_manifest(run_dir) or {}
    curve = read_curve(run_dir / "training_curves.csv")

    steps = [int(row["step"]) for row in curve if "step" in row]
    window = max(len(curve) // 10, 1) if curve else 0

    def window_mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row[key] for row in rows if key in row]
        return sum(values) / len(values) if values else None

    history_path = run_dir / "rvq_history.jsonl"
    history = (
        [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if history_path.exists()
        else []
    )
    collapse_steps = [int(row["step"]) for row in history if row.get("rvq_collapse_suspected")]
    final_history = history[-1] if history else {}

    after = (probe or {}).get("after", {}) or {}
    before = (probe or {}).get("before", {}) or {}

    return {
        "candidate": name,
        "stage": spec.get("stage"),
        "note": spec.get("note"),
        "complete": bool(train_metrics and probe and after),
        # --- representation ---
        "latent_frame_rate_hz": spec.get("latent_frame_rate_hz"),
        "downsample_factor": spec.get("downsample_factor"),
        "num_quantizers": spec.get("num_quantizers"),
        "codebook_size": spec.get("codebook_size"),
        "bits_per_code": spec.get("bits_per_code"),
        "bitrate_bps": spec.get("bitrate_bps"),
        "pcm16_compression_ratio": spec.get("pcm16_compression_ratio"),
        "raw_codec_codes_per_10s": spec.get("raw_codec_codes_per_10s"),
        "raw_codec_codes_per_30s": spec.get("raw_codec_codes_per_30s"),
        "raw_codec_codes_per_180s": spec.get("raw_codec_codes_per_180s"),
        "strides": config.get("model", {}).get("strides"),
        # --- held-out probe, identical segments across candidates ---
        "probe_fingerprint": probe.get("probe_fingerprint"),
        "probe_segments": probe.get("segments"),
        "same_probe_before_and_after": probe.get("same_probe_before_and_after"),
        "val_recon_loss_before": before.get("recon_loss"),
        "val_recon_loss_after": after.get("recon_loss"),
        "val_l1_after": after.get("waveform_l1"),
        "val_mrstft_after": after.get("mrstft"),
        "val_snr_db_after": after.get("snr_db"),
        "val_si_sdr_db_after": after.get("si_sdr_db"),
        "val_log_spectral_distance_db_after": after.get("log_spectral_distance_db"),
        "val_transient_preservation_after": after.get("transient_preservation"),
        "val_hf_preservation_db_after": after.get("high_frequency_preservation_db"),
        "validation_improved": probe.get("improved"),
        # --- whole-evaluation metrics ---
        "eval_val_l1": val_metrics.get("waveform_l1"),
        "eval_val_mrstft": val_metrics.get("mrstft"),
        "eval_val_si_sdr_db": val_metrics.get("si_sdr_db"),
        "eval_val_transient": val_metrics.get("transient_preservation"),
        "eval_val_hf_db": val_metrics.get("high_frequency_preservation_db"),
        "eval_segments": val_metrics.get("evaluated_segments"),
        # --- RVQ ---
        "rvq_utilization_avg": val_metrics.get("codebook_utilization_avg"),
        "rvq_utilization_per_codebook": [e.get("utilization") for e in val_metrics.get("per_codebook", [])],
        "rvq_perplexity_per_codebook": [e.get("perplexity") for e in val_metrics.get("per_codebook", [])],
        "rvq_entropy_per_codebook": [e.get("entropy") for e in val_metrics.get("per_codebook", [])],
        "rvq_dead_codes_per_codebook": [e.get("dead_codes") for e in val_metrics.get("per_codebook", [])],
        "rvq_collapse_final": bool(final_history.get("rvq_collapse_suspected")) if final_history else None,
        "rvq_first_collapse_step": min(collapse_steps) if collapse_steps else None,
        "rvq_last_collapse_step": max(collapse_steps) if collapse_steps else None,
        "rvq_collapsed_snapshots": len(collapse_steps),
        "rvq_min_utilization": min(
            (float(row["codebook_utilization_avg"]) for row in history), default=None
        ),
        # --- training curve ---
        "train_loss_first_window": window_mean(curve[:window], "loss"),
        "train_loss_last_window": window_mean(curve[-window:], "loss") if window else None,
        "train_l1_first_window": window_mean(curve[:window], "waveform_l1"),
        "train_l1_last_window": window_mean(curve[-window:], "waveform_l1") if window else None,
        # --- cost ---
        "encode_decode_seconds": val_metrics.get("encode_decode_seconds"),
        "encode_decode_rtf": val_metrics.get("real_time_factor"),
        "peak_gpu_vram_gb": train_metrics.get("peak_gpu_memory_gb"),
        "train_wall_seconds": (metadata.get("throughput", {}) or {}).get("train_wall_seconds"),
        "train_steps_per_second": (metadata.get("throughput", {}) or {}).get("steps_per_second"),
        # --- integrity ---
        "run_id": manifest.get("run_id"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "resumes": len(manifest.get("resumes", [])),
        "all_resumes_authoritative": all(
            event.get("rng_restored") and event.get("authoritative")
            for event in manifest.get("resumes", [])
        ),
        "curve_rows": len(curve),
        "curve_unique_steps": len(set(steps)),
        "curve_duplicate_steps": len(steps) - len(set(steps)),
        "curve_run_ids": curve_run_ids(curve),
        "listening_examples": {k: v.get("original") for k, v in (examples or {}).items()},
        "listening_example_count": len(examples or {}),
    }


def integrity_ok(row: dict[str, Any]) -> bool:
    return bool(
        row["complete"]
        and row["curve_duplicate_steps"] == 0
        and len(row["curve_run_ids"]) <= 1
        and row["all_resumes_authoritative"]
    )


def write_comparison(rows: list[dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
    """Objective comparison table. No selection is made here; that is a human call."""
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        f"# {milestone('M04')}: candidate comparison",
        "",
        "All candidates share the same M02 train/validation manifests, the same training",
        "budget, the same deterministic held-out probe segments, and the same evaluation",
        "procedure. Each ran in its own isolated directory.",
        "",
        "## Representation and long-form cost",
        "",
        "| Candidate | Hz | factor | Q | K | bits | bitrate | ratio | codes/10s | codes/30s | codes/3min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['latent_frame_rate_hz'],1)} | {row['downsample_factor']} | "
            f"{row['num_quantizers']} | {row['codebook_size']} | {row['bits_per_code']} | "
            f"{fmt(row['bitrate_bps'],0)} | {fmt(row['pcm16_compression_ratio'],1)} | "
            f"{fmt(row['raw_codec_codes_per_10s'],0)} | {fmt(row['raw_codec_codes_per_30s'],0)} | "
            f"{fmt(row['raw_codec_codes_per_180s'],0)} |"
        )

    lines += [
        "",
        "## Held-out reconstruction quality (identical probe segments)",
        "",
        "| Candidate | recon before | recon after | L1 | MR-STFT | SNR dB | SI-SDR dB | LSD dB | transient | HF dB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['val_recon_loss_before'])} | {fmt(row['val_recon_loss_after'])} | "
            f"{fmt(row['val_l1_after'])} | {fmt(row['val_mrstft_after'])} | {fmt(row['val_snr_db_after'],2)} | "
            f"{fmt(row['val_si_sdr_db_after'],2)} | {fmt(row['val_log_spectral_distance_db_after'],2)} | "
            f"{fmt(row['val_transient_preservation_after'],3)} | {fmt(row['val_hf_preservation_db_after'],2)} |"
        )

    lines += [
        "",
        "## RVQ health (whole evaluation sample)",
        "",
        "| Candidate | utilization | per codebook | perplexity | dead codes | collapsed snaps | first | last | final collapse |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['rvq_utilization_avg'],3)} | "
            f"{[round(v,3) for v in row['rvq_utilization_per_codebook'] if v is not None]} | "
            f"{[round(v,1) for v in row['rvq_perplexity_per_codebook'] if v is not None]} | "
            f"{[int(v) for v in row['rvq_dead_codes_per_codebook'] if v is not None]} | "
            f"{row['rvq_collapsed_snapshots']} | {row['rvq_first_collapse_step']} | "
            f"{row['rvq_last_collapse_step']} | {row['rvq_collapse_final']} |"
        )

    lines += [
        "",
        "## Cost and integrity",
        "",
        "| Candidate | enc+dec RTF | peak VRAM GB | wall s | steps/s | run_id | curve rows | dup | resumes ok | integrity |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['encode_decode_rtf'],5)} | {fmt(row['peak_gpu_vram_gb'],4)} | "
            f"{fmt(row['train_wall_seconds'],1)} | {fmt(row['train_steps_per_second'],2)} | "
            f"`{row['run_id']}` | {row['curve_rows']} | {row['curve_duplicate_steps']} | "
            f"{row['all_resumes_authoritative']} | {integrity_ok(row)} |"
        )

    lines += [
        "",
        "## Listening examples",
        "",
        "Character buckets are spectral heuristics, not instrument ground truth. The same",
        "held-out source segments are used for every candidate, so the pairs are directly",
        "comparable A/B.",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['candidate']}`: {row['listening_example_count']} pairs in `{row['candidate']}/examples/`")

    lines += [
        "",
        "No selection is recorded in this file. Selection and rationale live in",
        "`docs/M04_CODEC_SELECTION.md` and must cite these numbers.",
        "",
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    md = output_root / "m04_comparison.md"
    js = output_root / "m04_comparison.json"
    md.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return md, js


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{milestone('M04')} candidate sweep.")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None)
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--rvq-log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--eval-max-segments", type=int, default=256)
    parser.add_argument("--validation-probe", type=int, default=64)
    parser.add_argument("--export-examples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-only", action="store_true", help="Aggregate existing runs without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidates:
        names = args.candidates
    elif args.stage == 1:
        names = STAGE_1
    elif args.stage == 2:
        names = STAGE_2
    else:
        names = STAGE_1 + STAGE_2

    output_root = Path(args.output_root)
    print(f"{milestone('M04')}\ncandidates: {', '.join(names)}\noutput root: {output_root}")

    if not args.report_only:
        for name in names:
            outcome = train_candidate(name, args)
            if outcome["returncode"] != 0:
                raise SystemExit(
                    f"Candidate {name} failed (return code {outcome['returncode']}). "
                    "Fix it or rerun to resume; other candidates are untouched."
                )

    rows = [collect_candidate(name, output_root) for name in names]
    md, js = write_comparison(rows, output_root)

    print("\n" + json.dumps(
        [
            {
                "candidate": r["candidate"], "hz": r["latent_frame_rate_hz"], "Q": r["num_quantizers"],
                "bitrate": r["bitrate_bps"], "val_recon_after": r["val_recon_loss_after"],
                "si_sdr_db": r["val_si_sdr_db_after"], "transient": r["val_transient_preservation_after"],
                "hf_db": r["val_hf_preservation_db_after"], "rvq_util": r["rvq_utilization_avg"],
                "collapse_final": r["rvq_collapse_final"], "integrity": integrity_ok(r),
            }
            for r in rows
        ],
        indent=2, sort_keys=True,
    ))
    print(f"\ncomparison: {md}\n            {js}")
    incomplete = [r["candidate"] for r in rows if not r["complete"]]
    if incomplete:
        print(f"\nincomplete candidates (rerun to resume): {', '.join(incomplete)}")


if __name__ == "__main__":
    main()
