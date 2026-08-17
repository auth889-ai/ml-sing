"""M03 — Neural Audio Codec & Discrete Audio Representation: Colab CUDA acceptance runner.

Trains the RVQ codec from randomly initialized SongForge weights on the real
M02 — Audio Preprocessing & Dataset Pipeline canonical manifest, verifies
checkpoint resume, exports held-out listening examples, and records the full
metric set required by the gate.

    python scripts/colab_m03_acceptance.py \
        --config configs/codec/codec_m03_tiny.yaml \
        --train-manifest "$M02_OUTPUT_DIR/manifests/train.jsonl" \
        --val-manifest   "$M02_OUTPUT_DIR/manifests/val.jsonl" \
        --output-dir     "$DRIVE_ROOT/outputs/codec_m03_acceptance" \
        --steps 4000

No pretrained codec weights are used or loaded anywhere in this path.
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

import torch

from songforge.milestones import milestone
from songforge.training.run import curve_run_ids, read_run_manifest

ROOT = Path(__file__).resolve().parents[1]

CUDA_TESTS = ("test_codec_cuda_smoke", "test_codec_cuda_amp_smoke")


def run_command(command: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def environment_report(require_cuda: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_total_vram_bytes": props.total_memory,
                "gpu_total_vram_gb": props.total_memory / (1024**3),
            }
        )
    else:
        report.update({"gpu_name": None, "gpu_total_vram_bytes": 0, "gpu_total_vram_gb": 0.0})
    print(json.dumps(report, indent=2, sort_keys=True))
    if require_cuda and not report["cuda_available"]:
        raise SystemExit(f"{milestone('M03')} requires CUDA. Select a Colab GPU runtime and rerun.")
    return report


def cuda_test_report(require_cuda: bool) -> dict[str, Any]:
    """Prove the CUDA and AMP codec tests actually ran instead of silently skipping."""
    result = run_command(
        [sys.executable, "-m", "pytest", "-v", "tests/test_codec.py", "-k", "cuda"]
    )
    output = result["stdout"] + result["stderr"]

    # pytest -v prints "tests/test_codec.py::<name> PASSED" (or SKIPPED) per test.
    outcomes = {}
    for name in CUDA_TESTS:
        outcome = "missing"
        for line in output.splitlines():
            if f"::{name}" not in line:
                continue
            if "PASSED" in line:
                outcome = "passed"
            elif "SKIPPED" in line:
                outcome = "skipped"
            elif "FAILED" in line or "ERROR" in line:
                outcome = "failed"
            break
        outcomes[name] = outcome

    all_passed = all(state == "passed" for state in outcomes.values())
    report = {
        "command": result["command"],
        "returncode": result["returncode"],
        "outcomes": outcomes,
        "all_cuda_tests_passed": all_passed,
        "stdout_tail": output[-3000:],
    }
    if require_cuda and not all_passed:
        raise SystemExit(
            "CUDA/AMP codec tests did not pass (they may have been skipped). "
            f"Outcomes: {outcomes}. {milestone('M03')} requires a real GPU runtime."
        )
    return report


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_curve(path: Path) -> list[dict[str, Any]]:
    """Parse the training curve, keeping `run_id` so splicing stays detectable."""
    import csv

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


def analyze_rvq_history(path: Path) -> dict[str, Any]:
    """Summarize codebook health across the whole run, not just the final step.

    The RVQ is expected to dip early while the encoder still emits near-constant
    latents and then recover. A temporary collapse is reported explicitly; only a
    collapse that survives to the end fails acceptance.
    """
    if not path.exists():
        return {"available": False, "reason": f"{path.name} not written"}

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"available": False, "reason": "no RVQ snapshots recorded"}

    collapse_steps = [int(row["step"]) for row in rows if row.get("rvq_collapse_suspected")]
    utilization = [(int(row["step"]), float(row["codebook_utilization_avg"])) for row in rows]
    perplexity = [(int(row["step"]), float(row["codebook_perplexity_avg"])) for row in rows]
    final = rows[-1]
    final_collapsed = bool(final.get("rvq_collapse_suspected"))

    return {
        "available": True,
        "rows": rows,
        "snapshots": len(rows),
        "first_step": int(rows[0]["step"]),
        "final_step": int(final["step"]),
        "collapsed_snapshots": len(collapse_steps),
        "first_collapse_step": min(collapse_steps) if collapse_steps else None,
        "last_collapse_step": max(collapse_steps) if collapse_steps else None,
        "final_collapse": final_collapsed,
        "temporary_collapse": bool(collapse_steps) and not final_collapsed,
        "recovered": bool(collapse_steps) and not final_collapsed,
        "never_collapsed": not collapse_steps,
        "min_utilization": min(value for _, value in utilization),
        "max_utilization": max(value for _, value in utilization),
        "final_utilization": utilization[-1][1],
        "min_perplexity": min(value for _, value in perplexity),
        "final_perplexity": perplexity[-1][1],
        "final_per_codebook": final.get("per_codebook", []),
        "utilization_curve": utilization,
        "perplexity_curve": perplexity,
    }


def validate_acceptance(output_dir: Path) -> dict[str, Any]:
    required = [
        "original.wav",
        "reconstructed.wav",
        "config.yaml",
        "checkpoint.pt",
        "checkpoint_last.pt",
        "optimizer_state.pt",
        "training_curves.csv",
        "metrics.jsonl",
        "metrics_summary.json",
        "train_metrics.json",
        "validation_metrics.json",
        "compression_stats.json",
        "experiment_metadata.json",
        "rvq_history.jsonl",
        "listening_examples.json",
        "run_manifest.json",
        "validation_before.json",
        "validation_probe.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]

    train_metrics = read_json(output_dir / "train_metrics.json") if not missing else {}
    val_metrics = read_json(output_dir / "validation_metrics.json") if not missing else {}
    compression = read_json(output_dir / "compression_stats.json") if not missing else {}
    metadata = read_json(output_dir / "experiment_metadata.json") if not missing else {}
    examples = read_json(output_dir / "listening_examples.json") if not missing else {}
    validation_probe = read_json(output_dir / "validation_probe.json") if not missing else {}
    curve = read_curve(output_dir / "training_curves.csv") if not missing else []
    rvq = analyze_rvq_history(output_dir / "rvq_history.jsonl")

    # Judge the trend on windowed means, not on the two endpoint batches.
    #
    # Per-step loss is a single batch of `batch_size` random segments drawn from
    # heterogeneous real music, so it is extremely noisy: the first Colab run on
    # BabySlakh spanned 0.0076 to 1.1865, a 156x spread. Comparing curve[0] to
    # curve[-1] therefore samples noise rather than learning, and reported a
    # rising loss on a run whose windowed L1 fell 38% and MR-STFT fell 32%.
    # Averaging the first and last decile is the standard way to read such a
    # curve and uses far more of the evidence than two points.
    window = max(len(curve) // 10, 1) if curve else 0

    def _window_mean(rows: list[dict[str, float]], key: str) -> float | None:
        values = [row[key] for row in rows if key in row]
        return sum(values) / len(values) if values else None

    first_loss = curve[0]["loss"] if curve else None
    final_loss = curve[-1]["loss"] if curve else None
    first_window = curve[:window]
    final_window = curve[-window:] if window else []

    first_window_loss = _window_mean(first_window, "loss")
    final_window_loss = _window_mean(final_window, "loss")
    first_window_l1 = _window_mean(first_window, "waveform_l1")
    final_window_l1 = _window_mean(final_window, "waveform_l1")
    first_window_mrstft = _window_mean(first_window, "mrstft")
    final_window_mrstft = _window_mean(final_window, "mrstft")

    loss_decreased = (
        first_window_loss is not None
        and final_window_loss is not None
        and final_window_loss < first_window_loss
    )

    # Trust the tracked history for the collapse verdict when it exists; fall
    # back to the final evaluation metrics otherwise.
    if rvq.get("available"):
        collapse = bool(rvq["final_collapse"])
    else:
        collapse = bool(train_metrics.get("rvq_collapse_suspected", 1.0)) or bool(
            val_metrics.get("rvq_collapse_suspected", 1.0)
        )

    used_m02_manifest = metadata.get("path_source", "").startswith("m02_manifest")

    # Experiment isolation: the curve and the RVQ history must each come from a
    # single run. A directory reused by two runs concatenates both, and the
    # windowed verdict would then compare one run's start to another's end.
    curve_steps = [row.get("step") for row in curve if row.get("step") is not None]
    curve_ids = curve_run_ids(curve)
    history_rows = rvq.get("rows", [])
    history_ids = curve_run_ids(history_rows)
    history_steps = [row.get("step") for row in history_rows]

    isolation = {
        "curve_rows": len(curve),
        "curve_unique_steps": len(set(curve_steps)),
        "curve_duplicate_steps": len(curve_steps) - len(set(curve_steps)),
        "curve_run_ids": curve_ids,
        "rvq_rows": len(history_rows),
        "rvq_duplicate_steps": len(history_steps) - len(set(history_steps)),
        "rvq_run_ids": history_ids,
        "run_manifest": read_run_manifest(output_dir),
    }
    # Every resume of an authoritative run must have restored the full training
    # state. A resume that silently restarted the RNG stream is a different
    # experiment wearing the same run_id.
    resumes = (isolation["run_manifest"] or {}).get("resumes", [])
    bad_resumes = [
        event for event in resumes if not (event.get("rng_restored") and event.get("authoritative"))
    ]
    isolation["resumes"] = len(resumes)
    isolation["resume_events"] = resumes
    isolation["non_authoritative_resumes"] = len(bad_resumes)
    isolation["rng_restored_on_every_resume"] = not bad_resumes

    # Steps must be contiguous: a gap means a session's rows were lost, not merely
    # duplicated.
    expected = set(range(int(min(curve_steps)), int(max(curve_steps)) + 1)) if curve_steps else set()
    isolation["curve_gaps"] = len(expected - {int(step) for step in curve_steps})

    isolation["ok"] = bool(
        len(curve_ids) <= 1
        and len(history_ids) <= 1
        and isolation["curve_duplicate_steps"] == 0
        and isolation["rvq_duplicate_steps"] == 0
        and isolation["curve_gaps"] == 0
        and not bad_resumes
    )
    if not isolation["ok"]:
        isolation["reason"] = (
            "Artifacts contain rows from more than one run "
            f"(curve run ids: {curve_ids or 'unstamped'}, "
            f"duplicate curve steps: {isolation['curve_duplicate_steps']}, "
            f"duplicate rvq steps: {isolation['rvq_duplicate_steps']}). "
            "This directory was reused; rerun into a fresh --output-dir."
        )

    return {
        "missing_artifacts": missing,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss_window_steps": window,
        "first_window_loss": first_window_loss,
        "final_window_loss": final_window_loss,
        "first_window_waveform_l1": first_window_l1,
        "final_window_waveform_l1": final_window_l1,
        "first_window_mrstft": first_window_mrstft,
        "final_window_mrstft": final_window_mrstft,
        "loss_decreased": loss_decreased,
        "rvq_collapse_suspected": collapse,
        "rvq_history": rvq,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "compression": compression,
        "metadata": metadata,
        "listening_examples": examples,
        "listening_example_count": len(examples),
        "used_m02_manifest": used_m02_manifest,
        "isolation": isolation,
        "validation_probe": validation_probe,
        "validation_improved": bool(validation_probe.get("improved")),
        "acceptance_pass": bool(
            not missing
            and loss_decreased
            and not collapse
            and used_m02_manifest
            and isolation["ok"]
            and bool(validation_probe.get("after"))
            and bool(validation_probe.get("same_probe_before_and_after"))
        ),
    }


def collect_required_metrics(acceptance: dict[str, Any]) -> dict[str, Any]:
    """The exact metric set the M03 gate asks to have recorded."""
    train = acceptance.get("train_metrics", {})
    val = acceptance.get("validation_metrics", {})
    compression = acceptance.get("compression", {})
    rvq = acceptance.get("rvq_history", {})
    metadata = acceptance.get("metadata", {})
    throughput = metadata.get("throughput", {})
    probe = acceptance.get("validation_probe", {}) or {}
    before = probe.get("before", {}) or {}
    after = probe.get("after", {}) or {}

    return {
        "validation_probe_segments": probe.get("segments"),
        "validation_probe_fingerprint": probe.get("probe_fingerprint"),
        "validation_same_probe_before_and_after": probe.get("same_probe_before_and_after"),
        "validation_recon_loss_before": before.get("recon_loss"),
        "validation_recon_loss_after": after.get("recon_loss"),
        "validation_l1_before": before.get("waveform_l1"),
        "validation_l1_after": after.get("waveform_l1"),
        "validation_mrstft_before": before.get("mrstft"),
        "validation_mrstft_after": after.get("mrstft"),
        "validation_snr_db_before": before.get("snr_db"),
        "validation_snr_db_after": after.get("snr_db"),
        "validation_improved": probe.get("improved"),
        "initial_train_loss": acceptance.get("first_loss"),
        "final_train_loss": acceptance.get("final_loss"),
        "loss_window_steps": acceptance.get("loss_window_steps"),
        "initial_train_loss_windowed": acceptance.get("first_window_loss"),
        "final_train_loss_windowed": acceptance.get("final_window_loss"),
        "initial_waveform_l1_windowed": acceptance.get("first_window_waveform_l1"),
        "final_waveform_l1_windowed": acceptance.get("final_window_waveform_l1"),
        "initial_mrstft_windowed": acceptance.get("first_window_mrstft"),
        "final_mrstft_windowed": acceptance.get("final_window_mrstft"),
        "train_waveform_l1": train.get("waveform_l1"),
        "train_mrstft": train.get("mrstft"),
        "train_snr_db": train.get("snr_db"),
        "validation_waveform_l1": val.get("waveform_l1"),
        "validation_mrstft": val.get("mrstft"),
        "validation_snr_db": val.get("snr_db"),
        "validation_spectral_convergence": val.get("spectral_convergence"),
        "latent_frame_rate_hz": compression.get("latent_frame_rate_hz"),
        "downsample_factor": compression.get("downsample_factor"),
        "num_codebooks": compression.get("code_streams"),
        "codebook_size": train.get("codebook_size"),
        "bits_per_code": compression.get("bits_per_code"),
        "discrete_bitrate_bps": compression.get("discrete_bitrate_bps"),
        "pcm16_compression_ratio": compression.get("pcm16_compression_ratio"),
        "codebook_utilization_avg": train.get("codebook_utilization_avg"),
        "codebook_utilization_min": train.get("codebook_utilization_min"),
        "codebook_utilization_per_codebook": [
            entry.get("utilization") for entry in train.get("per_codebook", [])
        ],
        "codebook_dead_codes_avg": train.get("codebook_dead_codes_avg"),
        "codebook_dead_codes_per_codebook": [
            entry.get("dead_codes") for entry in train.get("per_codebook", [])
        ],
        "codebook_entropy_avg": train.get("codebook_entropy_avg"),
        "codebook_entropy_per_codebook": [entry.get("entropy") for entry in train.get("per_codebook", [])],
        "codebook_perplexity_avg": train.get("codebook_perplexity_avg"),
        "codebook_perplexity_per_codebook": [
            entry.get("perplexity") for entry in train.get("per_codebook", [])
        ],
        "rvq_collapse_flag_final": acceptance.get("rvq_collapse_suspected"),
        "rvq_temporary_collapse": rvq.get("temporary_collapse"),
        "rvq_recovered": rvq.get("recovered"),
        "rvq_min_utilization_during_training": rvq.get("min_utilization"),
        "peak_gpu_vram_gb": train.get("peak_gpu_memory_gb"),
        "gpu_name": train.get("gpu_name"),
        "train_steps_per_second": throughput.get("steps_per_second"),
        "train_audio_seconds_per_second": throughput.get("audio_seconds_per_second"),
        "train_wall_seconds": throughput.get("train_wall_seconds"),
        "encode_decode_seconds": train.get("encode_decode_seconds"),
        "encode_decode_real_time_factor": train.get("real_time_factor"),
    }


def write_experiment_log(path: Path, report: dict[str, Any]) -> None:
    acceptance = report["acceptance"]
    metrics = report["metrics"]
    rvq = acceptance.get("rvq_history", {})
    status = "PASS" if acceptance["acceptance_pass"] and report["resume"]["returncode"] == 0 else "FAIL"

    def fmt(value, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}g}"
        return str(value)

    lines = [
        f"# Experiment Log — {milestone('M03')}",
        "",
        f"## {milestone('M03')}: Colab Final Acceptance",
        "",
        f"Status: **{milestone('M03')} = {status}**",
        "",
        f"Run: `{report['run_label']}`",
        "",
        "### Environment",
        "",
        "```json",
        json.dumps(report["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "### Data Source (M02 canonical manifest)",
        "",
        f"- Train manifest: `{report['train_manifest']}`",
        f"- Validation manifest: `{report['val_manifest']}`",
        f"- Path source recorded by trainer: `{acceptance.get('metadata', {}).get('path_source')}`",
        f"- Train files: `{acceptance.get('metadata', {}).get('train_file_count')}`",
        f"- Validation files: `{acceptance.get('metadata', {}).get('val_file_count')}`",
        "- Weights: randomly initialized SongForge codec. No pretrained codec weights are loaded.",
        "",
        "### Commands",
        "",
    ]
    for key in ("pytest", "cuda_tests", "train", "resume"):
        entry = report[key]
        lines.extend(
            ["```bash", " ".join(entry["command"]), "```", f"Return code: `{entry['returncode']}`", ""]
        )

    lines.extend(
        [
            "### Required Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Initial train loss (single batch, noisy) | `{fmt(metrics['initial_train_loss'])}` |",
            f"| Final train loss (single batch, noisy) | `{fmt(metrics['final_train_loss'])}` |",
            f"| Loss window (steps averaged each end) | `{fmt(metrics['loss_window_steps'])}` |",
            f"| Initial train loss (windowed mean) | `{fmt(metrics['initial_train_loss_windowed'])}` |",
            f"| Final train loss (windowed mean) | `{fmt(metrics['final_train_loss_windowed'])}` |",
            f"| Initial waveform L1 (windowed mean) | `{fmt(metrics['initial_waveform_l1_windowed'])}` |",
            f"| Final waveform L1 (windowed mean) | `{fmt(metrics['final_waveform_l1_windowed'])}` |",
            f"| Initial MR-STFT (windowed mean) | `{fmt(metrics['initial_mrstft_windowed'])}` |",
            f"| Final MR-STFT (windowed mean) | `{fmt(metrics['final_mrstft_windowed'])}` |",
            f"| Loss decreased (windowed) | `{acceptance['loss_decreased']}` |",
            f"| Validation probe segments (held-out) | `{fmt(metrics['validation_probe_segments'])}` |",
            f"| Validation probe fingerprint | `{metrics['validation_probe_fingerprint']}` |",
            f"| Same probe before and after | `{metrics['validation_same_probe_before_and_after']}` |",
            f"| Validation recon loss BEFORE | `{fmt(metrics['validation_recon_loss_before'])}` |",
            f"| Validation recon loss AFTER | `{fmt(metrics['validation_recon_loss_after'])}` |",
            f"| Validation L1 BEFORE | `{fmt(metrics['validation_l1_before'])}` |",
            f"| Validation L1 AFTER | `{fmt(metrics['validation_l1_after'])}` |",
            f"| Validation MR-STFT BEFORE | `{fmt(metrics['validation_mrstft_before'])}` |",
            f"| Validation MR-STFT AFTER | `{fmt(metrics['validation_mrstft_after'])}` |",
            f"| Validation improved from init | `{metrics['validation_improved']}` |",
            f"| Train waveform L1 | `{fmt(metrics['train_waveform_l1'])}` |",
            f"| Train MR-STFT | `{fmt(metrics['train_mrstft'])}` |",
            f"| Validation waveform L1 | `{fmt(metrics['validation_waveform_l1'])}` |",
            f"| Validation MR-STFT | `{fmt(metrics['validation_mrstft'])}` |",
            f"| Validation SNR dB | `{fmt(metrics['validation_snr_db'])}` |",
            f"| Latent frame rate Hz | `{fmt(metrics['latent_frame_rate_hz'])}` |",
            f"| Downsample factor | `{fmt(metrics['downsample_factor'])}` |",
            f"| Number of codebooks | `{fmt(metrics['num_codebooks'])}` |",
            f"| Codebook size | `{fmt(metrics['codebook_size'])}` |",
            f"| Bits per code | `{fmt(metrics['bits_per_code'])}` |",
            f"| Discrete bitrate bps | `{fmt(metrics['discrete_bitrate_bps'])}` |",
            f"| PCM16 compression ratio | `{fmt(metrics['pcm16_compression_ratio'])}` |",
            f"| Codebook utilization avg | `{fmt(metrics['codebook_utilization_avg'])}` |",
            f"| Codebook utilization per codebook | `{metrics['codebook_utilization_per_codebook']}` |",
            f"| Dead codes avg | `{fmt(metrics['codebook_dead_codes_avg'])}` |",
            f"| Dead codes per codebook | `{metrics['codebook_dead_codes_per_codebook']}` |",
            f"| Entropy avg | `{fmt(metrics['codebook_entropy_avg'])}` |",
            f"| Entropy per codebook | `{metrics['codebook_entropy_per_codebook']}` |",
            f"| Perplexity avg | `{fmt(metrics['codebook_perplexity_avg'])}` |",
            f"| Perplexity per codebook | `{metrics['codebook_perplexity_per_codebook']}` |",
            f"| Collapse flag (final) | `{metrics['rvq_collapse_flag_final']}` |",
            f"| Peak GPU VRAM GB | `{fmt(metrics['peak_gpu_vram_gb'])}` |",
            f"| Training throughput steps/s | `{fmt(metrics['train_steps_per_second'])}` |",
            f"| Training throughput audio-s/s | `{fmt(metrics['train_audio_seconds_per_second'])}` |",
            f"| Encode+decode seconds | `{fmt(metrics['encode_decode_seconds'])}` |",
            f"| Encode+decode real-time factor | `{fmt(metrics['encode_decode_real_time_factor'])}` |",
            "",
            "### Experiment Isolation",
            "",
            f"- Run id: `{(acceptance.get('isolation', {}).get('run_manifest') or {}).get('run_id')}`",
            (
                f"- Curve rows: `{acceptance.get('isolation', {}).get('curve_rows')}` "
                f"across `{len(acceptance.get('isolation', {}).get('curve_run_ids', []))}` run id(s)"
            ),
            f"- Duplicate curve steps: `{acceptance.get('isolation', {}).get('curve_duplicate_steps')}`",
            f"- Duplicate RVQ steps: `{acceptance.get('isolation', {}).get('rvq_duplicate_steps')}`",
            f"- Curve gaps: `{acceptance.get('isolation', {}).get('curve_gaps')}`",
            (
                f"- Resumes: `{acceptance.get('isolation', {}).get('resumes')}` "
                f"(all authoritative: `{acceptance.get('isolation', {}).get('rng_restored_on_every_resume')}`)"
            ),
            f"- Single-run artifacts: `{acceptance.get('isolation', {}).get('ok')}`",
            "",
            "### RVQ Behavior Throughout Training",
            "",
        ]
    )

    if rvq.get("available"):
        lines.extend(
            [
                f"- Snapshots recorded: `{rvq['snapshots']}` (steps {rvq['first_step']}-{rvq['final_step']})",
                f"- Never collapsed: `{rvq['never_collapsed']}`",
                f"- Temporary collapse observed: `{rvq['temporary_collapse']}`",
                f"- Recovered before end: `{rvq['recovered']}`",
                f"- Collapsed snapshots: `{rvq['collapsed_snapshots']}`",
                f"- First collapse step: `{rvq['first_collapse_step']}`",
                f"- Last collapse step: `{rvq['last_collapse_step']}`",
                f"- Minimum utilization during training: `{fmt(rvq['min_utilization'])}`",
                f"- Final utilization: `{fmt(rvq['final_utilization'])}`",
                f"- Final perplexity: `{fmt(rvq['final_perplexity'])}`",
                f"- Final collapse flag: `{rvq['final_collapse']}`",
                "",
                "",
                "In-training snapshots are computed on a **single batch**, so they are bounded by",
                "the number of latent frames in that batch and read lower than the final figures,",
                "which are computed over the whole split. Compare snapshots to snapshots, and use",
                "the per-codebook table above for the final verdict.",
                "",
                "Utilization curve (step, per-batch utilization):",
                "",
                "```text",
                "\n".join(f"{step:>7}  {value:.4f}" for step, value in rvq["utilization_curve"][:80]),
                "```",
                "",
                "Final per-codebook state:",
                "",
                "```json",
                json.dumps(rvq["final_per_codebook"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
        if rvq["temporary_collapse"]:
            lines.extend(
                [
                    "**Temporary collapse and recovery.** The residual codebook collapsed during early",
                    f"training (first at step {rvq['first_collapse_step']}, last at step {rvq['last_collapse_step']})",
                    "and recovered before the end of the run. This is the expected trajectory for this",
                    "codec: the codebook is seeded from real encoder outputs, but the encoder initially",
                    "emits latents that barely vary across time, so entries stay clustered until the",
                    "encoder learns time-varying structure. Acceptance is judged on the final state.",
                    "",
                ]
            )
    else:
        lines.extend([f"- RVQ history unavailable: `{rvq.get('reason')}`", ""])

    examples = acceptance.get("listening_examples", {})
    lines.extend(
        [
            "### Held-Out Listening Examples",
            "",
            f"Exported `{len(examples)}` held-out original/reconstructed pairs from the validation split.",
            "Buckets are spectral heuristics, not instrument ground truth.",
            "",
            "| Character | Segment | Waveform L1 | MR-STFT | SNR dB |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for character in sorted(examples):
        entry = examples[character]
        entry_metrics = entry.get("metrics", {})
        lines.append(
            f"| {character} | `{Path(entry.get('source_path', '')).name}` | "
            f"{fmt(entry_metrics.get('waveform_l1'), 4)} | {fmt(entry_metrics.get('mrstft'), 4)} | "
            f"{fmt(entry_metrics.get('snr_db'), 4)} |"
        )

    lines.extend(
        [
            "",
            "### Artifacts",
            "",
            f"Output directory: `{report['output_dir']}`",
            "",
            "- `original.wav`, `reconstructed.wav`",
            "- `examples/character_{percussive,harmonic,bass_heavy,mixed}_{original,reconstructed}.wav`",
            "- `examples/val_*_original.wav`, `examples/val_*_reconstructed.wav`",
            "- `checkpoint.pt`, `checkpoint_last.pt`, `optimizer_state.pt`",
            "- `training_curves.csv`, `metrics.jsonl`, `rvq_history.jsonl`",
            "- `train_metrics.json`, `validation_metrics.json`, `metrics_summary.json`",
            "- `compression_stats.json`, `experiment_metadata.json`, `listening_examples.json`",
            "- `m03_acceptance_report.json`",
            "",
            "### Checkpoint Resume",
            "",
            f"Resume run return code: `{report['resume']['returncode']}`",
            f"Resume output: `{report['resume_output_dir']}`",
            "",
            "### Subjective Listening Notes",
            "",
            "Not auto-filled. Listen to the four held-out A/B pairs and record percussion,",
            "harmonic texture, bass, mix, and any vocal observations here.",
            "",
            "### Long-Form Cost Note",
            "",
            f"This codec records about `{fmt(metrics['latent_frame_rate_hz'])}` latent frames/sec at 24 kHz",
            f"with downsample factor `{fmt(metrics['downsample_factor'])}`. That is the M03 baseline and a",
            "long-form generation cost concern to benchmark in M04, not to change during M03.",
            "",
            "### Proposed M04 Experiment Matrix",
            "",
            "| Candidate | Strides | Approx latent rate | Purpose |",
            "| --- | --- | ---: | --- |",
            "| M03 baseline | 2,4,5,5 (x200) | 120 Hz | Quality/control baseline. |",
            "| Lower-rate A | 2,4,5,8 (x320) | 75 Hz | Reduce token cost; check transient retention. |",
            "| Lower-rate B | 4,4,5,6 (x480) | 50 Hz | Stronger long-form compression candidate. |",
            "| Lower-rate C | 4,5,6,8 (x960) | 25 Hz | Only if decoder quality and RVQ usage hold. |",
            "",
            "Optimization target: perceptual reconstruction quality + low token/latent rate +",
            "stable RVQ usage + reasonable Colab training cost.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M03 final acceptance on Google Colab CUDA.")
    parser.add_argument("--config", default="configs/codec/codec_m03_tiny.yaml")
    parser.add_argument("--train-manifest", default=None, help="M02 canonical train manifest (required for PASS).")
    parser.add_argument("--val-manifest", default=None, help="M02 canonical validation manifest.")
    parser.add_argument("--audio-glob", default=None, help="Debug only; bypasses M02 and cannot pass acceptance.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="RVQ codebooks collapse early and need time to recover; 80 fails the collapse gate.",
    )
    parser.add_argument("--resume-extra-steps", type=int, default=25, help="Extra steps for the resume check.")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Only used without --val-manifest.")
    parser.add_argument("--rvq-log-every", type=int, default=25)
    parser.add_argument(
        "--validation-probe",
        type=int,
        default=64,
        help="Held-out segments evaluated before and after training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Steps between atomic checkpoints so a Colab disconnect is resumable.",
    )
    parser.add_argument(
        "--eval-max-segments",
        type=int,
        default=256,
        help="Cap segments evaluated per split; unbounded evaluation exhausted the runtime.",
    )
    parser.add_argument("--export-examples", type=int, default=5)
    parser.add_argument("--allow-cpu", action="store_true", help="Debug only; M03 acceptance requires CUDA.")
    parser.add_argument("--log-path", default="docs/EXPERIMENT_LOG.md")
    parser.add_argument("--run-label", default="colab-m03-acceptance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_manifest and not args.audio_glob:
        raise SystemExit("Provide --train-manifest (M02 canonical manifest) or --audio-glob for debugging.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    require_cuda = not args.allow_cpu

    environment = environment_report(require_cuda=require_cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pytest_result = run_command([sys.executable, "-m", "pytest", "-q"])
    if pytest_result["returncode"] != 0:
        raise SystemExit(f"Full test suite failed; not running {milestone('M03')} acceptance.")
    cuda_result = cuda_test_report(require_cuda=require_cuda)

    data_args: list[str] = []
    if args.train_manifest:
        data_args += ["--train-manifest", args.train_manifest]
        if args.val_manifest:
            data_args += ["--val-manifest", args.val_manifest]
    else:
        data_args += ["--audio-glob", args.audio_glob]

    common = [
        "--config", args.config,
        *data_args,
        "--device", device,
        "--val-fraction", str(args.val_fraction),
        "--rvq-log-every", str(args.rvq_log_every),
        "--run-label", args.run_label,
        "--validation-probe", str(args.validation_probe),
        "--checkpoint-every", str(args.checkpoint_every),
        "--eval-max-segments", str(args.eval_max_segments),
    ]

    # Resume an interrupted run of THIS experiment rather than starting a second
    # one. A Colab disconnect must not cost the run its identity: the same run_id
    # continues, and replayed rows are retired so steps stay unique.
    existing = read_run_manifest(output_dir)
    latest_checkpoint = output_dir / "checkpoint_latest.pt"
    resume_training = bool(existing and latest_checkpoint.exists())

    train_command = [
        sys.executable, "scripts/train_codec.py",
        *common,
        "--output-dir", str(output_dir),
        "--steps", str(args.steps),
        "--export-examples", str(args.export_examples),
    ]
    if resume_training:
        train_command += ["--resume", str(latest_checkpoint)]
        print(
            f"\nResuming existing run {existing.get('run_id')} from {latest_checkpoint.name} "
            "(same logical experiment)\n"
        )
    if device == "cuda":
        train_command.append("--amp")
    train_result = run_command(train_command)
    if train_result["returncode"] != 0:
        raise SystemExit(f"Training failed; {milestone('M03')} acceptance failed.")

    resume_output_dir = output_dir / "resume_check"
    resume_command = [
        sys.executable, "scripts/train_codec.py",
        *common,
        "--output-dir", str(resume_output_dir),
        "--resume", str(output_dir / "checkpoint.pt"),
        "--steps", str(args.steps + max(args.resume_extra_steps, 1)),
        "--export-examples", "2",
    ]
    if device == "cuda":
        resume_command.append("--amp")
    resume_result = run_command(resume_command)

    acceptance = validate_acceptance(output_dir)
    metrics = collect_required_metrics(acceptance)

    report = {
        "run_label": args.run_label,
        "environment": environment,
        "pytest": pytest_result,
        "cuda_tests": cuda_result,
        "train": train_result,
        "resume": resume_result,
        "resume_output_dir": str(resume_output_dir),
        "acceptance": acceptance,
        "metrics": metrics,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "output_dir": str(output_dir),
    }

    (output_dir / "m03_acceptance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    log_path = (
        output_dir / "EXPERIMENT_LOG_CPU_DEBUG.md"
        if args.allow_cpu and args.log_path == "docs/EXPERIMENT_LOG.md"
        else Path(args.log_path)
        if Path(args.log_path).is_absolute()
        else ROOT / args.log_path
    )
    write_experiment_log(log_path, report)
    # Keep a copy beside the artifacts so the log survives on Drive too.
    write_experiment_log(output_dir / "EXPERIMENT_LOG.md", report)

    print("\n" + json.dumps(metrics, indent=2, sort_keys=True))
    print("\n" + json.dumps(
        {
            "missing_artifacts": acceptance["missing_artifacts"],
            "loss_decreased": acceptance["loss_decreased"],
            "rvq_collapse_final": acceptance["rvq_collapse_suspected"],
            "rvq_temporary_collapse": acceptance["rvq_history"].get("temporary_collapse"),
            "rvq_recovered": acceptance["rvq_history"].get("recovered"),
            "used_m02_manifest": acceptance["used_m02_manifest"],
            "isolation_ok": acceptance["isolation"]["ok"],
            "curve_gaps": acceptance["isolation"]["curve_gaps"],
            "resumes": acceptance["isolation"]["resumes"],
            "rng_restored_on_every_resume": acceptance["isolation"]["rng_restored_on_every_resume"],
            "curve_rows": acceptance["isolation"]["curve_rows"],
            "curve_run_ids": acceptance["isolation"]["curve_run_ids"],
            "validation_improved": acceptance["validation_improved"],
            "listening_examples": acceptance["listening_example_count"],
            "resume_returncode": resume_result["returncode"],
            "acceptance_pass": acceptance["acceptance_pass"],
        },
        indent=2,
        sort_keys=True,
    ))
    print(f"\nreport : {output_dir / 'm03_acceptance_report.json'}")
    print(f"log    : {log_path}")

    if not acceptance["acceptance_pass"] or resume_result["returncode"] != 0:
        raise SystemExit(f"{milestone('M03')} = FAIL. Inspect {output_dir / 'm03_acceptance_report.json'}.")
    print(f"\n{milestone('M03')} = PASS")


if __name__ == "__main__":
    main()
