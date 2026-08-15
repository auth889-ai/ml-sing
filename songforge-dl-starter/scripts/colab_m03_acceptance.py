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


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
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
        raise SystemExit("M03 acceptance requires CUDA. Select a Colab GPU runtime and rerun.")
    return report


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_curve(path: Path) -> list[dict[str, float]]:
    import csv

    with path.open("r", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items() if key != "time"} for row in csv.DictReader(f)]


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
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    train_metrics = read_json(output_dir / "train_metrics.json") if not missing else {}
    val_metrics = read_json(output_dir / "validation_metrics.json") if not missing else {}
    compression = read_json(output_dir / "compression_stats.json") if not missing else {}
    curve = read_curve(output_dir / "training_curves.csv") if not missing else []
    first_loss = curve[0]["loss"] if curve else None
    final_loss = curve[-1]["loss"] if curve else None
    loss_decreased = first_loss is not None and final_loss is not None and final_loss < first_loss
    collapse = bool(train_metrics.get("rvq_collapse_suspected", 1.0)) or bool(
        val_metrics.get("rvq_collapse_suspected", 1.0)
    )
    return {
        "missing_artifacts": missing,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "rvq_collapse_suspected": collapse,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "compression": compression,
        "acceptance_pass": not missing and loss_decreased and not collapse,
    }


def write_experiment_log(path: Path, report: dict[str, Any]) -> None:
    status = "PASS" if report["acceptance"]["acceptance_pass"] and report["resume"]["returncode"] == 0 else "FAIL"
    train = report["acceptance"].get("train_metrics", {})
    val = report["acceptance"].get("validation_metrics", {})
    compression = report["acceptance"].get("compression", {})
    lines = [
        "# Experiment Log",
        "",
        "## M03 Colab Acceptance",
        "",
        f"Status: **M03 {status}**",
        "",
        "### Environment",
        "",
        "```json",
        json.dumps(report["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "### Commands",
        "",
    ]
    for key in ("pytest", "codec_pytest", "train", "resume"):
        lines.extend(["```bash", " ".join(report[key]["command"]), "```", f"Return code: `{report[key]['returncode']}`", ""])
    lines.extend(
        [
            "### Results",
            "",
            f"- First training loss: `{report['acceptance']['first_loss']}`",
            f"- Final training loss: `{report['acceptance']['final_loss']}`",
            f"- Loss decreased: `{report['acceptance']['loss_decreased']}`",
            f"- RVQ collapse suspected: `{report['acceptance']['rvq_collapse_suspected']}`",
            f"- Missing artifacts: `{report['acceptance']['missing_artifacts']}`",
            f"- Train waveform L1: `{train.get('waveform_l1')}`",
            f"- Train MR-STFT: `{train.get('mrstft')}`",
            f"- Validation waveform L1: `{val.get('waveform_l1')}`",
            f"- Validation MR-STFT: `{val.get('mrstft')}`",
            f"- Latent frame rate Hz: `{compression.get('latent_frame_rate_hz')}`",
            f"- Effective bitrate bps: `{compression.get('discrete_bitrate_bps')}`",
            f"- PCM16 compression ratio: `{compression.get('pcm16_compression_ratio')}`",
            f"- Codebook utilization avg train: `{train.get('codebook_utilization_avg')}`",
            f"- Dead codes avg train: `{train.get('codebook_dead_codes_avg')}`",
            f"- Codebook perplexity avg train: `{train.get('codebook_perplexity_avg')}`",
            f"- Peak GPU memory GB: `{train.get('peak_gpu_memory_gb')}`",
            f"- Real-time factor: `{train.get('real_time_factor')}`",
            "",
            "### Artifacts",
            "",
            f"Output directory: `{report['output_dir']}`",
            "",
            "- `original.wav`",
            "- `reconstructed.wav`",
            "- `examples/*_original.wav`",
            "- `examples/*_reconstructed.wav`",
            "- `checkpoint.pt`",
            "- `optimizer_state.pt`",
            "- `training_curves.csv`",
            "- `train_metrics.json`",
            "- `validation_metrics.json`",
            "- `compression_stats.json`",
            "- `experiment_metadata.json`",
            "",
            "### Subjective Listening Notes",
            "",
            "Not auto-filled. Listen to the A/B WAV pairs and record percussion, harmonic texture, bass, mix, and vocal observations here.",
            "",
            "### Long-Form Cost Note",
            "",
            "The current M03 codec records about 120 latent frames/sec at 24 kHz with downsample factor 200. This is a potential long-form generation cost concern and must be benchmarked in M04, not changed during M03 acceptance.",
            "",
            "### Proposed M04 Experiment Matrix",
            "",
            "| Candidate | Approx latent rate | Purpose |",
            "| --- | ---: | --- |",
            "| M03 baseline | 120 Hz | Quality/control baseline. |",
            "| Lower-rate A | ~75 Hz | Reduce token cost while checking vocal/transient retention. |",
            "| Lower-rate B | ~50 Hz | Stronger long-form compression candidate. |",
            "| Lower-rate C | ~25-40 Hz | Only if decoder quality and RVQ usage remain stable. |",
            "",
            "Optimization target: perceptual reconstruction quality + low token/latent rate + stable RVQ usage + reasonable Colab training cost.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M03 final acceptance on Google Colab CUDA.")
    parser.add_argument("--config", default="configs/codec/codec_m03_tiny.yaml")
    parser.add_argument("--audio-glob", required=True, help="Small approved real-music WAV/FLAC glob.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="RVQ codebooks start collapsed and need a few hundred steps to spread; 80 fails the collapse gate.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--allow-cpu", action="store_true", help="Debug only; M03 acceptance requires CUDA.")
    parser.add_argument("--log-path", default="docs/EXPERIMENT_LOG.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = environment_report(require_cuda=not args.allow_cpu)

    pytest_result = run_command([sys.executable, "-m", "pytest", "-q"], root)
    codec_pytest_result = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_codec.py"], root)
    if pytest_result["returncode"] != 0 or codec_pytest_result["returncode"] != 0:
        raise SystemExit("Tests failed; not running M03 training acceptance.")

    train_command = [
        sys.executable,
        "scripts/train_codec.py",
        "--config",
        args.config,
        "--audio-glob",
        args.audio_glob,
        "--output-dir",
        str(output_dir),
        "--steps",
        str(args.steps),
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
        "--val-fraction",
        str(args.val_fraction),
        "--export-examples",
        "5",
    ]
    if torch.cuda.is_available():
        train_command.append("--amp")
    train_result = run_command(train_command, root)
    if train_result["returncode"] != 0:
        raise SystemExit("Training failed; M03 acceptance failed.")

    resume_command = [
        sys.executable,
        "scripts/train_codec.py",
        "--config",
        args.config,
        "--audio-glob",
        args.audio_glob,
        "--output-dir",
        str(output_dir / "resume_check"),
        "--resume",
        str(output_dir / "checkpoint.pt"),
        "--steps",
        str(args.steps + 1),
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
        "--val-fraction",
        str(args.val_fraction),
        "--export-examples",
        "2",
    ]
    if torch.cuda.is_available():
        resume_command.append("--amp")
    resume_result = run_command(resume_command, root)

    acceptance = validate_acceptance(output_dir)
    report = {
        "environment": env,
        "pytest": pytest_result,
        "codec_pytest": codec_pytest_result,
        "train": train_result,
        "resume": resume_result,
        "acceptance": acceptance,
        "output_dir": str(output_dir),
    }
    (output_dir / "m03_acceptance_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_path = output_dir / "EXPERIMENT_LOG_CPU_DEBUG.md" if args.allow_cpu and args.log_path == "docs/EXPERIMENT_LOG.md" else root / args.log_path
    write_experiment_log(log_path, report)
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    if not acceptance["acceptance_pass"] or resume_result["returncode"] != 0:
        raise SystemExit(f"M03 acceptance FAIL. Inspect {output_dir / 'm03_acceptance_report.json'} and {log_path}.")
    print("M03 acceptance PASS")


if __name__ == "__main__":
    main()
