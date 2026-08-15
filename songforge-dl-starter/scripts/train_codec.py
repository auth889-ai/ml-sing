from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from torch.utils.data import DataLoader

from songforge.data.audio import AudioSegmentDataset, read_audio_paths, write_wav
from songforge.evaluation.audio import codec_timing_metrics, reconstruction_metrics
from songforge.losses.audio import codec_reconstruction_loss
from songforge.models.codec import NeuralCodec
from songforge.training.checkpoint import load_checkpoint, save_checkpoint
from songforge.training.seed import seed_everything


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", config)
    return {
        "sample_rate": int(model.get("sample_rate", 24000)),
        "channels": int(model.get("channels", 1)),
        "base_channels": int(model.get("base_channels", 32)),
        "latent_dim": int(model.get("latent_dim", 64)),
        "codebook_size": int(model.get("codebook_size", 256)),
        "num_quantizers": int(model.get("num_quantizers", 4)),
        "strides": tuple(model.get("strides", [2, 4, 5, 5])),
        "commitment_weight": float(model.get("commitment_weight", 0.25)),
    }


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def split_paths(paths: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    if len(paths) < 2 or val_fraction <= 0:
        return paths, paths[:1]
    val_count = max(1, round(len(paths) * val_fraction))
    val_count = min(val_count, len(paths) - 1)
    return paths[:-val_count], paths[-val_count:]


@torch.no_grad()
def evaluate_dataset(
    model: NeuralCodec,
    dataset: AudioSegmentDataset,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    max_examples: int,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    all_codes = []
    first_audio = None
    first_recon = None
    for idx in range(len(dataset)):
        audio = dataset[idx].unsqueeze(0).to(device)
        out = model(audio)
        recon = out["reconstruction"]
        row = reconstruction_metrics(recon, audio)
        rows.append(row)
        all_codes.append(out["codes"].detach().cpu())
        if first_audio is None:
            first_audio = audio.detach().cpu()
            first_recon = recon.detach().cpu()
        if idx < max_examples:
            example_dir = output_dir / "examples"
            write_wav(example_dir / f"{prefix}_{idx:02d}_original.wav", audio.cpu(), model.sample_rate)
            write_wav(example_dir / f"{prefix}_{idx:02d}_reconstructed.wav", recon.cpu(), model.sample_rate)

    mean_metrics = {
        key: sum(row[key] for row in rows) / max(len(rows), 1)
        for key in rows[0]
    }
    if all_codes:
        mean_metrics.update(model.quantizer.codebook_usage(torch.cat(all_codes, dim=0)))
    if first_audio is not None and first_recon is not None:
        mean_metrics.update(codec_timing_metrics(model, first_audio.to(device)))
    return mean_metrics


def gpu_metrics(device: torch.device) -> dict[str, float | str]:
    if device.type != "cuda":
        return {
            "device_type": device.type,
            "gpu_name": "none",
            "peak_gpu_memory_bytes": 0.0,
            "peak_gpu_memory_gb": 0.0,
        }
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "device_type": "cuda",
        "gpu_name": torch.cuda.get_device_name(index),
        "cuda_version": str(torch.version.cuda),
        "total_gpu_memory_bytes": float(props.total_memory),
        "total_gpu_memory_gb": float(props.total_memory / (1024**3)),
        "peak_gpu_memory_bytes": float(torch.cuda.max_memory_allocated(index)),
        "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated(index) / (1024**3)),
    }


def train(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    training = config.get("training", {})
    seed_everything(int(training.get("seed", 42)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp or training.get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    kwargs = model_kwargs(config)
    model = NeuralCodec(**kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_step = int(checkpoint.get("step", 0)) + 1

    paths = read_audio_paths(args.manifest, args.audio_glob)
    train_paths, val_paths = split_paths(paths, float(args.val_fraction))
    (output_dir / "train_files.txt").write_text("\n".join(str(path) for path in train_paths) + "\n", encoding="utf-8")
    (output_dir / "val_files.txt").write_text("\n".join(str(path) for path in val_paths) + "\n", encoding="utf-8")
    dataset = AudioSegmentDataset(
        paths=train_paths,
        sample_rate=kwargs["sample_rate"],
        channels=kwargs["channels"],
        segment_samples=int(training.get("segment_samples", kwargs["sample_rate"])),
    )
    loader = DataLoader(dataset, batch_size=int(training.get("batch_size", 2)), shuffle=True, drop_last=False)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps = int(args.steps or training.get("steps", 20))
    grad_clip = float(training.get("grad_clip", 1.0))
    fft_sizes = tuple(training.get("fft_sizes", [256, 512, 1024]))
    metrics_csv = output_dir / "training_curves.csv"
    metrics_jsonl = output_dir / "metrics.jsonl"

    first_batch = None
    step = start_step
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = batch.to(device)
            if first_batch is None:
                first_batch = batch[:1].detach().cpu()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(batch)
                losses = codec_reconstruction_loss(
                    out["reconstruction"],
                    batch,
                    out["vq_loss"],
                    waveform_weight=float(training.get("waveform_weight", 1.0)),
                    spectral_weight=float(training.get("spectral_weight", 1.0)),
                    vq_weight=float(training.get("vq_weight", 1.0)),
                    fft_sizes=fft_sizes,
                )
            scaler.scale(losses["loss"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            row = {
                "step": step,
                "loss": float(losses["loss"].detach().cpu().item()),
                "waveform_l1": float(losses["waveform_l1"].detach().cpu().item()),
                "mrstft": float(losses["mrstft"].detach().cpu().item()),
                "vq_loss": float(losses["vq_loss"].detach().cpu().item()),
                "time": int(time.time()),
            }
            append_metrics(metrics_csv, row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"step={step} loss={row['loss']:.5f} l1={row['waveform_l1']:.5f} mrstft={row['mrstft']:.5f}")
            step += 1

    sample = first_batch if first_batch is not None else dataset[0].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        sample = sample.to(device)
        out = model(sample)
        recon = out["reconstruction"]
        metrics = reconstruction_metrics(recon, sample)
        metrics.update(model.compression_stats(sample.shape[-1]))
        metrics.update(model.quantizer.codebook_usage(out["codes"]))
        metrics.update(codec_timing_metrics(model, sample))
        metrics.update(gpu_metrics(device))

    write_wav(output_dir / "original.wav", sample.cpu(), kwargs["sample_rate"])
    write_wav(output_dir / "reconstructed.wav", recon.cpu(), kwargs["sample_rate"])
    train_eval = evaluate_dataset(model, dataset, device, output_dir, "train", args.export_examples)
    val_dataset = AudioSegmentDataset(
        paths=val_paths,
        sample_rate=kwargs["sample_rate"],
        channels=kwargs["channels"],
        segment_samples=int(training.get("segment_samples", kwargs["sample_rate"])),
    )
    val_eval = evaluate_dataset(model, val_dataset, device, output_dir, "val", args.export_examples)
    train_eval.update(model.compression_stats(sample.shape[-1]))
    val_eval.update(model.compression_stats(sample.shape[-1]))
    train_eval.update(gpu_metrics(device))
    val_eval.update(gpu_metrics(device))

    metadata = {
        "config": args.config,
        "manifest": args.manifest,
        "audio_glob": args.audio_glob,
        "output_dir": str(output_dir),
        "steps_completed": step,
        "resumed_from": args.resume,
        "amp": amp,
        "device": str(device),
        "train_file_count": len(train_paths),
        "val_file_count": len(val_paths),
        "train_files": [str(path) for path in train_paths],
        "val_files": [str(path) for path in val_paths],
        "m03_latent_rate_note": "Current M03 codec uses about 120 latent frames/sec at 24 kHz with downsample factor 200; this is a long-form generation cost concern to benchmark in M04.",
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "train_metrics.json").write_text(json.dumps(train_eval, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "validation_metrics.json").write_text(json.dumps(val_eval, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "compression_stats.json").write_text(
        json.dumps(model.compression_stats(sample.shape[-1]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, step - 1, config, metrics)
    save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step - 1, config, metrics)
    torch.save(optimizer.state_dict(), output_dir / "optimizer_state.pt")
    print(f"wrote {output_dir / 'reconstructed.wav'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the M03 SongForge neural codec on local/Colab audio.")
    parser.add_argument("--config", default="configs/codec/codec_m03_tiny.yaml")
    parser.add_argument("--manifest", default=None, help="JSONL manifest with path fields.")
    parser.add_argument("--audio-glob", default=None, help="Glob of WAV/FLAC files. Quote this argument in shells.")
    parser.add_argument("--output-dir", default="outputs/codec_m03_smoke")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--export-examples", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
