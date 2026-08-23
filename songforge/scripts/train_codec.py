from __future__ import annotations

import argparse
import csv
import hashlib
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
from songforge.evaluation.audio import (
    audio_character_features,
    codec_timing_metrics,
    reconstruction_metrics,
    select_character_examples,
)
from songforge.losses.audio import codec_reconstruction_loss
from songforge.models.codec import NeuralCodec
from songforge.training.checkpoint import (
    RngRestoreError,
    load_checkpoint,
    save_checkpoint,
)
from songforge.training.run import (
    assert_fresh_run_dir,
    assert_resume_compatible,
    new_run_id,
    probe_fingerprint,
    truncate_csv_from_step,
    truncate_jsonl_from_step,
    write_run_manifest,
)
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


def resolve_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path], str]:
    """Prefer canonical manifests; fall back to ad-hoc globbing for debugging.

    A canonical split is song-disjoint, which is what makes validation genuinely
    held out. Splitting a glob by file order does not guarantee that.

    The returned label names the manifest directory rather than a milestone, so a
    run recorded against an expanded corpus is not filed under the milestone that
    happened to create the first one.
    """
    if args.train_manifest:
        corpus = Path(args.train_manifest).parent.parent.name or "manifest"
        train_paths = read_audio_paths(manifest=args.train_manifest)
        if args.val_manifest:
            val_paths = read_audio_paths(manifest=args.val_manifest)
        else:
            train_paths, val_paths = split_paths(train_paths, float(args.val_fraction))
            return train_paths, val_paths, f"manifest:{corpus}:train_only"
        return train_paths, val_paths, f"manifest:{corpus}"

    paths = read_audio_paths(args.manifest, args.audio_glob)
    train_paths, val_paths = split_paths(paths, float(args.val_fraction))
    return train_paths, val_paths, "manifest" if args.manifest else "audio_glob"


def select_probe_indices(dataset: AudioSegmentDataset, max_items: int) -> list[int]:
    """Deterministic held-out probe membership.

    Chosen by hashing each segment's own path, so the selection depends only on
    the dataset contents - not on ordering, machine, or how many Colab sessions
    the run took. The same segments are therefore measured before and after
    training, and after a disconnect/resume.
    """
    scored = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(str(dataset.paths[index]).encode("utf-8")).hexdigest(),
    )
    return sorted(scored[: min(max_items, len(dataset))])


@torch.no_grad()
def evaluate_validation_probe(
    model: NeuralCodec,
    dataset: AudioSegmentDataset,
    device: torch.device,
    indices: list[int],
) -> dict[str, Any]:
    """Reconstruction quality on exactly `indices` of the held-out validation set.

    Called once before the first optimizer step and once after training, on the
    same segment ids, so the before/after pair is a like-for-like measurement of
    what training bought. `recon_loss` is waveform L1 + MR-STFT, the
    reconstruction part of the training objective (no VQ term).
    """
    if not indices:
        return {}
    was_training = model.training
    model.eval()
    rows = []
    for index in indices:
        audio = dataset[index].unsqueeze(0).to(device)
        rows.append(reconstruction_metrics(model(audio)["reconstruction"], audio))
    if was_training:
        model.train()

    metrics: dict[str, Any] = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    metrics["recon_loss"] = metrics["waveform_l1"] + metrics["mrstft"]
    metrics["segments"] = float(len(rows))
    metrics["sample_ids"] = [Path(dataset.paths[index]).stem for index in indices]
    metrics["probe_fingerprint"] = probe_fingerprint(metrics["sample_ids"])
    return metrics


#: Listening categories requested for M04, mapped to Slakh instrument families.
#: A category with no matching held-out audio is reported, never substituted.
INSTRUMENT_CATEGORIES = {
    "piano": ("Piano",),
    "guitar": ("Guitar",),
    "bass": ("Bass",),
    "percussion": ("Drums",),
    "strings": ("Strings", "Strings (continued)"),
    "full_mix": ("Mixture",),
}


def family_by_path(manifest: str | None) -> dict[str, str]:
    """Map segment path -> instrument family, read from the manifest."""
    if not manifest:
        return {}
    from songforge.data.manifest import read_jsonl

    return {
        record.path: record.instrument_family
        for record in read_jsonl(manifest)
        if record.instrument_family
    }


@torch.no_grad()
def export_instrument_examples(
    model: NeuralCodec,
    dataset: AudioSegmentDataset,
    device: torch.device,
    output_dir: Path,
    families: dict[str, str],
) -> dict[str, Any]:
    """Export one held-out pair per real instrument category.

    Selection is deterministic (lowest hashed path within a family), so every
    candidate codec is compared on identical source audio. Categories absent
    from the held-out data are reported as unavailable rather than filled in
    with something that merely sounds similar.
    """
    if not families or len(dataset) == 0:
        return {"exported": {}, "unavailable": sorted(INSTRUMENT_CATEGORIES)}

    by_family: dict[str, list[int]] = {}
    for index, path in enumerate(dataset.paths):
        family = families.get(str(path))
        if family:
            by_family.setdefault(family, []).append(index)

    example_dir = output_dir / "examples"
    exported: dict[str, Any] = {}
    unavailable: list[str] = []
    for category, accepted in INSTRUMENT_CATEGORIES.items():
        candidates = [i for family in accepted for i in by_family.get(family, [])]
        if not candidates:
            unavailable.append(category)
            continue
        index = min(
            candidates,
            key=lambda i: hashlib.sha256(str(dataset.paths[i]).encode("utf-8")).hexdigest(),
        )
        audio = dataset[index].unsqueeze(0).to(device)
        recon = model(audio)["reconstruction"]
        original = example_dir / f"instrument_{category}_original.wav"
        reconstructed = example_dir / f"instrument_{category}_reconstructed.wav"
        write_wav(original, audio.cpu(), model.sample_rate)
        write_wav(reconstructed, recon.cpu(), model.sample_rate)
        exported[category] = {
            "segment_index": index,
            "source_path": str(dataset.paths[index]),
            "instrument_family": families.get(str(dataset.paths[index])),
            "original": str(original),
            "reconstructed": str(reconstructed),
            "metrics": reconstruction_metrics(recon, audio, model.sample_rate),
        }
    return {"exported": exported, "unavailable": sorted(unavailable)}


@torch.no_grad()
def export_character_examples(
    model: NeuralCodec,
    dataset: AudioSegmentDataset,
    device: torch.device,
    output_dir: Path,
    max_candidates: int = 64,
) -> dict[str, Any]:
    """Export one held-out original/reconstructed pair per spectral character.

    Gives the listening test percussion-heavy, harmonic, bass-heavy and mixed
    material instead of whichever segments happened to sort first.
    """
    if len(dataset) == 0:
        return {}
    model.eval()
    candidates = []
    for index in range(min(len(dataset), max_candidates)):
        audio = dataset[index]
        candidates.append(
            {"index": index, "features": audio_character_features(audio, model.sample_rate)}
        )

    chosen = select_character_examples(candidates)
    example_dir = output_dir / "examples"
    exported: dict[str, Any] = {}

    for character, entry in chosen.items():
        audio = dataset[entry["index"]].unsqueeze(0).to(device)
        out = model(audio)
        recon = out["reconstruction"]
        original_path = example_dir / f"character_{character}_original.wav"
        reconstructed_path = example_dir / f"character_{character}_reconstructed.wav"
        write_wav(original_path, audio.cpu(), model.sample_rate)
        write_wav(reconstructed_path, recon.cpu(), model.sample_rate)
        exported[character] = {
            "segment_index": entry["index"],
            "source_path": str(dataset.paths[entry["index"]]),
            "original": str(original_path),
            "reconstructed": str(reconstructed_path),
            "features": entry["features"],
            "metrics": reconstruction_metrics(recon, audio),
        }
    return exported


@torch.no_grad()
def evaluate_dataset(
    model: NeuralCodec,
    dataset: AudioSegmentDataset,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    max_examples: int,
    max_segments: int = 0,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    all_codes = []
    first_audio = None
    first_recon = None
    # Bounded, deterministic sample. Evaluating every segment meant thousands of
    # Drive-resident reads per run, which is what exhausted the Colab runtime
    # before any final evidence was persisted. The bound is recorded in the
    # metrics so the sample size is never mistaken for the full split.
    indices = select_probe_indices(dataset, max_segments) if max_segments else list(range(len(dataset)))
    for idx in indices:
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
    mean_metrics["evaluated_segments"] = float(len(rows))
    mean_metrics["available_segments"] = float(len(dataset))
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

    # Experiment isolation: a fresh run may not append to another run's curves.
    # Only an explicit --resume may continue an existing directory, and only
    # when the checkpoint came from a compatible config.
    if not args.resume:
        assert_fresh_run_dir(output_dir)

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
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_step = 0
    run_id = new_run_id(args.run_label)
    resumed_run_id = None
    rng_restored = False
    superseded_rows = 0
    if args.resume:
        # Strict by default: an acceptance resume that silently restarts the RNG
        # stream, or drops optimizer/scaler state, is no longer the same
        # experiment. --allow-rng-fallback exists only for debugging and marks
        # the run non-authoritative.
        strict_rng = not args.allow_rng_fallback
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, map_location=device, scaler=scaler, strict_rng=strict_rng
        )
        resumed_run_id = assert_resume_compatible(checkpoint, config)
        if resumed_run_id:
            run_id = resumed_run_id
        rng_restored = bool(checkpoint.get("rng_restored"))
        if strict_rng and not rng_restored:
            raise RngRestoreError("authoritative resume requires a restored RNG stream")
        start_step = int(checkpoint.get("step", 0)) + 1
        # A checkpoint saved at step N while training ran on to N+k leaves replayed
        # rows behind. Retire them (kept as .superseded evidence) so steps appear
        # exactly once across however many Colab sessions this run spans.
        superseded_rows = truncate_csv_from_step(output_dir / "training_curves.csv", start_step)
        superseded_rows += truncate_jsonl_from_step(output_dir / "rvq_history.jsonl", start_step)
        superseded_rows += truncate_jsonl_from_step(output_dir / "metrics.jsonl", start_step)
        print(
            f"resuming run {run_id} at step {start_step} "
            f"(rng_restored={rng_restored}, superseded rows retired={superseded_rows})"
        )

    resume_event = None
    if args.resume:
        resume_event = {
            "resumed_from": str(args.resume),
            "resumed_at_step": start_step,
            "rng_restored": rng_restored,
            "strict_rng": not args.allow_rng_fallback,
            "superseded_rows_retired": superseded_rows,
            "authoritative": not args.allow_rng_fallback,
        }
    write_run_manifest(
        output_dir,
        run_id=run_id,
        run_label=args.run_label,
        config=config,
        resumed_from=args.resume,
        resumed_run_id=resumed_run_id,
        start_step=start_step,
        device=str(device),
        amp=amp,
        resume_event=resume_event,
    )
    print(f"run_id: {run_id}" + (f" (resumed from {resumed_run_id})" if resumed_run_id else ""))

    train_paths, val_paths, path_source = resolve_paths(args)
    print(f"data source: {path_source} ({len(train_paths)} train / {len(val_paths)} val files)")
    (output_dir / "train_files.txt").write_text("\n".join(str(path) for path in train_paths) + "\n", encoding="utf-8")
    (output_dir / "val_files.txt").write_text("\n".join(str(path) for path in val_paths) + "\n", encoding="utf-8")
    dataset = AudioSegmentDataset(
        paths=train_paths,
        sample_rate=kwargs["sample_rate"],
        channels=kwargs["channels"],
        segment_samples=int(training.get("segment_samples", kwargs["sample_rate"])),
    )
    val_dataset = AudioSegmentDataset(
        paths=val_paths,
        sample_rate=kwargs["sample_rate"],
        channels=kwargs["channels"],
        segment_samples=int(training.get("segment_samples", kwargs["sample_rate"])),
    )
    # Deterministic held-out probe membership, identical before and after training
    # and across a disconnect/resume.
    probe_indices = select_probe_indices(val_dataset, args.validation_probe)
    probe_path = output_dir / "validation_before.json"
    if args.resume and probe_path.exists():
        # Keep the original BEFORE measurement; it belongs to step 0 of this run.
        validation_before = json.loads(probe_path.read_text(encoding="utf-8"))
        print(f"reusing BEFORE probe from this run (fingerprint {validation_before.get('probe_fingerprint')})")
    else:
        validation_before = evaluate_validation_probe(model, val_dataset, device, probe_indices)
        probe_path.write_text(json.dumps(validation_before, indent=2, sort_keys=True), encoding="utf-8")
        if validation_before:
            print(
                f"validation probe BEFORE ({int(validation_before['segments'])} held-out segments, "
                f"fingerprint {validation_before['probe_fingerprint']}): "
                f"recon_loss={validation_before['recon_loss']:.5f} "
                f"l1={validation_before['waveform_l1']:.5f} mrstft={validation_before['mrstft']:.5f}"
            )

    loader = DataLoader(dataset, batch_size=int(training.get("batch_size", 2)), shuffle=True, drop_last=False)
    steps = int(args.steps or training.get("steps", 20))
    grad_clip = float(training.get("grad_clip", 1.0))
    fft_sizes = tuple(training.get("fft_sizes", [256, 512, 1024]))
    metrics_csv = output_dir / "training_curves.csv"
    metrics_jsonl = output_dir / "metrics.jsonl"

    rvq_history_path = output_dir / "rvq_history.jsonl"
    rvq_log_every = max(int(args.rvq_log_every), 1)
    checkpoint_every = max(int(args.checkpoint_every), 0)

    first_batch = None
    step = start_step
    train_start = time.perf_counter()
    samples_seen = 0
    audio_seconds_seen = 0.0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = batch.to(device)
            samples_seen += batch.size(0)
            audio_seconds_seen += batch.size(0) * batch.size(-1) / float(kwargs["sample_rate"])
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
                # Stamped on every row so a spliced curve is detectable, not just unlikely.
                "run_id": run_id,
            }
            append_metrics(metrics_csv, row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            # Track RVQ health throughout training, not only at the end: the
            # codebook typically collapses early and recovers, and acceptance
            # has to be able to see that happen rather than infer it.
            if step % rvq_log_every == 0 or step == steps - 1:
                usage = model.quantizer.codebook_usage(out["codes"].detach())
                usage_row = {
                    "step": step,
                    "run_id": run_id,
                    "loss": row["loss"],
                    "codebook_utilization_avg": usage["codebook_utilization_avg"],
                    "codebook_utilization_min": usage["codebook_utilization_min"],
                    "codebook_perplexity_avg": usage["codebook_perplexity_avg"],
                    "codebook_perplexity_min": usage["codebook_perplexity_min"],
                    "codebook_entropy_avg": usage["codebook_entropy_avg"],
                    "codebook_dead_codes_avg": usage["codebook_dead_codes_avg"],
                    "codebook_unique_avg": usage["codebook_unique_avg"],
                    "rvq_collapse_suspected": usage["rvq_collapse_suspected"],
                    "per_codebook": usage["per_codebook"],
                }
                with rvq_history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(usage_row) + "\n")
                print(
                    f"step={step} loss={row['loss']:.5f} l1={row['waveform_l1']:.5f} "
                    f"mrstft={row['mrstft']:.5f} util={usage['codebook_utilization_avg']:.3f} "
                    f"ppl={usage['codebook_perplexity_avg']:.2f} "
                    f"collapse={int(usage['rvq_collapse_suspected'])}"
                )
            else:
                print(f"step={step} loss={row['loss']:.5f} l1={row['waveform_l1']:.5f} mrstft={row['mrstft']:.5f}")

            # Periodic atomic checkpoint. Written via a temp file + os.replace, so
            # a runtime that dies mid-write leaves the previous one intact. This is
            # what makes the run survive a Colab disconnect as ONE experiment.
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                save_checkpoint(
                    output_dir / "checkpoint_latest.pt", model, optimizer, step, config,
                    {"loss": row["loss"]}, run_id=run_id, run_label=args.run_label,
                    scaler=scaler, extra={"periodic": True},
                )
                print(f"  checkpoint_latest.pt written at step {step}", flush=True)
            step += 1

    train_seconds = time.perf_counter() - train_start
    steps_run = max(step - start_step, 1)
    throughput = {
        "train_wall_seconds": float(train_seconds),
        "steps_run": int(steps_run),
        "steps_per_second": float(steps_run / max(train_seconds, 1e-8)),
        "segments_per_second": float(samples_seen / max(train_seconds, 1e-8)),
        "audio_seconds_per_second": float(audio_seconds_seen / max(train_seconds, 1e-8)),
        "audio_seconds_processed": float(audio_seconds_seen),
    }

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
    train_eval = evaluate_dataset(
        model, dataset, device, output_dir, "train", args.export_examples, args.eval_max_segments
    )
    val_eval = evaluate_dataset(
        model, val_dataset, device, output_dir, "val", args.export_examples, args.eval_max_segments
    )
    train_eval.update(model.compression_stats(sample.shape[-1]))
    val_eval.update(model.compression_stats(sample.shape[-1]))
    train_eval.update(gpu_metrics(device))
    val_eval.update(gpu_metrics(device))
    metrics.update(throughput)
    train_eval.update(throughput)

    # Same held-out segment ids as the BEFORE probe, so the pair is comparable.
    validation_after = evaluate_validation_probe(model, val_dataset, device, probe_indices)
    same_probe = bool(
        validation_before.get("probe_fingerprint")
        and validation_before.get("probe_fingerprint") == validation_after.get("probe_fingerprint")
    )
    validation_probe = {
        "segments": validation_before.get("segments"),
        "probe_fingerprint": validation_before.get("probe_fingerprint"),
        "same_probe_before_and_after": same_probe,
        "sample_ids": validation_before.get("sample_ids", []),
        "before": validation_before,
        "after": validation_after,
        "delta": {
            key: validation_after[key] - validation_before[key]
            for key in ("recon_loss", "waveform_l1", "mrstft", "spectral_convergence", "snr_db")
            if key in validation_before and key in validation_after
        },
        "improved": bool(
            validation_before
            and validation_after
            and validation_after["recon_loss"] < validation_before["recon_loss"]
        ),
    }
    (output_dir / "validation_probe.json").write_text(
        json.dumps(validation_probe, indent=2, sort_keys=True), encoding="utf-8"
    )
    if validation_after:
        print(
            f"validation probe AFTER  ({int(validation_after['segments'])} held-out segments): "
            f"recon_loss={validation_after['recon_loss']:.5f} "
            f"l1={validation_after['waveform_l1']:.5f} mrstft={validation_after['mrstft']:.5f} "
            f"| improved={validation_probe['improved']}"
        )

    # Held-out listening examples come from the validation split only.
    character_examples = export_character_examples(model, val_dataset, device, output_dir)
    (output_dir / "listening_examples.json").write_text(
        json.dumps(character_examples, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"exported {len(character_examples)} held-out character examples: {sorted(character_examples)}")

    instrument_examples = export_instrument_examples(
        model, val_dataset, device, output_dir, family_by_path(args.val_manifest)
    )
    (output_dir / "instrument_examples.json").write_text(
        json.dumps(instrument_examples, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"exported {len(instrument_examples['exported'])} held-out instrument examples: "
        f"{sorted(instrument_examples['exported'])}"
        + (f" | unavailable in this corpus: {instrument_examples['unavailable']}"
           if instrument_examples["unavailable"] else "")
    )

    metadata = {
        "run_id": run_id,
        "run_label": args.run_label,
        "resumed_run_id": resumed_run_id,
        "validation_probe": validation_probe,
        "config": args.config,
        "manifest": args.manifest,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "path_source": path_source,
        "audio_glob": args.audio_glob,
        "output_dir": str(output_dir),
        "steps_completed": step,
        "resumed_from": args.resume,
        "amp": amp,
        "device": str(device),
        "throughput": throughput,
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
    save_checkpoint(
        output_dir / "checkpoint_last.pt", model, optimizer, step - 1, config, metrics,
        run_id=run_id, run_label=args.run_label, scaler=scaler,
    )
    save_checkpoint(
        output_dir / "checkpoint.pt", model, optimizer, step - 1, config, metrics,
        run_id=run_id, run_label=args.run_label, scaler=scaler,
    )
    torch.save(optimizer.state_dict(), output_dir / "optimizer_state.pt")
    print(f"wrote {output_dir / 'reconstructed.wav'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the M03 SongForge neural codec on local/Colab audio.")
    parser.add_argument("--config", default="configs/codec/codec_m03_tiny.yaml")
    parser.add_argument("--manifest", default=None, help="JSONL manifest with path fields.")
    parser.add_argument(
        "--train-manifest", default=None, help="M02 canonical train manifest (preferred over --audio-glob)."
    )
    parser.add_argument("--val-manifest", default=None, help="M02 canonical validation manifest.")
    parser.add_argument("--audio-glob", default=None, help="Glob of WAV/FLAC files. Debug only; bypasses M02.")
    parser.add_argument("--rvq-log-every", type=int, default=10, help="Steps between RVQ health snapshots.")
    parser.add_argument("--run-label", default="codec-run", help="Human-readable label recorded in the run id.")
    parser.add_argument(
        "--validation-probe",
        type=int,
        default=64,
        help="Held-out segments evaluated before and after training for like-for-like evidence.",
    )
    parser.add_argument(
        "--allow-rng-fallback",
        action="store_true",
        help="Debug only: continue a resume with a fresh RNG stream. Marks the run non-authoritative.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Steps between atomic checkpoints; lets a run survive a Colab disconnect. 0 disables.",
    )
    parser.add_argument(
        "--eval-max-segments",
        type=int,
        default=256,
        help="Cap segments evaluated per split. 0 evaluates everything (slow on Drive).",
    )
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
