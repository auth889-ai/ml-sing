"""Validated, resume-aware tensor preprocessing for SongForge corpora.

Why this exists
---------------
Upstream's two-pass preprocessor (``acestep.training_v2.preprocess``) is only
*partially* resumable, and the gap cost this project four hours of L4 time:

* **Pass 1** writes ``{stem}.tmp.pt`` intermediates and skips a sample only
  when the *final* ``{stem}.pt`` already exists. A runtime that dies during
  pass 1 therefore leaves thousands of valid intermediates that the next run
  ignores and recomputes from scratch.
* **Pass 2** consumes the in-memory list of intermediates that *this run's*
  pass 1 produced — it never adopts intermediates found on disk. So orphaned
  ``.tmp.pt`` files are dead weight forever.
* Both passes write with a bare ``torch.save``. A process killed mid-write
  leaves a truncated ``.pt`` that later looks "present" and gets skipped,
  silently poisoning the training set.

This module fixes all three without forking upstream:

1. **Atomic writes** — ``torch.save`` is wrapped so every tensor lands via
   ``temp file -> flush+fsync -> os.replace``. A kill leaves either the old
   file or the new one, never a half-written one.
2. **Validation** — existing tensors are verified before being trusted.
   Corrupt ones are deleted and regenerated rather than skipped.
3. **Orphan adoption** — valid intermediates are handed straight to pass 2,
   so a recycle costs at most the sample that was in flight.

Usage
-----
    python scripts/resume_preprocess.py \\
        --dataset-json  <corpus>/dataset.json \\
        --tensor-output <drive>/tensors \\
        --checkpoint-dir /content/checkpoints \\
        --model-variant xl_turbo \\
        --max-duration 240

Rerunning the identical command after any interruption is always safe and
always cheap: finished work is verified and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

#: Keys every finished tensor must carry (from upstream's pass-2 writer).
FINAL_KEYS = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
    "metadata",
)
#: Keys pass 2 reads out of a pass-1 intermediate.
TMP_KEYS = ("target_latents", "attention_mask", "metadata")



# --- atomic writes --------------------------------------------------------


def install_atomic_save() -> None:
    """Make every ``torch.save`` in this process crash-safe.

    Writes to a sibling temp file, fsyncs it, then ``os.replace``s it over the
    target. ``os.replace`` is atomic within a filesystem, so a reader either
    sees the complete previous file or the complete new one.
    """
    import torch

    if getattr(torch.save, "_songforge_atomic", False):
        return
    original = torch.save

    def atomic_save(obj, f, *args, **kwargs):
        # Only intercept plain filesystem paths; file objects and buffers
        # keep upstream behaviour.
        if not isinstance(f, (str, os.PathLike)):
            return original(obj, f, *args, **kwargs)
        target = Path(f)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.partial-{os.getpid()}")
        try:
            with open(tmp, "wb") as handle:
                original(obj, handle, *args, **kwargs)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return None

    atomic_save._songforge_atomic = True
    torch.save = atomic_save


# --- validation -----------------------------------------------------------


def validate_quick(path: Path) -> bool:
    """Cheap structural check: is this a complete torch zip archive?

    Reads only the zip central directory, so it costs a couple of seeks per
    file rather than a full read. This is what catches the realistic failure
    mode — a write truncated by a killed runtime or a full disk.

    Completeness is judged by the archive itself, never by a size threshold:
    a byte-count floor would condemn a legitimately small tensor to be
    deleted and regenerated on every single run.
    """
    try:
        if not path.is_file():
            return False
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.endswith("data.pkl") for name in names):
                return False
            # testzip() reads every member; too slow here. The central
            # directory parsing above already proves the file was closed.
        return True
    except (zipfile.BadZipFile, OSError, EOFError):
        return False


def validate_full(path: Path, required_keys: tuple[str, ...]) -> bool:
    """Deep check: loads the tensor, verifies keys and numeric sanity."""
    import torch

    if not validate_quick(path):
        return False
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(blob, dict):
        return False
    for key in required_keys:
        if key not in blob:
            return False
    for key, value in blob.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            if value.is_floating_point() and not torch.isfinite(value).all():
                return False
    return True


def is_valid(path: Path, required_keys: tuple[str, ...], full: bool) -> bool:
    return validate_full(path, required_keys) if full else validate_quick(path)


# --- planning -------------------------------------------------------------


def load_dataset(dataset_json: Path) -> list[dict]:
    payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("samples", "data", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise SystemExit(f"{dataset_json}: no sample list found in JSON object")
    if not isinstance(payload, list):
        raise SystemExit(f"{dataset_json}: expected a list of samples")
    return payload


def audio_path_of(sample: dict) -> str | None:
    return sample.get("audio_path") or sample.get("filename") or sample.get("path")


def plan(samples: list[dict], out_dir: Path, full: bool) -> dict:
    """Partition the corpus into done / adoptable / to-process.

    Tensor names follow upstream: ``{audio file stem}.pt`` (and the pass-1
    intermediate ``{stem}.tmp.pt``).
    """
    done: list[str] = []
    adopt: list[Path] = []
    todo: list[dict] = []
    repaired: list[str] = []
    missing_audio: list[str] = []

    for sample in samples:
        audio = audio_path_of(sample)
        if not audio:
            continue
        stem = Path(audio).stem
        final = out_dir / f"{stem}.pt"
        tmp = out_dir / f"{stem}.tmp.pt"

        if final.exists():
            if is_valid(final, FINAL_KEYS, full):
                done.append(stem)
                tmp.unlink(missing_ok=True)  # finished; intermediate is litter
                continue
            final.unlink(missing_ok=True)
            repaired.append(stem)

        if tmp.exists():
            if is_valid(tmp, TMP_KEYS, full):
                adopt.append(tmp)
                continue
            tmp.unlink(missing_ok=True)
            repaired.append(stem)

        if not Path(audio).exists():
            missing_audio.append(audio)
            continue
        todo.append(sample)

    return {
        "done": done,
        "adopt": adopt,
        "todo": todo,
        "repaired": repaired,
        "missing_audio": missing_audio,
    }


# --- execution ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--tensor-output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-variant", default="xl_turbo")
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument(
        "--full-validate",
        action="store_true",
        help="Load every existing tensor and check keys + finiteness "
             "(slow on a large corpus; default is a structural check that "
             "catches truncation).",
    )
    parser.add_argument("--ace-root", default="/content/ACE-Step-1.5")
    parser.add_argument("--report", default=None, help="Write a JSON report here.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Report the resume plan and exit without using the GPU.")
    args = parser.parse_args()

    sys.path.insert(0, args.ace_root)

    out_dir = Path(args.tensor_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = Path(args.dataset_json)

    samples = load_dataset(dataset_json)
    started = time.time()

    print(f"[resume] corpus:     {len(samples)} samples")
    print(f"[resume] tensors:    {out_dir}")
    print(f"[resume] validation: {'full' if args.full_validate else 'structural'}")

    state = plan(samples, out_dir, args.full_validate)
    print(f"[resume] already done:        {len(state['done'])}")
    print(f"[resume] intermediates found: {len(state['adopt'])} (pass 2 only)")
    print(f"[resume] to process:          {len(state['todo'])}")
    if state["repaired"]:
        print(f"[resume] corrupt, deleted:    {len(state['repaired'])}")
    if state["missing_audio"]:
        print(f"[resume] MISSING AUDIO:       {len(state['missing_audio'])}")

    report = {
        "dataset_json": str(dataset_json),
        "tensor_output": str(out_dir),
        "corpus_samples": len(samples),
        "already_done": len(state["done"]),
        "adopted_intermediates": len(state["adopt"]),
        "to_process": len(state["todo"]),
        "corrupt_regenerated": len(state["repaired"]),
        "missing_audio": state["missing_audio"][:50],
        "validation": "full" if args.full_validate else "structural",
    }

    if args.plan_only:
        _write_report(args.report, report)
        return 0

    if not state["todo"] and not state["adopt"]:
        print("[resume] nothing to do — every tensor is present and valid.")
        report["processed"] = 0
        report["failed"] = 0
        report["elapsed_seconds"] = round(time.time() - started, 1)
        _write_report(args.report, report)
        return 0

    install_atomic_save()
    print("[resume] atomic torch.save installed (temp -> fsync -> rename)")

    from acestep.training_v2.gpu_utils import detect_gpu
    from acestep.training_v2.preprocess import _pass2_heavy, preprocess_audio_files

    processed = failed = 0

    # 1. Adopt orphaned intermediates: pass 2 only, no re-encoding of audio.
    if state["adopt"]:
        gpu = detect_gpu(args.device, args.precision)
        print(f"[resume] adopting {len(state['adopt'])} intermediates via pass 2")
        done_count, fail_count = _pass2_heavy(
            intermediates=state["adopt"],
            out_path=out_dir,
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            device=gpu.device,
            precision=gpu.precision,
            progress_callback=None,
            cancel_check=None,
        )
        processed += done_count
        failed += fail_count
        print(f"[resume] adopted: {done_count} finalized, {fail_count} failed")

    # 2. Process the genuinely unprocessed remainder through both passes.
    if state["todo"]:
        filtered = out_dir / ".resume_todo.json"
        filtered.write_text(json.dumps(state["todo"]), encoding="utf-8")
        print(f"[resume] preprocessing {len(state['todo'])} remaining samples")
        result = preprocess_audio_files(
            audio_dir=None,
            output_dir=str(out_dir),
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            max_duration=args.max_duration,
            dataset_json=str(filtered),
            device=args.device,
            precision=args.precision,
        )
        processed += int(result.get("processed", 0))
        failed += int(result.get("failed", 0))
        filtered.unlink(missing_ok=True)

    final_count = sum(
        1 for p in out_dir.glob("*.pt") if not p.name.endswith(".tmp.pt")
    )
    report.update({
        "processed": processed,
        "failed": failed,
        "final_tensors_on_disk": final_count,
        "elapsed_seconds": round(time.time() - started, 1),
    })
    print(f"[resume] processed={processed} failed={failed} "
          f"final_tensors={final_count} in {report['elapsed_seconds']}s")
    _write_report(args.report, report)
    return 0 if failed == 0 else 1


def _write_report(path: str | None, report: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[resume] report: {target}")


if __name__ == "__main__":
    raise SystemExit(main())
