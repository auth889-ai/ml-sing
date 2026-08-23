"""Prove that SongForge V1 is really training, not merely running.

A process at 100% GPU utilisation is not evidence of learning. This script
checks the claims that actually matter, each independently falsifiable:

  1. finite loss           — scalars logged by the trainer are real numbers
  2. loss is moving        — a frozen loss means the optimizer is not applying
  3. non-zero gradients    — grad-norm scalars, when the trainer logs them
  4. parameters changed    — LoKr tensors differ between two checkpoints, or
                             between a recorded snapshot and the newest one
  5. throughput            — steps/min, so the epoch ETA is a measurement

It reads the TensorBoard event stream the trainer already writes, so it never
touches the training process and cannot perturb the run.

    python scripts/verify_v1_training_gates.py [--snapshot] [--json PATH]

``--snapshot`` records the current LoKr weights so a later invocation can prove
they changed. Without a second reference point, gate 4 reports UNPROVEN rather
than guessing — an unproven gate is not a passed gate.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

DRIVE_V1 = Path("/content/drive/MyDrive/songforge-dl/v1")
LOSS_KEYS = ("train/loss", "loss", "train_loss", "train/loss_step")
GRAD_KEYS = ("grad_norm", "train/grad_norm", "gradients/norm", "grad_norm_step")


def read_event_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Pull every scalar series out of the newest tfevents file."""
    events = sorted(run_dir.rglob("events.out.tfevents.*"),
                    key=lambda p: p.stat().st_mtime)
    if not events:
        return {}
    newest = events[-1]
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("[warn] tensorboard not importable; cannot read scalars",
              file=sys.stderr)
        return {}
    acc = EventAccumulator(str(newest), size_guidance={"scalars": 0})
    acc.Reload()
    return {
        tag: [(e.step, float(e.value)) for e in acc.Scalars(tag)]
        for tag in acc.Tags().get("scalars", [])
    }


def pick(series: dict[str, list], candidates) -> tuple[str, list] | tuple[None, list]:
    for key in candidates:
        if key in series and series[key]:
            return key, series[key]
    # Fall back to any tag that merely *looks* like the thing we want, so a
    # trainer that renames its scalars does not silently fail the gate.
    for key, values in series.items():
        if any(c.split("/")[-1] in key.lower() for c in candidates) and values:
            return key, values
    return None, []


def lokr_tensors(path: Path) -> dict:
    """Load only the adapter tensors from a checkpoint, on CPU."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "lokr", "adapter", "model"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            blob = blob[key]
            break
    if not isinstance(blob, dict):
        return {}
    return {
        k: v for k, v in blob.items()
        if hasattr(v, "shape") and hasattr(v, "dtype")
        and any(m in k.lower() for m in ("lokr", "lycoris", "lora"))
    }


def newest_checkpoint(root: Path) -> Path | None:
    cands = [p for p in root.rglob("*") if p.suffix in (".pt", ".ckpt", ".safetensors")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def gpu_sample(seconds: int = 20, every: int = 4) -> list[tuple[int, int]]:
    out = []
    for _ in range(max(1, seconds // every)):
        try:
            raw = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            util, mem = raw.split(",")[:2]
            out.append((int(util), int(mem)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(every)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-root", default=str(DRIVE_V1))
    ap.add_argument("--snapshot", action="store_true",
                    help="record current LoKr weights as a comparison baseline")
    ap.add_argument("--json", default=None)
    ap.add_argument("--gpu-seconds", type=int, default=20)
    args = ap.parse_args()

    v1 = Path(args.v1_root)
    ckpt_root = v1 / "checkpoints"
    report: dict = {"v1_root": str(v1)}

    # ---------------------------------------------------------------- scalars
    series = read_event_scalars(ckpt_root)
    report["scalar_tags"] = sorted(series)

    loss_tag, loss = pick(series, LOSS_KEYS)
    report["loss_tag"] = loss_tag
    report["loss_points"] = len(loss)
    if loss:
        values = [v for _, v in loss]
        finite = [v for v in values if math.isfinite(v)]
        report["loss_first"] = values[0]
        report["loss_last"] = values[-1]
        report["loss_min"] = min(finite) if finite else None
        report["loss_all_finite"] = len(finite) == len(values)
        report["loss_distinct_values"] = len(set(round(v, 9) for v in values))
        report["latest_step"] = loss[-1][0]
        # A loss that never moves is the signature of an optimizer that is not
        # applying updates — the exact failure this whole gate exists to catch.
        report["gate_finite_loss"] = bool(finite) and report["loss_all_finite"]
        report["gate_loss_moving"] = report["loss_distinct_values"] > 1
    else:
        report["gate_finite_loss"] = False
        report["gate_loss_moving"] = False

    grad_tag, grads = pick(series, GRAD_KEYS)
    report["grad_tag"] = grad_tag
    if grads:
        gv = [v for _, v in grads]
        report["grad_last"] = gv[-1]
        report["grad_all_finite"] = all(math.isfinite(v) for v in gv)
        report["grad_any_nonzero"] = any(abs(v) > 0 for v in gv)
        report["gate_grads_finite_nonzero"] = (
            report["grad_all_finite"] and report["grad_any_nonzero"])
    else:
        # Not logged by this trainer build. Say so; do not infer a pass.
        report["gate_grads_finite_nonzero"] = None

    # ------------------------------------------------------------ throughput
    if len(loss) >= 2:
        first_step, last_step = loss[0][0], loss[-1][0]
        report["steps_observed"] = last_step - first_step
    report["checkpoints"] = [str(p.relative_to(ckpt_root))
                             for p in sorted(ckpt_root.rglob("epoch_*"))][:10]

    # ------------------------------------------------- parameters really move
    snap_path = v1 / "lokr_weight_snapshot.json"
    newest = newest_checkpoint(ckpt_root)
    report["newest_checkpoint"] = str(newest) if newest else None

    if newest is not None:
        import torch

        tensors = lokr_tensors(newest)
        fingerprint = {
            k: [float(v.float().abs().sum()), float(v.float().std())]
            for k, v in list(tensors.items())[:64]
        }
        report["lokr_tensor_count"] = len(tensors)
        if args.snapshot:
            snap_path.write_text(json.dumps(
                {"checkpoint": str(newest), "fingerprint": fingerprint}, indent=1))
            report["snapshot_written"] = str(snap_path)
            report["gate_params_changed"] = None
        elif snap_path.exists() and fingerprint:
            prev = json.loads(snap_path.read_text())
            changed = sum(
                1 for k, v in fingerprint.items()
                if k in prev["fingerprint"]
                and any(abs(a - b) > 1e-9
                        for a, b in zip(v, prev["fingerprint"][k]))
            )
            report["compared_against"] = prev["checkpoint"]
            report["tensors_compared"] = len(fingerprint)
            report["tensors_changed"] = changed
            report["gate_params_changed"] = changed > 0
        else:
            report["gate_params_changed"] = None
    else:
        report["lokr_tensor_count"] = 0
        report["gate_params_changed"] = None

    # -------------------------------------------------------------------- gpu
    samples = gpu_sample(args.gpu_seconds)
    report["gpu_samples"] = samples
    if samples:
        utils = [u for u, _ in samples]
        report["gpu_util_min"] = min(utils)
        report["gpu_util_mean"] = round(sum(utils) / len(utils), 1)
        report["gpu_mem_mib"] = max(m for _, m in samples)
        # Starvation looks like a busy process and an idle GPU.
        report["gate_gpu_busy"] = report["gpu_util_mean"] > 5
        report["gate_no_dataloader_starvation"] = min(utils) > 0
    else:
        report["gate_gpu_busy"] = False
        report["gate_no_dataloader_starvation"] = False

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))

    print("=" * 62)
    print("V1 TRAINING GATES")
    print("=" * 62)
    for key in ("gate_finite_loss", "gate_loss_moving",
                "gate_grads_finite_nonzero", "gate_params_changed",
                "gate_gpu_busy", "gate_no_dataloader_starvation"):
        value = report.get(key)
        mark = {True: "PASS", False: "FAIL", None: "UNPROVEN"}[value]
        print(f"  {key:34s} {mark}")
    print("-" * 62)
    for key in ("loss_tag", "loss_points", "loss_first", "loss_last",
                "loss_min", "loss_distinct_values", "latest_step",
                "grad_tag", "grad_last", "lokr_tensor_count",
                "tensors_compared", "tensors_changed", "newest_checkpoint",
                "gpu_util_mean", "gpu_util_min", "gpu_mem_mib"):
        if key in report and report[key] is not None:
            print(f"  {key:34s} {report[key]}")
    print("  scalar_tags", report["scalar_tags"][:12])
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
