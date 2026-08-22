"""Stage training tensors from Drive to local disk, with verification.

Drive is the durable store; it must not sit in the per-step read path. The
Colab FUSE mount stalls under sustained random reads — observed directly:
the dataloader blocked with the GPU pinned at 0% while a plain ``ls`` on the
mount timed out after 15 s. Copying the corpus to local disk once removes
Drive from every training step.

Verification is the point of this script, not the copy. It refuses to report
success unless the local corpus is complete and loadable:

  * local count == Drive count (unique final tensors, ``.tmp.pt`` excluded)
  * no zero-byte or truncated files
  * a random sample really loads through torch and carries the expected keys
  * sampled tensors are non-empty and finite

    python scripts/v1_stage_tensors.py \\
        --source /content/drive/MyDrive/songforge-dl/v1/tensors \\
        --dest   /content/tensors_local --sample 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FINAL_KEYS = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
    "metadata",
)


def final_tensors(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.pt") if not p.name.endswith(".tmp.pt"))


def structurally_ok(path: Path) -> bool:
    """Complete torch zip archive? Catches truncation without a full read."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with zipfile.ZipFile(path) as archive:
            return any(n.endswith("data.pkl") for n in archive.namelist())
    except (zipfile.BadZipFile, OSError, EOFError):
        return False


def deep_check(path: Path) -> tuple[bool, str]:
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {type(exc).__name__}: {exc}"
    if not isinstance(blob, dict):
        return False, "not a dict"
    missing = [k for k in FINAL_KEYS if k not in blob]
    if missing:
        return False, f"missing keys: {missing}"
    shapes = {}
    for key, value in blob.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False, f"{key} is empty"
            if value.is_floating_point() and not torch.isfinite(value).all():
                return False, f"{key} has non-finite values"
            shapes[key] = tuple(value.shape)
    return True, json.dumps(shapes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--sample", type=int, default=25,
                        help="how many local tensors to fully torch.load")
    parser.add_argument("--report", default=None)
    parser.add_argument("--workers", type=int, default=16,
                        help="parallel copy workers; Drive FUSE is latency-bound")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    source, dest = Path(args.source), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()

    src_files = final_tensors(source)
    src_count = len(src_files)
    if src_count == 0:
        print("FATAL: no tensors found on the source (is Drive mounted?)", file=sys.stderr)
        return 1
    print(f"[stage] source tensors: {src_count}")

    # Copy in parallel. Drive's FUSE mount is latency-bound per file, not
    # bandwidth-bound: a serial loop spends nearly all its time waiting on
    # round-trips and leaves the link idle. Measured on the live L4 runtime, the
    # serial version moved ~12 tensors/min, which is ~5 h for the 3,626-tensor
    # corpus — longer than the training run it exists to feed. Overlapping the
    # waits recovers that time; the per-file semantics below are unchanged.
    #
    # Threads (not processes) are correct here: the work is I/O, so the GIL is
    # released during the copy, and shutil.copyfile on a pre-created temp name
    # keeps each worker independent.
    pending = []
    skipped = 0
    for src in src_files:
        target = dest / src.name
        if target.exists() and target.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        pending.append(src)

    copied = failed = 0
    done = skipped

    def stage_one(src: Path) -> tuple[str, str | None]:
        """Copy one tensor. Atomic: a kill never leaves a half file behind."""
        target = dest / src.name
        tmp = target.with_name(f".{target.name}.partial")
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, target)
            return src.name, None
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            return src.name, f"{type(exc).__name__}: {exc}"

    if pending:
        print(f"[stage] copying {len(pending)} tensors with {args.workers} workers "
              f"({skipped} already present)")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, error in pool.map(stage_one, pending):
                done += 1
                if error is None:
                    copied += 1
                else:
                    failed += 1
                    print(f"[stage] COPY FAIL {name}: {error}", file=sys.stderr)
                if done % 250 == 0:
                    rate = done / max(time.time() - started, 1e-6) * 60
                    print(f"[stage] {done}/{src_count}  ({rate:.0f}/min)", flush=True)

    local_files = final_tensors(dest)
    local_count = len(local_files)
    stray_tmp = list(dest.glob("*.tmp.pt")) + list(dest.glob(".*.partial"))
    zero_byte = [p.name for p in local_files if p.stat().st_size == 0]
    broken = [p.name for p in local_files if not structurally_ok(p)]

    random.seed(args.seed)
    sample = random.sample(local_files, min(args.sample, local_count)) if local_count else []
    sample_failures = []
    shapes_seen = None
    for path in sample:
        ok, detail = deep_check(path)
        if not ok:
            sample_failures.append(f"{path.name}: {detail}")
        elif shapes_seen is None:
            shapes_seen = detail

    report = {
        "source": str(source),
        "dest": str(dest),
        "source_count": src_count,
        "local_count": local_count,
        "copied": copied,
        "already_present": skipped,
        "copy_failures": failed,
        "stray_tmp_or_partial": [p.name for p in stray_tmp],
        "zero_byte": zero_byte,
        "structurally_broken": broken[:20],
        "sampled": len(sample),
        "sample_failures": sample_failures,
        "example_shapes": shapes_seen,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    ok = (
        local_count == src_count
        and failed == 0
        and not zero_byte
        and not broken
        and not sample_failures
        and not stray_tmp
    )
    report["verified"] = ok

    print(json.dumps(report, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")

    if not ok:
        print("STAGING VERIFICATION FAILED", file=sys.stderr)
        return 1
    print(f"STAGING VERIFIED: {local_count} tensors at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
