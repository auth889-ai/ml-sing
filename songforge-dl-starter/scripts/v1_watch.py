"""Continuous training monitor — writes status.json so humans don't poll.

Runs alongside training and records what a status check would otherwise cost
a round trip to discover: stage, epoch, step, latest loss, gradient norm,
last checkpoint, GPU memory, and any error the trainer printed.

Status is written to LOCAL disk first and mirrored to Drive best-effort. The
mirror is deliberately non-blocking: the Colab FUSE mount stalls under load,
and a monitor that hangs on its own status write is worse than no monitor.

    python scripts/v1_watch.py \\
        --log /content/drive/MyDrive/songforge-dl/v1/logs/t4_train.log \\
        --checkpoints /content/drive/MyDrive/songforge-dl/v1/checkpoints \\
        --local-status /content/v1_status.json \\
        --drive-status /content/drive/MyDrive/songforge-dl/v1/status.json \\
        --interval 60
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

# The trainer's exact step-line format is not contractual, so match loosely
# and keep the raw line: a status file that silently reports nothing because
# a log format changed is a trap.
STEP_RE = re.compile(r"\bstep[\s:=]+(\d+)", re.I)
EPOCH_RE = re.compile(r"\bepoch[\s:=]+(\d+)", re.I)
LOSS_RE = re.compile(r"\bloss[\s:=]+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", re.I)
GRAD_RE = re.compile(r"grad(?:ient)?[_\s]*norm[\s:=]+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", re.I)
ERROR_RE = re.compile(r"(Traceback|\bERROR\b|FAILED|CUDA out of memory)", re.I)


def tail(path: Path, limit: int = 400) -> list[str]:
    try:
        with path.open("r", errors="replace") as handle:
            return handle.readlines()[-limit:]
    except OSError:
        return []


def gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()[0]
        used, util = (x.strip() for x in out.split(","))
        return {"vram_used_mib": int(used), "gpu_util_pct": int(util)}
    except Exception:  # noqa: BLE001
        return {"vram_used_mib": None, "gpu_util_pct": None}


def trainer_alive() -> int:
    try:
        out = subprocess.run(["pgrep", "-fc", "train.py fixed"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip() or 0)
    except Exception:  # noqa: BLE001
        return -1


def latest_checkpoint(root: Path) -> str | None:
    try:
        candidates = [p for p in root.rglob("epoch_*") if p.is_dir()]
        if not candidates:
            return None
        return str(max(candidates, key=lambda p: p.stat().st_mtime))
    except OSError:
        return None


def build_status(args, peak: dict) -> dict:
    lines = tail(Path(args.log))
    step = epoch = loss = grad = None
    last_metric_line = None
    for line in lines:
        if (m := STEP_RE.search(line)):
            step = int(m.group(1))
            last_metric_line = line.strip()[-300:]
        if (m := EPOCH_RE.search(line)):
            epoch = int(m.group(1))
        if (m := LOSS_RE.search(line)):
            loss = float(m.group(1))
        if (m := GRAD_RE.search(line)):
            grad = float(m.group(1))

    errors = [l.strip()[-300:] for l in lines if ERROR_RE.search(l)][-3:]
    gpu = gpu_stats()
    if gpu["vram_used_mib"]:
        peak["vram"] = max(peak.get("vram", 0), gpu["vram_used_mib"])
    if gpu["gpu_util_pct"]:
        peak["util"] = max(peak.get("util", 0), gpu["gpu_util_pct"])

    alive = trainer_alive()
    if errors and alive <= 0:
        stage = "failed"
    elif alive > 0:
        stage = "training"
    elif step is not None:
        stage = "stopped"
    else:
        stage = "starting"

    return {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "latest_loss": loss,
        "latest_grad_norm": grad,
        "last_metric_line": last_metric_line,
        "last_checkpoint": latest_checkpoint(Path(args.checkpoints)),
        "trainer_processes": alive,
        **gpu,
        "peak_vram_mib": peak.get("vram"),
        "peak_gpu_util_pct": peak.get("util"),
        "errors": errors,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--local-status", required=True)
    parser.add_argument("--drive-status", default=None)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    peak: dict = {}
    local = Path(args.local_status)
    while True:
        status = build_status(args, peak)
        tmp = local.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
        tmp.replace(local)
        if args.drive_status:
            # Best effort only; a stalled mount must not stall the monitor.
            try:
                shutil.copyfile(local, args.drive_status)
            except Exception:  # noqa: BLE001
                pass
        if status["stage"] in ("failed", "stopped") and status["trainer_processes"] == 0:
            # Record the terminal state once, then stop spinning.
            time.sleep(args.interval)
            final = build_status(args, peak)
            if final["trainer_processes"] == 0:
                local.write_text(json.dumps(final, indent=1), encoding="utf-8")
                return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
