"""Run identity and experiment isolation.

A training run owns its output directory. Curves and histories are opened in
append mode so a long run can checkpoint incrementally, which means pointing a
second independent run at the same directory silently splices two experiments
into one file. That happened on a real Colab M03 run: `training_curves.csv`
ended up with 8000 rows covering steps 0-3999 twice, and the windowed loss
verdict compared one run's start against another run's end.

The rules enforced here:

* A fresh run refuses to start in a directory that already holds run artifacts.
  Nothing is deleted - the caller picks a new directory.
* Appending is legal only under an explicit `--resume`, and only when the
  checkpoint's config fingerprint matches the current config.
* Every curve and history row carries the `run_id`, so a spliced file is
  detectable after the fact rather than merely improbable.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

RUN_MANIFEST_NAME = "run_manifest.json"

#: Files that mean "a run already wrote here".
RUN_ARTIFACTS = (
    "run_manifest.json",
    "training_curves.csv",
    "rvq_history.jsonl",
    "metrics.jsonl",
    "checkpoint.pt",
    "checkpoint_last.pt",
)


class RunIsolationError(RuntimeError):
    """Raised when a run would contaminate, or continue, the wrong experiment."""


def config_fingerprint(config: dict[str, Any] | None) -> str:
    """Stable short hash of the parts of a config that define the experiment."""
    payload = json.dumps(config or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def new_run_id(run_label: str = "run") -> str:
    """Unique id for one training run."""
    return f"{run_label}-{uuid.uuid4().hex[:12]}"


def existing_run_artifacts(output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    return [name for name in RUN_ARTIFACTS if (output_dir / name).exists()]


def assert_fresh_run_dir(output_dir: str | Path) -> None:
    """A new run must not start on top of another run's artifacts."""
    found = existing_run_artifacts(output_dir)
    if not found:
        return
    raise RunIsolationError(
        f"{output_dir} already contains artifacts from a previous run: {', '.join(found)}.\n"
        "A fresh run must not append to another run's curves or history.\n"
        "Fix: point --output-dir at a new directory (one per run, e.g. .../<run-label>), "
        "or pass --resume to explicitly continue that run.\n"
        "The existing files were left untouched."
    )


def write_run_manifest(
    output_dir: str | Path,
    run_id: str,
    run_label: str,
    config: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    """Record who owns this directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "run_label": run_label,
        "config_fingerprint": config_fingerprint(config),
        "created_at": int(time.time()),
        **extra,
    }
    (output_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def read_run_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / RUN_MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_resume_compatible(checkpoint: dict[str, Any], config: dict[str, Any] | None) -> str:
    """Refuse to resume a checkpoint that came from a different experiment.

    Returns the resumed run_id so the continued run keeps the original identity.
    """
    expected = config_fingerprint(config)
    found = checkpoint.get("config_fingerprint")
    if found is None:
        # Checkpoints written before run identity existed still carry the config.
        found = config_fingerprint(checkpoint.get("config"))
    if found != expected:
        raise RunIsolationError(
            "Refusing to resume: checkpoint config does not match the current config "
            f"(checkpoint {found}, current {expected}).\n"
            "Resuming across incompatible configs would mix two experiments in one curve."
        )
    return str(checkpoint.get("run_id") or "")


def curve_run_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Distinct run ids present in a curve or history, in first-seen order."""
    seen: list[str] = []
    for row in rows:
        value = row.get("run_id")
        if value and value not in seen:
            seen.append(str(value))
    return seen


def assert_single_run(rows: list[dict[str, Any]], what: str = "curve") -> None:
    """Fail when a file mixes rows from more than one run, or repeats a step."""
    ids = curve_run_ids(rows)
    if len(ids) > 1:
        raise RunIsolationError(
            f"{what} contains rows from {len(ids)} different runs ({', '.join(ids[:4])}). "
            "Artifacts from independent runs were spliced together."
        )
    steps = [row.get("step") for row in rows if row.get("step") is not None]
    duplicates = len(steps) - len(set(steps))
    if duplicates:
        raise RunIsolationError(
            f"{what} repeats {duplicates} step value(s); it holds more than one run's rows."
        )
