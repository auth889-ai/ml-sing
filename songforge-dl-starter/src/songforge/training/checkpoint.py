from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that affects training, so a resume continues the same stream."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_byte_tensor(value: Any) -> torch.Tensor:
    """RNG states must be CPU uint8 tensors.

    `torch.load(map_location="cuda")` moves every saved tensor to the GPU,
    including the RNG state, and `set_rng_state` then rejects it. Forcing it back
    to CPU is what lets a GPU resume restore the stream.
    """
    return torch.as_tensor(value, dtype=torch.uint8).cpu().contiguous()


class RngRestoreError(RuntimeError):
    """Raised when a required RNG stream cannot be restored on an authoritative resume."""


def restore_rng_state(state: dict[str, Any] | None, strict: bool = True) -> bool:
    """Restore RNG streams captured by `capture_rng_state`.

    Strict by default. An acceptance run that silently continues on a fresh RNG
    stream is no longer the same experiment, so every required stream - python,
    numpy, torch CPU, and CUDA whenever the checkpoint carries it and a GPU is
    present - must come back or the resume fails loudly.

    `strict=False` exists only for debugging and non-authoritative runs.
    """
    if not state:
        if strict:
            raise RngRestoreError(
                "checkpoint carries no RNG state; an authoritative resume cannot continue "
                "the same experiment without it"
            )
        return False

    required = ["python", "numpy", "torch"]
    if torch.cuda.is_available() and state.get("torch_cuda"):
        required.append("torch_cuda")
    missing = [name for name in required if name not in state or state[name] is None]
    if missing and strict:
        raise RngRestoreError(f"checkpoint is missing RNG state for: {', '.join(missing)}")

    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(_cpu_byte_tensor(state["torch"]))
        cuda_state = state.get("torch_cuda")
        if cuda_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_cpu_byte_tensor(s) for s in cuda_state])
    except (RuntimeError, TypeError, ValueError) as exc:
        if strict:
            raise RngRestoreError(f"could not restore RNG state: {exc}") from exc
        print(f"warning: could not restore RNG state ({exc}); continuing with a fresh stream")
        return False
    return True


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    run_id: str | None = None,
    run_label: str | None = None,
    scaler: Any | None = None,
    rng: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically save everything needed to resume the same logical experiment.

    Written to a sibling ``.tmp`` file and then ``os.replace``d, so a runtime that
    dies mid-write leaves the previous checkpoint intact rather than a truncated
    file. Carries the AMP scaler and RNG streams alongside the weights, because
    resuming without them continues a different experiment in all but name.
    """
    from .run import config_fingerprint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "metrics": metrics or {},
        "run_id": run_id,
        "run_label": run_label,
        "config_fingerprint": config_fingerprint(config),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state() if rng else None,
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    scaler: Any | None = None,
    restore_rng: bool = True,
    strict_rng: bool = True,
) -> dict[str, Any]:
    """Load weights, optimizer, AMP scaler and RNG streams.

    On an authoritative resume every piece of state is required: missing
    optimizer or AMP scaler state would silently change the experiment just as
    surely as a reset RNG stream.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        if "optimizer" not in checkpoint or checkpoint["optimizer"] is None:
            if strict_rng:
                raise RngRestoreError("checkpoint carries no optimizer state; cannot resume authoritatively")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        if not checkpoint.get("scaler"):
            if strict_rng:
                raise RngRestoreError("checkpoint carries no AMP scaler state; cannot resume authoritatively")
        else:
            scaler.load_state_dict(checkpoint["scaler"])
    elif scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])

    checkpoint["rng_restored"] = (
        restore_rng_state(checkpoint.get("rng_state"), strict=strict_rng) if restore_rng else False
    )
    return checkpoint
