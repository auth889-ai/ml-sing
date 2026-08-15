from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    run_id: str | None = None,
    run_label: str | None = None,
) -> None:
    """Save weights plus the run identity that produced them.

    `run_id` and `config_fingerprint` let a later resume prove it is continuing
    the same experiment rather than splicing an unrelated one.
    """
    from .run import config_fingerprint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "metrics": metrics or {},
            "run_id": run_id,
            "run_label": run_label,
            "config_fingerprint": config_fingerprint(config),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
