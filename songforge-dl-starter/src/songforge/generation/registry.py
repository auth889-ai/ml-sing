"""Adapter registry, and the benchmark prompt set loader.

Keeping adapters behind a name lets the benchmark and the product select a
foundation by string, so swapping the selected model is a config change rather
than an import change.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .adapter import FoundationAdapter
from .request import Section, SongRequest, VocalSpec

_ADAPTERS: dict[str, Callable[..., FoundationAdapter]] = {}


def register(name: str) -> Callable[[Callable[..., FoundationAdapter]], Callable[..., FoundationAdapter]]:
    def decorate(factory: Callable[..., FoundationAdapter]) -> Callable[..., FoundationAdapter]:
        _ADAPTERS[name] = factory
        return factory

    return decorate


def available() -> list[str]:
    _import_adapters()
    return sorted(_ADAPTERS)


def build(name: str, **kwargs: Any) -> FoundationAdapter:
    _import_adapters()
    if name not in _ADAPTERS:
        raise SystemExit(f"unknown adapter {name!r}; available: {', '.join(sorted(_ADAPTERS)) or 'none'}")
    return _ADAPTERS[name](**kwargs)


def _import_adapters() -> None:
    """Import adapter modules for their registration side effects.

    Each import is guarded: a foundation whose heavy dependencies are missing
    should not stop the others from being listed or run.
    """
    from importlib import import_module

    for module in ("null", "acestep", "musicgen", "stable_audio"):
        try:
            import_module(f"songforge.generation.adapters.{module}")
        except ImportError:
            continue


def load_prompts(path: str | Path) -> list[SongRequest]:
    """Load the benchmark prompt set as SongRequests, defaults applied."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults = payload.get("defaults") or {}
    requests: list[SongRequest] = []

    for entry in payload.get("prompts") or []:
        vocal = entry.get("vocal", defaults.get("vocal"))
        structure = tuple(Section(**section) for section in (entry.get("structure") or ()))
        request = SongRequest(
            prompt=" ".join(str(entry["prompt"]).split()),
            lyrics=entry.get("lyrics"),
            genre=tuple(entry.get("genre") or ()),
            mood=tuple(entry.get("mood") or ()),
            instruments=tuple(entry.get("instruments") or ()),
            vocal=VocalSpec(**vocal) if isinstance(vocal, dict) else None,
            bpm=entry.get("bpm"),
            key=entry.get("key"),
            duration_seconds=float(entry.get("duration_seconds", defaults.get("duration_seconds", 60))),
            structure=structure,
            seed=int(entry.get("seed", defaults.get("seed", 0))),
            extra={
                "id": entry["id"],
                "title": entry.get("title", entry["id"]),
                "expects_instruments": list(entry.get("expects_instruments") or ()),
            },
        )
        request.validate()
        requests.append(request)

    ids = [r.extra["id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate prompt ids in {path}")
    return requests
