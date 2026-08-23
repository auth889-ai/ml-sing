from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_DATASET_FIELDS = {
    "name",
    "role",
    "source_url",
    "license",
    "access",
    "split_policy",
    "integrity",
}

REQUIRED_LICENSE_FIELDS = {
    "name",
    "url",
    "commercial_allowed",
    "requires_user_acceptance",
}


@dataclass(frozen=True)
class DatasetRegistry:
    datasets: dict[str, dict[str, Any]]
    default_order: list[str]
    global_split_policy: dict[str, Any]


def load_dataset_registry(path: str | Path) -> DatasetRegistry:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Dataset registry must contain a non-empty 'datasets' mapping")

    default_order = raw.get("default_order", list(datasets))
    global_split_policy = raw.get("global_split_policy", {})
    registry = DatasetRegistry(datasets=datasets, default_order=default_order, global_split_policy=global_split_policy)
    errors = validate_dataset_registry(registry)
    if errors:
        raise ValueError("\n".join(errors))
    return registry


def validate_dataset_registry(registry: DatasetRegistry) -> list[str]:
    errors: list[str] = []
    if not registry.default_order:
        errors.append("default_order must name at least one dataset")

    for dataset_id in registry.default_order:
        if dataset_id not in registry.datasets:
            errors.append(f"default_order references unknown dataset: {dataset_id}")

    for dataset_id, spec in registry.datasets.items():
        missing = REQUIRED_DATASET_FIELDS.difference(spec)
        if missing:
            errors.append(f"{dataset_id} missing required fields: {', '.join(sorted(missing))}")
            continue

        license_spec = spec.get("license", {})
        missing_license = REQUIRED_LICENSE_FIELDS.difference(license_spec)
        if missing_license:
            errors.append(f"{dataset_id}.license missing required fields: {', '.join(sorted(missing_license))}")

        if not spec.get("source_url"):
            errors.append(f"{dataset_id}.source_url must be set")
        if not isinstance(spec.get("split_policy"), dict):
            errors.append(f"{dataset_id}.split_policy must be a mapping")
        if not isinstance(spec.get("integrity"), dict):
            errors.append(f"{dataset_id}.integrity must be a mapping")

    return errors


def datasets_requiring_acceptance(registry: DatasetRegistry) -> list[str]:
    return [
        dataset_id
        for dataset_id, spec in registry.datasets.items()
        if spec.get("license", {}).get("requires_user_acceptance") or spec.get("access", {}).get("gated")
    ]
