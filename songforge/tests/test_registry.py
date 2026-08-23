from pathlib import Path

from songforge.data.registry import datasets_requiring_acceptance, load_dataset_registry


def test_default_dataset_registry_validates():
    registry = load_dataset_registry(Path("configs/data/datasets.yaml"))
    assert "babyslakh" in registry.datasets
    assert registry.datasets["babyslakh"]["license"]["name"] == "CC-BY-4.0"
    assert "gtsinger" in datasets_requiring_acceptance(registry)
    assert registry.global_split_policy["track_disjoint"] is True
