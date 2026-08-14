from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from songforge.data.registry import datasets_requiring_acceptance, load_dataset_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SongForge dataset registry metadata.")
    parser.add_argument(
        "--registry",
        default="configs/data/datasets.yaml",
        help="Path to dataset registry YAML.",
    )
    args = parser.parse_args()

    registry = load_dataset_registry(Path(args.registry))
    print(f"OK: {len(registry.datasets)} datasets registered")
    gated = datasets_requiring_acceptance(registry)
    if gated:
        print("Requires user license/terms acceptance: " + ", ".join(gated))


if __name__ == "__main__":
    main()
