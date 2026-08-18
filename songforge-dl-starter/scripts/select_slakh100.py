"""Select the Slakh-100 track list from extracted metadata, before any audio.

Runs on a tree containing only the ``*/TrackXXXXX/metadata.yaml`` files
(pass 1 of the acquisition plan in docs/SLAKH100_DESIGN.md), applies the
eligibility rules and instrument quotas from configs/datasets/slakh100.yaml,
and emits the exact 100 track ids per official redux split — plus a tar
member list for the selective extraction pass. Deterministic: same metadata
in, same selection out.

    python scripts/select_slakh100.py \
        --slakh-root /content/slakh_meta \
        --config configs/datasets/slakh100.yaml \
        --output slakh100_selection.json

Unmet quotas are reported, never silently ignored: with a 100-song budget an
infeasible quota is a finding about the corpus, not something to hide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}
#: Slakh splits strings across two class names; both are strings.
FAMILY_ALIASES = {"Strings (continued)": "Strings"}


def track_profile(track_dir: Path, eligibility: dict) -> dict | None:
    """Family set + eligibility verdict for one track, from metadata alone."""
    metadata_path = track_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    stems = metadata.get("stems") or {}
    if not isinstance(stems, dict):
        return None

    families: set[str] = set()
    audible_families: set[str] = set()
    rendered_all = True
    min_lufs = float(eligibility.get("target_family_min_lufs", -30))
    for entry in stems.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("audio_rendered") is False:
            rendered_all = False
            continue
        family = entry.get("inst_class")
        if not family:
            continue
        family = FAMILY_ALIASES.get(family, family)
        families.add(family)
        loudness = entry.get("integrated_loudness")
        if not isinstance(loudness, (int, float)) or loudness >= min_lufs:
            audible_families.add(family)

    eligible = len(stems) >= int(eligibility.get("min_stems", 6))
    if eligibility.get("all_stems_rendered", True) and not rendered_all:
        eligible = False
    return {
        "track_id": track_dir.name,
        "families": sorted(families),
        "audible_families": sorted(audible_families),
        "stems": len(stems),
        "eligible": eligible,
    }


def greedy_select(tracks: list[dict], count: int, quotas: dict[str, int]) -> list[dict]:
    """Greedy weighted max-coverage over quota families, deterministic.

    Weights are inverse family prevalence in the eligible pool, so a track
    carrying a rare quota family beats one carrying only common families.
    Quota credit requires the family to be *audible* (louder than the LUFS
    floor), not merely present.
    """
    prevalence: dict[str, int] = {}
    for track in tracks:
        for family in track["audible_families"]:
            prevalence[family] = prevalence.get(family, 0) + 1

    remaining = dict(quotas)
    pool = sorted(tracks, key=lambda t: t["track_id"])
    chosen: list[dict] = []
    while pool and len(chosen) < count:
        def gain(track: dict) -> float:
            score = 0.0
            for family in track["audible_families"]:
                if remaining.get(family, 0) > 0:
                    score += 1.0 / max(prevalence.get(family, 1), 1)
            # Tie-break toward family diversity once quotas are satisfied.
            return score + 1e-6 * len(track["families"])

        best = max(pool, key=lambda t: (gain(t), t["track_id"]))
        pool.remove(best)
        chosen.append(best)
        for family in best["audible_families"]:
            if remaining.get(family, 0) > 0:
                remaining[family] -= 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the Slakh-100 subset.")
    parser.add_argument("--slakh-root", required=True,
                        help="dir containing train/ validation/ test/ with metadata.yaml files")
    parser.add_argument("--config", default=str(ROOT / "configs" / "datasets" / "slakh100.yaml"))
    parser.add_argument("--output", default="slakh100_selection.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    strat = config["instrument_stratification"]
    eligibility = strat.get("eligibility", {})
    split_counts: dict[str, int] = config["splits"]
    train_quotas: dict[str, int] = dict(strat.get("train_quotas", {}))
    vt = strat.get("val_test_quotas_each", {})
    vt_quotas = {family: int(vt.get("other_rare_families", 0)) for family in train_quotas}
    if "Strings" in vt:
        vt_quotas["Strings"] = int(vt["Strings"])

    root = Path(args.slakh_root)
    selection: dict[str, list[str]] = {}
    report: dict[str, dict] = {}
    for split, count in split_counts.items():
        split_dir = root / SPLIT_DIRS[split]
        if not split_dir.is_dir():
            raise SystemExit(f"missing split directory: {split_dir}")
        profiles = [
            profile
            for track_dir in sorted(split_dir.iterdir())
            if track_dir.is_dir()
            and (profile := track_profile(track_dir, eligibility)) is not None
        ]
        eligible = [p for p in profiles if p["eligible"]]
        if len(eligible) < count:
            raise SystemExit(
                f"{split}: only {len(eligible)} eligible tracks for a quota of {count}"
            )
        quotas = train_quotas if split == "train" else vt_quotas
        chosen = greedy_select(eligible, count, quotas)
        selection[split] = [t["track_id"] for t in chosen]

        coverage: dict[str, int] = {}
        for track in chosen:
            for family in track["audible_families"]:
                coverage[family] = coverage.get(family, 0) + 1
        unmet = {
            family: needed - coverage.get(family, 0)
            for family, needed in quotas.items()
            if coverage.get(family, 0) < needed
        }
        report[split] = {
            "scanned": len(profiles),
            "eligible": len(eligible),
            "selected": len(chosen),
            "family_coverage": dict(sorted(coverage.items())),
            "unmet_quotas": unmet,
        }
        if unmet:
            print(f"WARNING {split}: unmet quotas {unmet} — corpus limitation, recorded")

    members = [
        f"{SPLIT_DIRS[split]}/{track_id}"
        for split, ids in selection.items()
        for track_id in ids
    ]
    out = Path(args.output)
    out.write_text(json.dumps({"selection": selection, "report": report}, indent=2),
                   encoding="utf-8")
    out.with_suffix(".tar_members.txt").write_text("\n".join(members) + "\n",
                                                   encoding="utf-8")
    total = sum(len(ids) for ids in selection.values())
    print(f"selected {total} tracks -> {out} (+ .tar_members.txt for selective extraction)")


if __name__ == "__main__":
    main()
