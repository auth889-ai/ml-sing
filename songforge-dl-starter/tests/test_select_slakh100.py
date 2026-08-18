"""Slakh-100 selection: deterministic, split-respecting, quota-honest."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "select_slakh100", ROOT / "scripts" / "select_slakh100.py"
)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

FAMILIES = ["Piano", "Guitar", "Bass", "Drums", "Strings", "Strings (continued)",
            "Brass", "Reed", "Pipe", "Organ", "Synth Pad", "Synth Lead",
            "Chromatic Percussion"]


def make_track(base: Path, name: str, families: list[str], *,
               rendered: bool = True, lufs: float = -15.0) -> None:
    stems = {
        f"S{i:02d}": {
            "inst_class": family,
            "audio_rendered": rendered,
            "integrated_loudness": lufs,
            "midi_program_name": family.lower(),
        }
        for i, family in enumerate(families)
    }
    track = base / name
    track.mkdir(parents=True)
    (track / "metadata.yaml").write_text(yaml.safe_dump({"stems": stems}), encoding="utf-8")


def build_corpus(root: Path, per_split: dict[str, int]) -> None:
    counter = 0
    for split_key, split_dir in sel.SPLIT_DIRS.items():
        base = root / split_dir
        for i in range(per_split[split_key]):
            counter += 1
            # Core four always present; rotate rare families for variety.
            rare = [FAMILIES[4 + (counter + j) % 9] for j in range(3)]
            make_track(base, f"Track{counter:05d}",
                       ["Piano", "Guitar", "Bass", "Drums", *rare])


def write_config(path: Path, counts: dict[str, int]) -> None:
    config = {
        "splits": counts,
        "instrument_stratification": {
            "eligibility": {"min_stems": 6, "all_stems_rendered": True,
                            "target_family_min_lufs": -30},
            "train_quotas": {"Strings": 4, "Brass": 2, "Reed": 2},
            "val_test_quotas_each": {"Strings": 1, "other_rare_families": 1},
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def run(root: Path, config: Path, out: Path) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "select_slakh100.py"),
         "--slakh-root", str(root), "--config", str(config), "--output", str(out)],
        check=True, capture_output=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


class TestSelection:
    def test_selects_counts_per_official_split_and_is_deterministic(self, tmp_path):
        root = tmp_path / "slakh"
        build_corpus(root, {"train": 12, "val": 4, "test": 4})
        config = tmp_path / "config.yaml"
        write_config(config, {"train": 8, "val": 2, "test": 2})

        first = run(root, config, tmp_path / "a.json")
        second = run(root, config, tmp_path / "b.json")
        assert first["selection"] == second["selection"]
        assert len(first["selection"]["train"]) == 8
        assert len(first["selection"]["val"]) == 2
        assert len(first["selection"]["test"]) == 2
        # Split membership follows the directory, never crosses.
        train_dirs = {p.name for p in (root / "train").iterdir()}
        assert set(first["selection"]["train"]) <= train_dirs

    def test_quota_families_preferred_and_coverage_reported(self, tmp_path):
        root = tmp_path / "slakh"
        build_corpus(root, {"train": 12, "val": 4, "test": 4})
        config = tmp_path / "config.yaml"
        write_config(config, {"train": 8, "val": 2, "test": 2})
        result = run(root, config, tmp_path / "out.json")
        coverage = result["report"]["train"]["family_coverage"]
        assert coverage.get("Strings", 0) >= 4  # aliased class counts as Strings
        assert result["report"]["train"]["unmet_quotas"] == {}

    def test_ineligible_tracks_are_excluded(self, tmp_path):
        root = tmp_path / "slakh"
        build_corpus(root, {"train": 10, "val": 4, "test": 4})
        make_track(root / "train", "Track99998",
                   ["Piano", "Guitar", "Bass", "Drums", "Strings", "Brass"],
                   rendered=False)  # unrendered stem -> ineligible
        make_track(root / "train", "Track99999", ["Piano", "Guitar"])  # too few stems
        config = tmp_path / "config.yaml"
        write_config(config, {"train": 8, "val": 2, "test": 2})
        result = run(root, config, tmp_path / "out.json")
        chosen = set(result["selection"]["train"])
        assert "Track99998" not in chosen
        assert "Track99999" not in chosen

    def test_fails_loudly_when_not_enough_eligible_tracks(self, tmp_path):
        root = tmp_path / "slakh"
        build_corpus(root, {"train": 3, "val": 4, "test": 4})
        config = tmp_path / "config.yaml"
        write_config(config, {"train": 8, "val": 2, "test": 2})
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "select_slakh100.py"),
             "--slakh-root", str(root), "--config", str(config),
             "--output", str(tmp_path / "out.json")],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
        assert "eligible" in proc.stderr

    def test_tar_member_list_matches_selection(self, tmp_path):
        root = tmp_path / "slakh"
        build_corpus(root, {"train": 12, "val": 4, "test": 4})
        config = tmp_path / "config.yaml"
        write_config(config, {"train": 8, "val": 2, "test": 2})
        out = tmp_path / "out.json"
        result = run(root, config, out)
        members = out.with_suffix(".tar_members.txt").read_text().splitlines()
        assert len(members) == 12
        assert f"train/{result['selection']['train'][0]}" in members
        assert all(m.split("/")[0] in {"train", "validation", "test"} for m in members)
