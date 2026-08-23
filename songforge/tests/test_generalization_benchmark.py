"""The Generalization Benchmark: breadth guarantees and holdout hygiene.

The frozen eight are a regression test; this benchmark is what keeps SongForge
a free-form generator. These tests pin its structural promises: enough
prompts, every coverage category present, a real held-out subset, and —
critically — held-out prompt ids never referenced anywhere in training or
tooling code, so "never used for training or prompt tuning" is enforced by CI
rather than by memory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "generalization_prompts.yaml"
FROZEN = ROOT / "benchmarks" / "prompts.yaml"

EXPECTED_CATEGORIES = {
    "solo", "arrangement", "vocals", "full_band", "cinematic",
    "electronic", "acoustic", "genre_fusion", "multilingual", "compositional",
}

#: The difficult combinations the benchmark explicitly promised to contain.
REQUIRED_HARD_COMBOS = {
    "piano_violin_female_vocal",
    "guitar_strings_drums",
    "orchestral_synth_hybrid",
    "acoustic_male_vocal",
    "violin_led_rock",
    "cinematic_piano_climax",
}


def load() -> dict:
    return yaml.safe_load(BENCH.read_text(encoding="utf-8"))


def prompts() -> list[dict]:
    return load()["prompts"]


class TestShape:
    def test_size_is_forty_to_sixty(self):
        assert 40 <= len(prompts()) <= 60

    def test_ids_unique_and_disjoint_from_frozen_eight(self):
        ids = [p["id"] for p in prompts()]
        assert len(ids) == len(set(ids))
        frozen = {p["id"] for p in yaml.safe_load(FROZEN.read_text(encoding="utf-8"))["prompts"]}
        assert not frozen & set(ids)

    def test_every_category_present_with_at_least_three_prompts(self):
        by_category: dict[str, int] = {}
        for p in prompts():
            by_category[p["category"]] = by_category.get(p["category"], 0) + 1
        assert set(by_category) == EXPECTED_CATEGORIES
        assert all(count >= 3 for count in by_category.values()), by_category

    def test_every_prompt_has_text_and_evaluation_targets(self):
        for p in prompts():
            assert p["prompt"].strip(), p["id"]
            assert p["expects_instruments"], p["id"]

    def test_same_seed_discipline_as_frozen_benchmark(self):
        assert load()["defaults"]["seed"] == 20260818

    def test_required_hard_combos_present(self):
        ids = {p["id"] for p in prompts()}
        assert REQUIRED_HARD_COMBOS <= ids

    def test_multilingual_covers_at_least_three_non_english_languages(self):
        languages = {
            p["vocal"]["language"]
            for p in prompts()
            if p.get("category") == "multilingual" and p.get("vocal")
        }
        assert len(languages - {"en"}) >= 3, languages
        with_lyrics = [
            p for p in prompts()
            if p.get("category") == "multilingual" and p.get("lyrics")
        ]
        assert len(with_lyrics) >= 3


class TestHoldout:
    def test_at_least_ten_heldout_spanning_six_categories(self):
        held = [p for p in prompts() if p.get("heldout")]
        assert len(held) >= 10
        assert len({p["category"] for p in held}) >= 6

    def test_dev_tier_still_has_breadth(self):
        dev = [p for p in prompts() if not p.get("heldout")]
        assert len({p["category"] for p in dev}) == len(EXPECTED_CATEGORIES)

    def test_heldout_ids_never_referenced_outside_benchmark_and_tests(self):
        """The leak guard. A held-out prompt id appearing in configs, scripts,
        or src means someone wired it into training or prompt tuning."""
        held_ids = {p["id"] for p in prompts() if p.get("heldout")}
        assert held_ids
        offenders: list[str] = []
        for directory in ("configs", "scripts", "src", "deploy", "notebooks"):
            base = ROOT / directory
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix in {".pyc", ".wav", ".png"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for held_id in held_ids:
                    if held_id in text:
                        offenders.append(f"{path.relative_to(ROOT)}: {held_id}")
        assert not offenders, (
            "Held-out generalization prompts are referenced outside the "
            f"benchmark: {offenders}. They must never be used for training or "
            "prompt tuning."
        )
