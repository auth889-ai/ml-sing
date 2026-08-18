"""The listening gate: parsing, ranking, attribution, and the one-intervention rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "listening_review", ROOT / "scripts" / "listening_review.py"
)
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)


def sheet(tmp_path: Path, rows: list[str]) -> Path:
    header = "id," + ",".join(lr.ALL_DIMENSIONS) + f",{lr.NOTES_COLUMN}"
    path = tmp_path / "review.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def row(track: str, **scores: object) -> str:
    cells = [str(scores.get(d, "")) for d in lr.ALL_DIMENSIONS]
    return ",".join([track, *cells, str(scores.get("note", ""))])


class TestReadSheet:
    def test_blank_is_unscored_and_na_is_not_applicable(self, tmp_path):
        path = sheet(tmp_path, [row("piano", overall_realism=7, vocal_realism="N/A")])
        rows = lr.read_sheet(path)
        assert rows["piano"]["overall_realism"] == 7.0
        assert rows["piano"]["vocal_realism"] == lr.NOT_APPLICABLE
        assert rows["piano"]["instrument_realism"] is None

    def test_legend_row_is_skipped(self, tmp_path):
        path = sheet(tmp_path, [row("_legend", overall_realism="not a number"), row("piano")])
        assert list(lr.read_sheet(path)) == ["piano"]

    def test_out_of_range_rejected(self, tmp_path):
        path = sheet(tmp_path, [row("piano", overall_realism=11)])
        with pytest.raises(SystemExit, match="outside 1-10"):
            lr.read_sheet(path)

    def test_non_number_rejected(self, tmp_path):
        path = sheet(tmp_path, [row("piano", overall_realism="good")])
        with pytest.raises(SystemExit, match="not a number"):
            lr.read_sheet(path)

    def test_repo_sheet_holds_the_recorded_partial_scores(self):
        """The 2026-08-18 coarse review: overall_realism only, nothing invented."""
        rows = lr.read_sheet(ROOT / "benchmarks" / "listening_review.csv")
        assert set(rows) == {"piano", "violin", "guitar", "rock", "edm",
                             "cinematic", "vocal", "rich_mix"}
        assert rows["violin"]["overall_realism"] == 7.0
        assert rows["rich_mix"]["overall_realism"] == 7.0
        for track in ("piano", "guitar", "rock", "edm", "cinematic", "vocal"):
            assert rows[track]["overall_realism"] == 3.5
        # Every other required dimension stays unscored — no score was fabricated.
        for track, scores in rows.items():
            for dimension in lr.REQUIRED_DIMENSIONS:
                if dimension == "overall_realism":
                    continue
                assert scores[dimension] in (None, lr.NOT_APPLICABLE), (track, dimension)
        filled, expected, missing = lr.completeness(rows)
        # 8 tracks x 8 required dims, minus the 12 N/A vocal cells on 6 instrumentals
        assert expected == 8 * 8 - 12
        assert filled == 8
        assert len(missing) == expected - 8


class TestCompleteness:
    def test_na_cells_do_not_count_as_missing(self, tmp_path):
        path = sheet(tmp_path, [row(
            "piano", overall_realism=7, instrument_realism=6, vocal_realism="N/A",
            lyrics_intelligibility="N/A", prompt_adherence=8, structure_coherence=8,
            spectral_clarity=5, artifact_freedom=9,
        )])
        filled, expected, missing = lr.completeness(lr.read_sheet(path))
        assert (filled, expected, missing) == (6, 6, [])

    def test_optional_dimensions_never_gate(self, tmp_path):
        path = sheet(tmp_path, [row(
            "vocal", overall_realism=7, instrument_realism=7, vocal_realism=6,
            lyrics_intelligibility=6, prompt_adherence=7, structure_coherence=7,
            spectral_clarity=7, artifact_freedom=7,
        )])
        filled, expected, missing = lr.completeness(lr.read_sheet(path))
        assert missing == []  # phrasing/pitch/emotion blanks are fine
        assert filled == expected == 8


def scored_rows(**area_means: float) -> dict[str, dict[str, object]]:
    """A full 8-track sheet where every dimension of an area scores its mean."""
    dim_score: dict[str, float] = {}
    for key, mean in area_means.items():
        for dimension in lr.WEAKNESS_AREAS[key]["dimensions"]:
            dim_score[dimension] = mean
    rows: dict[str, dict[str, object]] = {}
    for track in ("piano", "violin", "guitar", "rock", "edm", "cinematic", "vocal", "rich_mix"):
        scores: dict[str, object] = {"_note": ""}
        for dimension in lr.ALL_DIMENSIONS:
            vocal_dim = dimension in ("vocal_realism", "lyrics_intelligibility",
                                      "phrasing", "pitch_stability", "emotion")
            if vocal_dim and track not in lr.VOCAL_TRACKS:
                scores[dimension] = lr.NOT_APPLICABLE
            else:
                scores[dimension] = dim_score.get(dimension, 8.0)
        rows[track] = scores
    return rows


class TestAreasAndAttribution:
    def test_worst_area_ranks_first(self):
        rows = scored_rows(A=4.0, B=8.0, C=8.0, D=8.0, E=8.0)
        areas = lr.rank_areas(rows)
        assert areas[0]["area"] == "A"
        assert areas[0]["mean"] == 4.0

    def test_insufficient_evidence_sinks_and_is_labelled(self):
        rows = scored_rows()
        for track in rows:  # wipe spectral_clarity everywhere -> area D has 0 cells
            rows[track]["spectral_clarity"] = None
        areas = lr.rank_areas(rows)
        d = next(a for a in areas if a["area"] == "D")
        assert not d["sufficient"]
        verdict = lr.attribute(d, rows, None)
        assert verdict["bucket"] == "insufficient evidence"

    def test_area_c_attributes_to_control_layer(self):
        rows = scored_rows(C=5.0)
        areas = lr.rank_areas(rows)
        c = next(a for a in areas if a["area"] == "C")
        assert lr.attribute(c, rows, None)["bucket"] == "prompt/control-layer problem"

    def test_severe_vocal_deficit_flags_pretrained_limitation(self):
        rows = scored_rows(B=3.0)
        b = next(a for a in lr.rank_areas(rows) if a["area"] == "B")
        assert lr.attribute(b, rows, None)["bucket"] == "pretrained-model limitation (suspected)"

    def test_dense_muffling_reattributes_area_d(self):
        rows = scored_rows(D=4.0)
        for track in rows:  # dense tracks muffled too
            rows[track]["spectral_clarity"] = 4.0
        d = next(a for a in lr.rank_areas(rows) if a["area"] == "D")
        assert lr.attribute(d, rows, None)["bucket"] == "pretrained-model limitation (suspected)"

    def test_missing_instruments_point_at_captions_not_training(self):
        rows = scored_rows()
        for track in rows:
            rows[track]["instrument_realism"] = 6.0
            rows[track]["instrument_presence"] = 3.0
        a = next(x for x in lr.rank_areas(rows) if x["area"] == "A")
        assert lr.attribute(a, rows, None)["bucket"] == "prompt/control-layer problem"


class TestIntervention:
    def test_weak_c_always_runs_first_with_training_candidate_named(self):
        rows = scored_rows(A=4.0, C=5.0)
        areas = lr.rank_areas(rows)
        attributions = {a["area"]: lr.attribute(a, rows, None) for a in areas}
        decision = lr.choose_intervention(areas, attributions)
        assert decision["intervention"] == "caseC_control_experiments"
        assert decision["training_candidate_after"]["case"] == "caseA_slakh_instrument"

    def test_pure_sparse_muffling_selects_case_d(self):
        rows = scored_rows(D=4.0)
        areas = lr.rank_areas(rows)
        attributions = {a["area"]: lr.attribute(a, rows, None) for a in areas}
        assert lr.choose_intervention(areas, attributions)["intervention"] == "caseD_sparse_acoustic"

    def test_no_weakness_means_no_training(self):
        rows = scored_rows()
        areas = lr.rank_areas(rows)
        attributions = {a["area"]: lr.attribute(a, rows, None) for a in areas}
        assert lr.choose_intervention(areas, attributions)["intervention"] == "none"

    def test_artifact_weakness_gets_settings_sweep_not_lora(self):
        rows = scored_rows(E=4.0)
        areas = lr.rank_areas(rows)
        attributions = {a["area"]: lr.attribute(a, rows, None) for a in areas}
        assert lr.choose_intervention(areas, attributions)["intervention"] == "inference_settings_sweep"
