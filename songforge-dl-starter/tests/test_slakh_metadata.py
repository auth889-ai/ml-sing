"""BabySlakh instrument metadata and weighted song-disjoint splitting (M04 data expansion).

Instrument labels must come from the corpus. Spectral heuristics can separate
"bass-heavy" from "percussive" but cannot tell a piano from a guitar, and a
fabricated label would put fiction into the evaluation record.
"""

import pytest
import yaml

from songforge.data.manifest import AudioRecord, assert_no_track_leakage
from songforge.data.slakh_metadata import (
    MIX_FAMILY,
    family_counts,
    find_track_dir,
    instrument_lookup,
    stem_metadata,
)
from songforge.data.splits import (
    SplitConfig,
    assert_group_disjoint,
    assign_splits,
    plan_weighted_group_splits,
)

METADATA = {
    "stems": {
        "S00": {
            "inst_class": "Piano", "midi_program_name": "Acoustic Grand Piano",
            "program_num": 0, "is_drum": False, "plugin_name": "grand.nkm",
        },
        "S01": {
            "inst_class": "Guitar", "midi_program_name": "Electric Guitar (clean)",
            "program_num": 27, "is_drum": False, "plugin_name": "elektrik.nkm",
        },
        "S02": {
            "inst_class": "Drums", "midi_program_name": "Standard Kit",
            "program_num": 0, "is_drum": True,
        },
    }
}


@pytest.fixture
def track(tmp_path):
    directory = tmp_path / "Track00007"
    (directory / "stems").mkdir(parents=True)
    (directory / "metadata.yaml").write_text(yaml.safe_dump(METADATA), encoding="utf-8")
    for name in ("S00", "S01", "S02"):
        (directory / "stems" / f"{name}.wav").write_bytes(b"")
    (directory / "mix.wav").write_bytes(b"")
    stem_metadata.cache_clear() if hasattr(stem_metadata, "cache_clear") else None
    return directory


# --- track discovery ----------------------------------------------------


def test_find_track_dir_walks_up_to_the_track(track):
    assert find_track_dir(track / "stems" / "S00.wav") == track


def test_find_track_dir_returns_none_outside_a_track(tmp_path):
    assert find_track_dir(tmp_path / "random" / "audio.wav") is None


# --- real labels --------------------------------------------------------


def test_piano_stem_is_labelled_from_metadata(track):
    meta = stem_metadata(track / "stems" / "S00.wav")
    assert meta.instrument_family == "Piano"
    assert meta.instrument_name == "Acoustic Grand Piano"
    assert meta.midi_program == 0
    assert meta.is_drum is False
    assert meta.source_track_id == "Track00007"
    assert meta.stem_id == "S00"


def test_guitar_stem_is_distinguished_from_piano(track):
    """The distinction a spectral heuristic cannot make."""
    piano = stem_metadata(track / "stems" / "S00.wav")
    guitar = stem_metadata(track / "stems" / "S01.wav")
    assert piano.instrument_family != guitar.instrument_family
    assert guitar.instrument_family == "Guitar"
    assert guitar.midi_program == 27


def test_drum_stem_is_flagged(track):
    meta = stem_metadata(track / "stems" / "S02.wav")
    assert meta.is_drum is True
    assert meta.instrument_family == "Drums"


def test_mix_is_labelled_as_a_mixture(track):
    meta = stem_metadata(track / "mix.wav")
    assert meta.instrument_family == MIX_FAMILY
    assert meta.stem_id is None
    assert meta.is_drum is False


def test_unknown_stem_stays_unlabelled_rather_than_guessed(track):
    meta = stem_metadata(track / "stems" / "S99.wav")
    assert meta.source_track_id == "Track00007"
    assert meta.instrument_family is None
    assert meta.instrument_name is None


def test_missing_metadata_file_yields_no_labels(tmp_path):
    directory = tmp_path / "Track00009" / "stems"
    directory.mkdir(parents=True)
    meta = stem_metadata(directory / "S00.wav")
    assert meta.instrument_family is None


def test_lookup_adapter_returns_a_plain_dict(track):
    payload = instrument_lookup(track / "stems" / "S01.wav")
    assert payload["instrument_family"] == "Guitar"
    assert payload["source_track_id"] == "Track00007"


def test_lookup_outside_a_track_is_empty(tmp_path):
    assert instrument_lookup(tmp_path / "loose.wav") == {}


def test_family_counts_reports_unknown_separately(track, tmp_path):
    counts = family_counts(
        [track / "stems" / "S00.wav", track / "stems" / "S01.wav", tmp_path / "loose.wav"]
    )
    assert counts["Piano"] == 1
    assert counts["Guitar"] == 1
    assert counts["unknown"] == 1


# --- weighted song-disjoint splitting -----------------------------------


def make_records(weights: dict[str, int]) -> list[AudioRecord]:
    records = []
    for track_id, count in weights.items():
        for index in range(count):
            records.append(
                AudioRecord(
                    id=f"{track_id}-{index}", path=f"/d/{track_id}/{index}.wav", split="unassigned",
                    source="babyslakh", license="CC-BY-4.0", track_id=track_id,
                    duration_seconds=2.0,
                )
            )
    return records


def test_weighted_split_balances_duration_not_song_count():
    """Counting songs treats a 30 s song and a 6 min song alike; duration must win."""
    weights = {"TrackA": 1000.0, "TrackB": 100.0, "TrackC": 100.0, "TrackD": 100.0, "TrackE": 100.0}
    plan = plan_weighted_group_splits(weights, SplitConfig(0.8, 0.1, 0.1, seed=7, strategy="weighted"))
    totals: dict[str, float] = {}
    for key, split in plan.items():
        totals[split] = totals.get(split, 0.0) + weights[key]
    assert totals["train"] >= totals.get("val", 0)
    assert totals["train"] >= totals.get("test", 0)
    assert set(plan.values()) == {"train", "val", "test"}


def test_weighted_split_keeps_whole_songs_intact():
    records = make_records({"T1": 40, "T2": 5, "T3": 5, "T4": 6, "T5": 7})
    assigned = assign_splits(records, SplitConfig(0.8, 0.1, 0.1, seed=3, strategy="weighted"))

    by_track: dict[str, set[str]] = {}
    for record in assigned:
        by_track.setdefault(record.track_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_track.values())
    assert_no_track_leakage(assigned)
    assert_group_disjoint(assigned, "song")


def test_weighted_split_gives_training_the_most_segments():
    records = make_records({f"T{i}": 10 + i for i in range(12)})
    assigned = assign_splits(records, SplitConfig(0.8, 0.1, 0.1, seed=11, strategy="weighted"))
    counts = {s: sum(1 for r in assigned if r.split == s) for s in ("train", "val", "test")}
    assert counts["train"] > counts["val"]
    assert counts["train"] > counts["test"]


def test_weighted_split_is_deterministic():
    records = make_records({f"T{i}": 5 + i for i in range(8)})
    config = SplitConfig(seed=5, strategy="weighted")
    first = [r.split for r in assign_splits(records, config)]
    second = [r.split for r in assign_splits(list(reversed(records)), config)]
    by_id = {r.id: s for r, s in zip(reversed(records), second)}
    assert first == [by_id[r.id] for r in records]


def test_weighted_strategy_is_accepted_by_validation():
    SplitConfig(strategy="weighted").validate()
    with pytest.raises(ValueError):
        SplitConfig(strategy="nonsense").validate()
