import pytest

from songforge.data.dedup import assert_no_cross_split_duplicates
from songforge.data.manifest import (
    AudioRecord,
    assert_no_track_leakage,
    assert_singer_disjoint,
)
from songforge.data.splits import (
    SplitConfig,
    assert_group_disjoint,
    assign_splits,
    group_key,
    plan_group_splits,
    split_report,
)


def make_records(tracks: int = 10, segments: int = 3, singers: int | None = None) -> list[AudioRecord]:
    records = []
    for track in range(tracks):
        singer = f"singer{track % singers:02d}" if singers else None
        for segment in range(segments):
            records.append(
                AudioRecord(
                    id=f"t{track:03d}s{segment}",
                    path=f"/data/t{track:03d}_{segment}.wav",
                    split="unassigned",
                    source="babyslakh",
                    license="CC-BY-4.0",
                    track_id=f"Track{track:05d}",
                    singer_id=singer,
                    segment_index=segment,
                )
            )
    return records


# --- determinism --------------------------------------------------------


def test_split_plan_is_deterministic_for_a_seed():
    keys = [f"Track{i:05d}" for i in range(20)]
    config = SplitConfig(seed=7)
    assert plan_group_splits(keys, config) == plan_group_splits(keys, config)


def test_split_plan_ignores_input_order():
    keys = [f"Track{i:05d}" for i in range(20)]
    config = SplitConfig(seed=7)
    assert plan_group_splits(keys, config) == plan_group_splits(list(reversed(keys)), config)


def test_different_seeds_give_different_plans():
    keys = [f"Track{i:05d}" for i in range(30)]
    assert plan_group_splits(keys, SplitConfig(seed=1)) != plan_group_splits(keys, SplitConfig(seed=2))


def test_assign_splits_is_reproducible():
    records = make_records()
    config = SplitConfig(seed=13)
    assert [r.split for r in assign_splits(records, config)] == [r.split for r in assign_splits(records, config)]


# --- song-disjoint ------------------------------------------------------


def test_song_disjoint_split_has_no_track_leakage():
    assigned = assign_splits(make_records(tracks=12, segments=4), SplitConfig(seed=3))

    assert_no_track_leakage(assigned)
    assert_group_disjoint(assigned, "song")


def test_all_segments_of_a_track_share_one_split():
    assigned = assign_splits(make_records(tracks=8, segments=5), SplitConfig(seed=5))

    by_track: dict[str, set[str]] = {}
    for record in assigned:
        by_track.setdefault(record.track_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_track.values())


def test_quota_strategy_fills_every_split():
    assigned = assign_splits(make_records(tracks=10, segments=2), SplitConfig(0.6, 0.2, 0.2, seed=11))
    assert {record.split for record in assigned} == {"train", "val", "test"}


def test_quota_strategy_respects_requested_proportions():
    assigned = assign_splits(make_records(tracks=100, segments=1), SplitConfig(0.8, 0.1, 0.1, seed=42))
    counts = {split: sum(1 for r in assigned if r.split == split) for split in ("train", "val", "test")}
    assert counts["train"] == pytest.approx(80, abs=2)
    assert counts["val"] == pytest.approx(10, abs=2)
    assert counts["test"] == pytest.approx(10, abs=2)


def test_two_way_split_is_supported():
    assigned = assign_splits(make_records(tracks=10), SplitConfig(train=0.8, val=0.2, test=0.0, seed=9))
    assert {record.split for record in assigned} == {"train", "val"}


def test_single_group_corpus_does_not_crash():
    assigned = assign_splits(make_records(tracks=1, segments=3), SplitConfig(seed=2))
    assert len({record.split for record in assigned}) == 1
    assert_group_disjoint(assigned, "song")


# --- singer-disjoint ----------------------------------------------------


def test_singer_disjoint_split_has_no_singer_leakage():
    records = make_records(tracks=12, segments=3, singers=6)
    assigned = assign_splits(records, SplitConfig(seed=4, mode="singer"))

    assert_group_disjoint(assigned, "singer")
    assert_singer_disjoint(assigned)


def test_singer_mode_keeps_songs_disjoint_too():
    records = make_records(tracks=12, segments=3, singers=4)
    assigned = assign_splits(records, SplitConfig(seed=6, mode="singer"))

    assert_no_track_leakage(assigned)


def test_all_songs_by_one_singer_land_in_one_split():
    records = make_records(tracks=12, segments=2, singers=4)
    assigned = assign_splits(records, SplitConfig(seed=8, mode="singer"))

    by_singer: dict[str, set[str]] = {}
    for record in assigned:
        by_singer.setdefault(record.singer_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_singer.values())


def test_song_mode_can_leak_singers_across_splits():
    """Why singer mode exists: song-disjoint alone does not protect a performer."""
    records = make_records(tracks=12, segments=2, singers=3)
    assigned = assign_splits(records, SplitConfig(seed=4, mode="song"))
    with pytest.raises(ValueError, match="Singer leakage"):
        assert_singer_disjoint(assigned)


def test_group_key_selection():
    record = make_records(tracks=1, singers=1)[0]
    assert group_key(record, "song") == record.track_id
    assert group_key(record, "singer") == record.singer_id


def test_singer_mode_falls_back_to_track_when_singer_missing():
    records = make_records(tracks=6, segments=2)  # singer_id is None
    assigned = assign_splits(records, SplitConfig(seed=1, mode="singer"))
    assert_no_track_leakage(assigned)


# --- leakage detection --------------------------------------------------


def test_track_leakage_is_detected():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "Track1"),
        AudioRecord("2", "b.wav", "val", "x", "cc", "Track1"),
    ]
    with pytest.raises(ValueError, match="Track leakage"):
        assert_no_track_leakage(records)
    with pytest.raises(ValueError, match="Song leakage"):
        assert_group_disjoint(records, "song")


def test_singer_leakage_is_detected():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "Track1", singer_id="s1"),
        AudioRecord("2", "b.wav", "test", "x", "cc", "Track2", singer_id="s1"),
    ]
    with pytest.raises(ValueError, match="Singer leakage"):
        assert_singer_disjoint(records)


def test_cross_split_duplicate_audio_is_detected():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "Track1", audio_sha256="deadbeef"),
        AudioRecord("2", "b.wav", "test", "x", "cc", "Track2", audio_sha256="deadbeef"),
    ]
    with pytest.raises(ValueError, match="Duplicate audio"):
        assert_no_cross_split_duplicates(records)


def test_duplicates_inside_one_split_are_allowed():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "Track1", audio_sha256="deadbeef"),
        AudioRecord("2", "b.wav", "train", "x", "cc", "Track2", audio_sha256="deadbeef"),
    ]
    assert_no_cross_split_duplicates(records)


# --- reporting ----------------------------------------------------------


def test_split_report_counts_and_confirms_no_leakage():
    assigned = assign_splits(make_records(tracks=10, segments=3), SplitConfig(seed=21))
    report = split_report(assigned, "song")

    assert report["ok"] is True
    assert report["leakage"] is None
    assert sum(entry["segments"] for entry in report["splits"].values()) == 30
    assert sum(entry["tracks"] for entry in report["splits"].values()) == 10


def test_split_report_flags_leakage():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "Track1"),
        AudioRecord("2", "b.wav", "val", "x", "cc", "Track1"),
    ]
    report = split_report(records, "song")
    assert report["ok"] is False
    assert "leakage" in report["leakage"].lower()


# --- config validation --------------------------------------------------


def test_invalid_split_config_is_rejected():
    with pytest.raises(ValueError):
        SplitConfig(mode="album").validate()
    with pytest.raises(ValueError):
        SplitConfig(train=0, val=0, test=0).validate()
    with pytest.raises(ValueError):
        SplitConfig(train=-1).validate()


def test_split_config_from_dict_ignores_unknown_keys():
    config = SplitConfig.from_dict({"train": 0.7, "val": 0.2, "test": 0.1, "seed": 5, "nonsense": True})
    assert config.train == 0.7 and config.seed == 5
