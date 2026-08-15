from pathlib import Path

import pytest

from songforge.data.dedup import (
    duplicate_report,
    file_sha256,
    find_duplicate_groups,
    waveform_sha256,
)
from songforge.data.fixtures import (
    build_singer_corpus,
    build_slakh_like_corpus,
    write_clipped_wav,
    write_silent_wav,
    write_tone_wav,
)
from songforge.data.manifest import assert_provenance_complete, validate_records
from songforge.data.preprocess import (
    UNASSIGNED_SPLIT,
    PreprocessConfig,
    derive_singer_id,
    derive_track_id,
    find_audio_files,
    normalize_tags,
    preprocess_file,
    preprocess_paths,
    provenance_from_registry,
)
from songforge.data.registry import load_dataset_registry

BABYSLAKH_PROVENANCE = {
    "dataset_id": "babyslakh",
    "dataset_name": "BabySlakh",
    "source_url": "https://zenodo.org/records/4603844",
    "license_name": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
}


def tiny_config(**overrides) -> PreprocessConfig:
    base = {
        "sample_rate": 8000,
        "channels": 1,
        "segment_seconds": 1.0,
        "min_source_seconds": 0.5,
    }
    base.update(overrides)
    return PreprocessConfig(**base)


# --- config -------------------------------------------------------------


def test_config_derives_segment_and_hop_samples():
    config = tiny_config(segment_seconds=2.0, hop_seconds=None)
    assert config.segment_samples == 16000
    assert config.hop_samples == 16000
    assert tiny_config(segment_seconds=2.0, hop_seconds=0.5).hop_samples == 4000


def test_config_loads_from_project_yaml():
    config = PreprocessConfig.from_yaml(Path("configs/data/preprocess_m02.yaml"))
    assert config.sample_rate == 24000
    assert config.channels == 1
    assert config.drop_silent is True


# --- ids and metadata ---------------------------------------------------


def test_track_id_uses_slakh_track_directory():
    path = Path("/data/raw/babyslakh/Track00007/stems/S03.wav")
    assert derive_track_id(path, "babyslakh") == "Track00007"


def test_all_stems_of_one_song_share_a_track_id():
    mix = derive_track_id(Path("/d/Track00001/mix.wav"), "babyslakh")
    stem = derive_track_id(Path("/d/Track00001/stems/S00.wav"), "babyslakh")
    assert mix == stem == "Track00001"


def test_track_id_is_stable_across_calls():
    path = Path("/data/raw/babyslakh/Track00007/mix.wav")
    assert derive_track_id(path, "babyslakh") == derive_track_id(path, "babyslakh")


def test_singer_id_from_gtsinger_layout():
    path = Path("/data/raw/gtsinger/english/Singer03/Song01/vocal.wav")
    assert derive_singer_id(path, "gtsinger") == "english__Singer03"


def test_singer_id_is_none_for_instrumental_corpora():
    assert derive_singer_id(Path("/d/Track00001/mix.wav"), "babyslakh") is None


def test_tags_are_normalized():
    assert normalize_tags([" Rock ", "rock", "Post Punk", ""]) == ("post_punk", "rock")
    assert normalize_tags(None) == ()


def test_provenance_from_real_registry_carries_license():
    registry = load_dataset_registry(Path("configs/data/datasets.yaml"))
    provenance = provenance_from_registry(registry, "babyslakh")

    assert provenance["dataset_id"] == "babyslakh"
    assert provenance["license_name"] == "CC-BY-4.0"
    assert provenance["source_url"]
    assert provenance["requires_user_acceptance"] is False


def test_provenance_marks_gated_datasets():
    registry = load_dataset_registry(Path("configs/data/datasets.yaml"))
    assert provenance_from_registry(registry, "gtsinger")["requires_user_acceptance"] is True


# --- single file preprocessing -----------------------------------------


def test_preprocess_file_segments_deterministically(tmp_path):
    source = write_tone_wav(tmp_path / "Track00001" / "mix.wav", seconds=4.0, sample_rate=16000)
    records, skipped = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert skipped == []
    assert len(records) == 4
    assert [record.segment_index for record in records] == [0, 1, 2, 3]
    assert [record.start_sample for record in records] == [0, 8000, 16000, 24000]
    assert all(record.num_samples == 8000 for record in records)
    assert all(record.sample_rate == 8000 for record in records)
    assert all(record.split == UNASSIGNED_SPLIT for record in records)


def test_preprocess_resamples_to_target_rate(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "mix.wav", seconds=2.0, sample_rate=44100)
    records, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert records
    assert all(record.sample_rate == 8000 for record in records)
    assert records[0].preprocessing["source_sample_rate"] == 44100


def test_preprocess_writes_playable_segment_files(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "mix.wav", seconds=2.0, sample_rate=16000)
    records, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    for record in records:
        assert Path(record.path).exists()
        assert Path(record.path).stat().st_size > 0


def test_preprocess_output_is_byte_identical_across_runs(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "mix.wav", seconds=3.0, sample_rate=16000)

    first, _ = preprocess_file(source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "a", dataset_id="babyslakh")
    second, _ = preprocess_file(source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "b", dataset_id="babyslakh")

    assert [record.id for record in first] == [record.id for record in second]
    assert [record.audio_sha256 for record in first] == [record.audio_sha256 for record in second]
    assert [Path(record.path).read_bytes() for record in first] == [
        Path(record.path).read_bytes() for record in second
    ]


def test_record_ids_are_stable_and_unique(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "mix.wav", seconds=4.0, sample_rate=16000)
    records, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    ids = [record.id for record in records]
    assert len(set(ids)) == len(ids)
    rerun, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out2", dataset_id="babyslakh"
    )
    assert ids == [record.id for record in rerun]


def test_provenance_and_license_are_propagated_to_every_record(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "mix.wav", seconds=2.0, sample_rate=16000)
    records, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert records
    for record in records:
        assert record.license == "CC-BY-4.0"
        assert record.source == "babyslakh"
        assert record.provenance["source_url"] == BABYSLAKH_PROVENANCE["source_url"]
        assert record.provenance["license_url"]
    assert_provenance_complete(records)


def test_preprocessing_settings_are_recorded_for_reproducibility(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "mix.wav", seconds=2.0, sample_rate=16000)
    records, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    preprocessing = records[0].preprocessing
    assert preprocessing["sample_rate"] == 8000
    assert preprocessing["segment_seconds"] == 1.0
    assert preprocessing["normalize"] == "peak"
    assert preprocessing["version"]


# --- rejection paths ----------------------------------------------------


def test_silent_segments_are_dropped(tmp_path):
    source = write_silent_wav(tmp_path / "Track1" / "silent.wav", seconds=3.0, sample_rate=16000)
    records, skipped = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert records == []
    assert skipped
    assert all(entry["stage"] == "silence" for entry in skipped)


def test_silent_segments_can_be_kept_and_flagged(tmp_path):
    source = write_silent_wav(tmp_path / "Track1" / "silent.wav", seconds=2.0, sample_rate=16000)
    records, skipped = preprocess_file(
        source,
        tiny_config(drop_silent=False),
        BABYSLAKH_PROVENANCE,
        tmp_path / "out",
        dataset_id="babyslakh",
    )

    assert len(records) == 2
    assert all(record.silent for record in records)
    assert skipped == []


def test_source_clipping_is_recorded_even_after_normalization(tmp_path):
    """`clipping_ratio` describes the stored segment; source clipping is kept separately.

    Peak normalization pulls a clipped master down to -1 dBFS, so the written
    audio is no longer at full scale. The original defect must still be visible
    in the record, which is what `preprocessing.source_clipping_ratio` is for.
    """
    source = write_clipped_wav(tmp_path / "Track1" / "hot.wav", seconds=2.0, sample_rate=16000)
    records, _ = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert records
    assert records[0].preprocessing["source_clipping_ratio"] > 0.5
    assert all(record.clipping_ratio == 0.0 for record in records)


def test_clipping_is_measured_on_stored_segments_when_not_normalized(tmp_path):
    source = write_clipped_wav(tmp_path / "Track1" / "hot.wav", seconds=2.0, sample_rate=16000)
    records, _ = preprocess_file(
        source, tiny_config(normalize="none"), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert records
    assert all(record.clipping_ratio > 0.5 for record in records)


def test_clipped_sources_can_be_rejected(tmp_path):
    source = write_clipped_wav(tmp_path / "Track1" / "hot.wav", seconds=2.0, sample_rate=16000)
    records, skipped = preprocess_file(
        source,
        tiny_config(drop_clipped=True, max_clipping_ratio=0.01),
        BABYSLAKH_PROVENANCE,
        tmp_path / "out",
        dataset_id="babyslakh",
    )

    assert records == []
    assert skipped[0]["stage"] == "clipping"


def test_corrupt_file_is_skipped_not_raised(tmp_path):
    from songforge.data.fixtures import write_corrupt_wav

    source = write_corrupt_wav(tmp_path / "Track1" / "corrupt.wav")
    records, skipped = preprocess_file(
        source, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "out", dataset_id="babyslakh"
    )

    assert records == []
    assert skipped[0]["stage"] in ("validation", "decode")


def test_file_shorter_than_one_segment_is_skipped(tmp_path):
    source = write_tone_wav(tmp_path / "Track1" / "short.wav", seconds=0.6, sample_rate=16000)
    records, skipped = preprocess_file(
        source,
        tiny_config(segment_seconds=2.0),
        BABYSLAKH_PROVENANCE,
        tmp_path / "out",
        dataset_id="babyslakh",
    )

    assert records == []
    assert skipped[0]["stage"] == "segmentation"


# --- corpus level -------------------------------------------------------


def test_preprocess_corpus_skips_broken_and_keeps_good(tmp_path):
    corpus = build_slakh_like_corpus(
        tmp_path / "raw", tracks=3, stems_per_track=1, seconds=2.0, include_broken=True
    )
    result = preprocess_paths(
        find_audio_files(corpus.root),
        tiny_config(),
        BABYSLAKH_PROVENANCE,
        tmp_path / "out",
        dataset_id="babyslakh",
        source_root=corpus.root,
    )

    assert result.stats["tracks"] == 3
    assert result.stats["segments"] == 12  # 3 tracks x 2 files x 2 segments
    assert result.stats["skipped_files"] == 2  # corrupt + empty
    assert validate_records(result.records) == []


def test_preprocess_is_independent_of_input_order(tmp_path):
    corpus = build_slakh_like_corpus(tmp_path / "raw", tracks=3, stems_per_track=1, seconds=2.0)
    paths = find_audio_files(corpus.root)

    forward = preprocess_paths(
        paths, tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "a", dataset_id="babyslakh"
    )
    reverse = preprocess_paths(
        list(reversed(paths)), tiny_config(), BABYSLAKH_PROVENANCE, tmp_path / "b", dataset_id="babyslakh"
    )

    assert [record.id for record in forward.records] == [record.id for record in reverse.records]


def test_singer_corpus_populates_singer_ids(tmp_path):
    corpus = build_singer_corpus(tmp_path / "raw", singers=2, songs_per_singer=1, seconds=2.0)
    result = preprocess_paths(
        find_audio_files(corpus.root),
        tiny_config(),
        {"dataset_id": "gtsinger", "source_url": "https://example.test", "license_name": "CC-BY-NC-SA-4.0"},
        tmp_path / "out",
        dataset_id="gtsinger",
        source_root=corpus.root,
    )

    assert result.records
    assert {record.singer_id for record in result.records} == set(corpus.singer_ids)


# --- duplicate detection hooks -----------------------------------------


def test_identical_audio_produces_identical_content_hash(tmp_path):
    first = write_tone_wav(tmp_path / "a.wav", seconds=1.0, sample_rate=16000, frequency=300.0)
    second = write_tone_wav(tmp_path / "b.wav", seconds=1.0, sample_rate=16000, frequency=300.0)

    from songforge.data.media import decode_audio

    left, rate = decode_audio(first)
    right, _ = decode_audio(second)
    assert waveform_sha256(left, rate) == waveform_sha256(right, rate)
    assert file_sha256(first) == file_sha256(second)


def test_different_audio_produces_different_content_hash(tmp_path):
    from songforge.data.media import decode_audio

    left, rate = decode_audio(write_tone_wav(tmp_path / "a.wav", seconds=1.0, sample_rate=16000, frequency=300.0))
    right, _ = decode_audio(write_tone_wav(tmp_path / "b.wav", seconds=1.0, sample_rate=16000, frequency=700.0))
    assert waveform_sha256(left, rate) != waveform_sha256(right, rate)


def test_duplicate_sources_are_detectable_in_a_corpus(tmp_path):
    write_tone_wav(tmp_path / "raw" / "Track00001" / "mix.wav", seconds=2.0, sample_rate=16000, frequency=300.0)
    write_tone_wav(tmp_path / "raw" / "Track00002" / "mix.wav", seconds=2.0, sample_rate=16000, frequency=300.0)

    result = preprocess_paths(
        find_audio_files(tmp_path / "raw"),
        tiny_config(),
        BABYSLAKH_PROVENANCE,
        tmp_path / "out",
        dataset_id="babyslakh",
    )

    groups = find_duplicate_groups(result.records)
    assert groups, "identical audio in two tracks should be flagged"
    assert duplicate_report(result.records)["duplicate_groups"] == len(groups)


def test_distinct_corpus_reports_no_duplicates(tmp_path):
    corpus = build_slakh_like_corpus(tmp_path / "raw", tracks=3, stems_per_track=1, seconds=2.0)
    result = preprocess_paths(
        find_audio_files(corpus.root),
        tiny_config(),
        BABYSLAKH_PROVENANCE,
        tmp_path / "out",
        dataset_id="babyslakh",
    )
    report = duplicate_report(result.records)
    assert report["duplicate_groups"] == 0
    assert report["ok"] is True


def test_empty_input_raises(tmp_path):
    with pytest.raises(ValueError):
        from songforge.data.audio import read_audio_paths

        read_audio_paths(None, str(tmp_path / "nothing" / "*.wav"))
