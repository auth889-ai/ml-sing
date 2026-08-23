import json

import pytest

from songforge.data.manifest import (
    MANIFEST_SCHEMA,
    AudioRecord,
    assert_no_track_leakage,
    assert_provenance_complete,
    assert_singer_disjoint,
    manifest_summary,
    read_jsonl,
    segment_id,
    stable_id,
    validate_records,
    write_jsonl,
    write_split_manifests,
)

PROVENANCE = {
    "dataset_id": "babyslakh",
    "source_url": "https://zenodo.org/records/4603844",
    "license_name": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
}


def full_record(index: int = 0, split: str = "train", **overrides) -> AudioRecord:
    payload = {
        "id": f"rec{index:03d}",
        "path": f"/processed/audio/Track00001/rec{index:03d}.wav",
        "split": split,
        "source": "babyslakh",
        "license": "CC-BY-4.0",
        "track_id": "Track00001",
        "singer_id": None,
        "tags": ("drums", "rock"),
        "source_path": "/raw/Track00001/mix.wav",
        "segment_index": index,
        "start_sample": index * 24000,
        "num_samples": 24000,
        "sample_rate": 24000,
        "channels": 1,
        "duration_seconds": 1.0,
        "peak": 0.9,
        "rms": 0.3,
        "peak_dbfs": -0.9,
        "rms_dbfs": -10.4,
        "clipping_ratio": 0.0,
        "silent": False,
        "audio_sha256": f"hash{index:03d}",
        "source_sha256": "sourcehash",
        "provenance": dict(PROVENANCE),
        "preprocessing": {"version": "m02.v1", "sample_rate": 24000},
    }
    payload.update(overrides)
    return AudioRecord(**payload)


# --- stable ids ---------------------------------------------------------


def test_stable_id_is_stable():
    assert stable_id("abc") == stable_id("abc")


def test_stable_id_differs_for_different_input():
    assert stable_id("abc") != stable_id("abd")


def test_stable_id_join_is_unambiguous():
    assert stable_id("a", "bc") != stable_id("ab", "c")


def test_segment_id_is_deterministic_and_position_sensitive():
    first = segment_id("babyslakh", "Track1", "/raw/mix.wav", 0, 0)
    assert first == segment_id("babyslakh", "Track1", "/raw/mix.wav", 0, 0)
    assert first != segment_id("babyslakh", "Track1", "/raw/mix.wav", 1, 24000)
    assert first != segment_id("gtsinger", "Track1", "/raw/mix.wav", 0, 0)


# --- round trip ---------------------------------------------------------


def test_manifest_round_trip_preserves_every_field(tmp_path):
    records = [full_record(index) for index in range(3)]
    path = write_jsonl(records, tmp_path / "all.jsonl")

    assert read_jsonl(path) == records


def test_round_trip_preserves_nested_provenance_and_preprocessing(tmp_path):
    path = write_jsonl([full_record(0)], tmp_path / "all.jsonl")
    restored = read_jsonl(path)[0]

    assert restored.provenance == PROVENANCE
    assert restored.preprocessing["version"] == "m02.v1"
    assert restored.tags == ("drums", "rock")


def test_round_trip_preserves_optional_none_fields(tmp_path):
    path = write_jsonl([full_record(0, singer_id=None, source_sha256=None)], tmp_path / "all.jsonl")
    restored = read_jsonl(path)[0]

    assert restored.singer_id is None
    assert restored.source_sha256 is None


def test_manifest_lines_are_valid_json_with_sorted_keys(tmp_path):
    path = write_jsonl([full_record(0)], tmp_path / "all.jsonl")
    line = path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert payload["schema"] == MANIFEST_SCHEMA
    assert list(payload) == sorted(payload)


def test_minimal_m01_style_record_still_loads(tmp_path):
    """Records written before M02 must keep loading against the canonical schema."""
    legacy = {
        "id": "1",
        "path": "a.wav",
        "split": "train",
        "source": "babyslakh",
        "license": "CC-BY-4.0",
        "track_id": "Track1",
    }
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    record = read_jsonl(path)[0]
    assert record.id == "1"
    assert record.segment_index == 0
    assert record.provenance == {}


def test_unknown_fields_are_preserved_in_extra(tmp_path):
    path = tmp_path / "future.jsonl"
    payload = {
        "id": "1", "path": "a.wav", "split": "train", "source": "s",
        "license": "cc", "track_id": "T1", "future_field": 42,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert read_jsonl(path)[0].extra["future_field"] == 42


def test_malformed_line_raises_with_location(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        read_jsonl(path)


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "gaps.jsonl"
    path.write_text(json.dumps(full_record(0).to_dict()) + "\n\n\n", encoding="utf-8")
    assert len(read_jsonl(path)) == 1


# --- validation ---------------------------------------------------------


def test_valid_records_report_no_errors():
    assert validate_records([full_record(0), full_record(1)]) == []


def test_missing_required_field_is_reported():
    errors = validate_records([full_record(0, license="")])
    assert any("license" in error for error in errors)


def test_duplicate_id_with_different_path_is_reported():
    errors = validate_records([full_record(0), full_record(0, path="/other.wav")])
    assert any("duplicate record id" in error for error in errors)


def test_inconsistent_duration_is_reported():
    errors = validate_records([full_record(0, duration_seconds=99.0)])
    assert any("duration_seconds" in error for error in errors)


# --- provenance and licensing ------------------------------------------


def test_provenance_complete_passes_for_full_records():
    assert_provenance_complete([full_record(0)])


def test_missing_license_is_rejected():
    with pytest.raises(ValueError, match="license"):
        assert_provenance_complete([full_record(0, license="")])


def test_missing_provenance_block_is_rejected():
    with pytest.raises(ValueError, match="provenance missing"):
        assert_provenance_complete([full_record(0, provenance={})])


def test_partial_provenance_is_rejected():
    with pytest.raises(ValueError, match="source_url"):
        assert_provenance_complete([full_record(0, provenance={"dataset_id": "x", "license_name": "cc"})])


# --- leakage (kept from M01) -------------------------------------------


def test_track_leakage_detected():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "song1"),
        AudioRecord("2", "b.wav", "val", "x", "cc", "song1"),
    ]
    with pytest.raises(ValueError):
        assert_no_track_leakage(records)


def test_singer_leakage_detected():
    records = [
        AudioRecord("1", "a.wav", "train", "x", "cc", "song1", singer_id="s1"),
        AudioRecord("2", "b.wav", "test", "x", "cc", "song2", singer_id="s1"),
    ]
    with pytest.raises(ValueError):
        assert_singer_disjoint(records)


# --- summaries and split files -----------------------------------------


def test_manifest_summary_counts_splits_and_duration():
    records = [full_record(0, split="train"), full_record(1, split="train"), full_record(2, split="val")]
    summary = manifest_summary(records)

    assert summary["segments"] == 3
    assert summary["tracks"] == 1
    assert summary["total_seconds"] == 3.0
    assert summary["splits"]["train"]["segments"] == 2
    assert summary["splits"]["val"]["segments"] == 1
    assert summary["licenses"] == ["CC-BY-4.0"]


def test_write_split_manifests_creates_one_file_per_split(tmp_path):
    records = [full_record(0, split="train"), full_record(1, split="val"), full_record(2, split="test")]
    written = write_split_manifests(records, tmp_path / "manifests")

    assert set(written) == {"all", "train", "val", "test", "summary"}
    assert len(read_jsonl(written["all"])) == 3
    assert len(read_jsonl(written["train"])) == 1
    assert json.loads(written["summary"].read_text(encoding="utf-8"))["segments"] == 3


def test_split_manifest_files_round_trip(tmp_path):
    records = [full_record(index, split="train" if index < 2 else "val") for index in range(4)]
    written = write_split_manifests(records, tmp_path / "manifests")

    recovered = read_jsonl(written["train"]) + read_jsonl(written["val"])
    assert sorted(recovered, key=lambda record: record.id) == sorted(records, key=lambda record: record.id)
