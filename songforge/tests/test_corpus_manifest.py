"""Canonical multi-corpus manifest: validation, IO, licence and dedup gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from songforge.data.corpus_manifest import (  # noqa: E402
    CorpusRecord,
    assert_deployable,
    assert_no_cross_corpus_duplicates,
    read_corpus_manifest,
    write_corpus_manifest,
)


def record(**overrides) -> CorpusRecord:
    base = dict(
        dataset="slakh100", track_id="Track00001", audio_path="train/Track00001/S00.wav",
        licence="CC-BY-4.0", licence_class="permissive",
        source_url="https://zenodo.org/records/4599666", split="train",
        duplicate_hash="abc123", caption="solo Acoustic Grand Piano",
        instrument_tags=("Piano",),
    )
    base.update(overrides)
    return CorpusRecord(**base)


class TestValidation:
    def test_valid_record_passes(self):
        record().validate()

    def test_missing_required_field_fails(self):
        with pytest.raises(ValueError, match="source_url"):
            record(source_url="").validate()

    def test_bad_split_fails(self):
        with pytest.raises(ValueError, match="bad split"):
            record(split="dev").validate()

    def test_permissive_claim_requires_allowlisted_licence(self):
        with pytest.raises(ValueError, match="permissive allowlist"):
            record(licence="CC-BY-NC-SA-4.0").validate()

    def test_nc_licence_is_fine_as_research_only(self):
        record(licence="CC-BY-NC-SA-4.0", licence_class="research-only").validate()

    def test_quality_score_bounds(self):
        record(quality_score=0.8).validate()
        with pytest.raises(ValueError, match="quality_score"):
            record(quality_score=1.2).validate()


class TestRoundTrip:
    def test_jsonl_roundtrip_preserves_everything(self, tmp_path):
        original = record(bpm=120.0, key="C major", genre=("jazz",),
                          language="en", quality_score=0.9,
                          extra={"stem": "S00"})
        path = tmp_path / "corpus.jsonl"
        assert write_corpus_manifest(path, [original]) == 1
        loaded = list(read_corpus_manifest(path))
        assert loaded == [original]

    def test_write_refuses_invalid_records(self, tmp_path):
        with pytest.raises(ValueError):
            write_corpus_manifest(tmp_path / "bad.jsonl", [record(split="nope")])


class TestGates:
    def test_deployable_gate_rejects_research_only(self):
        records = [record(),
                   record(dataset="gtsinger", track_id="t1",
                          licence="CC-BY-NC-SA-4.0", licence_class="research-only")]
        with pytest.raises(ValueError, match="non-permissive"):
            assert_deployable(records)
        assert_deployable([record()])

    def test_cross_corpus_duplicate_detected(self):
        records = [record(),
                   record(dataset="fma_ccby", track_id="99999",
                          audio_path="fma/099999.mp3",
                          source_url="https://freemusicarchive.org/track/99999")]
        with pytest.raises(ValueError, match="duplicate audio across corpora"):
            assert_no_cross_corpus_duplicates(records)

    def test_same_corpus_repeat_is_allowed(self):
        # Stems of one track legitimately share a source hash prefix scheme.
        records = [record(), record(track_id="Track00001_S01",
                                    audio_path="train/Track00001/S01.wav")]
        assert_no_cross_corpus_duplicates(records)
