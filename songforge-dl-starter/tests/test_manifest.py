import pytest
from songforge.data.manifest import AudioRecord, assert_no_track_leakage, assert_singer_disjoint, stable_id


def test_stable_id_is_stable():
    assert stable_id("abc") == stable_id("abc")


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
