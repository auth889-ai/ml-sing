"""Experiment-isolation guarantees.

A real Colab M03 run reused an output directory and produced a
`training_curves.csv` holding 8000 rows: steps 0-3999 twice, from two
independent runs. The windowed loss verdict then compared one run's first
decile against the other run's last decile. These tests exist so that cannot
recur silently.
"""

import csv
import json

import pytest
import torch

from songforge.models.codec.model import NeuralCodec
from songforge.training.checkpoint import load_checkpoint, save_checkpoint
from songforge.training.run import (
    RUN_ARTIFACTS,
    RunIsolationError,
    assert_fresh_run_dir,
    assert_resume_compatible,
    assert_single_run,
    config_fingerprint,
    curve_run_ids,
    existing_run_artifacts,
    new_run_id,
    read_run_manifest,
    write_run_manifest,
)

CONFIG = {"model": {"base_channels": 8, "latent_dim": 16}, "training": {"seed": 42}}
OTHER_CONFIG = {"model": {"base_channels": 32, "latent_dim": 64}, "training": {"seed": 42}}


def tiny_codec() -> NeuralCodec:
    return NeuralCodec(
        sample_rate=24000, channels=1, base_channels=8, latent_dim=16,
        codebook_size=32, num_quantizers=2, strides=(2, 4, 5),
    )


def write_curve(path, run_id, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "run_id"])
        if not exists:
            writer.writeheader()
        for step in steps:
            writer.writerow({"step": step, "loss": 1.0 / (step + 1), "run_id": run_id})


def read_curve_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        return [
            {"step": int(row["step"]), "loss": float(row["loss"]), "run_id": row["run_id"]}
            for row in csv.DictReader(handle)
        ]


# --- run identity -------------------------------------------------------


def test_run_ids_are_unique():
    assert new_run_id("x") != new_run_id("x")


def test_run_id_carries_label():
    assert new_run_id("colab-m03-final-clean").startswith("colab-m03-final-clean-")


def test_config_fingerprint_is_stable_and_discriminating():
    assert config_fingerprint(CONFIG) == config_fingerprint(dict(CONFIG))
    assert config_fingerprint(CONFIG) != config_fingerprint(OTHER_CONFIG)


def test_run_manifest_round_trip(tmp_path):
    written = write_run_manifest(tmp_path, run_id="r1", run_label="lbl", config=CONFIG, start_step=0)
    restored = read_run_manifest(tmp_path)
    assert restored["run_id"] == "r1"
    assert restored["run_label"] == "lbl"
    assert restored["config_fingerprint"] == written["config_fingerprint"]


def test_read_run_manifest_missing_returns_none(tmp_path):
    assert read_run_manifest(tmp_path) is None


# --- a fresh run cannot start on top of another run ---------------------


def test_fresh_dir_is_accepted(tmp_path):
    assert_fresh_run_dir(tmp_path / "brand_new")


@pytest.mark.parametrize("artifact", RUN_ARTIFACTS)
def test_every_run_artifact_blocks_a_fresh_run(tmp_path, artifact):
    (tmp_path / artifact).write_text("x", encoding="utf-8")
    with pytest.raises(RunIsolationError, match="previous run"):
        assert_fresh_run_dir(tmp_path)


def test_blocked_fresh_run_names_the_offending_files(tmp_path):
    (tmp_path / "training_curves.csv").write_text("step,loss\n", encoding="utf-8")
    with pytest.raises(RunIsolationError, match="training_curves.csv"):
        assert_fresh_run_dir(tmp_path)


def test_guard_does_not_delete_existing_evidence(tmp_path):
    curve = tmp_path / "training_curves.csv"
    curve.write_text("step,loss\n0,1.0\n", encoding="utf-8")
    with pytest.raises(RunIsolationError):
        assert_fresh_run_dir(tmp_path)
    assert curve.read_text(encoding="utf-8") == "step,loss\n0,1.0\n"


def test_existing_run_artifacts_lists_what_is_there(tmp_path):
    (tmp_path / "checkpoint.pt").write_text("x", encoding="utf-8")
    (tmp_path / "rvq_history.jsonl").write_text("{}\n", encoding="utf-8")
    assert set(existing_run_artifacts(tmp_path)) == {"checkpoint.pt", "rvq_history.jsonl"}


# --- splicing detection -------------------------------------------------


def test_independent_runs_cannot_splice_curves(tmp_path):
    """Two runs appending to one file must be detectable, not silent."""
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(100))
    write_curve(curve, "run-B", range(100))  # the bug: second run appends

    rows = read_curve_rows(curve)
    assert len(rows) == 200
    assert curve_run_ids(rows) == ["run-A", "run-B"]
    with pytest.raises(RunIsolationError, match="different runs"):
        assert_single_run(rows, "curve")


def test_single_run_curve_passes(tmp_path):
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(100))
    rows = read_curve_rows(curve)
    assert curve_run_ids(rows) == ["run-A"]
    assert_single_run(rows, "curve")


def test_duplicate_steps_are_rejected_even_without_run_ids():
    """Older curves carry no run_id; repeated steps still prove concatenation."""
    rows = [{"step": s} for s in range(50)] + [{"step": s} for s in range(50)]
    with pytest.raises(RunIsolationError, match="repeats"):
        assert_single_run(rows, "curve")


def test_independent_runs_cannot_splice_rvq_history(tmp_path):
    history = tmp_path / "rvq_history.jsonl"
    with history.open("w", encoding="utf-8") as handle:
        for step in range(0, 100, 25):
            handle.write(json.dumps({"step": step, "run_id": "run-A"}) + "\n")
    with history.open("a", encoding="utf-8") as handle:
        for step in range(0, 100, 25):
            handle.write(json.dumps({"step": step, "run_id": "run-B"}) + "\n")

    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    with pytest.raises(RunIsolationError, match="different runs"):
        assert_single_run(rows, "rvq history")


def test_the_observed_colab_contamination_would_be_caught():
    """Reproduces the shape of the real failure: 8000 rows, every step twice."""
    rows = [{"step": s, "run_id": "run-1"} for s in range(4000)]
    rows += [{"step": s, "run_id": "run-2"} for s in range(4000)]
    assert len(rows) == 8000
    assert len({r["step"] for r in rows}) == 4000
    with pytest.raises(RunIsolationError):
        assert_single_run(rows, "curve")


# --- resume -------------------------------------------------------------


def test_explicit_resume_continues_the_same_run(tmp_path):
    model, restored = tiny_codec(), tiny_codec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=41, config=CONFIG, run_id="run-A", run_label="lbl")

    checkpoint = load_checkpoint(path, restored, restored_optimizer)
    assert assert_resume_compatible(checkpoint, CONFIG) == "run-A"
    assert int(checkpoint["step"]) + 1 == 42


def test_resume_appending_to_its_own_run_keeps_one_run_id(tmp_path):
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(100))
    write_curve(curve, "run-A", range(100, 150))  # legitimate continuation

    rows = read_curve_rows(curve)
    assert curve_run_ids(rows) == ["run-A"]
    assert_single_run(rows, "curve")


def test_resume_from_incompatible_config_is_rejected(tmp_path):
    model = tiny_codec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=10, config=CONFIG, run_id="run-A", run_label="lbl")

    checkpoint = load_checkpoint(path, tiny_codec(), None)
    with pytest.raises(RunIsolationError, match="does not match"):
        assert_resume_compatible(checkpoint, OTHER_CONFIG)


def test_resume_falls_back_to_config_hash_for_legacy_checkpoints():
    """Checkpoints written before run identity existed still carry their config."""
    legacy = {"config": CONFIG, "step": 5}
    assert assert_resume_compatible(legacy, CONFIG) == ""
    with pytest.raises(RunIsolationError):
        assert_resume_compatible(legacy, OTHER_CONFIG)


def test_checkpoint_records_run_identity(tmp_path):
    model = tiny_codec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=1, config=CONFIG, run_id="run-Z", run_label="lbl")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["run_id"] == "run-Z"
    assert checkpoint["run_label"] == "lbl"
    assert checkpoint["config_fingerprint"] == config_fingerprint(CONFIG)
