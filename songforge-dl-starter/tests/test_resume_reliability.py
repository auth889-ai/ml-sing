"""Reliability guarantees for M03 — Neural Audio Codec & Discrete Audio Representation.

Two Colab runtimes died mid-acceptance before any final evidence was persisted.
The run must therefore survive a disconnect as ONE logical experiment: periodic
atomic checkpoints, a resume that restores the full training state, and step
sequences that stay unique no matter how many sessions the run spans.
"""

import csv
import json
import random

import numpy as np
import pytest
import torch

from songforge.models.codec.model import NeuralCodec
from songforge.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from songforge.training.run import (
    RunIsolationError,
    assert_single_run,
    probe_fingerprint,
    truncate_csv_from_step,
    truncate_jsonl_from_step,
)

CONFIG = {"model": {"base_channels": 8}, "training": {"seed": 42}}


def tiny_codec() -> NeuralCodec:
    return NeuralCodec(
        sample_rate=24000, channels=1, base_channels=8, latent_dim=16,
        codebook_size=32, num_quantizers=2, strides=(2, 4, 5),
    )


def write_curve(path, run_id, steps):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "run_id"])
        if not exists:
            writer.writeheader()
        for step in steps:
            writer.writerow({"step": step, "loss": 1.0, "run_id": run_id})


def read_curve(path):
    with path.open("r", encoding="utf-8") as handle:
        return [{"step": int(r["step"]), "run_id": r["run_id"]} for r in csv.DictReader(handle)]


# --- deterministic probe membership -------------------------------------


def test_probe_fingerprint_is_order_sensitive_and_stable():
    assert probe_fingerprint(["a", "b"]) == probe_fingerprint(["a", "b"])
    assert probe_fingerprint(["a", "b"]) != probe_fingerprint(["b", "a"])
    assert probe_fingerprint(["a", "b"]) != probe_fingerprint(["a", "c"])


def test_probe_fingerprint_detects_membership_change():
    """A before/after pair over different segments must not look comparable."""
    before = probe_fingerprint([f"seg{i}" for i in range(64)])
    after_same = probe_fingerprint([f"seg{i}" for i in range(64)])
    after_drifted = probe_fingerprint([f"seg{i}" for i in range(1, 65)])
    assert before == after_same
    assert before != after_drifted


# --- checkpoint carries the whole training state ------------------------


def test_checkpoint_round_trips_scaler_and_rng():
    model, restored = tiny_codec(), tiny_codec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)

    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as tmp:
        path = P(tmp) / "checkpoint_latest.pt"
        save_checkpoint(
            path, model, optimizer, step=500, config=CONFIG,
            run_id="run-A", run_label="lbl", scaler=scaler,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["scaler"] is not None
        assert payload["rng_state"] is not None
        assert payload["step"] == 500
        assert payload["run_id"] == "run-A"

        checkpoint = load_checkpoint(path, restored, restored_optimizer, scaler=restored_scaler)
        assert checkpoint["rng_restored"] is True


def test_rng_state_round_trip_reproduces_the_same_stream():
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(state)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_checkpoint_write_is_atomic(tmp_path):
    """A dead runtime must not leave a truncated checkpoint behind."""
    model = tiny_codec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint_latest.pt"
    save_checkpoint(path, model, optimizer, step=1, config=CONFIG, run_id="r")
    first = path.read_bytes()
    save_checkpoint(path, model, optimizer, step=2, config=CONFIG, run_id="r")

    assert not (tmp_path / "checkpoint_latest.pt.tmp").exists()
    assert path.read_bytes() != first
    assert torch.load(path, map_location="cpu", weights_only=False)["step"] == 2


# --- resume cannot duplicate steps --------------------------------------


def test_resume_retires_replayed_curve_rows(tmp_path):
    """Checkpoint at 500 but training reached 700: rows 500-699 must not repeat."""
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(700))

    dropped = truncate_csv_from_step(curve, 500)
    assert dropped == 200

    rows = read_curve(curve)
    assert [r["step"] for r in rows] == list(range(500))

    write_curve(curve, "run-A", range(500, 900))  # resumed session
    rows = read_curve(curve)
    steps = [r["step"] for r in rows]
    assert steps == list(range(900))
    assert len(steps) == len(set(steps))
    assert_single_run(rows, "curve")


def test_retired_rows_are_preserved_not_deleted(tmp_path):
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(100))
    truncate_csv_from_step(curve, 50)

    superseded = tmp_path / "training_curves.csv.superseded"
    assert superseded.exists()
    assert len(read_curve(superseded)) == 50


def test_resume_retires_replayed_rvq_rows(tmp_path):
    history = tmp_path / "rvq_history.jsonl"
    with history.open("w", encoding="utf-8") as handle:
        for step in range(0, 700, 25):
            handle.write(json.dumps({"step": step, "run_id": "run-A"}) + "\n")

    dropped = truncate_jsonl_from_step(history, 500)
    assert dropped == 8  # steps 500..675

    rows = [json.loads(x) for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert max(r["step"] for r in rows) < 500
    assert (tmp_path / "rvq_history.jsonl.superseded").exists()

    with history.open("a", encoding="utf-8") as handle:
        for step in range(500, 900, 25):
            handle.write(json.dumps({"step": step, "run_id": "run-A"}) + "\n")
    rows = [json.loads(x) for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert_single_run(rows, "rvq history")


def test_truncation_is_a_noop_when_nothing_replayed(tmp_path):
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(100))
    assert truncate_csv_from_step(curve, 100) == 0
    assert not (tmp_path / "training_curves.csv.superseded").exists()


def test_truncation_on_missing_file_is_safe(tmp_path):
    assert truncate_csv_from_step(tmp_path / "absent.csv", 10) == 0
    assert truncate_jsonl_from_step(tmp_path / "absent.jsonl", 10) == 0


# --- a multi-session run stays ONE experiment ---------------------------


def test_three_session_run_yields_each_step_exactly_once(tmp_path):
    """Simulates two disconnects: 0-1361, 1250-2600, 2500-3999."""
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(1362))          # session 1, died at 1361

    truncate_csv_from_step(curve, 1250)                   # resume from checkpoint 1249
    write_curve(curve, "run-A", range(1250, 2601))        # session 2, died at 2600

    truncate_csv_from_step(curve, 2500)                   # resume from checkpoint 2499
    write_curve(curve, "run-A", range(2500, 4000))        # session 3, completes

    rows = read_curve(curve)
    steps = [r["step"] for r in rows]
    assert steps == list(range(4000))
    assert len(steps) == len(set(steps))
    assert {r["run_id"] for r in rows} == {"run-A"}
    assert_single_run(rows, "curve")


def test_a_second_independent_run_is_still_rejected(tmp_path):
    """Truncation must not become a loophole for splicing two experiments."""
    curve = tmp_path / "training_curves.csv"
    write_curve(curve, "run-A", range(500))
    write_curve(curve, "run-B", range(500, 1000))
    with pytest.raises(RunIsolationError, match="different runs"):
        assert_single_run(read_curve(curve), "curve")
