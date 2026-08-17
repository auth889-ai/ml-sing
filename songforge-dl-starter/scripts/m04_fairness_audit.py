"""M04 — High-Quality Codec Optimization & Latent-Rate Selection: fairness audit.

A latent-rate comparison is only evidence if the candidates differed in exactly
one thing. This script re-derives that from the run artifacts rather than from
the intent of the launch command: same corpus, same probe, same listening
sources, same budget, same losses and optimizer, same Q and K, one clean run_id
each, no duplicated or missing steps.

It answers "was this a fair comparison?" and nothing else. It selects no winner.

    python scripts/m04_fairness_audit.py \
        --output-root "$SONGFORGE_DRIVE/outputs/m04_stage1_authoritative_expanded" \
        --candidates m04_baseline_120hz_q2 m04_a_75hz_q2 m04_b_50hz_q2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from songforge.milestones import milestone

#: Optimizer/schedule fields that must be byte-identical across candidates. The
#: stride change is the whole point of the experiment; anything else is a confound.
OPTIMIZER_KEYS = (
    "batch_size", "learning_rate", "betas", "weight_decay", "grad_clip",
    "amp", "seed", "segment_samples",
)
#: Loss weights and STFT resolutions. A candidate optimising a different
#: objective is not comparable no matter how clean its curve looks.
LOSS_KEYS = ("waveform_weight", "spectral_weight", "vq_weight", "fft_sizes")
LISTENING_CATEGORIES = ("piano", "guitar", "bass", "percussion", "strings", "full_mix")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_curve(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def step_integrity(rows: list[dict[str, str]], expected_steps: int) -> dict[str, Any]:
    """Every step exactly once, none missing, all from one run."""
    steps = [int(float(row["step"])) for row in rows if row.get("step") not in (None, "")]
    unique = sorted(set(steps))
    duplicates = len(steps) - len(unique)
    gaps = [s for s in range(expected_steps) if s not in set(unique)] if expected_steps else []
    run_ids = sorted({row["run_id"] for row in rows if row.get("run_id")})
    return {
        "rows": len(steps),
        "unique_steps": len(unique),
        "duplicate_steps": duplicates,
        "missing_steps": len(gaps),
        "first_step": unique[0] if unique else None,
        "last_step": unique[-1] if unique else None,
        "run_ids": run_ids,
        "single_run": len(run_ids) == 1,
        "ok": duplicates == 0 and not gaps and len(run_ids) == 1,
    }


def collect(root: Path, name: str, expected_steps: int) -> dict[str, Any]:
    run_dir = root / name
    manifest = read_json(run_dir / "run_manifest.json")
    metadata = read_json(run_dir / "experiment_metadata.json")
    probe = read_json(run_dir / "validation_probe.json")
    instruments = read_json(run_dir / "instrument_examples.json")

    import yaml

    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = config or {}
    model = config.get("model") or {}
    training = config.get("training") or {}

    exported = (instruments or {}).get("exported") or {}
    return {
        "candidate": name,
        "exists": run_dir.exists(),
        "run_id": manifest.get("run_id"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "resumes": len(manifest.get("resumes") or []),
        "resume_events": manifest.get("resumes") or [],
        "data_source": metadata.get("data_source") or metadata.get("path_source"),
        "train_manifest": metadata.get("train_manifest"),
        "val_manifest": metadata.get("val_manifest"),
        "train_file_count": metadata.get("train_file_count"),
        "val_file_count": metadata.get("val_file_count"),
        "device": metadata.get("device"),
        "throughput": metadata.get("throughput"),
        "steps_completed": metadata.get("steps_completed"),
        "probe_fingerprint": probe.get("probe_fingerprint"),
        "probe_sample_ids": probe.get("sample_ids") or [],
        "listening_sources": {
            key: (value or {}).get("source_path") for key, value in exported.items()
        },
        "listening_unavailable": (instruments or {}).get("unavailable") or [],
        "num_quantizers": model.get("num_quantizers"),
        "codebook_size": model.get("codebook_size"),
        "strides": model.get("strides"),
        "optimizer": {key: training.get(key) for key in OPTIMIZER_KEYS},
        "losses": {
            **{key: training.get(key) for key in LOSS_KEYS},
            "commitment_weight": model.get("commitment_weight"),
        },
        "integrity": step_integrity(read_curve(run_dir / "training_curves.csv"), expected_steps),
    }


def agreement(rows: list[dict[str, Any]], key: str) -> tuple[bool, list[Any]]:
    values = [row.get(key) for row in rows]
    return all(json.dumps(v, sort_keys=True) == json.dumps(values[0], sort_keys=True)
               for v in values), values


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{milestone('M04')} fairness audit.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--expect-data-source", default="manifest:babyslakh_m04_expanded")
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = [collect(root, name, args.steps) for name in args.candidates]
    missing = [row["candidate"] for row in rows if not row["exists"]]
    if missing:
        raise SystemExit(f"missing run directories: {', '.join(missing)}")

    checks: list[tuple[str, bool, str]] = []

    same_train, train_values = agreement(rows, "train_manifest")
    checks.append(("same train manifest", same_train, str(train_values[0])))
    same_val, val_values = agreement(rows, "val_manifest")
    checks.append(("same validation manifest", same_val, str(val_values[0])))

    same_probe, probe_values = agreement(rows, "probe_fingerprint")
    checks.append(("same validation probe fingerprint", same_probe, str(probe_values[0])))
    same_ids, _ = agreement(rows, "probe_sample_ids")
    checks.append((
        "same probe sample ids", same_ids,
        f"{len(rows[0]['probe_sample_ids'])} held-out segments",
    ))

    same_listen, _ = agreement(rows, "listening_sources")
    listened = sorted(rows[0]["listening_sources"])
    checks.append(("same listening source ids", same_listen, ", ".join(listened) or "none"))

    same_q, q_values = agreement(rows, "num_quantizers")
    checks.append((
        "Q identical across candidates", same_q and q_values[0] == 2, f"Q={q_values[0]}",
    ))
    same_k, k_values = agreement(rows, "codebook_size")
    checks.append((
        "K identical across candidates", same_k and k_values[0] == 128, f"K={k_values[0]}",
    ))

    same_training, training_values = agreement(rows, "optimizer")
    checks.append((
        "identical optimizer configuration", same_training,
        (
            f"batch={training_values[0].get('batch_size')} "
            f"lr={training_values[0].get('learning_rate')} "
            f"betas={training_values[0].get('betas')} seed={training_values[0].get('seed')} "
            f"amp={training_values[0].get('amp')}"
        ),
    ))
    same_losses, loss_values = agreement(rows, "losses")
    checks.append((
        "identical loss configuration", same_losses,
        (
            f"waveform={loss_values[0].get('waveform_weight')} "
            f"spectral={loss_values[0].get('spectral_weight')} "
            f"vq={loss_values[0].get('vq_weight')} fft={loss_values[0].get('fft_sizes')}"
        ),
    ))

    strides = [row["strides"] for row in rows]
    strides_differ = len({json.dumps(s) for s in strides}) == len(strides)
    checks.append((
        "strides are the only intended difference", strides_differ,
        " | ".join(f"{row['candidate']}={row['strides']}" for row in rows),
    ))

    budgets_ok = all(row["integrity"]["last_step"] == args.steps - 1 for row in rows)
    checks.append((
        "same completed step budget", budgets_ok,
        f"all reached step {args.steps - 1}",
    ))

    run_ids = [row["run_id"] for row in rows]
    checks.append((
        "one isolated run_id per candidate",
        len(set(run_ids)) == len(run_ids) and all(run_ids),
        ", ".join(str(r) for r in run_ids),
    ))
    checks.append((
        "zero duplicated steps",
        all(row["integrity"]["duplicate_steps"] == 0 for row in rows), "",
    ))
    checks.append((
        "zero missing steps",
        all(row["integrity"]["missing_steps"] == 0 for row in rows), "",
    ))
    checks.append((
        "single run_id inside every curve",
        all(row["integrity"]["single_run"] for row in rows), "",
    ))

    labels = [row["data_source"] for row in rows]
    label_ok = all(label == args.expect_data_source for label in labels)
    checks.append((
        "corrected provenance label recorded", label_ok,
        ", ".join(str(label) for label in labels),
    ))
    checks.append((
        "no m02_manifest label in authoritative evidence",
        not any("m02_manifest" == str(label) for label in labels), "",
    ))

    print(f"{milestone('M04')}: Stage 1 authoritative fairness audit\n")
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    print("\nper-candidate integrity and cost")
    print(
        f"  {'candidate':<24}{'rows':>7}{'steps':>7}{'dup':>5}{'gaps':>6}"
        f"{'resumes':>9}{'steps/s':>10}  run_id"
    )
    for row in rows:
        i = row["integrity"]
        rate = (row.get("throughput") or {}).get("steps_per_second")
        print(
            f"  {row['candidate']:<24}{i['rows']:>7}{i['unique_steps']:>7}"
            f"{i['duplicate_steps']:>5}{i['missing_steps']:>6}{row['resumes']:>9}"
            f"{(f'{rate:.2f}' if isinstance(rate, (int, float)) else 'n/a'):>10}  {row['run_id']}"
        )

    print("\nlistening sources (identical across candidates means the A/B is honest)")
    for category in LISTENING_CATEGORIES:
        source = rows[0]["listening_sources"].get(category)
        if source is None:
            print(f"  {category:<12} UNAVAILABLE in the held-out split")
        else:
            print(f"  {category:<12} {source}")

    passed = all(ok for _, ok, _ in checks)
    report = {
        "fair": passed,
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
        "candidates": rows,
    }
    out = root / "m04_fairness_audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nfair comparison: {passed}")
    print(f"report: {out}")
    if not passed:
        raise SystemExit("fairness audit FAILED; the comparison is not authoritative evidence")


if __name__ == "__main__":
    main()
