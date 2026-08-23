# Colab Remote Training

This project should not fetch huge datasets or train flagship models on the local workstation. Use `notebooks/songforge_colab_remote.ipynb` from the Google account that owns the target Drive storage.

## What Codex Can and Cannot Do

Codex cannot sign into `anastasiasaat81@gmail.com` or operate that Google Colab account from this environment. The notebook is prepared so the account owner can run the cells in Colab after authorizing Drive and any dataset provider credentials.

## Getting the Code onto Colab

The notebook supports two delivery modes, set by `SETUP_MODE` in cell 1.

### `SETUP_MODE = "zip"` (default, no GitHub account needed)

1. On the workstation, build the archive from the repository root:

   ```bash
   zip -r songforge-colab.zip songforge-dl-starter \
     -x '*/.venv/*' '*/__pycache__/*' '*.pyc' '*/.pytest_cache/*' '*/.ruff_cache/*' \
        '*.egg-info/*' '*/outputs/*' '*/data/raw/*' '*/data/processed/*' '*/checkpoints/*' \
        '*.pt' '*.wav' '*.DS_Store'
   ```

2. Upload `songforge-colab.zip` to Drive at `MyDrive/songforge-dl/`.
3. Run the notebook. Cell 3 extracts it to `/content/songforge`.

The project is extracted to Colab local disk, not Drive. Drive I/O is slow and `pip install -e`
against a Drive path is unreliable. Only datasets, outputs, and checkpoints live on Drive.

### `SETUP_MODE = "git"`

Only works once the repository actually exists and the branch contains the M03 code. Set
`GITHUB_TOKEN` in cell 1 for a private repo, and clear the cell output afterwards. If the clone
fails, cell 3 now stops with an explicit error instead of letting later cells run from `/content`.

## Colab Steps

1. Open Colab from the target Google account.
2. Upload or open `notebooks/songforge_colab_remote.ipynb`.
3. Set `SETUP_MODE`, `DRIVE_ROOT`, and the dataset choices in cell 1.
4. Run cells 2-4: mount Drive, place the code, install dependencies, run the M00 and M01 gates.
5. Accept dataset terms in the notebook before running any GTSinger or MTG-Jamendo download cell.
6. Run only milestone-appropriate jobs:
   - M00: `pytest -q`
   - M01: `python scripts/validate_dataset_registry.py`
   - M02: `python scripts/run_preprocessing_acceptance.py`
   - M03: `python scripts/run_codec_acceptance.py` (trains from the M02 manifest)

## Milestone Status

M00 PASS, M01 PASS, M02 PASS on real BabySlakh. **M03 is the milestone under test** and
trains from the M02 canonical manifest. M04 has not started.

## M02 Acceptance

M02 PASS requires a real approved audio subset to complete:

```text
raw audio -> validation -> preprocessing -> segmentation
          -> canonical manifest -> train/val/test split -> Drive persistence
```

```bash
python scripts/run_preprocessing_acceptance.py \
  --dataset-id babyslakh \
  --audio-dir "$SONGFORGE_DATA/raw/babyslakh" \
  --output-dir "$SONGFORGE_DATA/processed/babyslakh_m02" \
  --limit-files 24 \
  --split-mode song
```

Rehearse the same pipeline with no download using `--synthetic`. That mode reports
`synthetic: true` and is deliberately **not** an M02 pass.

Expected Drive artifacts under `--output-dir`:

- `manifests/all.jsonl`, `manifests/train.jsonl`, `manifests/val.jsonl`, `manifests/test.jsonl`
- `manifests/manifest_summary.json`
- `audio/<track_id>/<record_id>.wav`
- `preprocess_report.json`
- `m02_acceptance_report.json`

Expected repo update: `docs/EXPERIMENT_LOG_M02.md`.

M02 passes only when every artifact is present, the manifest round-trips, provenance and
licence survive on every record, and there is no song leakage, singer leakage, or cross-split
duplicate audio. Preprocessing settings are recorded in each record, so the run is
reproducible from the manifest alone. See `docs/MANIFEST_SCHEMA.md`.

## Storage Layout

```text
/content/drive/MyDrive/songforge-dl/
  data/
    raw/
    processed/
    manifests/
  checkpoints/
  logs/
  repo/
```

Keep raw datasets and checkpoints out of git. The repository `.gitignore` already blocks `data/raw`, `data/processed`, checkpoints, logs, audio files, and PyTorch weight files.

## Push Results

Only push source code, configs, docs, small manifests, and experiment metadata. Do not push raw datasets, generated WAV/MP3/FLAC files, or large checkpoints.

## M03 Final Acceptance

M03 trains the RVQ codec from randomly initialized SongForge weights on the **M02 canonical
manifest**. No pretrained codec weights are loaded anywhere in this path. Bypassing M02 with
`--audio-glob` is debug-only and cannot pass acceptance: the runner records the trainer's
`path_source` and fails unless it starts with `m02_manifest`.

```bash
python scripts/run_codec_acceptance.py \
  --config configs/codec/codec_m03_tiny.yaml \
  --train-manifest "$SONGFORGE_DATA/processed/babyslakh_m02/manifests/train.jsonl" \
  --val-manifest   "$SONGFORGE_DATA/processed/babyslakh_m02/manifests/val.jsonl" \
  --output-dir     "$DRIVE_ROOT/outputs/codec_m03_acceptance" \
  --steps 4000 \
  --rvq-log-every 25
```

The runner requires CUDA, and requires the CUDA **and** AMP codec tests to actually pass. A
skip is treated as a failure, so a CPU runtime cannot produce a green M03.

### Metrics recorded

Initial/final train loss, validation loss, waveform L1, MR-STFT, SNR, spectral convergence,
latent frame rate, downsample factor, codebook count, codebook size, bits per code, bitrate,
compression ratio, per-codebook utilization / dead codes / entropy / perplexity, collapse
flag, peak GPU VRAM, training throughput, and encode/decode timing. All land in
`m03_acceptance_report.json` and the generated `docs/EXPERIMENT_LOG.md`.

### RVQ health is tracked throughout training

`train_codec.py` writes an RVQ snapshot to `rvq_history.jsonl` every `--rvq-log-every` steps,
so acceptance sees the whole trajectory rather than only the final step. Temporary collapse
and recovery are reported explicitly.

In-training snapshots are computed on a single batch, so they are bounded by the number of
latent frames in that batch and read lower than the final whole-split figures. Compare
snapshots to snapshots.

### Held-out listening examples

Four held-out validation pairs are exported as
`examples/character_{percussive,harmonic,bass_heavy,mixed}_{original,reconstructed}.wav`.
The buckets are spectral heuristics, not instrument ground truth: a segment path does not
reliably identify a BabySlakh stem.

### Why `--steps 4000` and not 80

The RVQ codebook is seeded from real encoder outputs on the first training batch, but the encoder
initially emits latents that barely vary across time, so the codebook still collapses to a handful
of entries during early training. It recovers only once the encoder learns time-varying latents.

Measured on a CPU dry run with the tiny config:

| Step | Unique codes | Perplexity | Utilization |
| ---: | ---: | ---: | ---: |
| 0 | 94 | 56.8 | 0.73 |
| 50 | 1 | 1.0 | 0.01 |
| 150 | 6 | 1.3 | 0.05 |
| 400 | 44 | 27.6 | 0.34 |

`validate_acceptance` fails the run when utilization `< 0.05` or perplexity `< 2.0`, so a
budget of 80 steps fails its own gate even though the codec is training correctly. On a real
BabySlakh corpus give it thousands of steps; on a Colab GPU this is still a short run. If the
final RVQ is still collapsed, raise `--steps` rather than loosening the gate.

Expected Drive artifacts:

- `original.wav`
- `reconstructed.wav`
- `config.yaml`
- `checkpoint.pt`
- `checkpoint_last.pt`
- `optimizer_state.pt`
- `training_curves.csv`
- `metrics.jsonl`
- `rvq_history.jsonl`
- `listening_examples.json`
- `examples/character_*_original.wav`, `examples/character_*_reconstructed.wav`
- `train_metrics.json`
- `validation_metrics.json`
- `metrics_summary.json`
- `compression_stats.json`
- `experiment_metadata.json`
- `m03_acceptance_report.json`

Expected repo update:

- `docs/EXPERIMENT_LOG.md`

M03 passes only after Colab CUDA tests pass, AMP path passes, real-music train/validation runs, loss decreases, checkpoint resume works, WAV artifacts open, RVQ does not obviously collapse, and artifacts persist in Drive.

The current M03 codec records about 120 latent frames/sec at 24 kHz. Keep that as the M03 baseline; do not change it during acceptance.
