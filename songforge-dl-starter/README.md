# SongForge-DL

Research-grade, API-free generative music project. The goal is to train and run **your own PyTorch weights** for music planning, neural audio representation, singing synthesis, and accompaniment generation.

## Non-negotiable rule

Inference must not call Suno, MusicGen, ACE-Step, OpenAI, ElevenLabs, or any other hosted generation API. Reference repositories/papers may be studied, but production inference must load local checkpoints created by this project.

## V1 target

1. Reconstruct 5-10 s music clips with our neural codec.
2. Generate symbolic melody/structure with our Transformer planner.
3. Generate short singing spectrograms conditioned on phonemes/notes.
4. Generate short instrumental/accompaniment token sequences.
5. Mix a 10-30 s demo song locally.

Do **not** target commercial-Suno quality in V1. Target correct ML design, reproducibility, measurable improvement, and defendable ownership of the trained models.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,audio]'
pytest -q
```

## Smoke train

```bash
python scripts/smoke_train_codec.py
python scripts/smoke_train_planner.py
```

## Milestone status

| Milestone | Status |
| --- | --- |
| M00 repository health | PASS |
| M01 dataset registry + licensing | PASS |
| M02 audio preprocessing | implemented, acceptance runs on Colab |
| M03 neural codec | prototype exists, kept as an **experimental spike**, not accepted |
| M04+ | not started |

M03 was built before M02. The code is preserved and still runs, but it is not an accepted
milestone; official M03 acceptance follows M02 sign-off. See `AGENTS.md` for the gates.

## Preprocess a dataset (M02)

```bash
python scripts/preprocess_dataset.py \
  --dataset-id babyslakh \
  --input-dir "$SONGFORGE_DATA/raw/babyslakh" \
  --output-dir "$SONGFORGE_DATA/processed/babyslakh_m02"
```

Output is the canonical manifest described in `docs/MANIFEST_SCHEMA.md`. Heavy downloads and
preprocessing belong on Colab; see `docs/COLAB_REMOTE_TRAINING.md`.

## Repository map

- `configs/`: experiment configuration
- `src/songforge/data/`: canonical manifest, dataset registry, M02 preprocessing
  (`media.py` validation, `dsp.py` resampling/segmentation, `preprocess.py` pipeline,
  `splits.py` leakage-free splitting, `dedup.py` duplicate hooks, `fixtures.py` synthetic corpora)
- `src/songforge/models/codec/`: neural codec + vector quantizer
- `src/songforge/models/planner/`: symbolic/autoregressive song Transformer
- `src/songforge/models/singer/`: singing diffusion components
- `src/songforge/models/accompaniment/`: instrumental token generator
- `src/songforge/training/`: reusable trainer/checkpoint logic
- `scripts/`: CLI entry points
- `tests/`: unit/smoke tests
- `docs/`: architecture, datasets, manifest schema, roadmap, evaluation, Codex build contract

## Build order

Read `AGENTS.md` first. It defines strict milestones and PASS gates for Codex.
