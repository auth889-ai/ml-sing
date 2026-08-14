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

## Repository map

- `configs/`: experiment configuration
- `src/songforge/data/`: manifests, license checks, preprocessing
- `src/songforge/models/codec/`: neural codec + vector quantizer
- `src/songforge/models/planner/`: symbolic/autoregressive song Transformer
- `src/songforge/models/singer/`: singing diffusion components
- `src/songforge/models/accompaniment/`: instrumental token generator
- `src/songforge/training/`: reusable trainer/checkpoint logic
- `scripts/`: CLI entry points
- `tests/`: unit/smoke tests
- `docs/`: architecture, datasets, roadmap, evaluation, Codex build contract

## Build order

Read `AGENTS.md` first. It defines strict milestones and PASS gates for Codex.
