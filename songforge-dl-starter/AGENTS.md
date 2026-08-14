# Codex Build Contract — SongForge-DL

You are implementing a research-grade, API-free generative music system. Preserve this repository structure and implement one milestone at a time. Never replace a missing model with a hosted inference API.

## Definition of "our model"

A component counts as ours only when:
- architecture code lives in `src/songforge/models/`;
- training code computes a loss and calls backward/optimizer step;
- weights are saved under `checkpoints/`;
- inference loads those local weights;
- tests cover shapes and at least one optimization step.

Pretrained weights may only be used in an explicitly marked baseline experiment, never in the flagship `ours_*` path.

## Milestones / PASS gates

### M00 Repository health
PASS when `pytest -q` passes and all configs parse.

### M01 Dataset registry + licensing
Implement download instructions, manifests, checksums where possible, license metadata, split rules, duplicate/singer leakage checks. Do not auto-download gated/restricted datasets without user acceptance.

### M02 Audio preprocessing
Implement decode, mono/stereo policy, resampling, peak/RMS checks, segmentation, mel/STFT helpers, deterministic train/val/test manifests.

### M03 Neural codec
Implement encoder -> RVQ -> decoder. Begin with 24 kHz mono. Train on short clips. Loss: waveform L1 + multi-resolution STFT; adversarial loss is a later milestone.
PASS: reconstruction loss falls on a tiny overfit set and reconstructed WAV is emitted.

### M04 Codec quality
Add residual blocks, improved RVQ, commitment/codebook losses, dead-code monitoring, bandwidth metrics, checkpoint/resume and objective evaluation.

### M05 Symbolic tokenizer
Implement MIDI event representation: BOS/EOS, NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY, TEMPO, BAR, optional CHORD/SECTION tokens. Round-trip MIDI -> tokens -> MIDI test required.

### M06 Song planner Transformer
Decoder-only Transformer over symbolic tokens with optional style/section conditioning.
PASS: overfit tiny MIDI set, then generate valid token sequences with masking and sampling controls.

### M07 Singing preprocessing
Parse GTSinger annotations into phoneme/note/pitch/duration/style tensors. Enforce singer-disjoint split for generalization experiments.

### M08 Singing model
Start with acoustic model predicting mel spectrogram; then implement diffusion noise prediction conditioned on phonemes, notes, pitch and singer/style embeddings.
PASS: diffusion loss decreases and inference generates a mel tensor of expected shape.

### M09 Vocoder
V1 may use Griffin-Lim only as a debugging baseline. Flagship path must implement/train a local neural vocoder or a jointly learned decoder. Any pretrained vocoder must be labeled baseline-only.

### M10 Accompaniment generator
Condition on style + tempo + harmonic/melodic plan. Generate symbolic arrangement first, then optionally codec-token audio generation.
PASS: locally generated accompaniment WAV/MIDI without external API.

### M11 Integration
planner -> singer -> accompaniment -> local mixer. Emit `plan.json`, `melody.mid`, `vocal.wav`, `instrumental.wav`, `final_song.wav`.

### M12 Evaluation
Implement objective metrics, ablations, listening-test export, seed reproducibility, model/data cards, and experiment table.

### M13 Long-form
Hierarchical section planner and chunked generation with overlap/context. Target 30-60 s before attempting minutes.

## Engineering rules

- PyTorch only for core neural models unless an experiment explicitly says otherwise.
- Hydra/OmegaConf-style config discipline is preferred; no hard-coded dataset paths.
- All random experiments expose a seed.
- Save config + git commit + metric history next to each checkpoint.
- Unit tests must stay CPU-runnable.
- Heavy training scripts must support CUDA AMP and gradient accumulation.
- Never commit raw datasets or large checkpoints.
- Every dataset entry in a manifest must include source and license fields.
- No train/validation leakage by track identity; singer-disjoint splits for singing evaluation.

## First commands Codex should run

```bash
pip install -e '.[dev,audio]'
pytest -q
python scripts/smoke_train_codec.py
python scripts/smoke_train_planner.py
```

Then implement M01 before adding model complexity.
