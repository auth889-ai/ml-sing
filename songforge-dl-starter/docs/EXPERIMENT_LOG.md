# Experiment Log

## M03 Colab Acceptance

Status: **M03 FAIL - Google Colab acceptance not executed yet**

Reason: this Codex environment cannot sign into or operate the requested Google Colab Pro account or mount its Google Drive. The acceptance runner is implemented, but the CUDA/Colab real-music run must be executed from Colab.

Exact Colab command template:

```bash
python scripts/colab_m03_acceptance.py \
  --config configs/codec/codec_m03_tiny.yaml \
  --audio-glob "$SONGFORGE_DATA/raw/babyslakh/**/*.wav" \
  --output-dir "$DRIVE_ROOT/outputs/codec_m03_acceptance" \
  --steps 80
```

Required status before M03 PASS:

- full pytest passes in Colab
- CUDA codec path passes
- AMP codec path passes
- real-music training executes
- loss decreases
- held-out validation reconstruction works
- checkpoint resume works
- valid WAV artifacts persist in Drive
- RVQ does not obviously collapse
- compression measurements are reproducible
- subjective A/B listening notes are recorded

### Long-Form Cost Note

The current M03 codec records about 120 latent frames/sec at 24 kHz with downsample factor 200. This is a potential long-form generation cost concern. Do not change it during M03 acceptance.

### Proposed M04 Experiment Matrix

Do not run M04 yet. Proposed future matrix:

| Candidate | Approx latent rate | Purpose |
| --- | ---: | --- |
| M03 baseline | 120 Hz | Quality/control baseline. |
| Lower-rate A | ~75 Hz | Reduce token cost while checking vocal/transient retention. |
| Lower-rate B | ~50 Hz | Stronger long-form compression candidate. |
| Lower-rate C | ~25-40 Hz | Only if decoder quality and RVQ usage remain stable. |

Optimization target: perceptual reconstruction quality + low token/latent rate + stable RVQ usage + reasonable Colab training cost.
