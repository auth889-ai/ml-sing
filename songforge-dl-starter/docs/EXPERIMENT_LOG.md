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

---

## M04 — High-Quality Codec Optimization & Latent-Rate Selection: Stage 1 authoritative

Full evidence: [M04_STAGE1_RESULT.md](M04_STAGE1_RESULT.md).
Corpus: [M04_DATA_EXPANSION.md](M04_DATA_EXPANSION.md).

Ran the proposed matrix's first three rows at equal 4000-step budgets on an
expanded 20-track / 229-WAV / 17,558-segment BabySlakh corpus, not the 11-song
M02 acceptance slice. Fairness audit 17/17 PASS: identical manifests, probe
fingerprint `fe6c1fc7ff8eede1`, listening sources, Q=2, K=128, optimizer and
loss blocks; strides the only difference; one run_id each; zero duplicate and
zero missing steps.

| candidate | bitrate | codes/30 s | recon after | SI-SDR dB | transient | RVQ util | final collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 120 Hz / Q2 | 1680 bps | 7,200 | **0.1207** | **+0.66** | **0.709** | 1.000 | False |
| 75 Hz / Q2 | 1050 bps | 4,500 | 0.1333 | −3.19 | 0.636 | 1.000 | False |
| 50 Hz / Q2 | 700 bps | 3,000 | 0.1461 | −12.99 | 0.533 | 1.000 | False |

All three showed temporary RVQ collapse early (from step 0–25) and fully
recovered: zero dead codes and full utilization at the end.

**Outcome: CASE 3.** Degradation is monotone with latent rate across
reconstruction, L1, MR-STFT, SNR, SI-SDR and transients, while HF preservation
is non-monotone — so the damage is temporal, not high-frequency rolloff. RVQ is
healthy in every candidate (the 50 Hz run has the *highest* perplexity and
entropy), so the losses are not a starved quantizer. Because Q and K were fixed,
lower frame rates also cut total capacity, and Stage 1 cannot separate frames/s
from bits/s. Stage 2 (75 Hz/Q4, 50 Hz/Q4) is the test for that — the CASE 4
question. 25 Hz/Q8 is **not** justified yet.

Codec not frozen. Stage 2 not launched. No selection recorded.

### Colab throughput note

These runs measured 16.77–17.33 steps/s; the M03 single-session clean run
measured 2.01 steps/s. Same model and training blocks (byte-identical), same
batch and audio-seconds per step, same torch 2.11.0+cu128 on Tesla T4, and Drive
reads measured at ~400 segments/s against a ~35 segments/s demand, so I/O was
not the bottleneck. The difference is session-level runtime conditions, not
experiment design. Absolute steps/s is therefore not comparable across Colab
sessions; all three Stage 1 candidates ran back-to-back in one session with zero
resumes, so their relative timings are sound.
