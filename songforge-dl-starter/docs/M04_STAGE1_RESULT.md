# M04 — High-Quality Codec Optimization & Latent-Rate Selection: Stage 1 authoritative result

**No codec is selected or frozen by this document.** It records what the Stage 1
comparison measured and what that evidence does and does not support.

Output root: `$DRIVE/outputs/m04_stage1_authoritative_expanded/`

## Fairness audit

A latent-rate comparison is evidence only if the candidates differed in exactly
one thing. `scripts/m04_fairness_audit.py` re-derives that from the run
artifacts rather than trusting the launch command. **17 / 17 checks PASS,
`fair: true`.**

| check | result |
| --- | --- |
| same train manifest | `babyslakh_m04_expanded/manifests/train.jsonl` |
| same validation manifest | `babyslakh_m04_expanded/manifests/val.jsonl` |
| same validation probe fingerprint | `fe6c1fc7ff8eede1` |
| same probe sample ids | 64 held-out segments, identical set |
| same listening source ids | bass, full_mix, guitar, percussion, piano, strings |
| Q identical | Q = 2 |
| K identical | K = 128 |
| identical optimizer configuration | batch 2, lr 3e-4, betas [0.9, 0.95], seed 42, AMP on |
| identical loss configuration | waveform 1.0, spectral 1.0, vq 1.0, FFT [256, 512, 1024] |
| strides the only intended difference | [2,4,5,5] / [2,4,5,8] / [2,4,6,10] |
| same completed step budget | all reached step 3999 |
| one isolated run_id per candidate | 3 distinct ids |
| zero duplicated steps | 0 |
| zero missing steps | 0 |
| single run_id inside every curve | yes |
| corrected provenance label recorded | `manifest:babyslakh_m04_expanded` |
| no `m02_manifest` label in authoritative evidence | confirmed absent |

The provenance label matters: `resolve_paths` previously hardcoded
`m02_manifest` for any manifest pair, so a run on the expanded corpus filed
itself under the wrong milestone. Fixed before these runs; the Stage 1 *pilot*
artifacts still carry the old label.

## Corpus

Real BabySlakh (CC-BY-4.0, Zenodo 4603844) via the canonical
`songforge.audio.v1` manifests at `processed/babyslakh_m04_expanded/`.
The frozen M02 acceptance manifests were not modified.

| | |
| --- | ---: |
| source tracks | 20 |
| source WAVs | 229 |
| segments | 17,558 |
| usable duration | 585.27 min (9.75 h) |
| segments with a corpus instrument label | 100.0% |

| split | tracks | WAVs | segments | minutes | segments % | duration % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 16 | 182 | 13,645 | 454.83 | 77.71 | 77.71 |
| val | 2 | 22 | 2,056 | 68.53 | 11.71 | 11.71 |
| test | 2 | 25 | 1,857 | 61.90 | 10.58 | 10.58 |

Song leakage **zero**. Duplicate leakage **zero**. Licence and provenance
complete on every record.

Instrument families present in the corpus (14): Guitar 2,902 · Piano 2,510 ·
Mixture 2,376 · Bass 2,265 · Strings (continued) 2,228 · Drums 2,167 ·
Synth Pad 682 · Organ 623 · Pipe 408 · Brass 402 · Chromatic Percussion 349 ·
Synth Lead 295 · Reed 244 · Strings 107.

> **Coverage limit.** The validation split is 2 songs and covers **8 of the 14**
> families. Every number below is measured on those 8. This is not universal
> instrument coverage and must not be reported as such.

## Representation and long-form cost

| Candidate | latent Hz | downsample | Q | K | bits/code | bitrate | compression | codes/10 s | codes/30 s | codes/3 min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 120.0 | 200 | 2 | 128 | 7 | 1680 bps | 228.6:1 | 2,400 | 7,200 | 43,200 |
| A | 75.0 | 320 | 2 | 128 | 7 | 1050 bps | 365.7:1 | 1,500 | 4,500 | 27,000 |
| B | 50.0 | 480 | 2 | 128 | 7 | 700 bps | 548.6:1 | 1,000 | 3,000 | 18,000 |

"Raw codec codes", not "tokens" — these are pre-tokenizer quantizer indices.

## Learning — held-out probe, same 64 segment IDs before and after

| metric | baseline 120 Hz | A 75 Hz | B 50 Hz |
| --- | ---: | ---: | ---: |
| reconstruction before | 0.2656 | 0.2359 | 0.2214 |
| reconstruction after | **0.1207** | 0.1333 | 0.1461 |
| waveform L1 before | 0.11890 | 0.09222 | 0.08441 |
| waveform L1 after | **0.04978** | 0.05815 | 0.06771 |
| MR-STFT before | 0.1467 | 0.1436 | 0.1370 |
| MR-STFT after | **0.07091** | 0.07517 | 0.07837 |

The *before* values differ between candidates because an untrained model of a
different stride stack is a different starting point; only the *after* column is
a comparison, and the probe segments are identical.

### Training windows (first vs last decile, not single batches)

| candidate | loss first → last | L1 first → last |
| --- | --- | --- |
| baseline 120 Hz | 0.20526 → 0.19778 (−3.6%) | 0.08121 → 0.05224 (−35.7%) |
| A 75 Hz | 0.20180 → 0.23185 (**+14.9%**) | 0.08067 → 0.06618 (−18.0%) |
| B 50 Hz | 0.21050 → 0.18482 (−12.2%) | 0.08264 → 0.06841 (−17.2%) |

Candidate A's windowed *training* loss rose while its L1 fell and its held-out
probe improved. Training-window loss is noisy at this scale; the held-out probe
is the load-bearing signal and it improved for all three.

## Perceptual / objective — held-out probe after training

| metric | baseline 120 Hz | A 75 Hz | B 50 Hz |
| --- | ---: | ---: | ---: |
| SNR dB | **+3.68** | +2.19 | +0.97 |
| SI-SDR dB | **+0.66** | −3.19 | −12.99 |
| log spectral distance dB (lower better) | 19.12 | **19.03** | 20.07 |
| transient preservation (higher better) | **0.709** | 0.636 | 0.533 |
| HF preservation dB | −3.68 | −6.86 | **−2.00** |

SI-SDR before training was ≈ −54 dB for all three, so the after-values are real
learning, not scale artefacts. SI-SDR is the metric that separates these
candidates most sharply, and it is scale-invariant, so it is not explained away
by gain differences.

## RVQ health (whole evaluation sample)

| candidate | utilization | per codebook | perplexity | entropy (max 4.852) | dead codes | collapsed snapshots | first | last | final collapse |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| baseline 120 Hz | 1.000 | [1.0, 1.0] | [78.9, 80.2] | [4.368, 4.384] | [0, 0] | 14 | 25 | 1725 | **False** |
| A 75 Hz | 1.000 | [1.0, 1.0] | [76.5, 89.9] | [4.338, 4.499] | [0, 0] | 17 | 25 | 3700 | **False** |
| B 50 Hz | 1.000 | [1.0, 1.0] | [85.9, 96.2] | [4.453, 4.566] | [0, 0] | 18 | 0 | 1575 | **False** |

Every candidate shows **temporary collapse early and full recovery**: collapse
snapshots appear from step 0–25, the last one at step 1725 / 3700 / 1575, and
all three end at full utilization with zero dead codes. No candidate finished
collapsed. The quantizer is not starved in any candidate — notably, the *lowest*
latent rate has the *highest* perplexity and entropy, i.e. its codebooks are
working hardest.

## Compute

| candidate | training wall s | steps/s | peak VRAM GB | enc+dec RTF | interruptions | resumes | data location |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline 120 Hz | 238.5 | 16.77 | 0.0973 | 0.00852 | 0 | 0 | Google Drive |
| A 75 Hz | 230.8 | 17.33 | 0.0997 | 0.00536 | 0 | 0 | Google Drive |
| B 50 Hz | 230.8 | 17.33 | 0.0774 | 0.00486 | 0 | 0 | Google Drive |

All three ran back-to-back in a single Colab session on one Tesla T4, with zero
interruptions and zero resumes. Training data and probes were read from Drive
(`/content/drive/...`), not from a local cache, for all three.

### Why ~17 steps/s here and ~2 steps/s in M03

Measured, not assumed:

- The M03 single-session clean run recorded 4000 steps in 1990.6 s = **2.01
  steps/s**. (The M03 *authoritative* run's throughput block covers only its
  final 1-step session and is not a usable rate; the widely quoted "~0.9
  steps/s" was elapsed multi-session wall-clock across 4 interruptions and 3
  resumes divided by steps, which is not a training-loop rate at all.)
- **Not the data volume per step.** Both processed 8000.0 audio-seconds over
  4000 steps — batch 2 × 1 s, identical.
- **Not I/O.** Measured on this runtime: 300 segment reads from
  `babyslakh_m04_expanded` took 0.75 s (399/s) and from the frozen
  `babyslakh_m02` 0.73 s (413/s); a warm repeat was 0.74 s. Both corpora read at
  ~400 segments/s, while the training loop consumed only ~35 segments/s. Drive
  was never the bottleneck, and the two corpora are not measurably different.
  *(This disproved the cache-warmth explanation I initially expected.)*
- **Not the model or hyperparameters.** `codec_m03_tiny.yaml` and
  `m04_baseline_120hz_q2.yaml` have byte-identical `model` and `training` blocks
  (base_channels 16, latent_dim 32, strides [2,4,5,5], Q2, K128, batch 2,
  lr 3e-4, AMP on).
- **Not the software or GPU class.** Both torch 2.11.0+cu128 on Tesla T4.

So the same computation on the same hardware and software ran ~8× faster in this
session. The remaining explanation is session-level runtime conditions on a
shared Colab T4, which I did not isolate further and do not claim to have proven.

**What this means for the comparison:** absolute steps/s is not stable across
Colab sessions and should not be compared across them. All three Stage 1
candidates ran consecutively in one session with zero resumes, so their timings
are comparable *to each other*, which is what the experiment requires.

## Listening comparison

Same held-out source file per category for every candidate — verified by the
fairness audit, and `m04_listening_matrix.py` refuses to emit a row whose
candidates reconstructed different audio. Instrument identity comes from the
BabySlakh `metadata.yaml`, never from spectral heuristics.

`$DRIVE/outputs/m04_stage1_authoritative_expanded/listening/`

| category | family | files |
| --- | --- | --- |
| piano | Piano | `piano__source.wav`, `piano__120hz.wav`, `piano__75hz.wav`, `piano__50hz.wav` |
| guitar | Guitar | `guitar__source.wav`, `guitar__120hz.wav`, `guitar__75hz.wav`, `guitar__50hz.wav` |
| bass | Bass | `bass__source.wav`, `bass__120hz.wav`, `bass__75hz.wav`, `bass__50hz.wav` |
| percussion | Drums | `percussion__source.wav`, `percussion__120hz.wav`, `percussion__75hz.wav`, `percussion__50hz.wav` |
| strings | Strings (continued) | `strings__source.wav`, `strings__120hz.wav`, `strings__75hz.wav`, `strings__50hz.wav` |
| full_mix | Mixture | `full_mix__source.wav`, `full_mix__120hz.wav`, `full_mix__75hz.wav`, `full_mix__50hz.wav` |

All six requested categories were available in the held-out split; none had to be
reported as missing. Index: `listening/README.md`, machine-readable
`listening/listening_matrix.json`.

### Per-category objective metrics (SI-SDR dB / transient / HF dB)

| category | 120 Hz | 75 Hz | 50 Hz |
| --- | --- | --- | --- |
| piano | −1.02 / 0.769 / −0.94 | −5.61 / 0.784 / −2.92 | −7.36 / 0.609 / +1.65 |
| guitar | +1.99 / 0.030 / −6.45 | −1.29 / **−0.121** / −12.40 | −4.10 / 0.173 / −8.73 |
| bass | +8.34 / 0.900 / −2.98 | +3.71 / 0.654 / −2.27 | −0.82 / 0.453 / +3.23 |
| percussion | +5.24 / 0.919 / −29.54 | −4.67 / 0.878 / −33.71 | −2.26 / 0.913 / −25.67 |
| strings | +1.89 / 0.281 / −9.65 | +0.36 / **−0.012** / −12.04 | −1.22 / 0.193 / −7.45 |
| full_mix | +2.63 / 0.835 / −20.00 | −5.10 / 0.765 / −24.57 | −11.93 / 0.586 / −18.20 |

120 Hz has the best SI-SDR in all six categories. Guitar and strings show
*negative* onset-envelope correlation at 75 Hz — transient structure is not
merely degraded there but decorrelated. Numbers do not replace listening; these
are a guide to what to listen for.

## Interpretation

**Classification: CASE 3**, with CASE 4 as the specific hypothesis to test.

Reading the evidence:

1. **Degradation is monotone with latent rate** on reconstruction (0.1207 →
   0.1333 → 0.1461), L1, MR-STFT, SNR, SI-SDR and transient preservation. This
   is a consistent signal, not a single noisy metric.
2. **SI-SDR falls sharply**: +0.66 → −3.19 → −12.99 dB. 120 Hz is the only
   candidate above 0 dB. Per the instruction not to dismiss SI-SDR: at equal
   budget these relative differences are large, and SI-SDR is scale-invariant,
   so this is temporal-waveform damage rather than a gain artefact.
3. **Transients degrade monotonically**: 0.709 → 0.636 → 0.533, with guitar and
   strings going negative at 75 Hz. Consistent with loss of temporal resolution.
4. **HF preservation does not tell the same story** (−3.68 / −6.86 / −2.00, with
   50 Hz best). So this is not primarily a high-frequency-rolloff failure; the
   damage is temporal.
5. **RVQ is healthy everywhere** — utilization 1.000, zero dead codes, no final
   collapse, and the lowest rate has the *highest* perplexity and entropy. The
   losses are therefore not caused by a broken or starved quantizer.

Point 5 is what makes CASE 3 the right call rather than CASE 5. Because Q and K
were held fixed, halving the frame rate also cut total capacity (1680 → 1050 →
700 bps). Stage 1 cannot separate "too few frames per second" from "too few bits
per second", and the saturated codebooks at 50 Hz are a hint that added depth
could help. That is precisely the CASE 4 question, and the test for it is
**75 Hz/Q4 (2100 bps)** and **50 Hz/Q4 (1400 bps)** — both restoring capacity at
or above the 120 Hz/Q2 baseline while keeping the lower frame rate.

Explicitly **not** CASE 1 (75 Hz is not close to 120 Hz: −3.9 dB SI-SDR, −0.073
transient) and **not** CASE 2 (50 Hz is clearly not close).

**25 Hz/Q8 should not be run now.** It exists in the matrix, but until Stage 2
establishes whether added depth recovers 75/50 Hz, the temporal-resolution vs
codebook-depth question it probes is not yet scientifically useful.

### Limits of this evidence

- 20 BabySlakh tracks, validation covering 8 of 14 instrument families. This
  supports a representation study. It does **not** demonstrate universal genre
  coverage, universal instrument coverage, professional mastering quality, or
  multilingual singing — those need the later multi-dataset stages.
- The codec is deliberately tiny (base_channels 16, latent_dim 32) and 4000
  steps is not convergence. Absolute quality is low across the board; what is
  being compared is *relative* behaviour at equal budget, which is the question
  M04 asks.
- M04 answers "what latent audio representation should downstream SongForge
  models use?" It does not claim SongForge can already generate every kind of
  song.

## Status — provisional conclusion, Stage 2 deferred

**Provisional custom-codec conclusion: 120 Hz / Q2 / K128 is the best quality
observed among the custom codecs tested.** 75 Hz/Q2 and 50 Hz/Q2 reduce raw-code
pressure but cause meaningful quality degradation at this budget.

This is **not** a universal codec freeze. It is the current standing of the
custom-representation research track.

**Stage 2 (75 Hz/Q4, 50 Hz/Q4) is deferred, not cancelled.** The CASE 3 analysis
above still stands as the right next experiment *for the codec track*, but the
project objective has changed to maximum final-song quality within a fixed
100-hour budget, and the custom codec is not currently on the critical path to
that. Stage 2 should be run only if a downstream SongForge experiment shows the
custom codec actually gating final product quality.

All Stage 1 evidence is preserved and unmodified.

M05 — Musical Representation & Tokenization **not started**.

Related: [M04_DATA_EXPANSION.md](M04_DATA_EXPANSION.md),
[M03_RESULT_FROZEN.md](M03_RESULT_FROZEN.md) (unmodified).
