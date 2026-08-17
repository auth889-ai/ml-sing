# M03 — Neural Audio Codec & Discrete Audio Representation: FROZEN RESULT

**Status: PASS.** This document is the frozen record of the authoritative acceptance run.
It is evidence, not a working note: do not edit the numbers. Later milestones cite this file
as the M03 control.

## Authoritative run

| | |
| --- | --- |
| run_id | `colab-m03-final-authoritative-45ad1eec79c8` |
| run_label | `colab-m03-final-authoritative` |
| config fingerprint | `f9965a9bf8dd6d01` |
| validation probe fingerprint | `ae5bd101532b86ad` (64 held-out segments) |
| config | `configs/codec/codec_m03_tiny.yaml` |
| output | `$DRIVE_ROOT/outputs/codec_m03_acceptance/colab-m03-final-authoritative/` |
| `acceptance_pass` | **true** |

Executed across several Colab sessions as **one logical experiment**, with strict RNG /
optimizer / AMP-scaler restoration on resume and crash-tail reconciliation.

## Execution

Tesla T4, 14.563 GB VRAM, CUDA 12.8, PyTorch 2.11.0+cu128. 4000 steps completed
(0–3999). 4 observed runtime interruptions, 3 successful resumes. Peak VRAM 0.044 GB.

## Integrity

Curve 4000 rows / 4000 unique steps / **0 duplicates** / **0 gaps**; RVQ history 161 rows,
0 duplicates; a single run_id in both; `isolation.ok = true`; RNG restored on every
authoritative resume.

## Data

Real BabySlakh (CC-BY-4.0, Zenodo 4603844) through the
M02 — Audio Preprocessing & Dataset Pipeline canonical manifests.
`used_m02_manifest = true`. 7274 train / 992 validation segments, song-disjoint.

## Tests

Full suite 181 passed. `test_codec_cuda_smoke` **passed**, `test_codec_cuda_amp_smoke`
**passed** (executed, not skipped).

## Training (400-step windows)

| | first | last | change |
| --- | ---: | ---: | ---: |
| loss | 0.229098 | 0.203119 | −11.3% |
| waveform L1 | 0.091006 | 0.052604 | −42.2% |
| MR-STFT | 0.136125 | 0.084919 | −37.6% |

## Held-out validation — same 64 sample IDs before and after

| | before | after | change |
| --- | ---: | ---: | ---: |
| reconstruction loss | 0.281044 | 0.146376 | −47.9% |
| waveform L1 | 0.118988 | 0.060500 | −49.2% |
| MR-STFT | 0.162056 | 0.085876 | −47.0% |
| SNR dB | −2.810 | +3.636 | +6.45 dB |

`same_probe_before_and_after = true`, `validation_improved = true`.

## RVQ

Initial utilization 0.809 → minimum 0.0117 → final per-codebook [0.359, 0.406] on the
per-batch snapshot. **Temporary collapse steps 25–2225, recovered, final collapse `false`.**

Whole-evaluation figures: utilization **[1.0, 1.0]**, perplexity [80.54, 74.09], entropy
[4.389, 4.305], **0 dead codes** of 128 per codebook.

Per-batch utilization reads far lower than whole-evaluation utilization because a batch
contains only a few hundred latent frames. Compare snapshots to snapshots.

## Codec representation (the M04 control)

| | |
| --- | --- |
| latent frame rate | 120 Hz |
| downsample factor | 200 |
| codebooks (Q) | 2 |
| codebook size (K) | 128 |
| bits per code | 7 |
| bitrate | 1680 bps |
| PCM16 compression ratio | 228.57× |
| encode+decode | 0.00739 s (RTF 0.0074) |

Raw codec codes: 2400 per 10 s, 7200 per 30 s, 43 200 per 3 min.

## Artifacts

`checkpoint.pt`, `checkpoint_last.pt`, `checkpoint_latest.pt`, `run_manifest.json`,
`training_curves.csv`, `rvq_history.jsonl`, `validation_before.json`,
`validation_probe.json`, `metrics_summary.json`, `train_metrics.json`,
`validation_metrics.json`, `compression_stats.json`, `experiment_metadata.json`,
`listening_examples.json`, `m03_acceptance_report.json`, `EXPERIMENT_LOG.md`, and
`examples/character_{bass_heavy,harmonic,mixed,percussive}_{original,reconstructed}.wav`.

## Limitations carried into M04

- Trained on 11 BabySlakh songs (120 source files), not the full corpus.
- Objective metrics only; no listening test has been scored.
- 120 Hz means 43 200 raw codec codes for a 3-minute song, which is the long-form cost
  concern M04 — High-Quality Codec Optimization & Latent-Rate Selection must address.
