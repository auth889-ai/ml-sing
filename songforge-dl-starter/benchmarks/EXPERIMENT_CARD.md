# Experiment card — SongForge V1 (first intervention)

**Status: COMPLETE — filled 2026-08-18 from the partial listening review.**
One field (trainable parameter count) is recorded at trainer init, before any
optimization step; everything else is fixed here. Training may be prepared and
launched against this card. Do not edit after the run starts.

| field | value |
| --- | --- |
| hypothesis | A LoKr adapter trained on 100 balanced multi-instrument Slakh songs at native 44.1 kHz improves instrument realism and overall quality on the six weak frozen tracks without regressing the two strong ones or broad free-form capability. |
| selected weakness | Broad realism/general quality: area E scored 4.38/10 over 8 cells — the only area with sufficient evidence. Six of eight tracks (piano, guitar, rock, edm, cinematic, vocal) at ~3.5; violin and rich_mix at ~7. |
| evidence from listening | Coarse overall_realism scores only (listener, 2026-08-18): violin 7, rich_mix 7, all others 3.5. Per-dimension cells deliberately left blank — areas A–D report insufficient evidence and no missing score was invented. |
| objective corroboration | piano measured band-limited (95% rolloff 1406 Hz) — corroborates fidelity deficit on the weakest sparse track. Whisper lyric recall 95.7/95.2% with all sections in order — lyric *dropping* is not the problem, deprioritizing Case C. No clipping/NaN/looping on any track — gross artifacts unlikely dominant. Violin rolloff 5484 Hz yet scored 7 — the rolloff metric alone is not the quality story. |
| cause attribution | The strong tracks are the internal control: violin and rich_mix were generated with the *identical* inference settings, seed discipline, and pipeline as the six weak tracks. Settings-level causes (steps/shift/dtype) would degrade all eight, so the split points at domain/timbre coverage of the pretrained model — weak on sparse piano/guitar, band textures, synths, and brass/percussion-led orchestral; strong on lush string textures. Classified **LoRA-addressable (broad instrument realism)**; prompt-adherence and artifact contributions cannot be excluded (unscored) but cannot be first targets on this evidence. |
| intervention | Case A — broad instrument-realism LoKr (`configs/lora/caseA_slakh_instrument.yaml`). NOT Case B (rich_mix's vocal scored 7 while four instrumentals scored 3.5 — not vocal-dominant), NOT Case D (dense tracks are as weak as sparse ones — not sparse-only), NOT Case C (recall is 95%+ and adherence is unscored — no evidence), NOT a settings sweep alone (refuted by the internal control above). |
| dataset | `slakh100` — 100 tracks (80/10/10 from official redux splits), balanced over 14 instrument families by metadata quotas, **native 44.1 kHz** (a realism adapter must never train on 16 kHz sources — it would learn the band-limit it exists to fix). CC-BY-4.0, deployable line. Smallest set that covers all weak domains; BabySlakh rejected (16 kHz), full Slakh rejected (not needed for V1). |
| train/val/test split | Song-disjoint via official redux split directories: 80 train / 10 val / 10 test tracks; quotas per `configs/datasets/slakh100.yaml`. |
| adapter method | LoKr, dim 64, alpha 128, lr 0.03, batch 1, grad-accum 4, bf16, gradient checkpointing, on ACE-Step 1.5 XL-turbo (XL adapters only — 2B is incompatible). |
| trainable parameter count | Recorded from the trainer's init printout at step 0 into this row before optimization begins (LoKr dim-64 order of magnitude: tens of millions on the XL DiT). |
| estimated GPU / runtime / storage | Colab L4 (T4 forbidden — bf16 NaN). Download ~104.3 GB transient to local disk (resumable `wget -c`), ~5 GB kept, ~16 GB processed, ~22 GB peak Drive, ~130 GB peak Colab disk. Training ≤ 8000 steps, est. 8–14 h wall across sessions, atomic checkpoints every 30 min, `--resume-from` after every drop. |
| acceptance metric | Frozen-8 rerun (adapter the only change): mean overall of the six weak tracks improves **≥ 1.0** on a blind listen; piano 95% rolloff rises above 1406 Hz; Whisper recall within 2 points of 95.7/95.2. |
| rejection criterion | violin or rich_mix drops **> 0.5** overall; OR any new objective flag (clipping/NaN/looping); OR after frozen-8 passes, the 51-prompt Generalization Benchmark (held-out tier scored first-touch) shows broad degradation — objective scorecard regressions or a failed blind spot-check. Any of these kills V1 regardless of weak-track gains. |
| frozen baseline comparison | `foundation_benchmarks/ace_step_15_baseline_frozen/` — same 8 prompts, lyrics, seed 20260818, 60 s, bf16, 8 steps, shift 3.0; only the adapter varies. |

Rules carried over from `docs/FINETUNING_PLAN.md`:

- The baseline directory is read-only and never regenerated.
- The adapter is kept only if the ablation shows improvement without
  unacceptable regression ("keep-if-better").
- Held-out generalization prompts are never used for training or prompt
  tuning (CI-enforced); they are scored only at evaluation time.
- The goal is broad free-form quality, never optimizing for the eight
  benchmark songs.
