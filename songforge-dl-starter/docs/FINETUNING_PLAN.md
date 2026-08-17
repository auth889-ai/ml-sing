# SongForge fine-tuning plan — weakness-driven, gated on listening

**Status: prepared, not started.** No training runs until the listening review
is recorded. This document is deliberately contingent: it maps each *possible*
weakness to the intervention that would address it, so that once the scores
exist the first experiment can start immediately without another planning round.

Foundation: **ACE-Step 1.5 XL-turbo**, provisional primary. Not re-opened unless
listening reveals a material problem.

---

## 1. The gate

`benchmarks/listening_review.csv` must be filled in first.
`scripts/listening_review.py` refuses to run on an empty sheet, on purpose: a
fabricated weakness ranking would aim the entire fine-tuning phase at the wrong
problem, and every hour after that would compound the error.

```bash
python scripts/listening_review.py \
    --sheet benchmarks/listening_review.csv \
    --objective "$DRIVE/foundation_benchmarks/ace_step_15/objective_analysis.json"
```

It prints the score table, ranks the dimensions worst-first, and corroborates
with the objective flags without blending the two.

## 2. The ablation, fixed in advance

The claim we have to be able to defend is *"our ML work improved the system"*,
so the comparison is specified now, before we know the result.

| | |
| --- | --- |
| control | `$DRIVE/foundation_benchmarks/ace_step_15_baseline_frozen/` — 8 WAVs, read-only, MD5s recorded |
| treatment | same eight prompts through the LoRA-adapted model |
| held constant | prompt set, seed 20260818, duration 60 s, dtype bf16, 8 steps, shift 3.0, GPU |
| varied | the LoRA adapter alone |
| measured | objective scorecard, Whisper lyric recall, and a fresh blind listening pass |

The baseline directory is frozen and write-protected. It is never regenerated,
because a control that moves is not a control — the same discipline the M04
fairness audit enforced.

## 3. Weakness → intervention map

Selected once the scores land. Readiness matters as much as fit: with a fixed
budget, an intervention that can start today beats a better one that needs a
200 GB download first.

| if the weakness is | likely cause | intervention | data | licence | ready? |
| --- | --- | --- | --- | --- | --- |
| **requested instruments not audible** | weak caption→instrument association | LoRA on labelled stems with precise instrument captions | Slakh2100 stems, 14 families | CC-BY-4.0 ✅ | **now** |
| **piano muffled / dull** | poor sparse-acoustic rendering | LoRA on solo piano stems | Slakh2100 `Piano` (2,510 segs) | CC-BY-4.0 ✅ | **now** |
| **violin/strings synthetic** | limited string realism | LoRA on string stems | Slakh2100 `Strings` + `Strings (continued)` (2,335) | CC-BY-4.0 ✅ | **now** |
| **drums weak / abrupt** | percussion rendering and fills | LoRA on drum stems | Slakh2100 `Drums` (2,167) | CC-BY-4.0 ✅ | **now** |
| **arrangement too simple** | defaults to sparse textures | LoRA on full mixes with multi-instrument captions | Slakh2100 `Mixture` (2,376) | CC-BY-4.0 ✅ | **now** |
| **genre adherence poor** | caption conditioning too weak | LoRA on genre-labelled audio | MTG-Jamendo | per-track CC, must filter | needs download |
| **vocals flat / no emotion** | expressiveness not learned | LoRA on expressive singing | GTSinger | 🚩 **CC-BY-NC** | see §5 |
| **lyrics dropped** | syllable/duration mismatch | **prompt-side first**: pad duration, shorten lines, blank lines between tags | none | — | **now, free** |
| **poor long-form structure** | >2 min coherence | **measure first** at 120/180/240 s before assuming a training fix | none | — | **now, free** |

Two entries deliberately start with something other than training. Dropped
lyrics and long-form structure are cheap to characterise and may not need a LoRA
at all; spending GPU hours on a problem that a prompt change fixes would be the
single most wasteful thing we could do with the remaining budget.

## 4. Why Slakh2100 is the default first dataset

Not because it is best in the abstract, but because it is *ready*:

- **Already preprocessed.** 20 tracks, 229 WAVs, 17,558 segments, canonical
  manifests, song-disjoint splits, zero leakage — built and verified in M04.
- **Already labelled.** 14 instrument families read from `metadata.yaml`, 100%
  coverage, real labels rather than spectral guesses.
- **CC-BY-4.0** — permissive, so a checkpoint trained on it stays unencumbered.
- Instrument-level stems are exactly the right shape for the instrument-realism
  and instrument-presence weaknesses, which are the most likely findings given
  the objective band-limiting already measured on `piano` and `violin`.

`scripts/build_acestep_lora_dataset.py` converts our manifests into ACE-Step's
training layout, deriving captions **only** from corpus metadata:

```
Track00001__S00.wav  +  .caption.txt "solo Acoustic Grand Piano"
                        .lyrics.txt  (empty, explicitly)
                        .json        {caption, language}
```

## 5. Datasets we are NOT downloading yet, and why

**Lakh MIDI — no path into ACE-Step.** ACE-Step has *no melody and no chord
conditioning*; both were closed upstream as not-planned. Symbolic melody and
harmony data therefore has nowhere to enter this foundation. Lakh only becomes
useful if we build our own conditioning stage, which is a much later decision.
Downloading it now would cost storage and hours for zero effect on song quality.

**GTSinger — would encumber our checkpoint.** It is CC-BY-NC, as is nearly every
open singing corpus (OpenSinger, Opencpop, M4Singer, ACE-Opencpop). A LoRA
trained on it is non-commercial *regardless* of ACE-Step's MIT weights. If vocal
expressiveness scores as a top-3 weakness, the options are: (a) accept it as an
explicitly-labelled research-only adapter, or (b) find permissive vocal data.
This is a decision for you, not one to make silently.
`build_acestep_lora_dataset.py` refuses non-permissive corpora unless
`--allow-nonpermissive` is passed.

**MTG-Jamendo — only if genre adherence scores badly.** Large download, per-track
licences that must be filtered individually. Our manifest schema already carries
per-record licence so the gate works, but the cost is only justified by a scored
weakness.

**No blind merging.** Each dataset gets its own manifest, its own
`DATASET_CARD.json` recording goal, licences and provenance, and its own adapter.
Mixing corpora would make it impossible to attribute an improvement to anything.

## 6. Training path

Official ACE-Step LoRA/LoKr only:

- in-repo Gradio "Train LoRA" tab, `POST /v1/training/start_lokr`, or the
  `training_v2` module
- LoRA lr 1e-4; LoKr lr 0.03, `lokr_linear_dim` 64, alpha 128 — LoKr is roughly
  10× faster and is the sensible default under time pressure
- batch 1, grad-accum 4, `save_every_n_epochs 5`, `--resume-from` for Colab drops
- ~16 GB minimum, 20 GB recommended; the L4's 23 GB is inside the recommended tier
- **2B and XL adapters are not interchangeable.** We are on XL, so that is fixed
  before any training run.

🚩 **Side-Step is CC BY-NC-SA**, not MIT, despite being the popular low-VRAM
trainer. It is not used.

## 7. Selection rule for the first experiment

Pick the weakness that is highest-ranked **and** satisfies all three:

1. addressable by a LoRA on audio (not by a control we do not have),
2. has permissive, already-prepared data,
3. fits a single Colab session with resumable checkpoints.

If the top-ranked weakness fails (2) or (3), take the next one and record why —
the same rule that deferred M04 Stage 2 rather than pretending it was cancelled.

## 8. Out of scope

- Custom codec research — not reopened unless downstream work genuinely needs it.
- Benchmarking further foundations — not unless listening finds a material problem.
- Any claim of professional, universal, or Suno-comparable quality.
