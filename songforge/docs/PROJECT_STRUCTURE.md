# SongForge project structure

How the pieces fit, where things live, and the rules that keep a
multi-dataset, multi-adapter project from collapsing into an unattributable
mess. The organising principle: **one canonical manifest schema in the middle**,
with corpus-specific readers on one side and trainer-specific writers on the
other, so adding a dataset never means touching training and adding a foundation
never means touching data.

```
  corpus                    canonical                      trainer
  ────────                  ─────────                      ────────
  Slakh2100  ─┐                                        ┌─ ACE-Step LoRA
  FMA        ─┤   adapters   songforge.audio.v1  gates │
  NSynth     ─┼──────────▶   AudioRecord (JSONL) ──────┼─ our codec (M03/M04)
  vocal sets ─┤                                        │
  ...        ─┘                                        └─ evaluation
```

---

## 1. Repository

```
songforge/
├── src/songforge/
│   ├── data/                    # corpus → canonical manifest
│   │   ├── manifest.py          # AudioRecord: THE schema. One, project-wide.
│   │   ├── registry.py          # dataset registry: ids, licences, URLs
│   │   ├── preprocess.py        # decode, resample, segment, provenance
│   │   ├── splits.py            # song/singer-disjoint, quota|weighted|hash
│   │   ├── dedup.py             # content hashing, cross-split duplicates
│   │   ├── media.py, dsp.py     # decoding backends, pure-torch resampling
│   │   └── slakh_metadata.py    # ← per-corpus reader (add one per dataset)
│   ├── generation/              # our control surface over a foundation
│   │   ├── request.py           # SongRequest: prompt, lyrics, controls
│   │   ├── capabilities.py      # NATIVE | PROMPT | IGNORED, per control
│   │   ├── adapter.py           # FoundationAdapter + LicensePosition
│   │   └── adapters/            # ← one per foundation (acestep, null, …)
│   ├── training/                # run isolation, atomic + resumable checkpoints
│   ├── evaluation/              # objective metrics, song scorecard
│   ├── models/codec/            # our M03/M04 research codec
│   └── milestones.py
├── scripts/                     # every CLI, no logic hidden in notebooks
├── configs/
│   ├── codec/                   # M03/M04 codec configs
│   ├── lora/                    # ← one YAML per LoRA experiment
│   └── datasets/                # ← one recipe per dataset build
├── benchmarks/                  # prompts.yaml + listening sheets
├── docs/  (+ docs/adr/)
└── tests/
```

**Rules.**

1. `AudioRecord` is the only manifest schema. A new corpus gets a *reader* in
   `data/`, never a competing record type.
2. A new foundation gets an adapter in `generation/adapters/`. It must declare
   capabilities and licence, and tests assert the declaration.
3. Anything runnable is a script with `--help`. Notebooks orchestrate; they
   never hold logic.
4. Nothing branches on an instrument or genre name to synthesise audio. Those
   names are conditioning text and evaluation targets only.

## 2. Google Drive (200 GB budget)

Drive holds what must survive a runtime drop. `/content` holds what must be
fast. Nothing is stored in both.

```
MyDrive/songforge-dl/
├── data/
│   ├── raw/<dataset>/                     # downloaded corpora
│   ├── processed/<dataset>_<variant>/
│   │   ├── manifests/{all,train,val,test}.jsonl
│   │   ├── manifest_summary.json
│   │   └── dataset_gate.json              # the seven gates
│   └── lora/<experiment>/                 # ACE-Step trainer layout
│       └── DATASET_CARD.json              # goal, licences, provenance
├── foundation_benchmarks/
│   ├── ace_step_15_baseline_frozen/       # CONTROL. read-only. never regenerate.
│   └── <experiment>/                      # treatment, same 8 prompts + seed
├── checkpoints/<experiment>/              # LoRA weights + resume state
└── outputs/                               # final songs
```

**Storage discipline.** Decoded/resampled working copies and training shards
live in `/content` and are disposable. Drive keeps raw sources, manifests,
checkpoints, evidence and final audio. If storage binds, we buy training hours
and data diversity, not redundant copies.

## 3. The dataset pipeline, and where the gates sit

```
download ──▶ preprocess_dataset.py ──▶ manifests ──▶ dataset_gate.py ──▶ build_acestep_lora_dataset.py
   │              │                                        │                        │
   │              └ provenance, licence,                    │                        └ licence gate again,
   │                song-disjoint splits,                   │                          captions from metadata
   │                duplicate hashes                        │
   └ report size/licence/storage BEFORE                     └ LICENSE PROVENANCE DUPLICATE QUALITY
     any large download                                       METADATA SPLIT-LEAKAGE ACE-STEP
```

`dataset_gate.py` returns one of three verdicts, and the distinction is the
whole point:

- **ALL GATES PASS** → usable for the deployable adapter.
- **RESEARCH ONLY** → everything passes except LICENSE. Usable for a clearly
  labelled research adapter, **never merged** with the deployable one.
- **BLOCKED** → fix it before training.

## 4. Adapter architecture

One LoRA is not asked to learn everything. Each is a separate experiment with
its own dataset, its own ablation, and its own licence status.

```
ACE-Step 1.5 XL-turbo   (PRETRAINED, MIT)
        │
        ├── instrument/arrangement LoRA   ← Slakh2100, CC-BY-4.0, deployable
        ├── vocal LoRA                    ← licence-dependent; may be research-only
        ├── genre/style LoRA              ← only if genre scores as a weakness
        │
        └── SongForge control layer       (BUILT BY US)
                 │
                 └── final_song.wav
```

An adapter is added **only if ablation proves it helps**, and is dropped if it
does not. Two adapter lines are kept permanently separate:

| line | data | may be deployed |
| --- | --- | --- |
| `SongForge-Permissive` | CC0 / CC-BY only | yes |
| `SongForge-Research-*` | includes NC / unclear data | no — research evidence only |

Their legal status is never merged, and a checkpoint never silently moves
between lines.

## 5. Experiment naming and isolation

Every training or generation run gets its own directory and its own `run_id`,
reusing the isolation machinery built for M03/M04:

```
<track>_<goal>_<variant>       e.g. lora01_slakh_instrument_r16
```

- fresh-run guard refuses to write into a non-empty directory
- `run_id` stamped on every curve and history row
- atomic checkpoints, strict RNG/optimizer/scaler restore, resumable
- a fairness audit re-derives from artifacts that candidates differed in exactly
  one thing

That machinery already caught a contaminated run once; it stays.

## 6. Attribution, kept current

`docs/ATTRIBUTION.md` maintains three lists, and nothing moves from PRETRAINED
to TRAINED BY US merely because we ran inference with it:

- **PRETRAINED** — ACE-Step 1.5 (MIT/MIT), Whisper (evaluation only, never in
  the generation path)
- **TRAINED BY US** — M03 codec, M04 candidates, and every LoRA we train
- **BUILT BY US** — data pipeline, manifests, control surface, evaluation,
  experiment tracking, integration, UI

## 7. Toolchain

All free and local: PyTorch, torchaudio, diffusers, Hugging Face Hub, official
ACE-Step tooling, FFmpeg, soundfile, numpy/scipy, Whisper for lyric recall,
Gradio for UI, and JSON/CSV/Markdown for tracking. No paid SaaS where a local
open tool does the same job.

Explicitly avoided: **Side-Step** (CC BY-NC-SA despite being the popular
low-VRAM trainer) — the official ACE-Step Gradio/LoKr path is MIT.

## 8. Current status

| stage | state |
| --- | --- |
| M00–M02 | complete |
| M03 codec | **frozen**, PASS |
| M04 Stage 1 | complete; Stage 2 deferred, not cancelled |
| foundation | ACE-Step 1.5 XL-turbo, provisional primary |
| baseline | 8 songs generated, verified, **frozen as control** |
| listening review | **awaiting scores** — gates the first LoRA |
| Slakh expansion | awaiting size/licence report before download |
| permissive-data research | running in background |
