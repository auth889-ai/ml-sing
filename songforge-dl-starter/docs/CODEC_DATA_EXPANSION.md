# M04 — High-Quality Codec Optimization & Latent-Rate Selection: data expansion

The M04 candidate comparison is not run on the M02 acceptance slice. That slice
was 11 songs and 120 files, sized for a milestone gate rather than for choosing a
representation the rest of the project has to live with. A codec selected there
would be a choice about 11 songs, not a project-wide choice.

The M02 acceptance manifests remain **frozen and unmodified** at
`$SONGFORGE_DATA/processed/babyslakh_m02/`. M04 writes to its own directory,
`$SONGFORGE_DATA/processed/babyslakh_m04_expanded/`.

## Build command

```bash
python scripts/preprocess_dataset.py \
    --dataset-id babyslakh \
    --input-dir  "$SONGFORGE_DATA/raw/babyslakh/babyslakh_16k" \
    --output-dir "$SONGFORGE_DATA/processed/babyslakh_m04_expanded" \
    --instrument-metadata slakh \
    --split-strategy weighted
```

No `--limit-files`: the whole available corpus is used. Wall clock on a Colab T4
with Drive-backed output: 13 minutes.

## What the corpus contains

| | count |
| --- | ---: |
| source tracks | 20 |
| source WAVs | 229 |
| segments | 17,558 |
| audio | 585.27 min (9.75 h) |
| segments carrying a real instrument label | 100.0% |

Against the frozen M02 slice this is roughly 1.8× the songs, 1.9× the source
files and 1.9× the segments.

### Archive residue

The download is an unzipped macOS-created archive, so it carries AppleDouble
stubs (`._name.wav`) and a `__MACOSX` tree. These have a `.wav` suffix, are not
audio, and sort *before* the real files — an unfiltered limited run would read
nothing but junk. `find_audio_files` now excludes both. In this corpus the filter
removed 0 files inside `babyslakh_16k` itself, and the guard is regression-tested
rather than left to luck.

## Splits

Whole songs, split by usable duration rather than by song count, so one long song
cannot quietly skew a split.

| split | tracks | WAVs | segments | minutes | segments % | duration % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 16 | 182 | 13,645 | 454.83 | 77.71 | 77.71 |
| val | 2 | 22 | 2,056 | 68.53 | 11.71 | 11.71 |
| test | 2 | 25 | 1,857 | 61.90 | 10.58 | 10.58 |

Requested 80/10/10; achieved 77.7/11.7/10.6. The gap is the cost of keeping songs
intact — with 20 songs the smallest movable unit is 5% of the corpus.

### Integrity

| check | result |
| --- | --- |
| song leakage | none |
| duplicate audio across splits | none |
| licence + provenance on every record | complete |
| segment ids stable across rebuilds | yes (content-derived) |

Re-checkable independently at any time:

```bash
python scripts/report_codec_dataset.py \
    --manifests "$SONGFORGE_DATA/processed/babyslakh_m04_expanded/manifests"
```

## Instrument metadata

Labels are read from each track's `metadata.yaml`, which Slakh ships. All 20
tracks have one. Nothing is inferred from the audio: a spectral heuristic can
separate "bass-heavy" from "percussive", but it cannot tell a piano from a
guitar, and a guessed label would put fiction into the evaluation record. Stems
without an entry stay unlabelled.

| family | segments | | family | segments |
| --- | ---: | --- | --- | ---: |
| Guitar | 2,902 | | Synth Pad | 682 |
| Piano | 2,510 | | Organ | 623 |
| Mixture (rendered full mix) | 2,376 | | Pipe | 408 |
| Bass | 2,265 | | Brass | 402 |
| Strings (continued) | 2,228 | | Chromatic Percussion | 349 |
| Drums | 2,167 | | Synth Lead | 295 |
| | | | Reed | 244 |
| | | | Strings | 107 |

14 families. Slakh splits strings across `Strings` and `Strings (continued)`;
both are accepted as strings because both names come from the corpus.

**These labels describe data only.** No generation path branches on them. Per the
project goal, instrumentation must be learned from data and conditioning, never
implemented as fixed instrument branches.

## Effect on listening examples

Held-out examples are now selected by real instrument category — piano, guitar,
bass, percussion, strings, full mix — instead of only by spectral character.
Selection is deterministic within a family, so every codec candidate is compared
on identical source audio. A category absent from the held-out split is reported
as unavailable; it is never filled in with something that merely sounds similar.

Because the validation split is 2 songs, category coverage there is a property of
those songs and is reported per run rather than assumed.

## What this does not decide

Nothing. This is corpus preparation. The codec is **not frozen**, and no latent
rate or quantizer depth is selected here. Selection happens only after the
authoritative Stage 1 runs at equal budgets, and is recorded separately.

Related: [CODEC_RESULTS_FROZEN.md](CODEC_RESULTS_FROZEN.md) — the frozen
M03 — Neural Audio Codec & Discrete Audio Representation result, which this work
does not modify.
