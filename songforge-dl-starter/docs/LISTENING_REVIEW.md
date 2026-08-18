# Listening review — ACE-Step 1.5 pretrained baseline

Eight generated songs need human ears. Everything measurable has already been
measured; what remains cannot be reached by any script we have, and is left
blank rather than guessed.

**Files:** `$DRIVE_ROOT/foundation_benchmarks/ace_step_15_baseline_frozen/`
(read-only control copy — the working copy is `ace_step_15/`)

**Sheet to fill:** `benchmarks/listening_review.csv`

## Scoring

All dimensions **1–10, where 10 is best**. `artifact_freedom` is phrased as
*freedom from* artifacts, so higher is better there too — no dimension is
inverted. Cell conventions: **blank = not scored yet** (never a zero),
**N/A = does not apply to this track** (pre-filled for the vocal dimensions on
the six instrumental tracks).

Required on every track (the first eight columns):

| dimension | question |
| --- | --- |
| overall_realism | does the whole thing sound like real music by real players, or obviously synthetic |
| instrument_realism | do the individual instruments sound like the real instrument |
| vocal_realism | does the voice sound human (`vocal`, `rich_mix` only; N/A elsewhere) |
| lyrics_intelligibility | can you make out the words (N/A on instrumentals) |
| prompt_adherence | did it make the song that was asked for |
| structure_coherence | does it hold together musically over 60 s — sections, transitions, an actual song shape |
| spectral_clarity | bright and full-range, or muffled/dull as if behind a curtain (the measured band-limit flag — listen for this especially on piano and violin) |
| artifact_freedom | freedom from noise, distortion, glitches, warbles (10 = clean) |

Optional diagnostic columns — fill only if something stands out; they sharpen
which fix gets picked but never block the analysis:

| dimension | question |
| --- | --- |
| instrument_presence | are the *requested* instruments actually audible |
| arrangement | rich and layered, or thin and simple |
| phrasing | natural, connected vocal phrasing or mechanical and segmented |
| pitch_stability | steady pitch, or drifting and unstable |
| emotion | is there a performance, or is it flat |

The scoring rubric also sits beside every player on the private listening page
(the "SongForge First Eight" artifact), so scoring can happen while listening
without switching windows. The page carries the same frozen audio — nothing was
regenerated.

## What was requested, per track

| id | prompt summary | requested instruments | lyrics |
| --- | --- | --- | --- |
| piano | emotional grand piano ballad, 68 BPM, F minor | piano | — |
| violin | romantic orchestral, expressive solo violin, 76 BPM, D minor | violin, strings | — |
| guitar | acoustic fingerpicked folk, 92 BPM, G major | acoustic guitar, bass, percussion | — |
| rock | modern rock, distorted riffs + solo, 132 BPM, E minor | electric guitar, bass, drums | — |
| edm | EDM with build and drop, 128 BPM, A minor | synth, bass, drums | — |
| cinematic | epic orchestral, 90 BPM, C minor | strings, piano, brass, percussion | — |
| vocal | contemporary pop, clear female lead, 100 BPM, C major | vocals, piano, bass, drums | ✅ verse/chorus/bridge |
| rich_mix | cinematic pop rock, 96 BPM, A minor | vocals, piano, violin, strings, electric guitar, bass, drums, synth | ✅ verse/chorus |

## Listening in Colab

```python
from IPython.display import Audio, display
import json, os
B = "/content/drive/MyDrive/songforge-dl/foundation_benchmarks/ace_step_15"
for name in ["piano","violin","guitar","rock","edm","cinematic","vocal","rich_mix"]:
    print("=== " + name)
    display(Audio(os.path.join(B, name + ".wav")))
```

## What measurement already told us

Do not let these anchor the scores — they are context, and two of them are the
specific things worth listening *for*.

- **`piano` and `violin` measured band-limited** (95% spectral rolloff 1406 Hz
  and 5484 Hz, against 12–14 kHz for the dense prompts). If they sound muffled,
  that corroborates. If they sound fine, the metric is misleading on sparse
  material and should be recalibrated.
- **Lyrics were recovered at ~95%** word recall by Whisper on both vocal tracks,
  with all sections present and in order. That measures *intelligibility to a
  transcriber*, which is not the same as sounding good — phrasing, emotion and
  vocal realism are exactly what the transcriber cannot tell us.
- No clipping, no NaN, correct 60.0 s duration, real stereo, non-looping on all
  eight.

## After scoring

```bash
python scripts/listening_review.py \
    --sheet benchmarks/listening_review.csv \
    --objective "$DRIVE/foundation_benchmarks/ace_step_15/objective_analysis.json"
```

Produces the ranked weaknesses that select the first fine-tuning experiment —
see [FINETUNING_PLAN.md](FINETUNING_PLAN.md).
