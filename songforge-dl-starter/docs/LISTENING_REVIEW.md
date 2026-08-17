# Listening review — ACE-Step 1.5 pretrained baseline

Eight generated songs need human ears. Everything measurable has already been
measured; what remains cannot be reached by any script we have, and is left
blank rather than guessed.

**Files:** `$DRIVE_ROOT/foundation_benchmarks/ace_step_15_baseline_frozen/`
(read-only control copy — the working copy is `ace_step_15/`)

**Sheet to fill:** `benchmarks/listening_review.csv`

## Scoring

All dimensions **1–10, where 10 is best**. `artifacts` is phrased as *freedom
from* artifacts, so higher is better there too — no dimension is inverted.
Leave a cell blank if you would rather not score it; blank means unscored, not
zero, and the summary reports what is missing.

Every track:

| dimension | question |
| --- | --- |
| overall | how good is this as a piece of music |
| realism | does it sound like real instruments or obviously synthetic |
| instrument_presence | are the *requested* instruments actually audible |
| arrangement | is it rich and layered, or thin and simple |
| prompt_adherence | did it make the song that was asked for |
| artifacts | freedom from noise, distortion, glitches (10 = clean) |
| coherence | does it hold together musically over 60 s |

`vocal` and `rich_mix` additionally:

| dimension | question |
| --- | --- |
| vocal_realism | does the voice sound human |
| lyric_intelligibility | can you make out the words |
| phrasing | natural, connected phrasing or mechanical and segmented |
| pitch_stability | steady pitch, or drifting and unstable |
| emotion | is there a performance, or is it flat |

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
