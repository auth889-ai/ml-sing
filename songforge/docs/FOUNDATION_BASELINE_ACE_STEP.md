# ACE-Step 1.5 foundation baseline

Pretrained baseline, no fine-tuning. This is the gate the foundation decision
rests on: **real generated audio**, not README claims.

## Configuration

| | |
| --- | --- |
| model | `ACE-Step/acestep-v15-xl-turbo-diffusers` — 4B XL DiT, turbo (guidance-distilled) |
| runtime | diffusers `AceStepPipeline` 0.39.0 (DiT half; no LM planner) |
| code licence | **MIT** |
| weights licence | **MIT** |
| GPU | NVIDIA L4, 23,034 MiB, compute capability 8.9 |
| dtype | bfloat16 |
| inference steps | 8 (turbo default) · shift 3.0 · CFG ignored by design on turbo |
| seed | 20260818, identical across all eight |
| duration | 60 s requested, 60.0 s delivered on all eight |
| **peak VRAM** | **15.38 GB** — 67% of the L4, no offload, no OOM |
| output | `$DRIVE_ROOT/foundation_benchmarks/ace_step_15/` |

Generation was 3.1–4.8 s per 60-second song (**RTF 0.05–0.08**, ~15–20× faster
than realtime). All eight in about 26 seconds of GPU time.

## The eight outputs

`piano.wav` `violin.wav` `guitar.wav` `rock.wav` `edm.wav` `cinematic.wav`
`vocal.wav` `rich_mix.wav` — plus `metadata.json`, `objective_analysis.json`,
`asr_transcripts.json`, `lyric_recall.json`.

| id | gen s | sec | sr | ch | peak dB | rms dB | clip | silent | roll95 Hz | flatness | stereo corr | repetition | flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| piano | 3.9 | 60.0 | 48k | 2 | −2.22 | −20.68 | 0 | 0.034 | **1406** | 0.026 | 0.248 | 0.00 | **band-limited** |
| violin | 3.2 | 60.0 | 48k | 2 | −1.80 | −16.59 | 0 | 0.033 | **5484** | 0.053 | 0.481 | 0.04 | **band-limited** |
| guitar | 3.1 | 60.0 | 48k | 2 | −1.98 | −14.73 | 0 | 0.032 | 11836 | 0.065 | 0.723 | 0.10 | none |
| rock | 3.1 | 60.0 | 48k | 2 | −1.09 | −14.23 | 0 | 0.033 | 13664 | 0.105 | 0.825 | 0.12 | none |
| edm | 3.1 | 60.0 | 48k | 2 | −2.39 | −14.95 | 0 | 0.038 | 14367 | 0.163 | 0.597 | 0.34 | none |
| cinematic | 3.1 | 60.0 | 48k | 2 | −2.98 | −12.81 | 0 | 0.037 | 9281 | 0.077 | 0.464 | 0.17 | none |
| vocal | 3.2 | 60.0 | 48k | 2 | −1.61 | −16.31 | 0 | 0.039 | 13406 | 0.120 | 0.776 | 0.10 | none |
| rich_mix | 3.2 | 60.0 | 48k | 2 | −2.78 | −17.74 | 0 | 0.038 | 11719 | 0.070 | 0.553 | −0.01 | none |

**Zero NaN/Inf samples across all eight files.** Zero clipped samples. Every file
is genuine 48 kHz stereo with a real stereo image (no dual-mono). Repetition
scores 0.00–0.34 indicate the material develops rather than looping. Flatness
0.026–0.163 confirms tonal music, well below the 0.5 noise threshold.

## Lyrics-to-song: verified, not assumed

Instrumental success proves nothing about singing, so this was tested directly
by transcribing the output with Whisper-small and comparing against the lyrics
that were supplied.

| track | unique supplied words recovered | recall |
| --- | ---: | ---: |
| `vocal` | 44 / 46 | **95.7%** |
| `rich_mix` | 20 / 21 | **95.2%** |

`vocal.wav` transcript — all three sections, in order, all ten lines:

> Street lights blur into the falling rain / I count the quiet miles again /
> Every door I close still knows my name / But I am not the same / So let it
> break, let it burn away / I am learning how to stay / Every ending taught me
> how to say / I will find another way / All the years I spent afraid / Turn to
> light / Turn to light

The only deviation is "closed" → "close", which may be the singer or the ASR.
`rich_mix` recovered all four supplied lines, with the opening line apparently
sung twice.

**Controls behaved correctly.** The instrumental prompts produced no lyrics at
all — Whisper fell into its characteristic hallucination on instrumental input
(`piano`: a looping "piano and piano play in bright rhythm"; `cinematic`: "thank
you for watching this video"). Neither returned any supplied lyric text, which
is what confirms the vocal tracks' recall is real signal rather than an artefact
of the transcriber.

This contradicts the dropped-lines complaint that dominates the project's issue
tracker. At 60 s with ten short lines the syllable density was comfortable; the
reported failure mode may need longer or denser lyrics to reproduce.

## What is NOT verified

Being explicit, because the foundation decision should not rest on inference:

- **I cannot listen to audio.** Every subjective dimension you asked for —
  musicality, instrument realism, vocal realism, phrasing, pitch stability,
  arrangement richness, artifacts — requires human ears. No score is recorded
  for them here, and none should be invented.
- **Instrument presence is not objectively verified.** Confirming that a violin
  is audible in `violin.wav` needs a music tagger (CLAP or PANNs) or a listener.
  The spectral profiles are *consistent* with the prompts — sparse solo material
  is dark, dense band material reaches 12–14 kHz — but consistency is not proof.
- The band-limited flags on `piano` and `violin` are a genuine objective finding
  and the most concerning result in the set. Caveat: in sparse quiet material
  most energy legitimately sits low, which inflates how severe a 95%-rolloff
  figure looks. A 1406 Hz rolloff for solo piano is still low enough that it
  should be audible as muffled, and it needs a listen to confirm.

## Strengths, on the evidence

1. **Lyrics-to-song genuinely works** at ~95% word recall on the L4/bf16 path.
   This is the single hardest capability and the one most models lack entirely.
2. **Speed is transformative for the 100-hour budget.** RTF 0.05 means a full
   evaluation sweep costs seconds, so ablations and A/B comparisons are
   effectively free. This changes what is achievable in the remaining time.
3. **Comfortable headroom on the L4** at 15.38 GB, leaving room for LoRA work
   without an A100.
4. **Technically clean output** — no clipping, no NaN, correct duration, real
   stereo, no dead air, non-looping.
5. **MIT on code and weights**, the most permissive position of any candidate.

## Weaknesses and open risks

1. **`piano` and `violin` render band-limited.** Needs a listen; if real, sparse
   acoustic material is a weak spot and a fine-tuning target.
2. **Genre, mood, instruments and vocal character are prompt text only** — no
   typed conditioning. Chord and melody conditioning do not exist at all. Real
   control over these is work SongForge must add, not something inherited.
3. Documented-but-unobserved here: weak vocal expressiveness, grid-locked
   timing, abrupt drum entries, harmonic stagnation on some genre captions.
4. **Training-data provenance is unaudited.** The commercial-safety claim is a
   vendor assertion the team declined to formalise.
5. Upstream project velocity is low (47 open PRs, near-zero commits mid-2026).

## Verdict

**ACE-Step 1.5 XL-turbo is strong enough to be SongForge's primary foundation**,
on this evidence: it sings supplied lyrics accurately, runs comfortably on
available hardware, is MIT on both code and weights, ships official LoRA/LoKr
training, and is fast enough to make experimentation cheap.

Two qualifications. The verdict is **provisional pending a listening pass** on
the eight files, particularly `piano` and `violin`. And it is a verdict about
*suitability as a foundation*, not about quality — nothing here demonstrates
professional or universal quality, and no such claim is made.

No other foundation needs benchmarking unless listening reveals a material
problem.
