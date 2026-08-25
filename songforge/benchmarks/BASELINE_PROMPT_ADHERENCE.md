# Baseline prompt adherence — frozen ACE-Step 1.5 XL-turbo

**This is the "before" measurement.** Every claim SongForge makes about
improving prompt-following is a delta against these numbers, so they are
recorded before any adapter exists rather than reconstructed afterwards.

## Conditions

| | |
| --- | --- |
| model | ACE-Step 1.5 XL-turbo, frozen, **no adapter** |
| hardware | NVIDIA L4, 23 GB, compute capability 8.9 |
| precision | bf16 (native at cc >= 8.0) |
| seed | 20260824, identical across prompts |
| duration / steps / guidance | 30 s / 60 / 7.0 |
| metric | CLAP (`laion/clap-htsat-unfused`) cosine similarity, audio vs prompt |
| render cost | 38 s for four 30 s tracks (~9.5 s each) |

## Result

| prompt | CLAP | peak | RMS | dynamics |
| --- | ---: | ---: | ---: | ---: |
| A — cinematic piano + violin | 0.578 | 0.500 | 0.064 | 0.67 |
| B — prog rock, violin lead | 0.618 | 0.776 | 0.134 | 0.41 |
| C — sparse acoustic | 0.560 | 0.699 | 0.073 | 0.70 |
| **D — dark electronic cinematic** | **0.284** | 0.796 | 0.123 | 0.81 |

## The finding

Three acoustic/orchestral prompts cluster at **0.56–0.62**. The electronic
prompt scores **0.284** — roughly half — while being perfectly healthy as
audio: its peak and RMS are the highest of the four and its dynamics the most
varied. So the model produced a confident, well-formed track that matches the
request far less closely than the others do.

That is a conditioning failure, not a fidelity failure, and it is consistent
with ACE-Step's published profile: it beats Suno v5 on overall SongEval
quality (8.12 vs 7.87) while trailing on style alignment (39.1 vs 46.8) and
lyric alignment (26.3 vs 34.2). Better sound, worse obedience.

## What this licenses us to claim, and what it does not

Four prompts is a signal, not a result. It says where to look; it cannot
support "we improved prompt adherence by X%". The generalization benchmark
(`benchmarks/generalization_prompts.yaml`, ~50 prompts with a CI-enforced
held-out tier) is what a published number has to come from, and
`scripts/measure_prompt_adherence.py` reports INCONCLUSIVE whenever the
bootstrap interval spans zero regardless of how good the mean looks.

## Hypothesis this motivates

If the gap is conditioning rather than capacity, then training an adapter on
narrative captions in the foundation's native distribution -- over a corpus
that deliberately includes electronic material -- should raise D more than it
raises A, B or C. A uniform lift would suggest something else is going on, and
a drop in A-C for a gain in D is a trade to be reported as such, not an
improvement.
