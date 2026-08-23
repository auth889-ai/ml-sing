# Foundation licence and hardware audit

Completed **before** downloading any weights.

The governing rule: **a repository's code licence tells you nothing about its
weights licence.** Meta ships MIT code with CC-BY-NC weights. Tencent ships one
custom licence over both that bars production use. A model card tag can be wrong.
Every row below records the two separately, and where possible the licence text
was read directly rather than taken from a badge.

Sources marked ✅ were verified by reading the licence file or model card
directly during this audit. Others come from the research sweep and are marked
accordingly. Anything unconfirmed is marked **UNVERIFIED** rather than guessed.

---

## Shortlist

Five candidates, not dozens. Two are disqualified as a product foundation but
kept for a specific role.

### 1. HeartMuLa-oss-3B — cleanest licence

| field | value |
| --- | --- |
| repos | `github.com/HeartMuLa/heartlib` · `huggingface.co/HeartMuLa/HeartMuLa-oss-3B` (recommended checkpoint: `-happy-new-year`) |
| **code licence** | **Apache 2.0** |
| **weights licence** | **Apache 2.0** ✅ *verified on the model card during this audit* |
| commercial use | **allowed, unrestricted.** No revenue threshold, no display requirement. |
| attribution | Apache-2.0 notice only |
| redistribution | allowed, including fine-tuned checkpoints |
| training data | ~100k h pretrain + 15k h SFT. **Provenance undisclosed — UNVERIFIED.** |
| research baseline | ✅ | 
| fine-tuning | ✅ | 
| product foundation | ✅ |

### 2. MiniMax Music 3 — best reported quality, conditional licence

| field | value |
| --- | --- |
| repos | `github.com/MiniMax-AI/MiniMax-Music3` (docs only) · `huggingface.co/MiniMaxAI/MiniMax-Music3` |
| **code licence** | MiniMax-Music3 Community License |
| **weights licence** | **same single licence**, which explicitly covers *"the model weights, parameters, configuration files, inference code and associated documentation"* ✅ *verified by reading the LICENSE file* |
| commercial use | **allowed below US $20M aggregate yearly revenue**; above that requires prior written authorization from MiniMax ✅ |
| attribution | **mandatory**: *"You shall prominently display 'MiniMax-Music3' on the user interface of commercial product or service that uses the Software."* ✅ |
| redistribution | allowed (MIT-shaped grant) |
| extra obligation | hosting generation for third parties requires implementing and periodically reviewing *"reasonable and proportionate technical and organizational safeguards"* ✅ — relevant if we publish a public demo |
| upstream components | Qwen3-8B (Apache-2.0), DiT from `stable-audio-tools` (MIT), VAE from Descript DAC (MIT) |
| training data | **undisclosed — UNVERIFIED** |
| research baseline | ✅ | 
| fine-tuning | ✅ (community tooling only; no official scripts) | 
| product foundation | ✅ with attribution |

### 3. ACE-Step 1.5 — best fine-tuning story, MIT throughout

| field | value |
| --- | --- |
| repos | `github.com/ace-step/ACE-Step-1.5` · HF `ACE-Step/acestep-v15-{base,sft,turbo}` (2B) and `-xl-{base,sft,turbo}` (4B) · LM planners `ACE-Step/acestep-5Hz-lm-{0.6B,1.7B,4B}` |
| **code licence** | **MIT** |
| **weights licence** | **MIT** — all official 1.5 and XL checkpoints |
| ⚠️ version trap | **legacy ACE-Step v1 is Apache-2.0; v1.5 is MIT.** Different licences — do not carry v1 terms forward. |
| commercial use | **allowed.** Model card: *"Trained on legally compliant datasets. Generated music can be used for commercial purposes."* |
| attribution | MIT notice for the software. **None required for generated audio.** |
| redistribution | unrestricted |
| training data | 27M-sample corpus. **No dataset list, no provenance audit.** The commercial-safety claim is an unaudited vendor assertion; a formal request for written output-rights terms (Discussion #1256, Jun 2026) has **zero replies**. |
| ⚠️ tooling trap | the popular low-VRAM trainer **Side-Step is CC BY-NC-SA 4.0**, not MIT. The official Gradio/LoKr path is MIT. |
| research baseline | ✅ |
| fine-tuning | ✅ **only candidate with official in-repo LoRA + LoKr training**, documented dataset format, `--resume-from` checkpointing |
| product foundation | ✅ |

**Controls — this is the most precisely known of any candidate.**

Genuinely typed conditioning inputs: `prompt`, `lyrics`, `audio_duration`,
`vocal_language`, **`bpm`**, **`keyscale`**, **`timesignature`**, seed,
`guidance_scale`, `reference_audio`.

Prompt text only, *not* typed fields: **genre, mood, instruments, timbre, vocal
gender/style.** Structure tags are bracket markers embedded in the lyrics string,
not a separate field.

Not supported at all: **chord progression** and **melody conditioning** (both
closed as not-planned). Relevant to the SongForge goal — if learned song planning
needs melody or harmony as real controls, ACE-Step offers no hook.

⚠️ **Seed reproducibility is currently broken** when the LM planner's `thinking`
is enabled (the default): the planner's sampling is unseeded. Fix PR #1283 is
open and unmerged. Must be validated before the benchmark relies on seeds.

**Capabilities:** 48 kHz stereo, 10–600 s, 50+ vocal languages, structure and
vocal-control tags, plus cover/repaint/stem-extract task types.

### 4. DiffRhythm 2 / v1.2 — clean licence, no fine-tuning path

| field | value |
| --- | --- |
| repos | `github.com/ASLP-lab/DiffRhythm2` · `huggingface.co/ASLP-lab/DiffRhythm2` |
| **code licence** | Apache 2.0 |
| **weights licence** | Apache 2.0 (README states code *and* DiT weights) |
| commercial use | allowed |
| training data | undisclosed; model card asks for originality verification and AI disclosure |
| research baseline | ✅ | 
| fine-tuning | ❌ **no training code released** | 
| product foundation | ⚠️ only if we never fine-tune — which conflicts with the ML-project requirement |

### 5. Stable Audio 3 Medium — instrumental only, best LoRA tooling

| field | value |
| --- | --- |
| repos | `github.com/Stability-AI/stable-audio-3` · `huggingface.co/stabilityai/stable-audio-3-medium` |
| **code licence** | MIT |
| **weights licence** | **Stability AI Community License** — different from the code |
| commercial use | allowed **below US $1M annual revenue** |
| attribution | **mandatory**: retain the Stability copyright notice and display *"Powered by Stability AI"* |
| extra | redistributes Google **T5Gemma** under Gemma Terms of Use, which carry their own restrictions |
| training data | 1,278,902 recordings — AudioSparx licensed + Freesound CC0/CC-BY; copyright-screened. **The best-documented provenance of any candidate.** |
| research baseline | ✅ | 
| fine-tuning | ✅ **best official LoRA support of any candidate** | 
| product foundation | ❌ **cannot sing** — instrumental by deliberate design |

---

## Ruled out, with reasons

| model | reason |
| --- | --- |
| **SongGeneration / LeVo 2** (Tencent) | Reported best-sounding open model, and **withdrawn** — official GitHub and HF both return 404/401. Licence bars *"any commercial or production purposes under any circumstances"*. Surviving mirrors are mis-tagged `unknown`. Do not build on this. |
| **All Meta: MusicGen, MAGNeT, JASCO, MelodyFlow** | **Weights are CC-BY-NC 4.0 despite MIT code.** Also: vocals were deliberately removed from the training corpus with source separation, so they cannot sing at all; 30 s hard cap; dependency stack pinned to 2023 and unlikely to install on a 2026 Colab image. |
| **JAM / JAM-0.5** | Weights under Stability AI Community License with *"Commercial use of JAM or its outputs is strictly prohibited"*. |
| **Shao (ex-Khala)** | **No code licence at all** (all rights reserved by default); weights CC-BY-NC described only as an intention, not a grant. |
| **Magenta RT 2** (Google) | Permissive (Apache code, CC-BY-4.0 weights) but **not lyrics-conditioned** — produces non-lexical vocal sounds only. |
| **Muse-0.6b** (Fudan) | Apache-2.0 weights and full training code released, but the corpus is **116k songs synthesised by Suno V5**, and no evidence is given that Suno's terms permit redistribution or training a competing generator. The permissive tag sits on top of an unevidenced upstream grant. |
| **Amphion / Vevo2** | Weights CC-BY-NC-**ND** — ND blocks distributing even a fine-tuned checkpoint. |
| **MuseCoco, Qwen-\*, Mureka, Seed-Music** | Not applicable: symbolic-MIDI-only, not music generators, or no open weights. |

### A licence pattern worth internalising

Nearly every open singing-voice dataset — OpenSinger, Opencpop, M4Singer,
**GTSinger**, ACE-Opencpop — is CC-BY-NC(-SA). **Any model trained on them
inherits non-commercial weights regardless of its code licence.** This directly
affects our own plans: GTSinger is on our dataset roadmap, and fine-tuning on it
would encumber our resulting checkpoint. Flagged now rather than discovered later.

---

## Hardware feasibility

**Correction to the working assumption.** The brief specified a Tesla T4 with
~14.5 GB. I checked the runtime dialog on this Colab Pro account: **A100 and L4
are both selectable** (H100 and G4 are greyed out). That materially changes
what is reachable.

**The decisive constraint is not VRAM — it is that the T4 is Turing (sm75) and
has neither bf16 nor FlashAttention-2.** The strongest candidates default to
both.

| model | params | T4 16 GB | L4 24 GB | A100 40 GB | LoRA VRAM |
| --- | ---: | --- | --- | --- | --- |
| HeartMuLa-oss-3B | 3.9B | ⚠️ 4-bit NF4 community build only | ✅ ~15 GB bf16 | ✅ | 12–16 GB (int8-quanto + bf16) |
| MiniMax Music 3 | ~11B | ❌ bf16 required; 8 GB path is slow group-offload | ✅ ~22–24 GB w/ offload | ✅ | 24 GB min, 48 GB recommended |
| DiffRhythm v1.2 | 1.1B | ✅ 8 GB with `--chunked` | ✅ | ✅ | n/a — no fine-tuning |
| Stable Audio 3 Medium | 1.4B | ⚠️ FA2 blocks Turing | ✅ 6.5 GB | ✅ | ~5.5 GB |
| ACE-Step 1.5 (2B turbo) | 2B | 🚩 **lyrics fail deterministically** | ✅ | ✅ | 16 GB min / 20 GB rec. |
| ACE-Step 1.5 XL (sft) | 4B | ❌ | ✅ 20–24 GB | ✅ | 20 GB+ |

**The T4 finding is now concrete, not theoretical.** ACE-Step on a T4 raises
`RuntimeError: Generation produced NaN or Inf latents` **whenever lyrics are
non-empty** — instrumental generation works, lyrics-to-song does not. The dtype
fallback is hardcoded (`major >= 8` → bf16 else fp16) with no CUDA override, and
the documented `ACESTEP_DTYPE=float32` escape is confirmed by a maintainer as
*"not currently wired into the DiT init dtype path"*. A measured T4 instrumental
run was ~168 s for a 30 s clip at 11.5 GB peak — 5.6× slower than realtime, for
the one mode that works. FlashAttention-2 is **not** required (SDPA and eager
paths exist); bf16 is the blocker.

**Working decision: L4 is the floor, not the T4.** Compute-unit cost differs
sharply (T4 ≈ 1.8 CU/hr, L4 ≈ 5 CU/hr, A100 ≈ 15 CU/hr), so the plan is L4 for
inference and benchmarking, A100 reserved for fine-tuning runs. Across a
100-hour sprint that difference is a real budget constraint, not a footnote.

---

## Where this points, pending ACE-Step

Two candidates clear both the licence and the capability bar:

- **HeartMuLa-oss-3B** — unrestricted Apache-2.0 on code *and* weights, 48 kHz
  stereo, 6 min, reported best-in-class lyric intelligibility, and it fits an L4
  for both inference and LoRA. Lowest legal risk by a wide margin.
- **MiniMax Music 3** — reported higher overall quality and uniquely explicit
  arrangement conditioning (global metadata / vocal details / section-level
  instrument evolution), at the cost of an attribution obligation, a revenue
  ceiling, and an A100 for comfortable fine-tuning.

Both must still be judged on **generated audio**, not on these tables. No
selection is recorded here.

## Status

No weights downloaded yet. Selection pending the ACE-Step audit and the
eight-prompt generation benchmark.
