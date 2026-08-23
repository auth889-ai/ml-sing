# What is pretrained, what we trained, what we built

SongForge uses open pretrained weights where they raise final quality, and the
repository has to make the boundary unambiguous. Three categories, kept current
as the project moves.

Rule: nothing moves from **PRETRAINED** to **TRAINED BY US** because we ran
inference with it. Only weights we actually updated count as ours.

---

## PRETRAINED — external weights, not ours

Populated once the foundation is selected. Every entry must cite its code
licence and its weights licence **separately** — see
[FOUNDATION_LICENSE_AUDIT.md](FOUNDATION_LICENSE_AUDIT.md).

| component | source | code licence | weights licence | role |
| --- | --- | --- | --- | --- |
| ACE-Step 1.5 XL-turbo | `ACE-Step/acestep-v15-xl-turbo-diffusers` | MIT | MIT | product foundation (provisional) |
| Whisper-small | `openai/whisper-small` | MIT | MIT | evaluation only — lyric intelligibility scoring, never in the generation path |

---

## TRAINED BY US — weights we produced

| component | what it is | evidence |
| --- | --- | --- |
| M03 neural audio codec | Conv encoder → RVQ → conv decoder, trained from random init on BabySlakh. No pretrained weights involved. | [CODEC_RESULTS_FROZEN.md](CODEC_RESULTS_FROZEN.md) |
| M04 codec candidates | Three latent-rate variants trained from scratch at equal budgets. | [CODEC_LATENT_RATE_RESULTS.md](CODEC_LATENT_RATE_RESULTS.md) |
| _LoRA / adapters on the selected foundation_ | pending | |

---

## BUILT BY US — engineering, no external weights

| component | what it does |
| --- | --- |
| `data/media.py`, `data/dsp.py` | Multi-backend decode, validation, pure-torch windowed-sinc resampling (deterministic across machines). |
| `data/manifest.py` | The canonical `songforge.audio.v1` record. One schema project-wide. |
| `data/splits.py` | Song- and singer-disjoint splitting; quota, weighted and hash strategies. |
| `data/slakh_metadata.py` | Real instrument labels read from the corpus, never inferred from spectra. |
| `data/preprocess.py` | Segmenting, provenance and licence propagation, archive-residue filtering. |
| `training/run.py`, `training/checkpoint.py` | Run isolation, atomic checkpoints, strict RNG/optimizer/scaler restore, crash-tail reconciliation. |
| `models/codec/*` | Codec architecture, including the data-dependent RVQ init and dead-code restart that fixed codebook collapse. |
| `generation/*` | SongForge's control surface: request format, capability declaration, and the resolver that refuses to present a control the model does not honour. |
| `evaluation/*` | Reconstruction metrics, instrument-aware example selection, objective song scorecard. |
| `scripts/*` | Preprocessing, training, sweeps, fairness audit, listening matrix, foundation benchmark. |
| Test suite | 232 tests covering the above. |

---

## Ablation requirement

A pretrained baseline alone proves nothing about our contribution. Every claim
that SongForge improved on the foundation must be backed by a paired comparison
on identical prompts, seeds and evaluation:

    pretrained baseline   vs   pretrained + our fine-tuning/conditioning

recorded the same way the M04 candidate comparison was: identical inputs, one
isolated run each, and a fairness audit over the artifacts.
