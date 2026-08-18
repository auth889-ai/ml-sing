# Evaluation plan

## Two benchmark tiers (fine-tuning era)

The eight songs in `benchmarks/prompts.yaml` are a **frozen regression
benchmark**: identical inputs before and after every change, existing to catch
breakage. They are *not* the product, not the target capabilities, and must
never become fixed product categories — SongForge is a free-form generator.

Breadth is measured by the **Generalization Benchmark**
(`benchmarks/generalization_prompts.yaml`): ~50 prompts across ten categories
(solo instruments, arrangements, vocals, full band, orchestral/cinematic,
electronic, acoustic, genre fusion, multilingual lyrics, compositional/unseen),
including deliberately difficult combinations. Two tiers inside it:

- **dev** — usable for prompt engineering and Case C control experiments,
  never as training audio;
- **heldout** — never used for training, prompt tuning, or any development
  decision; generated and scored only at evaluation time. Enforced by CI:
  `tests/test_generalization_benchmark.py` fails if a held-out prompt id
  appears anywhere in `configs/`, `scripts/`, `src/`, `deploy/` or
  `notebooks/`.

**Acceptance for any adapter:** improve its targeted weakness on the frozen
eight **and** retain broad capability on the generalization set — especially
the held-out prompts it has never seen. A model that aces eight songs and
degrades on unseen prompts is a regression, not a win. Never optimize for the
eight benchmark songs.

## Codec
- waveform L1 / SI-SDR (diagnostic)
- multi-resolution STFT distance
- codebook utilization/perplexity
- bitrate / tokens per second
- blind A/B listening export

## Planner
- token NLL / perplexity
- invalid-event rate
- pitch-class distribution
- note-duration distribution
- repetition / n-gram statistics
- structure validity (bars/sections)

## Singer
- diffusion validation loss
- F0 RMSE / voiced-unvoiced accuracy where annotation supports it
- phoneme-duration/alignment diagnostics
- mel/STFT distance
- MOS-style listening-study export

## Full song
- prompt/tag consistency using a separately trained evaluator only as an evaluation tool
- beat/tempo adherence
- clipping/loudness checks
- diversity across seeds
- human ratings: musicality, vocal intelligibility, prompt adherence, audio quality

Never report a single loss value as "Suno quality".
