# Experiment card — first intervention

**Status: TEMPLATE — not yet filled.** This card is completed from
`listening_review.summary.json` after the human scores exist, and **nothing is
downloaded and no training starts until every field below is filled.** A field
that cannot be filled honestly stays `TBD` and blocks the start.

| field | value |
| --- | --- |
| hypothesis | TBD — one falsifiable sentence, e.g. "a LoKr adapter trained on X improves Y without regressing Z" |
| selected weakness | TBD — area letter + name, from the ranking, never hand-picked |
| evidence from listening | TBD — area mean, worst cells, listener notes verbatim |
| objective corroboration | TBD — which measured flags agree/disagree with the scores |
| cause attribution | TBD — prompt/control-layer, pretrained limitation, LoRA-addressable, or insufficient evidence (from the script, with its reason) |
| intervention | TBD — caseA / caseB / caseC / caseD / inference-settings sweep |
| dataset | TBD — id + licence class; permissive only for the deployable line |
| train/val/test split | TBD — song-disjoint (singer-disjoint for Case B), counts per split |
| adapter method | TBD — lokr/lora, rank, alpha, lr (from the case config) |
| trainable parameter count | TBD — computed from the actual adapter init printout, not estimated by hand |
| estimated GPU / runtime / storage | TBD — GPU model, wall-clock with checkpoint-resume margin, peak Drive GB |
| acceptance metric | TBD — which dimensions must improve, by how much, measured how |
| rejection criterion | TBD — what regression kills the adapter regardless of gains |
| frozen baseline comparison | `foundation_benchmarks/ace_step_15_baseline_frozen/` — same 8 prompts, lyrics, seed 20260818, 60 s, bf16, 8 steps, shift 3.0; only the adapter varies |

Rules carried over from `docs/FINETUNING_PLAN.md`:

- Case C prompt-side experiments run before any training whenever area C is weak.
- The baseline directory is read-only and never regenerated.
- The adapter is kept only if the ablation shows improvement without
  unacceptable regression ("keep-if-better").
