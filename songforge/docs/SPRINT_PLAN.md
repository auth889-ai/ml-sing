# SongForge — 72-hour max-quality ship plan

**T0 = 2026-08-20 21:50 UTC. T+72 = 2026-08-23 21:50 UTC.**
Directive received 2026-08-20 21:49 UTC (supersedes the 96h ship directive,
whose hour-96 mark was 2026-08-23 01:00 UTC — the window moved ~21 h later).

Authoritative for schedule. `benchmarks/EXPERIMENT_CARD.md` stays authoritative
for V1 experiment identity; a V2 card will be written before V2 training.

## What is already built (verified 2026-08-20, do NOT rewrite)

| directive phase | status | code |
| --- | --- | --- |
| E free-form planner | EXISTS | `src/songforge/generation/planner.py` (249 L), `tests/test_planner.py` |
| F best-of-N ranking | EXISTS | `src/songforge/inference/ranking.py` (109 L) |
| F originality risk | EXISTS | `src/songforge/inference/originality.py` (159 L) + fingerprint DB builder |
| G finishing WAV/MP3 | EXISTS | `src/songforge/inference/finishing.py` (184 L) |
| H async job API | EXISTS | `deploy/backend/{app,jobs,config}.py` (738 L) |
| H Vercel frontend | EXISTS | `deploy/frontend/` Next.js + `vercel.json` |
| H deploy procedure | EXISTS | `deploy/RUNBOOK.md` |

E/F/G/H are therefore **verify + wire + deploy**, not build. This is what makes
the directive survivable in 72 h.

## Critical scheduling fact — tensorization is GPU-bound

ACE-Step preprocessing runs the DCAE + text encoders on GPU. V2 tensorization
therefore COMPETES with V1/V2 training for the single Colab L4. Download,
license gating, decode checks, segmentation and captioning are CPU/network and
genuinely parallel; the final `.pt` build is not.

GPU budget on one L4 (~33–37 h of the 72):
V1 train 6 h · V2 tensorize 6–8 h · V2 train 14–16 h · V3 4 h · eval gen 3 h.

**Mitigation (highest-leverage change): provision the production GPU host at
~T+2, not T+46.** It doubles GPU capacity (Colab = training, rented L4 =
tensorization + eval generation) AND forces deployment up early.

## Deploy-early risk inversion

The directive says GPU API + Vercel live by ~T+60. That leaves the single
must-not-cut deliverable dependent on everything upstream landing on time.

Instead: **go live at ~T+20 serving the V1 adapter**, then hot-swap the frozen
V2/V3 adapter by changing `SONGFORGE_LORA` and restarting. The public product
becomes independent of training slippage; adapter quality upgrades in place.

## Schedule

| window | UTC | work | GPU |
| --- | --- | --- | --- |
| T+0–6 | 08-20 21:50 → 08-21 04:00 | V1: stage tensors → optimizer-step gate → train 3 epochs → frozen-8 ablation → **V1 VERDICT** | Colab |
| T+0–22 | → 08-21 20:00 | V2 corpus: download, license gate, dedup, segment, caption, manifest | CPU/net |
| T+2 | 08-20 ~24:00 | Provision rented L4; RUNBOOK steps 1–6 | — |
| T+8–22 | → 08-21 20:00 | V2 tensorization (resume-aware, atomic) | rented L4 |
| T+18–22 | → 08-21 20:00 | **GPU API LIVE + VERCEL LIVE** on V1 adapter | — |
| T+22–38 | → 08-22 12:00 | V2 training; checkpoints scored on generated audio | Colab |
| T+38–46 | → 08-22 20:00 | Score V2; AT MOST ONE targeted V3 if justified | Colab |
| T+46–52 | → 08-23 04:00 | **BEST MODEL FROZEN**; hot-swap adapter into live backend | — |
| T+52–64 | → 08-23 16:00 | Baseline vs V1 vs V2/V3; frozen-8 + generalization + held-out | rented L4 |
| T+64–72 | → 08-23 21:50 | Final freeze, hashes, report, no new features | — |

## V2 corpus — sprint target

**8,000 unique 30–60 s segments** (band 6,000–10,000), recomputed from measured
preprocessing rate at T+8. Stratified/capped per track — representative
segments, never every song exploded into near-redundant slices.

| family | share | source | role |
| --- | --- | --- | --- |
| Slakh-redux | 25–30% | 800–1,200 unique tracks | arrangement, instrument interaction |
| FMA CC0/CC-BY | 30–40% | from the 3,088 selected pool | real finished music, production realism |
| Vocals | 10–15% | SingStyle111 (CC-BY, lyric-aligned) → VocalSet if rights gate passes | singing realism, intelligibility |
| Piano/strings/orchestra | 10–15% | Open Goldberg, Open WTC, Musopen Chopin + Musopen orchestral PD | grand piano, violin, cello, ensemble |
| Guitar | 5–10% | GuitarSet if rights verify, else verified permissive substitute | timbre, articulation |
| Drums | 5–10% | E-GMD diverse subset across kits × tempo × style — never all 132 GB | percussion realism |
| NSynth | auxiliary | — | timbre eval / classifier only, never main song training |

Weighted sampling, never plain concatenation. FMA must not drown the small
capability-specific corpora.

Gate chain per record: license → provenance → cross-corpus duplicate →
integrity/decode → silence → clipping/quality → split leakage → metadata →
rich caption → deployability. No fabricated license status; NC/ND/unknown never
enter the deployable model.

## Storage discipline

Drive (2 TB) = durable. Colab `/content` = fast temporary cache. Mac = code
only, no datasets or model weights, ever. Training reads from
`/content/tensors_local`; Drive FUSE never sits in the hot path.

## Blockers requiring the user

1. **Drive mount consent** in the Colab notebook — blocks everything.
2. **Rented GPU host** — account creation and payment are actions I cannot
   perform. Needs provisioning + an SSH endpoint handed over.
3. **Vercel auth** — `vercel login` is interactive; run as `! vercel login`.

## Milestones (report verbatim, nothing else)

TRAINING STARTED · V1 COMPLETE · V1 VERDICT · V2 CORPUS READY ·
V2 TENSORS READY · V2 TRAINING STARTED · V2 COMPLETE · BEST MODEL FROZEN ·
GPU API LIVE · VERCEL LIVE · 72-HOUR RELEASE COMPLETE

## Cut order if the clock slips

Cut first: second V3 · full E-GMD ingestion · extra auxiliary datasets · UI
polish · optional mastering experiments · excessive docs/tests.

Never cut: multi-corpus V2 · real music data · vocal/instrument diversity ·
V2 training · public GPU deployment · Vercel app · baseline comparison.
