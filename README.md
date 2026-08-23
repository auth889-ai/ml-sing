# SongForge

**Text to full song.** Type a free-form idea — *"piano intro building to a full-band climax"*, *"female vocal over strings"*, *"violin-led rock"* — and SongForge plans the arrangement, generates the audio, ranks several takes, screens the winner for originality, masters it, and hands back a WAV and an MP3.

The point of the project is not to call a music API. It is to **train our own model weights on a corpus we built and licence-cleared ourselves**, and to wrap them in the unglamorous engineering that turns a research checkpoint into something a person can actually use.

---

## What is ours, and what is not

Being precise about this matters more than sounding impressive.

| layer | who built it | detail |
|---|---|---|
| **Pretrained foundation** | *not ours* | ACE-Step 1.5 XL-turbo (MIT), used frozen. Never retrained from scratch. |
| **Trained ML** | **ours** | LoKr adapters trained by us on our corpus — 1,835,008 trainable parameters, 0.04% of the 4,991,023,206-parameter base. |
| **Data work** | **ours** | Six-family multi-corpus construction, licence census, ten-gate admission chain, balanced sampling. |
| **Engineering** | **ours** | Free-form planner, best-of-N ranking, originality screen, finishing chain, async job API, web app. |
| **Separate DL research** | **ours** | A custom neural audio codec, trained independently and reported separately. |

Adapting a strong open foundation is the correct engineering choice at this scale. Claiming to have trained a 5B-parameter music model from scratch in a weekend would not be.

---

## The problem

Generative music tools mostly fail in one of two ways. Either they are closed APIs — you cannot see the training data, cannot audit the licensing, and cannot run them yourself — or they are research checkpoints that produce a `.wav` in a notebook and nothing more.

Both leave the same gap: **a musician cannot use them, and a lawyer cannot clear them.**

SongForge targets that gap directly:

- **Auditable provenance.** Every second of deployable training audio is CC0 or CC BY, verified against the source's own metadata. NonCommercial, NoDerivatives, ShareAlike and unresolved records are excluded outright — for the Free Music Archive that meant discarding 97,735 of 106,574 tracks to keep 8,839 clean ones.
- **Originality screening.** Generated audio is checked against a corpus fingerprint database (PCM hash plus chroma-sequence similarity) before it is returned, so the system can flag a take that leans too close to its training data.
- **A real product surface.** Free-form prompt in, finished master out, through a web app — not a notebook.

---

## How it works

```
free-form prompt
      │
      ▼
  planner ─────────► genre · mood · instruments · BPM · key · structure · energy
      │              (typed controls always override inferred ones)
      ▼
  ACE-Step 1.5 (frozen)  +  SongForge LoKr adapter
      │
      ▼
  best-of-N generation ──► broken-take rejection
      │
      ▼
  originality screen (PCM hash + chroma similarity vs corpus DB)
      │
      ▼
  conservative finishing ──► WAV + MP3
```

Architecture: a **Next.js frontend on Vercel** talks to an **async FastAPI job API**, which drives the model on a **GPU host**. The model never runs inside Vercel — Vercel is frontend and control only.

Adapters are hot-swappable: the production backend loads whichever checkpoint `SONGFORGE_LORA` points at and **hard-fails if zero modules match**, so it can never silently serve the bare foundation model while claiming to serve ours.

---

## The corpus

Six capability families, weighted at sampling time rather than concatenated — otherwise the largest corpus drowns the small ones that exist precisely to fix known weaknesses.

| family | source | licence | role |
|---|---|---|---|
| arrangement | Slakh2100-redux | CC BY 4.0 | multi-instrument interaction, stem balance |
| real songs | Free Music Archive | CC0 / CC BY | production realism, genre diversity |
| vocals | VocalSet | CC BY 4.0 | singing realism, phrasing |
| piano · violin · cello · strings | MusicNet | CC BY 4.0 | acoustic and cinematic realism |
| guitar | GuitarSet | CC BY 4.0 | timbre, articulation, voicing |
| drums | Slakh isolated drum stems | CC BY 4.0 | percussion |

Every record passes ten gates in order: **licence → provenance → cross-corpus duplicate → integrity/decode → silence → clipping/quality → split leakage → metadata → rich caption → deployability.** A record failing any gate is excluded and counted in the corpus report. No gate is skipped for speed, and no licence status is asserted on assumption.

Segments are stratified and capped per track — a fourth 60-second slice of one arrangement teaches far less than the first slice of a new one, and costs the same GPU-seconds.

---

## Results so far

V1 is a **training-validation** model, deliberately trained on a single corpus (Slakh-100, 3,626 × 60 s segments at 44.1 kHz) to prove the pipeline end to end before spending GPU time on the full mix.

Training is verified from saved artifacts rather than asserted:

```
LoKr injected        : 1,835,008 / 4,991,023,206 params (0.04%), 256 modules
optimizer steps      : 907 per epoch  (3,626 samples ÷ 4 grad-accum)
gradients            : AdamW exp_avg_sq non-zero and finite for 768/768 tensors
learning rate        : 0.03 → 0.0236 (cosine)
loss                 : epoch 1 = 0.9691  →  epoch 2 = 0.9228
```

`exp_avg_sq` is the running mean of *squared* gradients, so a non-zero value is direct evidence that a parameter received real gradient signal — not an inference from a falling loss curve. Checkpoint selection uses **generated audio**, never loss alone.

V2 extends training to the full six-family corpus above.

---

## Run it

```bash
git clone https://github.com/auth889-ai/ml-sing.git
cd ml-sing/songforge
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q            # 288 passed, 2 skipped
```

**Backend (GPU host)**

```bash
pip install -r deploy/backend/requirements.txt
python -m acestep.model_downloader --dir /opt/checkpoints
python -m acestep.model_downloader --model acestep-v15-xl-turbo --dir /opt/checkpoints
export SONGFORGE_LORA=/opt/adapters/v_best/
uvicorn deploy.backend.app:app --host 0.0.0.0 --port 8000
# GET /health must report model_loaded: true before the frontend goes live
```

**Frontend (Vercel)**

```bash
cd deploy/frontend
# point SONGFORGE_BACKEND_HOST in vercel.json at the GPU host's HTTPS name
vercel deploy --prod
```

Full procedure, including the verified flash-attn wheel and the environment pins that matter, is in [`deploy/RUNBOOK.md`](songforge/deploy/RUNBOOK.md).

---

## Repository map

```
songforge/
  src/songforge/
    generation/planner.py      free-form prompt → typed musical controls
    generation/adapters/       adapter loading; hard-fails on 0 matched modules
    inference/ranking.py       best-of-N scoring
    inference/originality.py   PCM hash + chroma-sequence similarity
    inference/finishing.py     conservative mastering → WAV/MP3
  deploy/backend/              async FastAPI job API
  deploy/frontend/             Next.js web app
  scripts/                     corpus selection, ten-gate chain, preprocessing,
                               tensorization, training, verification
  configs/datasets/            corpus definitions with honest licence status
  docs/                        design records, data programme, evaluation
  benchmarks/                  frozen experiment cards, regression prompts
```

Documentation index: [`docs/`](songforge/docs) — corpus definition in
[`configs/datasets/v2_sprint.yaml`](songforge/configs/datasets/v2_sprint.yaml),
attribution in [`docs/ATTRIBUTION.md`](songforge/docs/ATTRIBUTION.md),
deployment in [`deploy/RUNBOOK.md`](songforge/deploy/RUNBOOK.md).

---

## Licence

MIT for the code — see [LICENSE](LICENSE), which also records every third-party
component and the licence of every training corpus.
