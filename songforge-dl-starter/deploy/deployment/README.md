# Deploying SongForge

Two halves, deployed separately:

| half | where | cost profile |
| --- | --- | --- |
| `deploy/frontend/` (Next.js) | Vercel free tier | free |
| `deploy/backend/` (FastAPI + GPU) | a machine with an Ampere+ GPU | the real cost |

The frontend proxies every `/api/songforge/*` request to the backend, so the
GPU host's address is a server-side environment variable, never shipped to the
browser.

## 1. Frontend on Vercel

```bash
cd deploy/frontend
vercel deploy          # or connect the repo in the Vercel dashboard
```

One environment variable: `SONGFORGE_API_URL` → the backend origin
(e.g. `https://abc123-8000.proxy.runpod.net`). Changing GPU hosts is a
redeploy of an env var, not of code.

Do **not** attempt to run generation inside Vercel functions: 4B weights,
~15.4 GB VRAM, ~30 s cold start — serverless has none of that.

## 2. Backend on a GPU host

Requirements: ≥16 GB VRAM, bf16 support (**Ampere or newer — T4 will not
work**, lyrics-to-song NaNs on Turing), ~40 GB disk (weights + HF cache +
job output).

### Option A — rented GPU (Runpod / Vast / Lambda), recommended for the demo

An L4 or A10 spot instance. Build and run the image:

```bash
docker build -f deploy/deployment/Dockerfile -t songforge-api .
docker run --gpus all -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  -e SONGFORGE_LORA=/models/lora -v ./models:/models \
  songforge-api
```

On Runpod, expose port 8000 and use the pod's proxy URL as
`SONGFORGE_API_URL`. Mount the HF cache volume so pod restarts do not
re-download ~15 GB of weights.

### Option B — Colab + tunnel, free but fragile

Fine for a supervised demo session; dies with the runtime (observed drops
every 10–40 min on this project, see `docs/COLAB_REMOTE_TRAINING.md`). Run
the backend in the notebook, then tunnel:

```bash
pip install fastapi uvicorn pydantic acestep cloudflared
uvicorn deploy.backend.app:app --host 0.0.0.0 --port 8000 &
cloudflared tunnel --url http://localhost:8000   # prints a public URL
```

Point `SONGFORGE_API_URL` at the printed URL. Expect to re-point it every
time the runtime drops — acceptable for a demo, not for a public link.
Colab must be reached as `colab.research.google.com/?authuser=1`.

### Option C — any owned box with an RTX 3090/4090 or better

`pip install -r deploy/backend/requirements.txt acestep`, then run uvicorn
behind Caddy/nginx for TLS. Cheapest per hour if the hardware exists.

## 3. Production checklist

- [ ] `SONGFORGE_LORA` set (or intentionally unset for baseline-only)
- [ ] CORS `allow_origins` in `deploy/backend/app.py` tightened to the Vercel origin
- [ ] `SONGFORGE_BLOCKED_TERMS` set (comma-separated artist-impersonation terms)
- [ ] HF cache on a persistent volume
- [ ] `/health` returns `model_loaded: true` after warmup, before the link is shared
- [ ] Attribution footer visible (ACE-Step is pretrained and MIT — not ours;
      see `docs/ATTRIBUTION.md`)

## 4. What the limits already handle

Queue depth, per-IP rate limit, daily quota, job timeout, max duration, and
6-hour audio retention are all in `deploy/backend/config.py`, overridable by
env var. Defaults assume exactly one shared GPU; raise nothing until there is
a second one.
