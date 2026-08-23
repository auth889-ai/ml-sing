# SongForge public deployment

Anyone opens the site, types a prompt, optionally supplies lyrics and controls,
gets a new song, and downloads it.

The single architectural constraint that shapes everything: **a 4B diffusion
model cannot run in a serverless function.** Vercel serves the frontend; the GPU
lives somewhere it can stay warm.

```
  browser
     │  POST /api/generate  ──▶ 202 { job_id }
     │  GET  /api/jobs/{id} ──▶ poll
     ▼
  Next.js on Vercel  ──proxy──▶  SongForge API (FastAPI, GPU host)
                                        │
                                        ├─ job queue, concurrency 1
                                        ├─ rate limit + daily quota
                                        ▼
                                 ACE-Step 1.5 XL-turbo
                                     + our LoRA
                                        │
                                        ▼
                                 job WAV on disk ──▶ /api/jobs/{id}/audio
```

## Why this shape

| decision | reason |
| --- | --- |
| GPU backend separate from Vercel | 4B weights, ~15.4 GB VRAM, ~30 s cold start. Serverless has none of that. |
| Job queue, not synchronous HTTP | A render occupies the only GPU. Blocking an HTTP worker for its duration is how the service falls over under two users. |
| Concurrency 1 | One GPU. Raising it without a second GPU adds latency and risks OOM mid-render. |
| Poll, not WebSocket | Generation is ~3–10 s at RTF 0.05. Polling is simpler and survives proxies. |
| Audio deleted after 6 h | A public demo that keeps every render fills its disk within days. |

## API

| method | path | purpose |
| --- | --- | --- |
| `POST` | `/api/generate` | submit; returns **202** with `job_id`, queue position, and which controls will actually be honoured |
| `GET` | `/api/jobs/{id}` | status, queue position, timings, control warnings |
| `GET` | `/api/jobs/{id}/audio` | the finished WAV |
| `GET` | `/api/capabilities` | what this deployment genuinely supports — the UI builds its form from this |
| `GET` | `/health` | liveness, model state, queue depth, limits |

### The capabilities endpoint is the point

`/api/capabilities` returns the same `Capabilities` declaration the library uses,
so the frontend can label each control by what it actually does:

- **BPM, key, time signature, duration, seed, vocal language** → real typed
  conditioning inputs
- **genre, mood, instruments, vocal character** → prompt text; the model may or
  may not comply
- **chord progression, melody** → not offered at all, because ACE-Step has no
  such conditioning

Every job also returns `control_warnings`. If someone asks for something this
deployment cannot honour, the response says so **before** the render. A knob
that silently does nothing is worse than a missing knob — that rule holds from
the library through to the interface.

## Operational limits

All overridable by environment variable; defaults are for one shared L4.

| setting | default | why |
| --- | ---: | --- |
| `SONGFORGE_MAX_DURATION` | 120 s | model allows 600 s; a shared GPU does not |
| `SONGFORGE_CONCURRENCY` | 1 | one GPU |
| `SONGFORGE_MAX_QUEUE` | 20 | beyond this, reject fast rather than promise slowly |
| `SONGFORGE_JOB_TIMEOUT` | 300 s | a hung job must not hold the only worker forever |
| `SONGFORGE_RPM` | 4 | per client per minute |
| `SONGFORGE_DAILY_QUOTA` | 30 | per client per day |
| `SONGFORGE_RETAIN_HOURS` | 6 | disk hygiene |
| `SONGFORGE_WARMUP` | 1 | load weights at startup so the first visitor is not the one who waits |

Rate limiting keys on the forwarded IP. That is trivially spoofable — it
discourages casual over-use and is **not** a security control. Anything stronger
needs real authentication, which this demo does not have.

## Running the backend

```bash
pip install fastapi uvicorn pydantic
uvicorn deploy.backend.app:app --host 0.0.0.0 --port 8000
```

GPU host options, cheapest first: a Colab session with a tunnel (fine for a
demo, dies with the runtime), a rented L4/A10 (Runpod, Lambda, Vast), or any box
with ≥16 GB VRAM and bf16 — **Turing/T4 will not work**, since lyrics-to-song
produces NaN latents there. Concrete host recipes, the Dockerfile, and the
production checklist live in [`deployment/`](deployment/README.md).

## Frontend

`frontend/` is deliberately minimal: a form, a submit, a poll, a player, a
download. UI polish is the last priority in the 100-hour plan, and every hour
spent there is an hour not spent on the music.

```bash
cd deploy/frontend && npm install && npm run dev
```

Set `SONGFORGE_API_URL` to the GPU backend. The Next.js route proxies so the
browser never talks to the GPU host directly and the backend origin stays
private.

## What is deployed, and whose work it is

| layer | origin |
| --- | --- |
| ACE-Step 1.5 XL-turbo weights | **pretrained**, MIT, not ours |
| LoRA adapter | **trained by us** (pending the listening gate) |
| control layer, capability honesty, API, queue, evaluation, frontend | **built by us** |

The deployed page must not imply the model is ours. See
[../docs/ATTRIBUTION.md](../docs/ATTRIBUTION.md).
