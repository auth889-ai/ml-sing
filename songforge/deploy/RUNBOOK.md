# SongForge public deployment runbook

Target architecture (hour-96 release):

    Vercel (static frontend, deploy/frontend/) 
      └─ /api/* rewrite → GPU host (FastAPI, deploy/backend/)
           └─ ACE-Step 1.5 XL-turbo + SONGFORGE_LORA adapter
                └─ best-of-N ranking → finishing → WAV/MP3

## GPU host (benchmark L4 first — the adapter is trained and verified on L4)

1. Provision one L4 (24 GB) instance with CUDA 12.8 drivers, Python 3.12.
2. Clone the repo; `pip install -r deploy/requirements.txt` (torch 2.10.0+cu128,
   diffusers, fastapi, uvicorn, soundfile, scipy, lycoris-lora) and ffmpeg.
3. flash-attn: install the VERIFIED cached wheel — never compile:
   `pip install wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl`
   (SHA256 f5ca2069…8e2f9d40, built for torch 2.10.0+cu128 / cc 8.9 — matches
   L4 only; other GPUs need a new wheel through the same verification gate).
   torchcodec is NOT needed for inference (training-side dependency).
4. Model weights: `python -m acestep.model_downloader --dir /opt/checkpoints`
   then `--model acestep-v15-xl-turbo`.
5. Environment:
   - `SONGFORGE_LORA=/opt/adapters/v_best/` (frozen checkpoint dir; the
     adapter hard-fails on 0 matched modules rather than silently serving
     the base model)
   - `SONGFORGE_OUTPUT=/var/songforge/jobs`, `SONGFORGE_MAX_DURATION=120`,
     `SONGFORGE_BEST_N=3`, rate limits per config.py defaults.
6. Run: `uvicorn deploy.backend.app:app --host 0.0.0.0 --port 8000`
   behind the host's HTTPS proxy. `GET /health` must report
   `model_loaded: true` before the frontend goes live.
7. Smoke: POST /api/generate (fast, 30 s) → poll → download WAV; then one
   `quality: best` job → confirm `ranking` and `finishing` appear in result.

## Vercel

1. `cd deploy/frontend && vercel deploy --prod` (project root = this dir).
2. Replace `SONGFORGE_BACKEND_HOST` in vercel.json with the GPU host's
   public HTTPS hostname before deploying. No secrets exist client-side;
   the rewrite is the only coupling.
3. Verify from a clean browser: generate (fast), stages advance, WAV plays,
   MP3 downloads, variation works, rate-limit error renders readably.

## Freeze checklist (hour 90–96)

- adapter checkpoint hash + corpus fingerprint recorded in the final report
- `GET /api/capabilities` reflects the frozen deployment
- benchmark results (frozen-8 regression + 51-prompt generalization) linked
- output TTL/cleanup verified running (`retain_hours`)
