# SongForge — Project Documentation

**Live app:** https://frontend-ashen-mu-51.vercel.app
**Source:** https://github.com/auth889-ai/ml-sing
**Licence:** MIT (code) — every training corpus CC0 or CC BY, verified per record

---

## 1. Purpose

Generative music tools fail in one of two ways. Either they are closed APIs —
you cannot see the training data, cannot audit the licensing, and cannot run
them yourself — or they are research checkpoints that emit a `.wav` in a
notebook and nothing more.

Both leave the same gap: **a musician cannot use them, and a lawyer cannot
clear them.**

SongForge targets that gap: a text-to-song system where every second of
training audio has verified permissive provenance, generated audio is screened
for similarity against the training corpus, and the whole thing is wrapped in a
product a non-technical person can use.

## 2. Target audience

- **Independent creators** who need original music for video, games or podcasts
  and cannot risk unclear rights.
- **Developers** who want a self-hostable generation stack rather than an API
  they cannot inspect.
- **Researchers** studying adapter-based specialisation of open music
  foundation models.

## 3. Main features

| feature | description |
|---|---|
| Free-form prompt | Plain language in, finished track out. No genre menus to learn. |
| Prompt composer | Genre, mood, instrument, vocal and structure chips fold into one rich caption, with a live "detail" meter |
| Planner | Derives genre, instruments, BPM, key, structure and energy; explicit user settings always override inferred ones |
| Best-of-N | Renders several candidates, rejects broken takes, returns the best |
| Originality screen | PCM hash + chroma-sequence similarity against a corpus fingerprint database |
| Finishing | DC removal, loudness normalisation, peak protection, click-safe fades → WAV + MP3 |
| Library | Saved songs kept per-browser; no account, no server-side profile |
| Reproducibility | Seed randomised per render, lockable for identical output |

---

## 4. What is ours, and what is not

Stated precisely, because the distinction matters.

| layer | origin | evidence |
|---|---|---|
| **Audio foundation** | **Not ours.** ACE-Step 1.5 XL-turbo (MIT), used frozen, never retrained | pip dependency |
| **LoKr adapter** | **Ours — trained** | 1,835,008 params (0.04% of 4,991,023,206) |
| **Neural audio codec** | **Ours — from scratch** | 5,068,481 params: Conv encoder → RVQ → Conv decoder |
| **Structure planner** | **Ours — from scratch** | Transformer encoder |
| **Diffusion singer** | **Ours — from scratch** | `models/singer/diffusion.py` |
| **Corpus + gates** | **Ours** | 106,574 tracks censused → 8,839 deployable |
| **Product** | **Ours** | planner, ranking, originality, finishing, API, web app |

Roughly **15,600 lines** of Python plus the Next.js application.

Adapting a strong open foundation is the correct engineering choice at this
scale — it is how essentially all applied ML ships. Claiming to have trained a
five-billion-parameter music model from scratch would not survive scrutiny: on
a single L4 that is on the order of two years of compute.

---

## 5. Research position

ACE-Step 1.5 XL scores **8.12 on SongEval**, above **Suno v5's 7.87**. But it
trails Suno on two specific axes:

| axis | Suno v5 | ACE-Step 1.5 |
|---|---|---|
| Overall quality | 7.87 | **8.12** |
| Style alignment | **46.8** | 39.1 |
| Lyric alignment | **34.2** | 26.3 |

Both trailing metrics are **prompt-following**, not audio fidelity — precisely
what adapter training on rich captions and real singing data addresses.

**Thesis:** close ACE-Step's style- and lyric-alignment gap with a
multi-corpus LoKr adapter trained on narrative captions in the foundation's
native conditioning format.

This also explains why V1 is a *validation* model rather than the product: its
captions were `[Instrumental]` on all 3,626 rows, and it contained no vocal
data at all — so it attacks neither weakness.

---

## 6. Data

### Licence census — Free Music Archive

FMA ships 106,574 tracks whose licences differ per track, and the maintainers
state they do not own all audio rights. So only the 342 MB metadata was
downloaded first, every row censused, and the subset built from the result.

```
Tracks censused        106,574
CC0 / public domain      1,820
CC BY                    7,019
  DEPLOYABLE             8,839   (606.1 hours)
Excluded — NonCommercial 93,713
Excluded — ShareAlike     2,802
Excluded — NoDerivatives    903
Excluded — unresolved       317
```

**97,735 of 106,574 tracks were discarded to keep 8,839 clean ones.**

### Six capability families

| family | source | licence | verified |
|---|---|---|---|
| Arrangement | Slakh2100-redux | CC BY 4.0 | ✓ |
| Real songs | FMA subset | CC0 / CC BY per track | ✓ census |
| Vocals | VocalSet | CC BY 4.0 | ✓ Zenodo API |
| Piano/violin/cello/strings | MusicNet | CC BY 4.0 | ✓ Zenodo API |
| Guitar | GuitarSet | CC BY 4.0 | ✓ Zenodo API |
| Drums | Slakh isolated stems | CC BY 4.0 | ✓ |

Licences were read from each source's own metadata, not inferred.

### Ten gates, in order

`licence → provenance → cross-corpus duplicate → integrity/decode → silence →
clipping/quality → split leakage → metadata → rich caption → deployability`

A record failing any gate is excluded and counted in the corpus report.

---

## 7. Results — V1

V1 is a training-validation model on Slakh-100 (3,626 × 60 s segments,
44.1 kHz), deliberately single-corpus to prove the pipeline before spending GPU
time on the full mix.

```
LoKr injected        1,835,008 / 4,991,023,206 params (0.04%), 256 modules
Optimizer steps      907 per epoch   (3,626 samples ÷ 4 grad-accum)
Gradients            exp_avg_sq non-zero and finite on 768/768 tensors
Learning rate        0.03 → 0.0236 (cosine)
Loss                 epoch 1  0.9691
                     epoch 2  0.9228
                     epoch 3  0.8809
```

`exp_avg_sq` is AdamW's running mean of *squared* gradients, so a non-zero
value is direct evidence a parameter received real gradient signal — not an
inference from a falling loss curve.

---

## 8. Installation

```bash
git clone https://github.com/auth889-ai/ml-sing.git
cd ml-sing/songforge-dl-starter
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q                      # 288 passed, 2 skipped
```

### Frontend

```bash
cd deploy/frontend
npm install
npm run dev                    # http://localhost:3000
```

### Backend (needs a GPU with compute capability ≥ 8.0)

```bash
pip install -r deploy/backend/requirements.txt
python -m acestep.model_downloader --dir /opt/checkpoints
python -m acestep.model_downloader --model acestep-v15-xl-turbo --dir /opt/checkpoints
export SONGFORGE_LORA=/opt/adapters/v_best/
uvicorn deploy.backend.app:app --host 0.0.0.0 --port 8000
```

Point the frontend at it:

```bash
vercel env add SONGFORGE_API_URL production     # https://your-gpu-host
```

**Hardware requirement:** bf16 needs compute capability ≥ 8.0 (Ampere or
newer). T4 (7.5) and P100 (6.0) produce NaNs on this model and are rejected by
the driver's environment gate.

---

## 9. User manual

1. Open the app → **Create**
2. Describe the song in plain words
3. Optionally refine with genre / mood / instrument / vocal / structure chips —
   the detail meter shows how much conditioning the model will receive
4. Set duration; leave the seed empty for a different song each time, or tick
   **Lock this seed** to reproduce one exactly
5. **Generate** → watch Planning → Rendering → Ranking → Mastering
6. Play, download WAV, or **Generate a variation**
7. Every finished song is saved to **Library**, in your browser only

---

## 10. Configuration

| variable | purpose | default |
|---|---|---|
| `SONGFORGE_API_URL` | GPU backend the frontend proxies to | `http://localhost:8000` |
| `SONGFORGE_LORA` | Adapter directory; hard-fails on 0 matched modules | none |
| `SONGFORGE_MAX_DURATION` | Per-request cap, seconds | 120 |
| `SONGFORGE_BEST_N` | Candidates rendered in Best mode | 3 |
| `SONGFORGE_CONCURRENCY` | Simultaneous jobs | 1 |

---

## 11. Privacy

No accounts, no email, no passwords, no third-party trackers. The saved library
lives in the visitor's browser under `songforge.library.v1` and never reaches a
server. Rendered audio is retained on the GPU host for a few hours and then
deleted. Prompts are sent to the backend to render the song and appear in that
request's log.

There is no login because there is no server-side user data — an account wall
would gate the interface without protecting anything.

---

## 12. Limitations — stated honestly

- **Live generation requires a GPU with compute capability ≥ 8.0.** The public
  deployment currently has no GPU attached; the app detects this and says so
  rather than failing opaquely.
- **V1 is instrumental only.** Slakh-100 contains no vocals, so V1 does not
  improve singing. V2's vocal corpus addresses this.
- **Drums in the V2 plan come from Slakh stems, not E-GMD.** The 89.8 GB E-GMD
  release did not fit the available disk. These are rendered, not acoustic, and
  that substitution is deliberate and disclosed.
- **Quality ceiling is the foundation's.** A 0.04% adapter steers a model; it
  does not replace one. SongForge does not claim parity with systems trained
  from scratch on far larger corpora and compute.
- **Originality screening reduces risk; it does not guarantee anything.** The
  correct claim is "originality-risk reduction", not "copyright safe".

---

## 13. References

- ACE-Step 1.5 — https://github.com/ace-step/ACE-Step-1.5 (MIT)
- Slakh2100 — https://zenodo.org/records/4599666 (CC BY 4.0)
- Free Music Archive — https://github.com/mdeff/fma
- VocalSet — https://zenodo.org/records/1193957 (CC BY 4.0)
- MusicNet — https://zenodo.org/records/5120004 (CC BY 4.0)
- GuitarSet — https://zenodo.org/records/3371780 (CC BY 4.0)
- LyCORIS / LoKr — https://github.com/KohakuBlueleaf/LyCORIS
