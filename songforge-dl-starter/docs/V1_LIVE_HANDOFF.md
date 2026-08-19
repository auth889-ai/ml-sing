# V1 LIVE HANDOFF — written 2026-08-18 ~21:55 UTC, mid-run

Purpose: let a fresh session resume the running SongForge V1 experiment with
zero repeated work. A flash-attn compile is RUNNING on the Colab runtime —
do not restart it; reconnect and check it first.

## CURRENT STATUS

| step | state |
| --- | --- |
| Slakh2100-redux 44.1 kHz download (104,322,767,708 bytes) | DONE (archive deleted after extraction) |
| Deterministic Slakh-100 selection (80/10/10, all quotas met, `unmet_quotas: {}`) | DONE |
| Selective extraction → Drive `raw/slakh100` (5.5 GB, 1,361 FLACs, ffprobe `44100,1,16`) | DONE |
| Preprocessing → `processed/slakh100_44k_lora` (23 GB, 60 s segments) | DONE |
| Seven dataset gates (per split + all): LICENSE, PROVENANCE, DUPLICATE, QUALITY, METADATA, SPLIT-LEAKAGE, ACE-STEP FORMAT | **ALL PASS** (`v1/gates_*.json`) |
| AUDIO FORMAT GATE (44.1k mono native → upstream 48k stereo conversion; no stereo loss possible) | **PASS** (evidence in `docs/SLAKH100_DESIGN.md`) |
| Trainer dataset JSON (`acestep_lora/slakh100/dataset.json`) | DONE — **7,252 samples** (train split only; upstream consumes ONLY this list, dir never scanned) |
| ACE-Step weights (bundle + `acestep-v15-xl-turbo`) → `/content/checkpoints` | DONE — 28 GB (t2 marker set) |
| flash-attn wheel | **VERIFIED + CACHED** (2026-08-19 ~00:00 UTC). `flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl`, SHA256 `f5ca206997514e0fd1b68d5111f775cd25700916584897f0443e814d8e2f9d40`, gate PASS (fresh-process import + CUDA bf16 fwd/bwd, finite outputs/grads) on Python 3.12.13 / torch 2.10.0+cu128 / CUDA 12.8 / nvcc r12.8 / L4 cc 8.9. Cached to Drive `v1/wheels/` with `flash_attn_env.json`. Future runtimes: `pip install v1/wheels/flash_attn*.whl` — never recompile. |
| torchcodec | requirements resolve to 0.11.0+cu128 which CANNOT load with torch 2.10.0+cu128 (core4: `undefined symbol torch_dtype_float4_e2m1fn_x2`; core5/6/7 need FFmpeg≥5, Colab has 4.4). **Fix: `torchcodec==0.10.0+cu128` — verified decode of a real processed WAV.** Pinned in `v1_train.sh` t1. |
| t3 tensor preprocess | **RUNNING** since 2026-08-19 00:00 UTC (driver pid 152675, `setsid`-detached, log `v1/logs/t3_preprocess_tensors.log`). t3 CLI fix applied: `--dataset-dir/--output-dir` are required even with `--preprocess`; `--yes` does not exist in this trainer (removed from t4). Monitor cell in the notebook: `grep "==" driver.log`, tensor `.pt` count, log tails, `pgrep -c train.py fixed`. |

Runtime: Colab Pro, account authuser=1, NVIDIA **L4** (bf16 OK), 12 vCPU,
52 GB RAM, VM hostname suffix `6e875b2e672c`. Colab notebook: `Untitled2.ipynb`
(scratch; only the Drive-mount cell matters). Local code copy lives at
`/content/` (scripts/, src/, configs/, benchmarks/) — byte-identical to repo
`main`; if the VM recycled, re-transfer `scripts/ src/ configs/ benchmarks/`
from this repo (clipboard-paste base64 zip; repo is private, no tokens in Colab).

## IMPORTANT PATHS

- Drive V1 root: `/content/drive/MyDrive/songforge-dl/v1/`
- Markers (stage skip-list): `v1/markers/*.done` — present: 00–06, 05 re-verified, t0, t1, t2. **t1's marker is set but its wheel was bad — after the new wheel verifies, t1 stays done (env already installed except flash-attn).**
- tmux: the Colab Terminal pane attaches to tmux session shown as `[0] 0:bash*` (default session `0`). `tmux ls` / `tmux attach -t 0` from any new terminal.
- Build dir: `/tmp/pip-wheel-*/flash_attn_*` (objects); wheel output: `/content/wheels_out/`
- Wheel cache (Drive, currently EMPTY on purpose): `v1/wheels/`
- ACE-Step repo clone: `/content/ACE-Step-1.5`; weights: `/content/checkpoints` (28 GB, local — re-download via t2 if VM recycled: `python -m acestep.model_downloader --dir /content/checkpoints` then `--model acestep-v15-xl-turbo`)
- Tensor output (t3, not yet run): `v1/tensors/`
- Training checkpoints (t4): `v1/checkpoints/`
- Logs: `v1/logs/*.log`, `/content/driver.log`, `/content/fa_rebuild2.log` (dead attempt #2), attempt #3 output is in the tmux scrollback
- Verification gate script (local Mac scratchpad): `scratchpad/fa_gate.sh` — regenerate from §RECOVERY step 4 if lost

## EXACT RECOVERY COMMANDS

1. **Reconnect tmux** (Colab Terminal pane auto-attaches; from a fresh terminal): `tmux attach -t 0` (or just open the Terminal pane — never close it with its X).
2. **Is the build alive?** `pgrep -c nvcc; find /tmp -path "*flash*" -name "*.o" | wc -l; jobs` (in the tmux shell). Object count rising ⇒ alive. If dead with no wheel: relaunch exactly `MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 FLASH_ATTN_CUDA_ARCHS=89 pip wheel flash-attn==2.8.3.post1 --no-build-isolation --no-deps --no-cache-dir -w /content/wheels_out` **in the tmux foreground/background — not via a separate nohup shell, and never `pip install flash-attn` bare (pip cache may hold a poisoned wheel; it was purged once already).**
3. **Wheel done?** `ls -la /content/wheels_out/` → `flash_attn-2.8.3.post1-*.whl`.
4. **FLASH-ATTN VERIFICATION GATE** (all must pass BEFORE caching):
   ```
   pip install /content/wheels_out/flash_attn*.whl
   sha256sum /content/wheels_out/flash_attn*.whl
   python - <<'EOF'
   import sys, torch, subprocess
   test = '''
   import torch
   from flash_attn import flash_attn_func
   q,k,v=[torch.randn(1,64,4,64,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
   o=flash_attn_func(q,k,v,causal=True); assert torch.isfinite(o).all()
   o.sum().backward()
   assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in (q,k,v))
   print("FWD-BWD-PASS")
   '''
   r = subprocess.run([sys.executable,"-c",test],capture_output=True,text=True)
   print(r.stdout, r.stderr[-300:]); assert r.returncode==0, "GATE FAIL"
   print("FLASH-ATTN GATE: PASS")
   EOF
   ```
   Record wheel name, SHA256, `python -V`, torch/CUDA versions, GPU, cc 8.9.
5. **Cache verified wheel:** `cp /content/wheels_out/flash_attn*.whl /content/drive/MyDrive/songforge-dl/v1/wheels/` (+ write a small env-metadata JSON beside it). Cache ONLY on gate PASS.
6. **Resume pipeline at t3:** `cd /content && nohup bash scripts/v1_colab_driver.sh > /content/driver.log 2>&1 &` — markers skip 00–06/t0–t2; t3 (tensor preprocess) runs, then t4 (training) auto-starts. Monitor via a notebook cell (`!grep "==" /content/driver.log | tail`); the terminal websocket flaps — the notebook channel is reliable, and `get_page_text` reads it cleanly.

## FROZEN EXPERIMENT (benchmarks/EXPERIMENT_CARD.md is authoritative)

- Base: ACE-Step 1.5 **XL-turbo** (native layout in `/content/checkpoints`), trainer = official `training_v2` CLI (`python train.py fixed`), MIT.
- Dataset: Slakh-100, 7,252 × 60 s train samples, captions from corpus metadata, lyrics `[Instrumental]`.
- LoKr dim **64** / alpha **128**, `--lokr-weight-decompose`, lr **0.03**, batch **1**, grad-accum **4**, **bf16**, epochs **3**, `--save-every 1`, seed **20260818**, `--yes`; auto `--resume-from` latest `v1/checkpoints/checkpoints/epoch_*`.
- Before full training: report tensor counts, matched modules > 0, trainable params > 0 + %, one optimizer smoke step (finite loss/grads, params change, peak VRAM). Loss decreasing does NOT equal success.
- Frozen-8 ablation: identical settings (seed 20260818, 60 s, bf16, 8 steps, shift 3.0), adapter the only change; control dir `foundation_benchmarks/ace_step_15_baseline_frozen` is read-only. Baseline human scores: violin≈7, rich_mix≈7, others≈3.5. Accept needs +1.0 mean on the weak six AND violin/rich_mix within 0.5, then the 51-prompt generalization benchmark (12 held-out prompts scored first-touch); verdict is **V1 ACCEPTED / REJECTED on generated audio quality and generalization**.
- Adapter loading for the ablation: `SONGFORGE_LORA` env → `src/songforge/generation/adapters/acestep.py` (hard-fails if 0 modules match).

## NEXT MILESTONES

FLASH-ATTN VERIFIED → TENSORS READY (report JSON records / tensor counts /
skips / GB) → LoKr attach proof → optimizer smoke step → TRAINING STARTED →
EPOCH COMPLETE ×3 (checkpoints on Drive) → V1 COMPLETE → baseline-vs-V1
generation → **V1 ACCEPTED / REJECTED**.
