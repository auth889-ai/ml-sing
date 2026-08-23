#!/usr/bin/env bash
# SongForge V1 training — ACE-Step 1.5 official training_v2 CLI (MIT), LoKr.
#
#   bash scripts/v1_train.sh <acestep_data_dir> <checkpoint_out_dir>
#
# Idempotent like the driver: each step markers on Drive, so a dropped runtime
# resumes with the same command. Trainer facts pinned from the upstream repo
# (commit 14c0211): headless CLI `train.py fixed`, dataset JSON + two-pass
# preprocess to .pt tensors, epoch-boundary checkpoints with --resume-from.
# The Gradio/Side-Step standalone (CC BY-NC-SA) is NOT used; acestep/training_v2
# inside the official repo is MIT.
set -uo pipefail

DATA_DIR="${1:?data dir}"
CKPT_OUT="${2:?checkpoint out dir}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/songforge-dl}"
V1="$DRIVE_ROOT/v1"
MARK="$V1/markers"
ACE=/content/ACE-Step-1.5
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$MARK" "$V1/logs" "$CKPT_OUT"

# Every training command runs through a pinned interpreter, never bare `python`.
# Colab's image moved to Python 3.13 / torch 2.11 on 2026-08-22, which the
# verified flash-attn wheel cannot load; provision_python312_training_env.sh
# rebuilds the verified 3.12 / torch 2.10 stack in a venv and this points at it.
# The notebook kernel stays 3.13 and is only a shell.
VENV="${VENV:-/content/venv-py312-acestep}"
PY="${SONGFORGE_PY:-$VENV/bin/python}"
[ -x "$PY" ] || PY=python

# Colab's kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline,
# and everything launched from a notebook cell inherits it. matplotlib inside
# the venv has no matplotlib_inline, so it raises at import time:
#   ValueError: Key backend: 'module://matplotlib_inline.backend_inline'
#              is not a valid value for backend
# The failure surfaces a long way from its cause — as a lightning ->
# torchmetrics -> matplotlib import chain crash during trainer startup, with
# nothing in the traceback naming Colab or the environment. Pin a headless
# backend so the trainer never depends on where it was launched from.
export MPLBACKEND=Agg

# Markers must live where their artifacts live. t1 (pip environment) and t2
# (28 GB of weights on local disk) are destroyed by a runtime recycle, but a
# marker on Drive survives it — so the driver would report "already done" on a
# fresh VM and then fail in training with nothing installed. Stages whose
# output is local therefore mark locally, and are correctly redone after a
# recycle; Drive-backed stages (dataset, gates, tensors) keep Drive markers.
LOCAL_MARK=/content/markers
mkdir -p "$LOCAL_MARK"

stage_local() {
  local name="$1" fn="$2"
  if [ -f "$LOCAL_MARK/$name.done" ]; then echo "== $name: already done (this VM)"; return 0; fi
  echo "== $name: starting $(date -u +%H:%M:%S)"
  if "$fn" 2>&1 | tee -a "$V1/logs/$name.log"; then
    touch "$LOCAL_MARK/$name.done"; echo "== $name: DONE"
  else
    echo "== $name: FAILED"; exit 1
  fi
}

stage() {
  local name="$1" fn="$2"
  if [ -f "$MARK/$name.done" ]; then echo "== $name: already done"; return 0; fi
  echo "== $name: starting $(date -u +%H:%M:%S)"
  if "$fn" 2>&1 | tee -a "$V1/logs/$name.log"; then
    touch "$MARK/$name.done"; echo "== $name: DONE"
  else
    echo "== $name: FAILED"; exit 1
  fi
}

t0_dataset_json() {
  "$PY" "$REPO_ROOT/scripts/build_acestep_training_json.py" \
      --manifest "$DRIVE_ROOT/processed/slakh100_44k_lora/manifests" \
      --audio-root "$DRIVE_ROOT/processed/slakh100_44k_lora" \
      --output "$DATA_DIR/dataset.json" \
      --split train
}

t1_install() {
  # Delegated: the whole environment question is one concern and it is messy
  # enough to deserve its own file. The provisioner creates the Python 3.12 /
  # torch 2.10 venv the cached flash-attn wheel was verified against, installs
  # ACE-Step into it, and re-runs the flash-attn fwd/bwd gate. It exits non-zero
  # on any gate failure, so a bad environment can never reach training.
  bash "$REPO_ROOT/scripts/provision_python312_training_env.sh" || return 1
  PY="$VENV/bin/python"
  [ -x "$PY" ] || { echo "provisioner did not produce $PY"; return 1; }
}

t2_weights() {
  cd "$ACE"
  # VAE + text encoder from the main bundle, plus the XL-turbo DiT. Local disk;
  # cheap to re-fetch relative to a Drive round-trip. The console script is not
  # always on PATH after an editable install, so invoke the module directly.
  "$PY" -m acestep.model_downloader --dir /content/checkpoints
  "$PY" -m acestep.model_downloader --model acestep-v15-xl-turbo --dir /content/checkpoints
  # The downloader stores the DiT under its HF repo name, but the trainer's
  # official --model-variant xl_turbo expects checkpoints/xl_turbo (observed:
  # "[FAIL] Model directory not found: /content/checkpoints/xl_turbo", exit 0).
  ln -sfn /content/checkpoints/acestep-v15-xl-turbo /content/checkpoints/xl_turbo
  du -sh /content/checkpoints
}

t3_preprocess_tensors() {
  cd "$ACE"
  # Tensors go to Drive so a runtime recycle does not repeat this pass.
  # --dataset-dir/--output-dir are required by the CLI even in preprocess mode
  # (verified against train.py fixed --help on the live runtime).
  "$PY" train.py fixed \
      --checkpoint-dir /content/checkpoints \
      --model-variant xl_turbo \
      --dataset-dir "$V1/tensors" \
      --output-dir "$CKPT_OUT" \
      --preprocess \
      --audio-dir "$DATA_DIR" \
      --dataset-json "$DATA_DIR/dataset.json" \
      --tensor-output "$V1/tensors" \
      --max-duration 240
  du -sh "$V1/tensors"
}

t4_train() {
  # The trainer sanitises every user path against a safe root that is simply
  # the working directory at import time (acestep/training/path_safety.py:
  # _SAFE_ROOT = _resolve(os.getcwd())). Running from the repo therefore
  # rejects our Drive paths outright:
  #   ValueError: Path escapes safe root: '.../v1/tensors' root='/content/ACE-Step-1.5'
  # Run from /content instead, which contains both the repo and the Drive
  # mount, so the guard still applies but admits the checkpoint/tensor dirs.
  cd /content

  # A starved trainer from an earlier attempt still holds VRAM and the dataset
  # lock; clear it before starting a new one.
  pkill -f "train.py fixed" 2>/dev/null && sleep 5 || true

  # Stage the tensors on LOCAL disk. Drive stays the durable store, but it must
  # not sit in the per-step read path: the Colab FUSE mount stalls under
  # sustained random reads (observed: dataloader blocked, GPU pinned at 0%,
  # while a plain `ls` on the mount timed out after 15 s). The stager verifies
  # count, zero-byte files, archive integrity and a random torch.load sample,
  # and fails loudly rather than training on a partial corpus.
  LOCAL_TENSORS=/content/tensors_local
  "$PY" "$REPO_ROOT/scripts/v1_stage_tensors.py" \
      --source "$V1/tensors" --dest "$LOCAL_TENSORS" \
      --sample 25 --report "$V1/staging_report.json" || {
    echo "FATAL: tensor staging failed verification"; return 1; }

  # Continuous status for humans, so nobody has to poll a notebook.
  pkill -f v1_watch.py 2>/dev/null || true
  setsid nohup "$PY" "$REPO_ROOT/scripts/v1_watch.py" \
      --log "$V1/logs/t4_train.log" \
      --checkpoints "$CKPT_OUT" \
      --local-status /content/v1_status.json \
      --drive-status "$V1/status.json" \
      --interval 60 > /content/v1_watch.log 2>&1 < /dev/null &
  echo "status monitor started (v1/status.json)"

  RESUME=()
  latest=$(ls -d "$CKPT_OUT"/checkpoints/epoch_* 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "${latest:-}" ]; then
    echo "resuming from $latest"
    RESUME=(--resume-from "$latest")
  fi
  # Per the frozen experiment card: LoKr dim 64 / alpha 128, lr 0.03 in the
  # weight-decompose regime, batch 1 x grad-accum 4, bf16. --save-every 1 keeps
  # the loss horizon inside one epoch on a runtime that drops every 10-40 min.
  # The trainer asks "Start training? [Y/n]" interactively (no CLI flag for it
  # in this build; a detached stdin hangs forever at the prompt) — answer via
  # stdin. printf, not `yes`: pipefail would turn yes's SIGPIPE into failure.
  printf 'Y\n' | "$PY" "$ACE/train.py" fixed \
      --checkpoint-dir /content/checkpoints \
      --model-variant xl_turbo \
      --adapter-type lokr \
      --dataset-dir "$LOCAL_TENSORS" \
      --output-dir "$CKPT_OUT" \
      --lokr-linear-dim 64 --lokr-linear-alpha 128 --lokr-weight-decompose \
      --lr 0.03 \
      --batch-size 1 --gradient-accumulation 4 \
      --precision bf16 \
      --epochs 3 --save-every 1 \
      --seed 20260818 "${RESUME[@]}"
}

stage t0_dataset_json      t0_dataset_json
stage_local t1_install     t1_install
stage_local t2_weights     t2_weights
stage t3_preprocess_tensors t3_preprocess_tensors
# NOTE: t4 has no marker on purpose — rerunning always resumes training from
# the latest checkpoint until the epoch budget completes.
#
# The exit status must be the TRAINER's, not tee's. Without PIPESTATUS this
# script returned 0 even when training died, so the driver marked 07_train
# "done" and every later run skipped training entirely — a failure that
# silently presents itself as a finished experiment.
t4_train 2>&1 | tee -a "$V1/logs/t4_train.log"
train_status=${PIPESTATUS[0]}
if [ "$train_status" -ne 0 ]; then
  echo "TRAINING FAILED (exit $train_status); refusing to report success"
  exit "$train_status"
fi
echo "training run finished; checkpoints in $CKPT_OUT"
