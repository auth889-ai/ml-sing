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
  python "$REPO_ROOT/scripts/build_acestep_training_json.py" \
      --manifest "$DRIVE_ROOT/processed/slakh100_44k_lora/manifests" \
      --audio-root "$DRIVE_ROOT/processed/slakh100_44k_lora" \
      --output "$DATA_DIR/dataset.json" \
      --split train
}

t1_install() {
  [ -d "$ACE" ] || git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "$ACE"
  cd "$ACE"
  python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,13), sys.version"
  # flash-attn compiles from source (~1 h for all archs). Reuse a wheel cached
  # on Drive from a previous VM; otherwise build only the L4's sm_89 with more
  # parallel jobs, then cache the result so a runtime recycle never pays twice.
  mkdir -p "$V1/wheels"
  pip install -q "$V1"/wheels/flash_attn*.whl 2>/dev/null && echo "flash-attn from Drive cache" || true
  # flash-attn MUST build without pip's build isolation: the isolated env
  # provisions its own torch and the extension links against the wrong ABI
  # (observed: undefined symbol _ZNK3c104cuda10CUDAStream5queryEv). Build the
  # wheel explicitly against the installed torch, then install everything else.
  export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.9" FLASH_ATTN_CUDA_ARCHS="89"
  if ! python -c "import flash_attn" 2>/dev/null; then
    pip install -q -r <(grep -v "^flash-attn" requirements.txt)
    pip wheel -q flash-attn --no-build-isolation --no-deps --no-cache-dir -w /content/wheels_out
    pip install -q /content/wheels_out/flash_attn*.whl
  fi
  pip install -q -r requirements.txt
  # requirements pins torchcodec>=0.9.1 which resolves to 0.11.0+cu128 — that
  # build cannot load against torch 2.10.0+cu128 (libtorchcodec_core4.so:
  # undefined symbol torch_dtype_float4_e2m1fn_x2; core5/6/7 need FFmpeg>=5,
  # Colab ships 4.4). 0.10.0+cu128 loads and decodes (verified on the L4 VM).
  pip install -q "torchcodec==0.10.0" --extra-index-url https://download.pytorch.org/whl/cu128
  pip install -q -e .
  cp -n /content/wheels_out/flash_attn*.whl "$V1/wheels/" 2>/dev/null || true
  python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)"
  python -c "import peft, lycoris; print('training deps ok')"
}

t2_weights() {
  cd "$ACE"
  # VAE + text encoder from the main bundle, plus the XL-turbo DiT. Local disk;
  # cheap to re-fetch relative to a Drive round-trip. The console script is not
  # always on PATH after an editable install, so invoke the module directly.
  python -m acestep.model_downloader --dir /content/checkpoints
  python -m acestep.model_downloader --model acestep-v15-xl-turbo --dir /content/checkpoints
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
  python train.py fixed \
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
  cd "$ACE"
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
  printf 'Y\n' | python train.py fixed \
      --checkpoint-dir /content/checkpoints \
      --model-variant xl_turbo \
      --adapter-type lokr \
      --dataset-dir "$V1/tensors" \
      --output-dir "$CKPT_OUT" \
      --lokr-linear-dim 64 --lokr-linear-alpha 128 --lokr-weight-decompose \
      --lr 0.03 \
      --batch-size 1 --gradient-accumulation 4 \
      --precision bf16 \
      --epochs 3 --save-every 1 \
      --seed 20260818 "${RESUME[@]}"
}

stage t0_dataset_json      t0_dataset_json
stage t1_install           t1_install
stage t2_weights           t2_weights
stage t3_preprocess_tensors t3_preprocess_tensors
# NOTE: t4 has no marker on purpose — rerunning always resumes training from
# the latest checkpoint until the epoch budget completes.
t4_train 2>&1 | tee -a "$V1/logs/t4_train.log"
echo "training run finished; checkpoints in $CKPT_OUT"
