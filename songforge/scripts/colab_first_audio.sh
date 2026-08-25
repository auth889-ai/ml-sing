#!/usr/bin/env bash
# One paste, from a fresh free-tier Colab T4 to a real generated WAV.
#
#   !bash <(curl -sL https://raw.githubusercontent.com/auth889-ai/ml-sing/main/songforge/scripts/colab_first_audio.sh)
#
# WHY A SEPARATE SCRIPT FROM colab_recover.sh
# That one restores a recycled VM that already has Drive laid out, and assumes
# the venv and weights were built by the V1 training path. This one assumes
# nothing: it works on a runtime that has never seen this project, and its only
# goal is the single unanswered question -- does the foundation generate audio
# on this machine.
#
# ON THE FREE TIER SPECIFICALLY
# A free T4 is compute capability 7.5, so bf16 is emulated and this model
# produces NaNs in it. The gate selects fp16 automatically, which is native
# from 7.0 up; 4,991,023,206 parameters at two bytes is roughly 10 GB against
# the T4's 15 GB, so it fits but not comfortably. If it fails, it should fail
# on memory, loudly, rather than by silently emitting noise.
#
# Flash-attention is not installed: the cached wheel was built for sm_89 and a
# T4 is sm_75. Inference falls back to PyTorch SDPA.
set -uo pipefail

REPO=/content/ml-sing
PROJECT="$REPO/songforge"
ACE=/content/ACE-Step-1.5
CKPT=/content/checkpoints
DRIVE=/content/drive/MyDrive/songforge-dl
MARK=/content/.first_audio_markers
mkdir -p "$MARK" "$CKPT"
export MPLBACKEND=Agg
export PATH="$HOME/.local/bin:$PATH"

say () { echo "== $(date -u +%H:%M:%S) $*"; }
step () {
  local name="$1"; shift
  [ -f "$MARK/$name" ] && { say "$name: already done"; return 0; }
  say "$name: starting"
  if "$@"; then touch "$MARK/$name"; say "$name: done";
  else say "$name: FAILED"; exit 1; fi
}

# Output goes to Drive when it is mounted, because a free runtime can vanish
# between two statements and the WAV is the entire point of the exercise.
OUT=/content/gate_out
[ -d "$DRIVE" ] && OUT="$DRIVE/gate_out"

s1_gpu () {
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || {
    echo "FATAL: no GPU. Runtime -> Change runtime type -> T4 GPU"; return 1; }
}

s2_repo () {
  if [ -d "$REPO/.git" ]; then git -C "$REPO" pull -q --ff-only || true
  else git clone -q https://github.com/auth889-ai/ml-sing "$REPO"; fi
  [ -d "$PROJECT" ] || { echo "FATAL: $PROJECT missing"; return 1; }
}

s3_acestep () {
  if [ -d "$ACE/.git" ]; then git -C "$ACE" pull -q --ff-only || true
  else git clone -q https://github.com/ace-step/ACE-Step-1.5 "$ACE"; fi
  # Colab already ships a working torch/CUDA; installing our own would waste
  # ten minutes and risk breaking a stack that is already correct for this GPU.
  pip install -q -e "$ACE" 2>&1 | tail -2 || \
    pip install -q -r "$ACE/requirements.txt" 2>&1 | tail -2 || true
  python -c "import acestep; print('acestep ok')"
}

s4_weights () {
  cd "$ACE" || return 1
  python -m acestep.model_downloader --dir "$CKPT"
  python -m acestep.model_downloader --model acestep-v15-xl-turbo --dir "$CKPT"
  ln -sfn "$CKPT/acestep-v15-xl-turbo" "$CKPT/xl_turbo"
  du -sh "$CKPT"
}

step s1_gpu      s1_gpu
step s2_repo     s2_repo
step s3_acestep  s3_acestep
step s4_weights  s4_weights

say "running base gate -> $OUT"
cd "$PROJECT" || exit 1
PYTHONPATH=src python scripts/run_generation_gates.py \
    --stage base --ace-dir "$ACE" --checkpoint-dir "$CKPT" \
    --out "$OUT" --duration 30
status=$?

echo
echo "=================================================================="
if [ -f "$OUT/base_gate.wav" ]; then
  ls -la "$OUT/base_gate.wav"
  echo "BASE AUDIO EXISTS -> $OUT/base_gate.wav"
  echo "Send back $OUT/base_gate.json"
else
  echo "NO AUDIO PRODUCED. The report says why:"
  echo "  $OUT/base_gate.json"
fi
echo "=================================================================="
exit $status
