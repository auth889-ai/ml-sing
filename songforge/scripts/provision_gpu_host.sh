#!/usr/bin/env bash
# Bring a bare Linux GPU host to the point where run_generation_gates.py runs.
#
#   bash scripts/provision_gpu_host.sh
#
# Written for a persistent rented box (Thunder Compute, Lambda, RunPod), not
# Colab. Everything lives under $HOME on local disk: a persistent host has no
# Drive mount, and the FUSE latency that shaped the Colab scripts does not
# apply here.
#
# Idempotent. Each step marks itself under ~/.songforge_markers and is skipped
# on a rerun, so a dropped SSH session is resumed with the same command rather
# than restarted.
#
# ON FLASH-ATTENTION
# The wheel cached during the Colab work was compiled for sm_89 (L4). An
# A6000 is sm_86 and cannot load it. Compiling from source takes roughly an
# hour of GPU-host time. Inference does not need it -- ACE-Step falls back to
# PyTorch scaled_dot_product_attention -- so this script does NOT install it
# and does not fail without it. Training is a different calculation; revisit
# it then, deliberately.
set -uo pipefail

REPO="${REPO:-$HOME/ml-sing}"
PROJECT="$REPO/songforge"
ACE="${ACE:-$HOME/ACE-Step-1.5}"
CKPT="${CKPT:-$HOME/checkpoints}"
VENV="${VENV:-$HOME/venv-acestep}"
PY="$VENV/bin/python"
MARK="$HOME/.songforge_markers"
mkdir -p "$MARK" "$CKPT"

step () {
  local name="$1"; shift
  if [ -f "$MARK/$name.done" ]; then echo "== $name: already done"; return 0; fi
  echo "== $name: starting $(date -u +%H:%M:%S)"
  if "$@"; then touch "$MARK/$name.done"; echo "== $name: DONE";
  else echo "== $name: FAILED"; exit 1; fi
}

s1_system () {
  sudo apt-get update -qq || true
  sudo apt-get install -y -qq git curl ffmpeg libsndfile1 build-essential || true
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

s2_repo () {
  export PATH="$HOME/.local/bin:$PATH"
  if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only
  else git clone https://github.com/auth889-ai/ml-sing "$REPO"; fi
  # The project directory was renamed from songforge-dl-starter to songforge;
  # fail loudly here rather than 40 minutes later inside the gate.
  [ -d "$PROJECT" ] || { echo "FATAL: $PROJECT missing after clone"; return 1; }
}

s3_venv () {
  export PATH="$HOME/.local/bin:$PATH"
  # Python 3.12 + torch 2.10/cu128 is the combination verified during V1.
  [ -x "$PY" ] || uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" --quiet \
      torch==2.10.0 torchaudio --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$PY" --quiet \
      soundfile numpy scipy pyyaml safetensors einops transformers diffusers \
      accelerate peft
  "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "no CUDA device visible"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
print(f"gpu {name}  cc {cap[0]}.{cap[1]}  vram {total/2**30:.1f} GiB")
# bf16 below 8.0 is emulated and this model produces NaNs in it.
print("bf16 native:", cap[0] >= 8)
EOF
}

s4_acestep () {
  export PATH="$HOME/.local/bin:$PATH"
  if [ -d "$ACE/.git" ]; then git -C "$ACE" pull --ff-only || true
  else git clone https://github.com/ace-step/ACE-Step-1.5 "$ACE"; fi
  uv pip install --python "$PY" --quiet -e "$ACE" || \
    uv pip install --python "$PY" --quiet -r "$ACE/requirements.txt" || true
  "$PY" -c "import acestep, pathlib; print('acestep at', pathlib.Path(acestep.__file__).parent)"
}

s5_weights () {
  cd "$ACE" || return 1
  "$PY" -m acestep.model_downloader --dir "$CKPT"
  "$PY" -m acestep.model_downloader --model acestep-v15-xl-turbo --dir "$CKPT"
  # The trainer and handler expect checkpoints/xl_turbo; the downloader stores
  # the DiT under its HF repo name.
  ln -sfn "$CKPT/acestep-v15-xl-turbo" "$CKPT/xl_turbo"
  du -sh "$CKPT"
}

step s1_system    s1_system
step s2_repo      s2_repo
step s3_venv      s3_venv
step s4_acestep   s4_acestep
step s5_weights   s5_weights

cat <<EOF

===================================================================
Provisioned. Base inference gate (no adapter) -- run this first:

  cd $PROJECT && PYTHONPATH=src $PY scripts/run_generation_gates.py \\
      --stage base --ace-dir $ACE --checkpoint-dir $CKPT \\
      --out \$HOME/gate_out --duration 30

Then, once base_gate.wav exists and is not silent:

  cd $PROJECT && PYTHONPATH=src $PY scripts/run_generation_gates.py \\
      --stage all --ace-dir $ACE --checkpoint-dir $CKPT \\
      --adapter <path-to-v1-lokr> --out \$HOME/gate_out
===================================================================
EOF
