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
# ON PYTHON
# Colab's image ships Python 3.13.15. ACE-Step requires <3.13 and so does
# nano-vllm, so BOTH refuse to install on the stock interpreter -- pip reports
# "requires a different Python" and acestep never lands, whatever else is
# fixed. The notebook kernel therefore stays 3.13 and is used only as a shell;
# a 3.12 interpreter is provisioned with uv and every real command runs through
# it. This mirrors provision_python312_training_env.sh, minus flash-attn, which
# that script needs for training and inference does not.
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
VENV=/content/venv-py312-acestep
PY="$VENV/bin/python"
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

s3_venv () {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo "FATAL: uv not on PATH"; return 1; }

  # A venv left over from an earlier attempt can hold a mismatched torchaudio
  # that no amount of reinstalling torch will correct, so honour an explicit
  # rebuild request rather than trusting whatever is already there.
  if [ "${REBUILD_VENV:-0}" = "1" ]; then rm -rf "$VENV"; fi
  if [ ! -x "$PY" ]; then
    uv python install 3.12 || return 1
    uv venv --python 3.12 "$VENV" || return 1
  fi
  "$PY" -c "import sys; assert sys.version_info[:2]==(3,12), sys.version" || return 1
  say "interpreter: $($PY -V)"

  # BOTH pinned, to the same minor. Pinning torch alone let uv resolve a
  # torchaudio built against a newer torch, and the mismatch does not surface
  # at install time -- it surfaces much later as
  #   libtorchaudio.abi3.so: undefined symbol: torch_dtype_float4_e2m1fn_x2
  # while ACE-Step's handler is importing, after the 28 GB download.
  uv pip install --python "$PY" --quiet \
      "torch==2.10.0" "torchaudio==2.10.0" \
      --index-url https://download.pytorch.org/whl/cu128 \
    || return 1

  # Prove the pair actually links before anything downstream depends on it.
  "$PY" - <<'EOF'
import torch, torchaudio
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)
assert torch.__version__.split("+")[0].rsplit(".", 1)[0] == \
       torchaudio.__version__.split("+")[0].rsplit(".", 1)[0], \
       "torch/torchaudio minor mismatch"
EOF
  "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "venv torch cannot see the GPU"
cap = torch.cuda.get_device_capability(0)
print(f"gpu {torch.cuda.get_device_name(0)}  cc {cap[0]}.{cap[1]}  bf16 native: {cap[0] >= 8}")
EOF
}

s4_acestep () {
  export PATH="$HOME/.local/bin:$PATH"
  if [ -d "$ACE/.git" ]; then git -C "$ACE" pull -q --ff-only || true
  else git clone -q https://github.com/ace-step/ACE-Step-1.5 "$ACE"; fi

  # On 3.12 this resolves; it was only ever rejected for the Python version.
  uv pip install --python "$PY" --quiet \
      "nano-vllm @ git+https://github.com/GeeeekExplorer/nano-vllm.git" \
    2>&1 | tail -2 || say "nano-vllm unavailable; continuing without it"

  # Still --no-deps + explicit list: flash-attn has no wheel for this pair and
  # compiling it costs about an hour before failing, and it is not needed for
  # inference (ACE-Step falls back to PyTorch SDPA).
  uv pip install --python "$PY" --quiet --no-deps -e "$ACE" 2>&1 | tail -2

  # The dependency list is PRINTED here and installed by uv below, not
  # installed from inside Python. A uv-created venv has no pip in it, so
  # `sys.executable -m pip install` fails with "No module named pip" -- and
  # because that failure was swallowed per-package, it installed nothing at
  # all while reporting success. The first sign was ModuleNotFoundError for
  # loguru at import time, a long way from the cause.
  "$PY" - "$ACE" > /tmp/acestep_deps.txt <<'PYDEP'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
deps, pyproject = [], root / "pyproject.toml"
if pyproject.exists():
    block = re.search(r"dependencies\s*=\s*\[(.*?)\]",
                      pyproject.read_text(encoding="utf-8"), re.S)
    if block:
        deps = re.findall(r'"([^"]+)"', block.group(1))
if not deps and (root / "requirements.txt").exists():
    deps = [l.strip() for l in (root / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]

skip = ("nano-vllm", "nano_vllm", "flash-attn", "flash_attn",
        "torch", "torchaudio", "torchvision")
for dep in deps:
    if not any(dep.lower().startswith(s) for s in skip):
        print(dep)
PYDEP

  count=$(wc -l < /tmp/acestep_deps.txt)
  say "installing $count declared dependencies into the venv"
  [ "$count" -gt 0 ] || { echo "FATAL: parsed zero dependencies from $ACE"; return 1; }
  xargs -a /tmp/acestep_deps.txt -d '\n' uv pip install --python "$PY" --quiet \
    || { echo "FATAL: dependency install failed"; return 1; }

  # Import the things whose absence previously surfaced only at model-load
  # time, so a broken environment fails here instead of after a 28 GB download.
  "$PY" -c "import numpy, soundfile, loguru; print('core deps ok')" \
    || { echo "FATAL: core dependencies missing after install"; return 1; }

  "$PY" -c "import acestep, pathlib; print('acestep ok at', pathlib.Path(acestep.__file__).parent)"
}

s5_weights () {
  cd "$ACE" || return 1
  "$PY" -m acestep.model_downloader --dir "$CKPT"
  "$PY" -m acestep.model_downloader --model acestep-v15-xl-turbo --dir "$CKPT"
  ln -sfn "$CKPT/acestep-v15-xl-turbo" "$CKPT/xl_turbo"
  du -sh "$CKPT"
}

step s1_gpu      s1_gpu
step s2_repo     s2_repo
step s3_venv     s3_venv
step s4_acestep  s4_acestep
step s5_weights  s5_weights

say "running base gate -> $OUT"
cd "$PROJECT" || exit 1
PYTHONPATH=src "$PY" scripts/run_generation_gates.py \
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
