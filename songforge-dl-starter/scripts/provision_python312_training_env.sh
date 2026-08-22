#!/usr/bin/env bash
# Rebuild the VERIFIED SongForge training environment on a Colab image that no
# longer ships the Python it was verified against.
#
#   bash scripts/provision_python312_training_env.sh
#
# WHY THIS EXISTS
# ---------------
# The V1 experiment was verified on Colab's Python 3.12 / torch 2.10.0+cu128
# image, and the flash-attn wheel cached on Drive
# (flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl, SHA256 f5ca2069…8e2f9d40)
# was compiled against exactly that pair. On 2026-08-22 the Colab image moved to
# Python 3.13.15 / torch 2.11.0+cu128 and Colab removed the fallback-runtime
# option, so the cached wheel is unusable on two counts at once: cp312 ABI tag,
# and a torch minor it was not linked against.
#
# The alternative was recompiling flash-attn for cp313/torch2.11 — roughly an
# hour of L4 time plus a fresh pass through the verification gate, on a sprint
# clock. Instead we reproduce the environment the wheel was verified in: a
# Python 3.12 interpreter provisioned by uv, torch 2.10.0+cu128 pinned into it,
# and the already-gate-passed wheel installed unchanged.
#
# The notebook kernel stays 3.13; it is only a shell. Every training command
# runs through $VENV/bin/python, which this script prints on success.
#
# The flash-attn verification gate is re-run here even though the wheel passed
# once before. A wheel that imports is not a wheel that computes: the gate does
# a fresh-process import, a CUDA bf16 forward AND backward, and asserts finite
# non-zero gradients. Caching a wheel that only imports is how the previous
# poisoned-wheel incident happened.
set -uo pipefail

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/songforge-dl}"
V1="$DRIVE_ROOT/v1"
VENV="${VENV:-/content/venv-py312-acestep}"
ACE="${ACE:-/content/ACE-Step-1.5}"
TORCH_VERSION="2.10.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
WHEEL_GLOB="$V1/wheels/flash_attn-*-cp312-cp312-linux_x86_64.whl"

say() { echo "== $(date -u +%H:%M:%S) $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. uv + 3.12
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || die "uv install failed"
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv not on PATH after install"

if [ ! -x "$VENV/bin/python" ]; then
  say "provisioning CPython 3.12 and creating $VENV"
  uv python install 3.12 || die "uv python install 3.12 failed"
  uv venv --python 3.12 "$VENV" || die "uv venv failed"
fi
PY="$VENV/bin/python"
"$PY" -c "import sys; assert sys.version_info[:2]==(3,12), sys.version" \
  || die "venv interpreter is not 3.12"
say "interpreter: $($PY -V)"

# Everything below installs INTO the venv. uv pip is used for speed but the
# target is pinned explicitly so nothing can leak into the 3.13 system env.
upip() { uv pip install --python "$PY" "$@"; }

# ------------------------------------------------------------------- 2. torch
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VERSION') else 1)" 2>/dev/null; then
  say "installing torch $TORCH_VERSION+cu128 (matches the cached wheel's link target)"
  upip "torch==$TORCH_VERSION" torchaudio --index-url "$TORCH_INDEX" || die "torch install failed"
fi
"$PY" - <<'EOF' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not visible from the venv"
assert torch.cuda.is_bf16_supported(), "bf16 unsupported on this GPU"
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
EOF

# --------------------------------------------------------------- 3. ACE-Step
[ -d "$ACE" ] || git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "$ACE" \
  || die "ACE-Step clone failed"

say "installing ACE-Step requirements (flash-attn excluded — cached wheel is used)"
grep -v "^flash-attn" "$ACE/requirements.txt" > /tmp/acestep_requirements_no_flash_attn.txt
upip -r /tmp/acestep_requirements_no_flash_attn.txt || die "requirements install failed"

# requirements.txt does not pin torch, so a transitive dependency can silently
# pull a different build and break the flash-attn ABI. Re-pin if that happened;
# an ABI mismatch here surfaces as an undefined-symbol crash mid-training.
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VERSION') else 1)" 2>/dev/null; then
  say "requirements moved torch off $TORCH_VERSION — re-pinning"
  upip "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX" || die "torch re-pin failed"
fi

# torchcodec 0.11.0+cu128 cannot load against this torch (undefined symbol
# torch_dtype_float4_e2m1fn_x2); 0.10.0 is the verified decoder. Only the
# preprocessing path needs it, but a broken import fails the trainer at import.
upip "torchcodec==0.10.0" --index-url "$TORCH_INDEX" 2>/dev/null \
  || say "WARNING: torchcodec 0.10.0 unavailable — tensors are already built, continuing"

upip -e "$ACE" || die "ACE-Step editable install failed"

# ---------------------------------------------------- 4. flash-attn + its gate
WHEEL=$(ls $WHEEL_GLOB 2>/dev/null | head -1)
[ -n "$WHEEL" ] || die "cached cp312 flash-attn wheel not found under $V1/wheels/"
say "installing cached wheel $(basename "$WHEEL")"
echo "sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
upip "$WHEEL" || die "flash-attn wheel install failed"

say "flash-attn verification gate (fresh process, CUDA bf16 fwd+bwd)"
"$PY" - <<'EOF' || die "FLASH-ATTN GATE FAILED — do not train on this environment"
import subprocess, sys
test = '''
import torch
from flash_attn import flash_attn_func
q, k, v = [torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
o = flash_attn_func(q, k, v, causal=True)
assert torch.isfinite(o).all(), "non-finite forward output"
o.sum().backward()
for t in (q, k, v):
    assert t.grad is not None, "missing gradient"
    assert torch.isfinite(t.grad).all(), "non-finite gradient"
    assert t.grad.abs().sum() > 0, "all-zero gradient"
import flash_attn
print("FWD-BWD-PASS", flash_attn.__version__)
'''
r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print(r.stderr[-2000:], file=sys.stderr)
    sys.exit(1)
print("FLASH-ATTN GATE: PASS")
EOF

"$PY" -c "import peft, lycoris; print('training deps ok')" || die "peft/lycoris missing"

# ------------------------------------------------------------------ 5. record
mkdir -p "$V1"
"$PY" - "$VENV" "$WHEEL" > "$V1/python312_training_env.json" <<'EOF'
import hashlib, json, subprocess, sys, torch
venv, wheel = sys.argv[1], sys.argv[2]
print(json.dumps({
    "venv": venv,
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "flash_attn_wheel": wheel.rsplit("/", 1)[-1],
    "flash_attn_sha256": hashlib.sha256(open(wheel, "rb").read()).hexdigest(),
    "reason": "Colab image moved to py3.13/torch2.11; verified stack rebuilt in a 3.12 venv",
}, indent=2))
EOF
cat "$V1/python312_training_env.json"

say "ENVIRONMENT READY"
echo "SONGFORGE_PY=$PY"
