# SongForge generation gates -- Kaggle kernel body.
#
# WHY KAGGLE
# Colab recycled the runtime mid-run and took /content with it: the venv, 28 GB
# of weights and the generated WAV. Kaggle gives 12-hour sessions, writes
# results to /kaggle/working where `kaggle kernels output` can fetch them, and
# ships Python 3.11 -- which already satisfies ACE-Step's <3.13 requirement, so
# none of the uv/3.12 provisioning is needed here.
#
# Everything below runs headless. There is no cell to babysit.

import json, os, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
ACE = Path("/kaggle/ace")
CKPT = WORK / "checkpoints"
REPO = Path("/kaggle/ml-sing")
OUT = WORK / "gate_out"
for d in (CKPT, OUT):
    d.mkdir(parents=True, exist_ok=True)


def run(cmd, check=True, quiet=False):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    if quiet and r.returncode != 0:
        print((r.stdout or "")[-2000:], (r.stderr or "")[-2000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"failed: {cmd}")
    return r


print("=" * 70, flush=True)
import torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}", flush=True)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"gpu {torch.cuda.get_device_name(0)}  cc {cap[0]}.{cap[1]}  "
          f"bf16 native: {cap[0] >= 8}", flush=True)
else:
    raise SystemExit("no GPU -- enable the accelerator in kernel settings")
print("=" * 70, flush=True)

# --- source -------------------------------------------------------------
if not REPO.exists():
    run(["git", "clone", "-q", "https://github.com/auth889-ai/ml-sing", str(REPO)])
if not ACE.exists():
    run(["git", "clone", "-q", "https://github.com/ace-step/ACE-Step-1.5", str(ACE)])

# --- dependencies -------------------------------------------------------
# Kaggle's torch matches its driver; replacing it is how the environment
# breaks. torchaudio must match torch's MINOR or libtorchaudio fails to
# dlopen with an undefined-symbol error much later, during model load.
tv = torch.__version__.split("+")[0]
print(f"pinning torchaudio to torch {tv}", flush=True)
run([sys.executable, "-m", "pip", "install", "-q", f"torchaudio=={tv}"], check=False)

# ACE-Step declares nano-vllm (GitHub only) and flash-attn (no wheel, and
# compiling costs about an hour before failing). Install --no-deps, then every
# other declared dependency explicitly.
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ACE)], check=False)

deps = []
req = ACE / "requirements.txt"
if req.exists():
    deps += [l.strip() for l in req.read_text().splitlines()
             if l.strip() and not l.startswith(("#", "-"))]
import re
pyproject = ACE / "pyproject.toml"
if pyproject.exists():
    block = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject.read_text(), re.S)
    if block:
        deps += re.findall(r'"([^"]+)"', block.group(1))

skip = ("nano-vllm", "nano_vllm", "flash-attn", "flash_attn",
        "torch==", "torchaudio==", "torchvision==", "torch>=", "torchaudio>=")
seen, wanted = set(), []
for d in deps:
    key = re.split(r"[<>=;\[]", d, 1)[0].strip().lower()
    if any(d.lower().startswith(s) for s in skip) or key in seen or not key:
        continue
    seen.add(key)
    wanted.append(d)
print(f"installing {len(wanted)} dependencies", flush=True)
run([sys.executable, "-m", "pip", "install", "-q"] + wanted, check=False)
run([sys.executable, "-m", "pip", "install", "-q", "peft", "lycoris-lora"], check=False)

import importlib
for mod in ("numpy", "soundfile", "loguru", "peft", "vector_quantize_pytorch"):
    importlib.import_module(mod)
print("core deps ok", flush=True)

# --- weights ------------------------------------------------------------
os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(CKPT)
os.chdir(ACE)
run([sys.executable, "-m", "acestep.model_downloader", "--dir", str(CKPT)])
run([sys.executable, "-m", "acestep.model_downloader",
     "--model", "acestep-v15-xl-turbo", "--dir", str(CKPT)])
print(subprocess.run(["du", "-sh", str(CKPT)], capture_output=True, text=True).stdout, flush=True)

# --- gates --------------------------------------------------------------
os.chdir(REPO / "songforge")
env = dict(os.environ, PYTHONPATH="src", ACESTEP_CHECKPOINTS_DIR=str(CKPT))
stage = os.environ.get("SONGFORGE_STAGE", "matrix")
cmd = [sys.executable, "scripts/run_generation_gates.py",
       "--stage", stage,
       "--ace-dir", str(ACE),
       "--checkpoint-dir", str(CKPT),
       "--out", str(OUT),
       "--duration", "30"]
adapter = os.environ.get("SONGFORGE_LORA", "")
if adapter:
    cmd += ["--adapter", adapter]
print(f"$ {' '.join(cmd)}", flush=True)
subprocess.run(cmd, env=env)

# --- inventory ----------------------------------------------------------
print("\n" + "=" * 70, flush=True)
wavs = sorted(OUT.rglob("*.wav"))
print(f"GENERATED {len(wavs)} WAV FILE(S)", flush=True)
for w in wavs:
    print(f"  {w.relative_to(WORK)}  {w.stat().st_size:,} bytes", flush=True)
print("=" * 70, flush=True)

# Keep the outputs small enough to download: drop the 28 GB of weights and the
# per-render scratch directories from the kernel's output snapshot.
import shutil
shutil.rmtree(CKPT, ignore_errors=True)
for scratch in OUT.glob("_*"):
    shutil.rmtree(scratch, ignore_errors=True)
