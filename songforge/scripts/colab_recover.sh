#!/usr/bin/env bash
# One command to bring a freshly recycled Colab VM back to a serving state.
#
#   bash /content/drive/MyDrive/songforge-dl/code/scripts/colab_recover.sh
#
# Colab recycled this project's runtime twice in one sprint, roughly every two
# to three hours. Each recycle destroys /content — the Python environment, the
# 28 GB of ACE-Step weights, any downloaded corpora and the running API — while
# Drive keeps the things that actually cost time to produce: trained
# checkpoints, tensors, the processed corpus, the licence reports and this
# code. Recovery was therefore a fifteen-step manual sequence performed under
# deadline pressure, which is exactly when steps get skipped.
#
# This script is that sequence, ordered cheapest-first so a partial recovery
# still leaves the VM more useful than it found it. Every stage is idempotent:
# running it twice is safe and nearly free.
set -uo pipefail

DRIVE=/content/drive/MyDrive/songforge-dl
CODE="$DRIVE/code"
VENV=/content/venv-py312-acestep
ACE=/content/ACE-Step-1.5
CKPT=/content/checkpoints
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg          # Colab's kernel exports an inline backend the venv cannot load

say() { echo "== $(date -u +%H:%M:%S) $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

[ -d "$DRIVE" ] || die "Drive not mounted — run drive.mount('/content/drive') first"

# ------------------------------------------------------------------ 1. code
say "staging code from Drive (survives recycles; the repo may be private)"
mkdir -p /content/songforge
cp -r "$CODE"/scripts "$CODE"/configs /content/songforge/ 2>/dev/null
chmod +x /content/songforge/scripts/*.sh 2>/dev/null
ls /content/songforge/scripts >/dev/null || die "no scripts staged from $CODE"

# ------------------------------------------------- 2. interpreter + ACE-Step
# Rebuilds the verified Python 3.12 / torch 2.10 stack and re-runs the
# flash-attn forward/backward gate. Exits non-zero if the gate fails, so a
# broken environment can never reach training or serving.
say "provisioning verified environment (~3 min when the wheel cache is warm)"
bash /content/songforge/scripts/provision_python312_training_env.sh || die "environment gate failed"
PY="$VENV/bin/python"
[ -x "$PY" ] || die "provisioner did not produce $PY"

# --------------------------------------------------------------- 3. weights
# ~28 GB. Local disk only — a Drive round-trip is slower than re-fetching.
if [ -d "$CKPT/xl_turbo" ]; then
  say "weights already present, skipping"
else
  say "downloading ACE-Step weights (~28 GB, ~17 min at observed throughput)"
  cd "$ACE" || die "ACE-Step clone missing"
  "$PY" -m acestep.model_downloader --dir "$CKPT" || die "base weight download failed"
  "$PY" -m acestep.model_downloader --model acestep-v15-xl-turbo --dir "$CKPT" \
    || die "xl-turbo download failed"
  # The downloader stores the DiT under its HF repo name; the trainer and the
  # serving adapter both expect checkpoints/xl_turbo.
  ln -sfn "$CKPT/acestep-v15-xl-turbo" "$CKPT/xl_turbo"
fi
du -sh "$CKPT" 2>/dev/null

# ------------------------------------------------------ 4. best adapter yet
BEST=$(ls -d "$DRIVE"/v1/checkpoints/checkpoints/epoch_* 2>/dev/null | sort -V | tail -1)
if [ -n "${BEST:-}" ]; then
  say "newest adapter: $(basename "$BEST")"
  echo "$BEST" > /content/best_adapter.txt
else
  say "WARNING: no trained adapter on Drive — serving would fall back to the bare foundation"
fi

say "RECOVERY COMPLETE"
echo "SONGFORGE_PY=$PY"
echo "SONGFORGE_LORA=${BEST:-<none>}"
