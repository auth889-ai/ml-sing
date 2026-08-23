#!/usr/bin/env bash
# SongForge V1 driver — idempotent, Colab-drop-safe.
#
# Every stage writes a marker to Drive on success and is skipped on rerun, so
# after any runtime drop the SAME command continues where it left off:
#
#   cd /content/ml-sing/songforge && bash scripts/v1_colab_driver.sh
#
# Experiment identity is frozen by benchmarks/EXPERIMENT_CARD.md. This script
# executes it; it does not decide anything.
set -uo pipefail

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/songforge-dl}"
V1="$DRIVE_ROOT/v1"
MARK="$V1/markers"
LOCAL=/content/v1_work
ARCHIVE_URL="https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
ARCHIVE=/content/slakh2100_flac_redux.tar.gz
mkdir -p "$MARK" "$V1/logs" "$LOCAL"

stage() {  # stage <name> <fn> — run once, marker on success, loud on failure
  local name="$1" fn="$2"
  if [ -f "$MARK/$name.done" ]; then echo "== $name: already done, skipping"; return 0; fi
  echo "== $name: starting $(date -u +%H:%M:%S)"
  if "$fn" 2>&1 | tee -a "$V1/logs/$name.log"; then
    touch "$MARK/$name.done"; echo "== $name: DONE"
  else
    echo "== $name: FAILED — fix and rerun this script"; exit 1
  fi
}

s00_env() {
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || return 1
  local gpu; gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
  case "$gpu" in
    *T4*) echo "FATAL: T4 detected — bf16 lyrics-to-song NaNs. Switch runtime to L4/A100."; return 1;;
  esac
  [ -d "$DRIVE_ROOT" ] || { echo "FATAL: Drive not mounted at $DRIVE_ROOT"; return 1; }
  python -c "import torch; assert torch.cuda.is_bf16_supported(), 'bf16 unsupported'"
  df -h /content | tail -1
}

s01_download() {
  # Resumable; rerun continues a partial file. ~104.3 GB to LOCAL disk only.
  local expected=104322767708
  local actual
  actual=$(stat -c%s "$ARCHIVE" 2>/dev/null || echo 0)
  if [ "$actual" != "$expected" ]; then
    wget -c -q --show-progress --progress=dot:giga -O "$ARCHIVE" "$ARCHIVE_URL"
  else
    echo "archive already complete ($actual bytes)"
  fi
  ls -l "$ARCHIVE"
  # Structure check promised by the design doc. `head` SIGPIPEs tar under
  # pipefail, so capture without a pipe.
  { tar -tzf "$ARCHIVE" 2>/dev/null || true; } | sed -n '1,20p;20q' > "$V1/archive_head.txt"
  [ -s "$V1/archive_head.txt" ] || { echo "empty archive listing"; return 1; }
  cat "$V1/archive_head.txt"
}

s02_select() {
  mkdir -p "$LOCAL/meta"
  tar -xzf "$ARCHIVE" -C "$LOCAL/meta" --wildcards '*/metadata.yaml'
  local root; root=$(find "$LOCAL/meta" -maxdepth 1 -mindepth 1 -type d | head -1)
  python scripts/select_slakh100.py --slakh-root "$root" \
      --output "$V1/slakh100_selection.json"
  cp "$V1/slakh100_selection.tar_members.txt" "$V1/tar_members.txt" 2>/dev/null || \
    cp slakh100_selection.tar_members.txt "$V1/" 2>/dev/null || true
  python - <<'EOF'
import json
s = json.load(open("/content/drive/MyDrive/songforge-dl/v1/slakh100_selection.json"))
print({k: len(v) for k, v in s["selection"].items()})
print(json.dumps(s["report"], indent=1)[:1500])
EOF
}

s03_extract() {
  local prefix; prefix=$(head -1 "$V1/archive_head.txt" | cut -d/ -f1)
  local raw="$DRIVE_ROOT/raw/slakh100"
  mkdir -p "$raw"
  # Extract exactly the selected 100 track dirs.
  sed "s|^|$prefix/|" "$V1/slakh100_selection.tar_members.txt" > "$LOCAL/members.txt"
  tar -xzf "$ARCHIVE" -C "$raw" --strip-components=1 \
      --wildcards $(sed 's|$|/*|' "$LOCAL/members.txt" | tr '\n' ' ')
  du -sh "$raw"
  rm -f "$ARCHIVE"   # archive is temporary; free ~104 GB
}

s04_preprocess() {
  python scripts/preprocess_dataset.py \
      --dataset-id slakh2100 \
      --input-dir "$DRIVE_ROOT/raw/slakh100" \
      --output-dir "$DRIVE_ROOT/processed/slakh100_44k_lora" \
      --config configs/data/preprocess_44k_lora.yaml \
      --instrument-metadata slakh
  du -sh "$DRIVE_ROOT/processed/slakh100_44k_lora"
}

s05_gates() {
  # One gate run per manifest file (the gate script reads files, not dirs),
  # and any failure fails the stage — no trailing command may mask it.
  local failed=0
  for m in "$DRIVE_ROOT"/processed/slakh100_44k_lora/manifests/*.jsonl; do
    echo "=== gates: $(basename "$m")"
    if ! python scripts/dataset_gate.py \
        --manifest "$m" --goal instrument-realism --min-seconds 10 \
        --output "$V1/gates_$(basename "$m" .jsonl).json"; then
      failed=1
    fi
  done
  [ "$failed" -eq 0 ]
}

s06_convert() {
  # The upstream trainer consumes the dataset JSON's audio_path list directly
  # (verified: with --dataset-json the audio dir is never scanned), so no
  # audio copies are made — the JSON points at the processed train split.
  mkdir -p "$DRIVE_ROOT/acestep_lora/slakh100"
  python scripts/build_acestep_training_json.py \
      --manifest "$DRIVE_ROOT/processed/slakh100_44k_lora/manifests" \
      --audio-root "$DRIVE_ROOT/processed/slakh100_44k_lora" \
      --output "$DRIVE_ROOT/acestep_lora/slakh100/dataset.json" \
      --split train
}

s07_train() {
  # Filled by scripts/v1_train.sh (ACE-Step official LoKr trainer; command
  # pinned from upstream docs). Checkpoints + trainer state go to
  # $V1/checkpoints so a dropped runtime resumes with the same command.
  if [ ! -f scripts/v1_train.sh ]; then
    echo "FATAL: scripts/v1_train.sh missing — trainer invocation not pinned yet"; return 1
  fi
  bash scripts/v1_train.sh "$DRIVE_ROOT/acestep_lora/slakh100" "$V1/checkpoints"
}

stage 00_env        s00_env
stage 01_download   s01_download
stage 02_select     s02_select
stage 03_extract    s03_extract
stage 04_preprocess s04_preprocess
stage 05_gates      s05_gates
stage 06_convert    s06_convert
stage 07_train      s07_train
echo "V1 pipeline complete. Next: ablation generation (baseline vs V1)."
