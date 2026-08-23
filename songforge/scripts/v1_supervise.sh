#!/usr/bin/env bash
# Keep the V1 pipeline running without a human watching it.
#
# The driver is already idempotent — every finished stage is marked on Drive
# and skipped — so a retry after a transient failure costs almost nothing and
# resumes training from the newest epoch checkpoint. This wrapper turns that
# property into unattended operation: transient trainer crashes, a stalled
# Drive mount that recovers, or an OOM on one batch no longer require someone
# to notice and relaunch by hand.
#
# It deliberately does NOT retry forever. A configuration error would
# otherwise spin here indefinitely and look like progress.
#
#   setsid nohup bash scripts/v1_supervise.sh > /content/supervise.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/songforge-dl}"
V1="$DRIVE_ROOT/v1"
MAX_RESTARTS="${MAX_RESTARTS:-6}"
BACKOFF="${BACKOFF:-60}"

mkdir -p "$V1/logs"
attempt=0
while [ "$attempt" -lt "$MAX_RESTARTS" ]; do
  attempt=$((attempt + 1))
  echo "=== supervisor: attempt $attempt/$MAX_RESTARTS at $(date -u +%FT%TZ)"
  bash "$REPO_ROOT/scripts/v1_colab_driver.sh"
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== supervisor: pipeline completed cleanly at $(date -u +%FT%TZ)"
    exit 0
  fi
  echo "=== supervisor: driver exited $status; retrying in ${BACKOFF}s"
  sleep "$BACKOFF"
done

echo "=== supervisor: giving up after $MAX_RESTARTS attempts — needs a human"
exit 1
