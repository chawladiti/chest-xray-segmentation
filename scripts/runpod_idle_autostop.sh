#!/usr/bin/env bash
# Pod-side idle auto-stop watchdog (runs ON the RunPod pod).
#
# Polls GPU utilization once a minute. If the GPU stays below the utilization
# threshold for RUNPOD_IDLE_MINUTES consecutive minutes, it stops the pod via
# runpodctl (halting GPU billing while keeping the /workspace volume). This is
# the "automatic turn-off" half of the on-demand cost management.
#
# Requires on the pod: runpodctl configured (scripts/runpod.sh autostop does this),
# and env: RUNPOD_POD_ID, RUNPOD_IDLE_MINUTES, RUNPOD_IDLE_GPU_UTIL_THRESHOLD.
set -uo pipefail

POD_ID="${RUNPOD_POD_ID:?RUNPOD_POD_ID required}"
IDLE_MINUTES="${RUNPOD_IDLE_MINUTES:-30}"
THRESHOLD="${RUNPOD_IDLE_GPU_UTIL_THRESHOLD:-5}"
export PATH="/usr/local/bin:/usr/bin:$PATH"

idle=0
echo "[$(date -u)] idle-autostop watchdog: pod=$POD_ID idle_minutes=$IDLE_MINUTES util_threshold=${THRESHOLD}%"
while true; do
  # max GPU utilization across all GPUs (integer %)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
         | sort -nr | head -1)
  util="${util:-0}"
  if [[ "$util" -le "$THRESHOLD" ]]; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  echo "[$(date -u)] gpu_util=${util}% idle_streak=${idle}/${IDLE_MINUTES}min"
  if [[ "$idle" -ge "$IDLE_MINUTES" ]]; then
    echo "[$(date -u)] GPU idle >= ${IDLE_MINUTES}min -> stopping pod $POD_ID"
    runpodctl stop pod "$POD_ID" 2>&1 | grep -v deprecated || \
      runpodctl pod stop "$POD_ID" 2>&1 | grep -v deprecated || true
    exit 0
  fi
  sleep 60
done
