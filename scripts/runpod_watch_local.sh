#!/usr/bin/env bash
# Local idle watchdog (runs on YOUR machine, not the pod).
#
# Polls the pod's GPU utilization over SSH once a minute. If the GPU stays below
# the threshold for RUNPOD_IDLE_MINUTES consecutive minutes, it stops the pod via
# local runpodctl (halting GPU billing; the /workspace volume is kept). This is
# the reliable "automatic turn-off" for on-demand cost control, because the pod's
# minimal container has no cron/systemd/tmux to supervise a pod-side daemon.
#
# Secrets stay local (read from .env). Start it detached with:
#   scripts/runpod.sh autostop        # nohup background
# or run in the foreground with:
#   scripts/runpod.sh watch
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source "$REPO_ROOT/.env"; set +a; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing from .env}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID missing from .env}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/id_ed25519}"; SSH_KEY="${SSH_KEY/#\~/$HOME}"
IDLE_MINUTES="${RUNPOD_IDLE_MINUTES:-30}"
THRESHOLD="${RUNPOD_IDLE_GPU_UTIL_THRESHOLD:-5}"
export PATH="$HOME/.local/bin:$PATH"
GQL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

_endpoint() {
  curl -fsS -X POST "$GQL" -H 'Content-Type: application/json' \
    -d '{"query":"query { pod(input:{podId:\"'"$RUNPOD_POD_ID"'\"}) { desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } }"}' \
  | python3 -c 'import sys,json
d=json.load(sys.stdin)["data"]["pod"]
r=d.get("runtime") or {}
ep=""
for p in (r.get("ports") or []):
    if p.get("privatePort")==22 and p.get("isIpPublic") and p.get("type")=="tcp":
        ep=p["ip"]+" "+str(p["publicPort"])
print(d.get("desiredStatus",""), ep)'
}

echo "[$(date -u)] LOCAL idle watchdog: pod=$RUNPOD_POD_ID idle_minutes=$IDLE_MINUTES threshold=${THRESHOLD}%"
idle=0
while true; do
  read -r status ip port < <(_endpoint) || { echo "[$(date -u)] api error"; sleep 60; continue; }
  if [[ "$status" != "RUNNING" || -z "${ip:-}" ]]; then
    echo "[$(date -u)] pod not RUNNING (status=$status) -> watchdog exiting"; exit 0
  fi
  util=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
           -i "$SSH_KEY" -p "$port" "root@$ip" \
           'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1' 2>/dev/null)
  util="${util:-0}"; util="${util//[^0-9]/}"; util="${util:-0}"
  if [[ "$util" -le "$THRESHOLD" ]]; then idle=$((idle+1)); else idle=0; fi
  echo "[$(date -u)] gpu_util=${util}% idle_streak=${idle}/${IDLE_MINUTES}min"
  if [[ "$idle" -ge "$IDLE_MINUTES" ]]; then
    echo "[$(date -u)] idle >= ${IDLE_MINUTES}min -> stopping pod $RUNPOD_POD_ID"
    runpodctl pod stop "$RUNPOD_POD_ID" 2>&1 | grep -v deprecated || \
      runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | grep -v deprecated || true
    exit 0
  fi
  sleep 60
done
