#!/usr/bin/env bash
# RunPod on-demand A100 control (start / stop / status / ssh / deploy / autostop).
#
# Secrets live ONLY in .env (git-ignored): RUNPOD_API_KEY, RUNPOD_POD_ID, RUNPOD_SSH_KEY.
# The pod's public IP/SSH port change on every start, so they are fetched live
# from the RunPod GraphQL API — never hard-coded or committed.
#
# Usage:
#   scripts/runpod.sh start        # resume the on-demand pod, wait, print SSH
#   scripts/runpod.sh stop         # stop the pod (halts GPU billing; keeps volume)
#   scripts/runpod.sh status       # desiredStatus + uptime + $/hr
#   scripts/runpod.sh info         # print current SSH connection string
#   scripts/runpod.sh ssh [cmd]    # ssh into the pod (optionally run a command)
#   scripts/runpod.sh deploy       # copy this repo -> /workspace/chexlocalize-medical-imaging
#   scripts/runpod.sh autostop     # install+start the pod-side idle auto-stop watchdog
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="/workspace/chexlocalize-medical-imaging"

# --- load .env -------------------------------------------------------------
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing from .env}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID missing from .env}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/id_ed25519}"
SSH_KEY="${SSH_KEY/#\~/$HOME}"
export PATH="$HOME/.local/bin:$PATH"

GQL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

_gql() { curl -fsS -X POST "$GQL" -H 'Content-Type: application/json' -d "$1"; }

# Echo "IP PORT" for the public TCP mapping of container port 22 (empty if none).
_ssh_endpoint() {
  _gql '{"query":"query { pod(input:{podId:\"'"$RUNPOD_POD_ID"'\"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }"}' \
  | python3 -c 'import sys,json
d=json.load(sys.stdin)
r=(d.get("data",{}).get("pod") or {}).get("runtime") or {}
for p in (r.get("ports") or []):
    if p.get("privatePort")==22 and p.get("isIpPublic") and p.get("type")=="tcp":
        print(p["ip"],p["publicPort"]); break'
}

_ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i "$SSH_KEY")

_wait_running() {
  echo "waiting for pod to run..."
  for _ in $(seq 1 40); do
    st=$(_gql '{"query":"query { pod(input:{podId:\"'"$RUNPOD_POD_ID"'\"}) { desiredStatus runtime { uptimeInSeconds } } }"}' \
        | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"]["pod"];print(d["desiredStatus"], (d.get("runtime") or {}).get("uptimeInSeconds"))')
    echo "  status: $st"
    [[ "$st" == RUNNING* && "$st" != *None ]] && return 0
    sleep 6
  done
  echo "pod did not report a running SSH endpoint in time" >&2; return 1
}

cmd_status() {
  _gql '{"query":"query { pod(input:{podId:\"'"$RUNPOD_POD_ID"'\"}) { name desiredStatus costPerHr runtime { uptimeInSeconds } } }"}' \
  | python3 -m json.tool
}

cmd_info() {
  read -r ip port < <(_ssh_endpoint) || true
  if [[ -z "${ip:-}" ]]; then echo "No public SSH endpoint (pod stopped?). Try: $0 start"; return 1; fi
  echo "ssh root@${ip} -p ${port} -i ${SSH_KEY}"
}

cmd_start() {
  echo "starting pod $RUNPOD_POD_ID ..."
  runpodctl pod start "$RUNPOD_POD_ID" 2>&1 | grep -v deprecated || true
  _wait_running
  sleep 4
  echo; echo "connect with:"; cmd_info
}

cmd_stop() {
  echo "stopping pod $RUNPOD_POD_ID (GPU billing halts; /workspace volume kept) ..."
  runpodctl pod stop "$RUNPOD_POD_ID" 2>&1 | grep -v deprecated || true
}

cmd_ssh() {
  read -r ip port < <(_ssh_endpoint)
  [[ -z "${ip:-}" ]] && { echo "no SSH endpoint; run: $0 start" >&2; exit 1; }
  if [[ $# -gt 0 ]]; then ssh "${_ssh_opts[@]}" -p "$port" "root@$ip" "$@";
  else ssh "${_ssh_opts[@]}" -p "$port" "root@$ip"; fi
}

cmd_deploy() {
  read -r ip port < <(_ssh_endpoint)
  [[ -z "${ip:-}" ]] && { echo "no SSH endpoint; run: $0 start" >&2; exit 1; }
  echo "deploying tracked files -> ${REMOTE_DIR} ..."
  ssh "${_ssh_opts[@]}" -p "$port" "root@$ip" "mkdir -p '$REMOTE_DIR'"
  # ship git-tracked files only (no data/, no .venv, no secrets)
  ( cd "$REPO_ROOT" && git archive --format=tar HEAD ) \
    | ssh "${_ssh_opts[@]}" -p "$port" "root@$ip" "tar -x -C '$REMOTE_DIR'"
  echo "done. remote path: $REMOTE_DIR"
}

cmd_autostop() {
  read -r ip port < <(_ssh_endpoint)
  [[ -z "${ip:-}" ]] && { echo "no SSH endpoint; run: $0 start" >&2; exit 1; }
  echo "installing pod-side idle auto-stop watchdog ..."
  scp "${_ssh_opts[@]}" -P "$port" "$REPO_ROOT/scripts/runpod_idle_autostop.sh" "root@$ip:/root/runpod_idle_autostop.sh"
  local mins="${RUNPOD_IDLE_MINUTES:-30}" thr="${RUNPOD_IDLE_GPU_UTIL_THRESHOLD:-5}" pid="$RUNPOD_POD_ID" key="$RUNPOD_API_KEY"
  # Bake env into a wrapper and launch fully detached (setsid) so it survives the
  # SSH channel closing. runpodctl is configured so the pod can stop itself.
  ssh "${_ssh_opts[@]}" -p "$port" "root@$ip" 'bash -s' <<EOS
runpodctl config --apiKey '$key' >/dev/null 2>&1
chmod 600 /root/.runpod/config.toml 2>/dev/null || true
chmod +x /root/runpod_idle_autostop.sh
cat > /root/start_autostop.sh <<'WRAP'
#!/usr/bin/env bash
export RUNPOD_POD_ID=$pid
export RUNPOD_IDLE_MINUTES=$mins
export RUNPOD_IDLE_GPU_UTIL_THRESHOLD=$thr
exec /root/runpod_idle_autostop.sh
WRAP
chmod +x /root/start_autostop.sh
pkill -f runpod_idle_autostop.sh 2>/dev/null || true
setsid bash /root/start_autostop.sh >/root/idle_autostop.log 2>&1 < /dev/null &
sleep 2
pgrep -fa runpod_idle_autostop.sh || echo "WARNING: watchdog not running"
EOS
  echo "watchdog installed (log on pod: /root/idle_autostop.log)"
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  info)    cmd_info ;;
  ssh)     shift; cmd_ssh "$@" ;;
  deploy)  cmd_deploy ;;
  autostop) cmd_autostop ;;
  *) echo "usage: $0 {start|stop|status|info|ssh|deploy|autostop}"; exit 1 ;;
esac
