#!/bin/bash
# Head node entry for multinode colocate demo.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/nfs/kaiyuan}"
DEMO_ROOT="${DEMO_ROOT:-${REPO_ROOT}/colocate_weight_sync_demo}"
HEAD_IP="${HEAD_IP:?HEAD_IP required}"

exec > >(tee -a "${RUN_LOG:-${REPO_ROOT}/logs/colocate_demo_head.log}") 2>&1

BARRIER_DIR="${BARRIER_DIR:-${REPO_ROOT}/logs/colocate_demo_barrier}"
mkdir -p "${BARRIER_DIR}"
rm -f "${BARRIER_DIR}/test-done"
touch "${BARRIER_DIR}/head-ready"
sync 2>/dev/null || true

fix_hosts() {
  local tmp
  tmp="$(mktemp)"
  grep -v "^127.0.1.1" /etc/hosts > "${tmp}" || true
  echo "$1 $(hostname)" >> "${tmp}"
  cat "${tmp}" > /etc/hosts
  rm -f "${tmp}"
}
fix_hosts "${HEAD_IP}"

export NODE_RANK=0
[[ -z "${DIST_INIT_ADDR:-}" ]] && DIST_INIT_ADDR="${HEAD_IP}:$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"
[[ -z "${TRAIN_MASTER_PORT:-}" ]] && TRAIN_MASTER_PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"
export TRAIN_MASTER_ADDR="${TRAIN_MASTER_ADDR:-${HEAD_IP}}"
export BARRIER_DIR REPO_ROOT DEMO_ROOT
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens90f0np0,ens90f1np1,ens92f0np0,ens92f1np1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp130s0f0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3-4B}"

for _ in $(seq 1 600); do [[ -f "${BARRIER_DIR}/worker-ready" ]] && break; sleep 1; done
[[ -f "${BARRIER_DIR}/worker-ready" ]] || { echo "ERROR: worker not ready" >&2; exit 2; }

python3 "${DEMO_ROOT}/run_multinode.py" "$@"
code=$?
[[ "${code}" -eq 0 ]] && touch "${BARRIER_DIR}/test-done" && sync 2>/dev/null || true
exit "${code}"
