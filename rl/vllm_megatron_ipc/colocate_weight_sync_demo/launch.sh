#!/bin/bash
# Launch 2-node colocate weight sync (2x8 GPU, TP=16).
set -euo pipefail

HEAD_SSH="${HEAD_SSH:-A800-server}"
WORKER_SSH="${WORKER_SSH:-A800-server2}"
HEAD_IP="${HEAD_IP:-7.216.199.149}"
WORKER_IP="${WORKER_IP:-7.216.196.62}"
CONTAINER="${CONTAINER:-vime_v22}"
REPO="${REPO:-/data/nfs/kaiyuan}"
DEMO="${REPO}/colocate_weight_sync_demo"
LOG_DIR="${REPO}/logs/colocate_demo_$(date +%Y%m%d_%H%M%S)"
BARRIER_DIR="${LOG_DIR}/barrier"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3-4B}"
FREE_PORT='python3 -c "import socket;s=socket.socket();s.bind((\"\",0));print(s.getsockname()[1]);s.close()"'

mkdir -p "${LOG_DIR}/barrier"
DIST_PORT="$(ssh "${HEAD_SSH}" "docker exec ${CONTAINER} ${FREE_PORT}")"
DIST_INIT_ADDR="${HEAD_IP}:${DIST_PORT}"
TRAIN_PORT="$(ssh "${HEAD_SSH}" "docker exec ${CONTAINER} ${FREE_PORT}")"

ssh "${HEAD_SSH}" "rm -rf ${BARRIER_DIR} && mkdir -p ${BARRIER_DIR} && touch ${BARRIER_DIR}/head-ready && sync 2>/dev/null || true"

echo "=== colocate_weight_sync_demo multinode ==="
echo "DIST_INIT_ADDR=${DIST_INIT_ADDR} TRAIN_MASTER=${HEAD_IP}:${TRAIN_PORT}"
echo "LOG_DIR=${LOG_DIR}"

for host in "${HEAD_SSH}" "${WORKER_SSH}"; do
  ssh "${host}" "docker exec ${CONTAINER} bash -lc '
    pkill -9 -f vllm 2>/dev/null || true
    pkill -9 -f run_multinode 2>/dev/null || true
  '" || true
done
sleep 5

ssh "${WORKER_SSH}" "nohup docker exec ${CONTAINER} env \
  REPO_ROOT=${REPO} DEMO_ROOT=${DEMO} HEAD_IP=${HEAD_IP} WORKER_IP=${WORKER_IP} \
  DIST_INIT_ADDR=${DIST_INIT_ADDR} TRAIN_MASTER_ADDR=${HEAD_IP} TRAIN_MASTER_PORT=${TRAIN_PORT} \
  MODEL_PATH=${MODEL_PATH} BARRIER_DIR=${BARRIER_DIR} RUN_LOG=${LOG_DIR}/worker.log \
  bash ${DEMO}/worker.sh > ${LOG_DIR}/worker_nohup.log 2>&1 &"

sleep 8
set +e
ssh "${HEAD_SSH}" "docker exec ${CONTAINER} env \
  REPO_ROOT=${REPO} DEMO_ROOT=${DEMO} HEAD_IP=${HEAD_IP} WORKER_IP=${WORKER_IP} \
  DIST_INIT_ADDR=${DIST_INIT_ADDR} TRAIN_MASTER_ADDR=${HEAD_IP} TRAIN_MASTER_PORT=${TRAIN_PORT} \
  MODEL_PATH=${MODEL_PATH} BARRIER_DIR=${BARRIER_DIR} RUN_LOG=${LOG_DIR}/head.log \
  bash ${DEMO}/head.sh" 2>&1 | tee "${LOG_DIR}/head_remote.log"
HEAD_EXIT=${PIPESTATUS[0]}
set -e

echo "=== head exit: ${HEAD_EXIT} ==="
ssh "${HEAD_SSH}" "tail -40 ${LOG_DIR}/head.log" || true
ssh "${WORKER_SSH}" "tail -20 ${LOG_DIR}/worker.log" || true
exit "${HEAD_EXIT}"
