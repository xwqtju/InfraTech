# Multinode Colocate Weight Sync (TP=16)

Single-node: [`colocate_single_demo`](../colocate_single_demo/) (`demo.py run`).

This folder is **2 nodes × 8 GPUs, TP=16** only. Trainer reuses `colocate_single_demo/demo.py`.

## Docker

- **Recommended:** `inferactinc/public:vime-vllm-latest` (vLLM 0.21+, PyTorch preinstalled). Use the same image on both nodes.
- **Alternative:** `slimerl/slime:latest` — install vLLM inside the container

```bash
docker pull inferactinc/public:vime-vllm-latest
docker run --gpus all -it --network host inferactinc/public:vime-vllm-latest /bin/bash
```

## Run

From a host that can SSH to both nodes:

```bash
bash /data/nfs/kaiyuan/colocate_weight_sync_demo/launch.sh
```

See `head.sh` / `worker.sh` for env (`DIST_INIT_ADDR`, `TRAIN_MASTER_*`, NCCL/Gloo interfaces).

Success:

- head: `PASS multinode tp=16 response=...`
- worker: `PASS worker multinode tp=16`

## Notes

- Tensors use **local CUDA IPC** only; cross-node traffic is **Gloo metadata + HTTP on head**.
- NFS repo default: `/data/nfs/kaiyuan`

## Files

| File | Role |
|------|------|
| `run_multinode.py` | Head/worker entry (includes vLLM launch) |
| `shared.py` | Path to `colocate_single_demo` |
| `launch.sh` / `head.sh` / `worker.sh` | Two-node orchestration |

Shared IPC/HTTP: [`colocate_single_demo/demo.py`](../colocate_single_demo/demo.py).
