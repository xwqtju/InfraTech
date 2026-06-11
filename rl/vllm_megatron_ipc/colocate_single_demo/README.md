# Single-Node Colocate Weight Sync (HF / Torch)

All-in-one demo: `demo.py` (orchestration, trainer, IPC, vLLM HTTP, worker extension).

Multinode: [`colocate_weight_sync_demo`](../colocate_weight_sync_demo/). Megatron: [`colocate_megatron_weight_sync_demo`](../colocate_megatron_weight_sync_demo/).

## Docker

- **Recommended:** `inferactinc/public:vime-vllm-latest` (vLLM 0.21+, PyTorch preinstalled)
- **Alternative:** `slimerl/slime:latest` — install vLLM inside the container

```bash
docker pull inferactinc/public:vime-vllm-latest
docker run --gpus all -it --network host inferactinc/public:vime-vllm-latest /bin/bash
```

## Run

```bash
export MODEL_PATH=/root/models/Qwen3-4B
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=0,1

cd colocate_single_demo
python3 demo.py run --model-path "${MODEL_PATH}" --tp-size 2
```

Success: `PASS single tp=2 response=...`

## Verified (A100-server)

Qwen3-4B, TP=2, ~91s in Docker on A100-server.

```bash
tail -n 50 /data/nfs/kaiyuan/logs/colocate_single_demo.log
```

Sample tail (2026-05-29): IPC chunks → `finish_weight_update` → `wake_up` → `generate` → `PASS single tp=2`.

`RotaryEmbedding: Failed to load weights` is a non-fatal HF naming mismatch; use the Megatron demo for production-aligned names.

## Commands

| Command | Purpose |
|---------|---------|
| `demo.py run` | Full flow (use this) |
| `demo.py trainer` | torchrun entry only |

## Requirements

vLLM 0.21+, PyTorch, transformers, cloudpickle, requests
