# Single-Node Colocate Weight Sync (Megatron)

Training uses **Megatron** (`HfWeightIterator` + `megatron_to_hf`). vLLM/IPC helpers come from [`colocate_single_demo`](../colocate_single_demo/demo.py).

Requires **slime** (Megatron training utilities) and **Megatron-LM**.

## Docker

- **Recommended:** `inferactinc/public:vime-vllm-latest` (vLLM 0.21+, PyTorch, Megatron-LM preinstalled)
- **Alternative:** `slimerl/slime:latest` — install vLLM inside the container

```bash
docker pull inferactinc/public:vime-vllm-latest
docker run --gpus all -it --network host inferactinc/public:vime-vllm-latest /bin/bash
```

## Flow

```text
vLLM serve (sleep + IPC)
  -> POST /sleep?level=0
  -> torchrun megatron_trainer.py
       Megatron load torch_dist ckpt
       HF chunks -> CUDA IPC -> POST /update_weights
  -> POST /wake_up
  -> POST /inference/v1/generate
```

## One-time setup (torch_dist checkpoint)

```bash
export SLIME_ROOT=/data/nfs/kaiyuan/slime
source ${SLIME_ROOT}/scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  ${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/models/Qwen3-4B \
  --save /root/models/Qwen3-4B_torch_dist
```

## Run (TP=2)

```bash
export SLIME_ROOT=/data/nfs/kaiyuan/slime
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=0,1

cd /data/nfs/kaiyuan/colocate_megatron_weight_sync_demo
python3 run_single.py \
  --hf-checkpoint /root/models/Qwen3-4B \
  --ref-load /root/models/Qwen3-4B_torch_dist \
  --tp-size 2
```

Success: `PASS megatron-single tp=2 response=...`

## Verified (A100-server)

2026-05-30, compressed code, TP=2, ~85s, 16 IPC chunks.

```bash
tail -n 30 /data/nfs/kaiyuan/logs/colocate_megatron_compressed.log
```

NFS repo root: `/data/nfs/kaiyuan`. Set `SLIME_ROOT` to the directory that contains `slime/` and `tools/`.

## Files

| File | Role |
|------|------|
| `run_single.py` | Orchestrate vLLM + torchrun |
| `megatron_trainer.py` | Megatron load + IPC push |
| `shared.py` | Import path to `colocate_single_demo` |

## Env

| Var | Default |
|-----|---------|
| `SLIME_ROOT` | `/data/nfs/kaiyuan/slime` |
| `MEGATRON_MODEL_TYPE` | `qwen3-4B` |
| `HF_CHECKPOINT` / `REF_LOAD` | see `run_single.py` |
