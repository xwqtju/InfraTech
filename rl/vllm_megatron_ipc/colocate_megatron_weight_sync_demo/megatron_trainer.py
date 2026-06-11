#!/usr/bin/env python3
"""torchrun entry: Megatron checkpoint -> HF chunks -> CUDA IPC -> vLLM."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import shared  # noqa: F401
import torch
import torch.distributed as dist

from demo import VllmHttpClient, send_ipc_chunk


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vllm-url", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--weight-version", default="1")
    demo, rest = p.parse_known_args()
    return demo, rest


def main() -> int:
    demo, rest = _parse_args()
    sys.argv = [sys.argv[0], *rest]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=int(os.environ.get("DISTRIBUTED_TIMEOUT_MINUTES", "30"))),
        )
    from slime.utils.distributed_utils import get_gloo_group, init_gloo_group

    init_gloo_group()

    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import initialize_model_and_optimizer
    from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers
    from slime.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase
    from slime.utils.arguments import parse_args

    args = parse_args()
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    init(args)

    if demo.model_name is None:
        from transformers import AutoConfig

        model_name = type(AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)).__name__.lower()
    else:
        model_name = demo.model_name.lower()

    rank, world_size = dist.get_rank(), dist.get_world_size()
    coord = rank == 0
    slot_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    client = VllmHttpClient(demo.vllm_url) if coord else None

    model, _, _, _ = initialize_model_and_optimizer(args, role="actor")
    local_weights = dict(
        named_params_and_buffers(args, model, convert_to_global_name=args.megatron_to_hf_mode == "raw")
    )
    hf_iter = HfWeightIteratorBase.create(args=args, model=model, model_name=model_name, quantization_config=None)

    if coord:
        client.init_weight_transfer_ipc()
    dist.barrier(group=get_gloo_group())
    if coord:
        client.start_weight_update()
    dist.barrier(group=get_gloo_group())

    for chunk_idx, chunk in enumerate(hf_iter.get_hf_weight_chunks(local_weights)):
        print(f"[megatron rank={rank}] IPC chunk {chunk_idx} ({len(chunk)} tensors)", flush=True)
        send_ipc_chunk(
            client, chunk, slot_group=slot_group, slot_size=world_size,
            is_coordinator=coord, weight_version=demo.weight_version,
        )
        dist.barrier(group=get_gloo_group())

    if coord:
        client.finish_weight_update()
    dist.barrier(group=get_gloo_group())
    print(f"[megatron rank={rank}] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
