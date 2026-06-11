#!/usr/bin/env python3
"""Single-node Megatron + vLLM colocate weight sync (sleep -> IPC -> wake -> generate)."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

import shared  # noqa: F401
from demo import VllmHttpClient, ColocateWorkerExtension  # noqa: F401
from shared import DEFAULT_SLIME_ROOT, SINGLE_DEMO_ROOT

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT = "1 + 1 ="


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _visible_devices(n: int) -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    return ",".join(cvd.split(",")[:n]) if cvd else ",".join(str(i) for i in range(n))


def _vllm_cmd(model: str, host: str, port: int, tp: int, gpu_mem: float, max_len: int) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env["CUDA_VISIBLE_DEVICES"] = _visible_devices(tp)
    pp = env.get("PYTHONPATH", "")
    if SINGLE_DEMO_ROOT not in pp.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([SINGLE_DEMO_ROOT, pp])
    cmd = [
        "vllm", "serve", model,
        "--tensor-parallel-size", str(tp),
        "--port", str(port), "--host", host,
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enable-sleep-mode",
        "--weight-transfer-config", '{"backend":"ipc"}',
        "--worker-extension-cls", "demo.ColocateWorkerExtension",
        "--trust-remote-code", "--seed", "1234",
    ]
    return cmd, env


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        from vllm.utils.system_utils import kill_process_tree
        kill_process_tree(proc.pid)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _slime_args(tp: int, hf: str, ref: str) -> str:
    return (
        f"--debug-train-only --train-backend megatron "
        f"--hf-checkpoint {hf} --ref-load {ref} "
        f"--tensor-model-parallel-size {tp} --pipeline-model-parallel-size 1 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {tp} "
        f"--megatron-to-hf-mode raw --micro-batch-size 1 "
        f"--global-batch-size {tp} --rollout-batch-size {tp} --num-rollout 1 "
        f"--optimizer adam --attention-dropout 0.0 --hidden-dropout 0.0 "
        f"--attention-backend flash --accumulate-allreduce-grads-in-fp32 "
        f"--attention-softmax-in-fp32 "
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", default=os.environ.get("HF_CHECKPOINT", "/root/models/Qwen3-4B"))
    p.add_argument("--ref-load", default=os.environ.get("REF_LOAD", "/root/models/Qwen3-4B_torch_dist"))
    p.add_argument("--model-name", default=os.environ.get("MEGATRON_MODEL_NAME"))
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--gpu-mem-util", type=float, default=0.55)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-tokens", type=int, default=32)
    args = p.parse_args()

    tp = args.tp_size
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    slime_root = os.environ.get("SLIME_ROOT") or os.environ.get("VIME_ROOT", DEFAULT_SLIME_ROOT)
    model_type = os.environ.get("MEGATRON_MODEL_TYPE", "qwen3-4B")
    model_sh = os.path.join(slime_root, "scripts", "models", f"{model_type}.sh")
    trainer = os.path.join(DEMO_ROOT, "megatron_trainer.py")

    cmd, env = _vllm_cmd(args.hf_checkpoint, "127.0.0.1", port, tp, args.gpu_mem_util, 4096)
    print(f"[megatron-single] vLLM TP={tp} port={port}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    client = VllmHttpClient(url)

    try:
        client.wait_healthy(proc=proc)
        client.sleep(level=0)
        name_arg = f"--model-name {args.model_name} " if args.model_name else ""
        bash = (
            f"set -euo pipefail; cd {DEMO_ROOT}; source {model_sh}; "
            f"export PYTHONPATH={slime_root}:/root/Megatron-LM; "
            f"export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1; "
            f"torchrun --nproc_per_node={tp} {trainer} --vllm-url {url} {name_arg}"
            f"${{MODEL_ARGS[@]}} {_slime_args(tp, args.hf_checkpoint, args.ref_load)}"
        )
        subprocess.run(["bash", "-lc", bash], check=True)
        client.wake(tags=["weights", "kv_cache"])
        text = client.generate(args.prompt, max_tokens=args.max_tokens, model_path=args.hf_checkpoint)
        print(f"PASS megatron-single tp={tp} response={text!r}", flush=True)
        return 0
    except Exception as e:
        print(f"FAIL megatron-single: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        _stop(proc)


if __name__ == "__main__":
    raise SystemExit(main())
