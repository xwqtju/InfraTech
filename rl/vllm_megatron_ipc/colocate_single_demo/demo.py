#!/usr/bin/env python3
"""Single-node vLLM colocate weight sync demo (HF trainer, CUDA IPC).

Usage:
  python demo.py run --model-path /root/models/Qwen3-4B --tp-size 2
  torchrun --nproc_per_node=2 demo.py trainer --vllm-url http://127.0.0.1:PORT ...
"""

from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import cloudpickle
import requests
import torch
import torch.distributed as dist

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT = "1 + 1 ="


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded in vLLM subprocess via --worker-extension-cls)
# ---------------------------------------------------------------------------


class ColocateWorkerExtension:
    """Patch IPC receive before vLLM deserializes CUDA handles."""

    def __new__(cls, **kwargs):
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

        if not getattr(IPCWeightTransferEngine, "_demo_receive_patched", False):
            _orig = IPCWeightTransferEngine.receive_weights

            def _receive(self, update_info, load_weights, _orig=_orig):
                _orig(self, update_info, load_weights)

            IPCWeightTransferEngine.receive_weights = _receive
            IPCWeightTransferEngine._demo_receive_patched = True  # type: ignore[attr-defined]
        return super().__new__(cls)


# ---------------------------------------------------------------------------
# vLLM HTTP control plane (sleep / wake / weight transfer / generate)
# ---------------------------------------------------------------------------


class VllmHttpClient:
    def __init__(self, base_url: str, timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, *, json: dict | None = None, params: dict | None = None) -> dict:
        r = requests.post(f"{self.base_url}/{path.lstrip('/')}", json=json, params=params, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"ok": True, "raw": r.text}

    def wait_healthy(self, timeout_s: float = 600.0, proc: subprocess.Popen | None = None) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early: code={proc.returncode}")
            try:
                if requests.get(f"{self.base_url}/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM not healthy: {self.base_url}")

    def flush_cache(self) -> None:
        params = {"reset_running_requests": False, "reset_external": False}
        for _ in range(30):
            try:
                if requests.post(f"{self.base_url}/reset_prefix_cache", params=params, timeout=60).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("reset_prefix_cache failed")

    def sleep(self, level: int = 0) -> dict:
        self.flush_cache()
        return self._post("sleep", params={"level": level})

    def wake(self, tags: list[str] | None = None) -> dict:
        params = [("tags", t) for t in tags] if tags else None
        return self._post("wake_up", params=params)

    def init_weight_transfer_ipc(self) -> dict:
        return self._post("init_weight_transfer_engine", json={"init_info": {}})

    def start_weight_update(self) -> dict:
        return self._post("start_weight_update", json={"is_checkpoint_format": True})

    def finish_weight_update(self) -> dict:
        return self._post("finish_weight_update", json={})

    def update_weights(self, update_info: dict[str, Any], *, weight_version: str | None = None) -> dict:
        body: dict[str, Any] = {"update_info": update_info}
        if weight_version is not None:
            body["weight_version"] = weight_version
        return self._post("update_weights", json=body)

    def generate(self, prompt: str, max_tokens: int, *, model_path: str) -> str:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        r = requests.post(
            f"{self.base_url}/inference/v1/generate",
            json={
                "model": model_path,
                "token_ids": tok.encode(prompt, add_special_tokens=False),
                "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0},
            },
            timeout=120,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        out_ids = choice.get("token_ids") or []
        return tok.decode(out_ids, skip_special_tokens=True) if out_ids else choice.get("text", "")


# ---------------------------------------------------------------------------
# CUDA IPC payload (per-TP rank merge via Gloo all_gather_object)
# ---------------------------------------------------------------------------


def _gpu_uuid() -> str:
    return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)


def _build_ipc_info(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> tuple[dict[str, list], list[torch.Tensor]]:
    from torch.multiprocessing.reductions import reduce_tensor

    names, dtypes, shapes, handles, refs = [], [], [], [], []
    uid = _gpu_uuid()
    for name, tensor in named_tensors:
        names.append(name)
        dtypes.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        weight = tensor.detach().contiguous()
        refs.append(weight)
        _, ipc_args = reduce_tensor(weight)
        handles.append({uid: ipc_args})
    return {"names": names, "dtype_names": dtypes, "shapes": shapes, "ipc_handles": handles}, refs


def _ipc_http_payload(info: dict[str, list]) -> dict:
    return {
        "names": info["names"],
        "dtype_names": info["dtype_names"],
        "shapes": info["shapes"],
        "ipc_handles_pickled": base64.b64encode(cloudpickle.dumps(info["ipc_handles"])).decode("utf-8"),
    }


def send_ipc_chunk(
    client: VllmHttpClient | None,
    chunk: Sequence[tuple[str, torch.Tensor]],
    *,
    slot_group,
    slot_size: int,
    is_coordinator: bool,
    weight_version: str,
) -> None:
    local, refs = _build_ipc_info(chunk)
    if slot_size <= 1:
        if is_coordinator:
            client.update_weights(_ipc_http_payload(local), weight_version=weight_version)
        del refs
        return

    payload = base64.b64encode(cloudpickle.dumps(local)).decode("ascii")
    gathered: list[str | None] = [None] * slot_size
    dist.all_gather_object(gathered, payload, group=slot_group)
    if is_coordinator:
        infos = [cloudpickle.loads(base64.b64decode(p.encode("ascii"))) for p in gathered]
        merged_handles = []
        for i in range(len(infos[0]["names"])):
            combined: dict = {}
            for info in infos:
                combined.update(info["ipc_handles"][i])
            merged_handles.append(combined)
        merged = {**infos[0], "ipc_handles": merged_handles}
        client.update_weights(_ipc_http_payload(merged), weight_version=weight_version)
    dist.barrier(group=slot_group)
    del refs


# ---------------------------------------------------------------------------
# vLLM subprocess launcher (single node)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _visible_devices(n: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return ",".join(visible.split(",")[:n]) if visible else ",".join(str(i) for i in range(n))


def _vllm_env(visible: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env["CUDA_VISIBLE_DEVICES"] = visible
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    pp = env.get("PYTHONPATH", "")
    if DEMO_ROOT not in {p for p in pp.split(os.pathsep) if p}:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [DEMO_ROOT, pp]))
    return env


def _build_vllm_cmd(
    *,
    model_path: str,
    host: str,
    port: int,
    tp_size: int,
    visible: str,
    max_model_len: int,
    gpu_mem_util: float,
) -> list[str]:
    return [
        "vllm", "serve", model_path,
        "--tensor-parallel-size", str(tp_size),
        "--port", str(port),
        "--host", host,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--enable-sleep-mode",
        "--weight-transfer-config", '{"backend":"ipc"}',
        "--worker-extension-cls", "demo.ColocateWorkerExtension",
        "--trust-remote-code", "--seed", "1234",
        "--nnodes", "1", "--node-rank", "0",
    ]


def _stop_vllm(proc: subprocess.Popen) -> None:
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


# ---------------------------------------------------------------------------
# Trainer: HF load + IPC push (torchrun entry)
# ---------------------------------------------------------------------------


def _iter_chunks(state_dict: dict[str, torch.Tensor], chunk_size: int) -> Iterator[list[tuple[str, torch.Tensor]]]:
    items = list(state_dict.items())
    for i in range(0, len(items), chunk_size):
        yield [(k, v) for k, v in items[i : i + chunk_size]]


def cmd_trainer(args: argparse.Namespace) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != args.tp_size:
        raise RuntimeError(f"world_size={world_size} != tp_size={args.tp_size}")

    slot_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    coord = rank == 0
    client = VllmHttpClient(args.vllm_url) if coord else None

    if coord:
        client.init_weight_transfer_ipc()
    dist.barrier()
    if coord:
        client.start_weight_update()
    dist.barrier()

    from transformers import AutoModelForCausalLM

    device = torch.device(f"cuda:{local_rank}")
    print(f"[trainer rank={rank}] loading {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    state_dict = {k: v.to(device) for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()

    for chunk_idx, chunk in enumerate(_iter_chunks(state_dict, args.chunk_size)):
        print(f"[trainer rank={rank}] IPC chunk {chunk_idx}", flush=True)
        send_ipc_chunk(client, chunk, slot_group=slot_group, slot_size=world_size, is_coordinator=coord, weight_version=args.weight_version)
        dist.barrier()

    if coord:
        client.finish_weight_update()
    dist.barrier()
    print(f"[trainer rank={rank}] done", flush=True)
    dist.destroy_process_group()
    return 0


# ---------------------------------------------------------------------------
# Orchestrator: vLLM sleep -> trainer -> wake -> generate
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    tp, host = args.tp_size, "127.0.0.1"
    port = args.port or _free_port()
    visible = _visible_devices(tp)
    cmd = _build_vllm_cmd(
        model_path=args.model_path, host=host, port=port, tp_size=tp, visible=visible,
        max_model_len=args.max_model_len, gpu_mem_util=args.gpu_mem_util,
    )
    env = _vllm_env(visible)

    print(f"[single] vLLM TP={tp} port={port}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    base_url = f"http://{host}:{port}"
    client = VllmHttpClient(base_url)

    try:
        client.wait_healthy(proc=proc)
        client.sleep(level=0)

        train_env = _vllm_env(visible)
        subprocess.run(
            ["torchrun", f"--nproc_per_node={tp}", __file__, "trainer",
             "--vllm-url", base_url, "--model-path", args.model_path, "--tp-size", str(tp)],
            env=train_env, check=True,
        )

        client.wake(tags=["weights", "kv_cache"])
        text = client.generate(args.prompt, args.max_tokens, model_path=args.model_path)
        print(f"PASS single tp={tp} response={text!r}", flush=True)
        return 0
    except Exception as e:
        print(f"FAIL single: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        _stop_vllm(proc)


def main() -> int:
    p = argparse.ArgumentParser(description="Single-node vLLM colocate weight sync")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="full demo: vLLM + IPC sync + generate")
    run_p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/root/models/Qwen3-4B"))
    run_p.add_argument("--tp-size", type=int, default=2)
    run_p.add_argument("--port", type=int, default=0)
    run_p.add_argument("--max-model-len", type=int, default=4096)
    run_p.add_argument("--gpu-mem-util", type=float, default=0.55)
    run_p.add_argument("--prompt", default=DEFAULT_PROMPT)
    run_p.add_argument("--max-tokens", type=int, default=32)

    tr_p = sub.add_parser("trainer", help="torchrun entry: HF weights -> vLLM IPC")
    tr_p.add_argument("--vllm-url", required=True)
    tr_p.add_argument("--model-path", required=True)
    tr_p.add_argument("--tp-size", type=int, required=True)
    tr_p.add_argument("--chunk-size", type=int, default=8)
    tr_p.add_argument("--weight-version", default="1")

    args = p.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_trainer(args)


if __name__ == "__main__":
    raise SystemExit(main())
