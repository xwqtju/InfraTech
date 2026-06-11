#!/usr/bin/env python3
"""Cross-host vLLM TP=16 colocate weight sync (2 nodes x 8 GPUs)."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import shared  # noqa: F401
from demo import VllmHttpClient

PROMPT = "1 + 1 ="


@dataclass(frozen=True)
class _Topo:
    nnodes: int
    node_rank: int
    local_gpus: int
    tp: int

    @property
    def headless(self) -> bool:
        return self.node_rank != 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _visible(n: int) -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    return ",".join(cvd.split(",")[:n]) if cvd else ",".join(str(i) for i in range(n))


def _topo(engine_gpus: int, gpus_per_node: int, node_rank: int, tp: int) -> _Topo:
    nnodes = engine_gpus // gpus_per_node
    return _Topo(nnodes=nnodes, node_rank=node_rank, local_gpus=engine_gpus // nnodes, tp=tp)


def _parse_addr(addr: str) -> tuple[str, int]:
    if addr.startswith("["):
        host = addr[1 : addr.rindex("]")]
        port = int(addr[addr.rindex(":") + 1 :])
    else:
        host, port_s = addr.rsplit(":", 1)
        port = int(port_s)
    return host, port


def _vllm_cmd(
    model: str, host: str, port: int, topo: _Topo, dist_addr: str | None,
    gpu_mem: float, max_len: int,
) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env["CUDA_VISIBLE_DEVICES"] = _visible(topo.local_gpus)
    root = shared.SINGLE_DEMO_ROOT
    pp = env.get("PYTHONPATH", "")
    if root not in pp.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([root, pp])
    cmd = [
        "vllm", "serve", model,
        "--tensor-parallel-size", str(topo.tp),
        "--port", str(port), "--host", host.strip("[]"),
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enable-sleep-mode",
        "--weight-transfer-config", '{"backend":"ipc"}',
        "--worker-extension-cls", "demo.ColocateWorkerExtension",
        "--trust-remote-code", "--seed", "1234",
        "--nnodes", str(topo.nnodes), "--node-rank", str(topo.node_rank),
    ]
    if topo.nnodes > 1:
        mh, mp = _parse_addr(dist_addr or "")
        cmd += [
            "--master-addr", mh, "--master-port", str(mp),
            "--distributed-executor-backend", "mp",
            "--data-parallel-backend", "mp",
        ]
    if topo.headless:
        cmd.append("--headless")
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


def _wait_file(path: str, timeout: float) -> None:
    for _ in range(int(timeout)):
        if os.path.exists(path):
            return
        time.sleep(1)
    raise TimeoutError(path)


def _wait_alive(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited: {proc.returncode}")
        time.sleep(2)


def _run_trainer(url: str, model: str, tp: int, nnodes: int, node_rank: int, master: str, port: int) -> None:
    trainer = os.path.join(shared.SINGLE_DEMO_ROOT, "demo.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([shared.SINGLE_DEMO_ROOT, env.get("PYTHONPATH", "")])
    subprocess.run(
        [
            "torchrun",
            f"--nnodes={nnodes}", f"--node_rank={node_rank}",
            f"--nproc_per_node={tp // nnodes}",
            f"--master_addr={master}", f"--master_port={port}",
            trainer, "trainer",
            "--vllm-url", url, "--model-path", model, "--tp-size", str(tp),
        ],
        env=env, check=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/root/models/Qwen3-4B"))
    p.add_argument("--engine-gpus", type=int, default=int(os.environ.get("ENGINE_GPUS", "16")))
    p.add_argument("--num-gpus-per-node", type=int, default=int(os.environ.get("NUM_GPUS_PER_NODE", "8")))
    p.add_argument("--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "16")))
    p.add_argument("--gpu-mem-util", type=float, default=0.45)
    p.add_argument("--timeout-s", type=float, default=2400.0)
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

    node_rank = int(os.environ["NODE_RANK"])
    dist_addr = os.environ["DIST_INIT_ADDR"]
    barrier = os.environ.get("BARRIER_DIR", "")
    topo = _topo(args.engine_gpus, args.num_gpus_per_node, node_rank, args.tp_size)
    port = _free_port() if node_rank == 0 else 0

    cmd, env = _vllm_cmd(
        args.model_path, "127.0.0.1", port, topo, dist_addr, args.gpu_mem_util, 4096,
    )
    print(f"[node={node_rank}] nnodes={topo.nnodes} local_gpus={topo.local_gpus} tp={topo.tp}", flush=True)
    proc = subprocess.Popen(cmd, env=env)

    try:
        if node_rank == 0:
            url = f"http://127.0.0.1:{port}"
            client = VllmHttpClient(url)
            client.wait_healthy(timeout_s=args.timeout_s, proc=proc)
            train_host = os.environ.get("TRAIN_MASTER_ADDR", dist_addr.split(":")[0])
            train_port = int(os.environ.get("TRAIN_MASTER_PORT", str(_free_port())))
            client.sleep(level=0)
            if barrier:
                open(os.path.join(barrier, "vllm-slept"), "w").close()
            _run_trainer(url, args.model_path, args.tp_size, topo.nnodes, 0, train_host, train_port)
            client.wake(tags=["weights", "kv_cache"])
            text = client.generate(args.prompt, max_tokens=args.max_tokens, model_path=args.model_path)
            print(f"PASS multinode tp={args.tp_size} response={text!r}", flush=True)
            if barrier:
                open(os.path.join(barrier, "test-done"), "w").close()
        else:
            _wait_alive(proc)
            if barrier:
                _wait_file(os.path.join(barrier, "vllm-slept"), args.timeout_s)
            train_host = os.environ.get("TRAIN_MASTER_ADDR", dist_addr.split(":")[0])
            train_port = int(os.environ["TRAIN_MASTER_PORT"])
            _run_trainer("http://127.0.0.1:1", args.model_path, args.tp_size, topo.nnodes, 1, train_host, train_port)
            if barrier:
                _wait_file(os.path.join(barrier, "test-done"), args.timeout_s)
            print(f"PASS worker multinode tp={args.tp_size}", flush=True)
        return 0
    except Exception as e:
        print(f"FAIL node={node_rank}: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        _stop(proc)


if __name__ == "__main__":
    raise SystemExit(main())
