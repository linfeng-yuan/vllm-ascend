#!/usr/bin/env python3
"""
vLLM Online DP Launch Script
"""
import argparse
import signal
import time
import os
import subprocess
import sys
from pathlib import Path

# Also initialize when this launcher is invoked outside the startup wrapper.
os.environ["PROMETHEUS_MULTIPROC_DIR"] = '/dev/shm/vllm_metrics'
try:
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)
except OSError as exc:
    print(f"[Prometheus] metrics directory creation failed; continuing: {exc}", file=sys.stderr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--pcp-size", type=int, default=1, help="Prefill context parallel size.")
    parser.add_argument("--dcp-size", type=int, default=1, help="Decode context parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default="12321", help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=8000, help="Starting port for the engine.")
    return parser.parse_args()

args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
pp_size = args.pp_size
pcp_size = args.pcp_size
dcp_size = args.dcp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
# 一个 DP 副本占 tp×pp×pcp 张卡；DCP 不额外占卡
gpus_per_dp_rank = tp_size * pp_size * pcp_size
template_path = Path(__file__).resolve().with_name("run_dp_template.sh")

def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        str(template_path),
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
        str(pp_size),
    ]
    child_env = os.environ.copy()
    child_env["AICLUSTERHUB_PCP_SIZE"] = str(pcp_size)
    child_env["AICLUSTERHUB_DCP_SIZE"] = str(dcp_size)
    return subprocess.Popen(command, start_new_session=True, env=child_env)

if __name__ == "__main__":
    if not template_path.is_file():
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * gpus_per_dp_rank

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    exit_code = 0
    try:
        for i in range(dp_size_local):
            dp_rank = dp_rank_start + i
            vllm_engine_port = vllm_start_port + i
            visible_devices = ",".join(str(x) for x in range(i * gpus_per_dp_rank, (i + 1) * gpus_per_dp_rank))
            processes.append(run_command(visible_devices, dp_rank, vllm_engine_port))

        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                exit_code = 1
                break
            time.sleep(0.2)
        if any(p.poll() not in (None, 0) for p in processes):
            exit_code = 1
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    sys.exit(exit_code)
