# DeepSeek-V4.1 Flash on Ascend 950DT：2P1D 部署

本文记录当前 DeepSeek-V4.1 Flash 2P1D 环境的最小启动配置。这里的 1D 是一个跨四台机器的 Decode 实例，其并行配置为 DP32/EP32/TP1。

## 环境

| 项目 | 配置 |
| --- | --- |
| 服务器 | 6 台，每台 8×Ascend 950DT 96 GiB |
| 系统 | openEuler 24.03，aarch64 |
| CANN | 9.2.0 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post7 |
| vLLM / vLLM-Ascend | 0.30.0 |
| 镜像 | `vllm-ascend:dsv41-a5-pr17927-4cbe4962c-freshso-20261008` |
| 模型 | `/home/weights/deepseek-ai--DeepSeek-V4.1-Flash` |
| 实验代码 | 2026-10-09 主线基线；Static Kernel 实验叠加 [`4058b6f`](https://github.com/linfeng-yuan/vllm-ascend/commit/4058b6f63fded255b59069d29606c421759fa091)，恢复 `torch.compile` 装饰器并支持动态输入 |

实验代码叠加以下四个公开 PR：

- [#18101 AICPU URMA Cube offload](https://github.com/vllm-project/vllm-ascend/pull/18101)
- [#18102 GateLinear for DeepSeek V4 routers](https://github.com/vllm-project/vllm-ascend/pull/18102)
- [#18113 Overlap Engram projection](https://github.com/vllm-project/vllm-ascend/pull/18113)
- [#18131 Remove TensorMove after DeepSeek V4.1 MXFP8 o_proj](https://github.com/vllm-project/vllm-ascend/pull/18131)

数据网卡均为 `enp39s0f0`。

| 节点 | 数据 IP | 容器 | 并行配置 |
| --- | --- | --- | --- |
| P0 | 197.33.45.1 | `dsv41-p0` | node-local DP8/EP8/TP1 |
| P1 | 197.33.46.1 | `dsv41-p1` | node-local DP8/EP8/TP1 |
| D0 | 197.33.47.1 | `dsv41-d0` | global DP 0–7 |
| D1 | 197.33.48.1 | `dsv41-d1` | global DP 8–15 |
| D2 | 197.33.49.1 | `dsv41-d2` | global DP 16–23 |
| D3 | 197.33.50.1 | `dsv41-d3` | global DP 24–31 |

P0 和 P1 是两个独立的 P 实例，每个实例为 DP8/EP8/TP1，共 16 个 P API。D0–D3 组成一个 external DP32/EP32/TP1 实例，共 32 个 D API。D 侧 LMHead TP 为 8。KV 传输使用 `MooncakeHybridConnector`，Proxy 使用 4 workers。

## 1. 启动容器

六台机器使用同一个镜像。先准备：

- `/etc/hccl_rootinfo.json`
- `/etc/hixlep/`
- 模型目录
- 本文下面的四个脚本
- [`load_balance_proxy_server_example.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

将脚本和 Proxy 文件放到 `/mnt/shared/l00517252/ylf/2p1d`。下面以 P0 为例，其余机器只修改容器名：

```bash
docker run -itd \
  --name dsv41-p0 \
  --net host \
  --shm-size 512g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/ummu \
  --device /dev/uburma \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
  -v /etc/hixlep:/etc/hixlep:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /home/weights:/home/weights:ro \
  -v /mnt/shared/l00517252/ylf/2p1d:/workspace/pd \
  vllm-ascend:dsv41-a5-pr17927-4cbe4962c-freshso-20261008 bash
```

其余节点只修改容器名：

```text
dsv41-p1
dsv41-d0
dsv41-d1
dsv41-d2
dsv41-d3
```

## 2. 公共环境

`common_env.sh`：

```bash
#!/usr/bin/env bash
source /root/.bashrc
set -euo pipefail

: "${LOCAL_IP:?set LOCAL_IP}"

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME=enp39s0f0
export TP_SOCKET_IFNAME=enp39s0f0
export HCCL_SOCKET_IFNAME=enp39s0f0
export HCCL_ALGO=level0:fullmesh
export HCCL_EXEC_TIMEOUT=2040
export HCCL_CONNECT_TIMEOUT=1200
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
unset ASCEND_GLOBAL_RESOURCE_CONFIG || true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
```

## 3. Prefill

`run_prefill.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
source /workspace/pd/common_env.sh

: "${P_GROUP:?set P_GROUP to 0 or 1}"

mkdir -p /workspace/pd/logs

export HCCL_BUFFSIZE=512

for local_rank in {0..7}; do
  port=$((8100 + local_rank))
  engine_id=$((P_GROUP * 100 + local_rank))

  ASCEND_RT_VISIBLE_DEVICES="$local_rank" \
  vllm serve /home/weights/deepseek-ai--DeepSeek-V4.1-Flash \
    --host 0.0.0.0 \
    --port "$port" \
    --served-model-name deepseek-v41-a5-v0300-2p1d \
    --data-parallel-size 8 \
    --data-parallel-rank "$local_rank" \
    --data-parallel-address "$LOCAL_IP" \
    --data-parallel-rpc-port 13346 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --max-model-len 1048576 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --block-size 128 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enforce-eager \
    --async-scheduling \
    --quantization deepseek_v4_fp8 \
    --tokenizer-mode deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --safetensors-load-strategy lazy \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeHybridConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_device\":\"npu\",\"kv_buffer_size\":1000000000,\"kv_port\":\"32000\",\"engine_id\":\"${engine_id}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":8,\"tp_size\":1},\"decode\":{\"dp_size\":32,\"tp_size\":1},\"ascend_local_comm_res_path\":\"/etc/hixlep\"}}" \
    --engram-config '{"cpu_offload":true,"embedding_across_dp":false,"dp_shared_memory":true}' \
    --additional-config '{"enable_cpu_binding":true,"enable_force_eplb":true,"combine_quant_mode":4,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true,"multistream_engram_overlap":false,"engram_lookup_backend":"aicpu_urma_cube_hbm","enable_dsa_cp":false,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
    > "/workspace/pd/logs/p-r${local_rank}.log" 2>&1 &
done

wait
```

分别启动两台 P：

```bash
# P0
docker exec -d -e LOCAL_IP=197.33.45.1 -e P_GROUP=0 \
  dsv41-p0 bash /workspace/pd/run_prefill.sh

# P1
docker exec -d -e LOCAL_IP=197.33.46.1 -e P_GROUP=1 \
  dsv41-p1 bash /workspace/pd/run_prefill.sh
```

## 4. Decode

`run_decode.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
source /workspace/pd/common_env.sh

: "${DP_START:?set DP_START}"

mkdir -p /workspace/pd/logs

export HCCL_BUFFSIZE=2048

for local_rank in {0..7}; do
  dp_rank=$((DP_START + local_rank))
  port=$((8200 + local_rank))
  engine_id=$((10 + dp_rank))

  ASCEND_RT_VISIBLE_DEVICES="$local_rank" \
  vllm serve /home/weights/deepseek-ai--DeepSeek-V4.1-Flash \
    --host 0.0.0.0 \
    --port "$port" \
    --served-model-name deepseek-v41-a5-v0300-2p1d \
    --data-parallel-size 32 \
    --data-parallel-rank "$dp_rank" \
    --data-parallel-address 197.33.47.1 \
    --data-parallel-rpc-port 13345 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --max-model-len 1048576 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 84 \
    --block-size 128 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization deepseek_v4_fp8 \
    --tokenizer-mode deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --safetensors-load-strategy lazy \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false,"rejection_sample_method":"synthetic","synthetic_acceptance_length":5.1}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[480,504]}' \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeHybridConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_device\":\"npu\",\"kv_buffer_size\":1000000000,\"kv_port\":\"32100\",\"engine_id\":\"${engine_id}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":8,\"tp_size\":1},\"decode\":{\"dp_size\":32,\"tp_size\":1},\"ascend_local_comm_res_path\":\"/etc/hixlep\"}}" \
    --engram-config '{"cpu_offload":true,"embedding_across_dp":false,"dp_shared_memory":true}' \
    --additional-config '{"enable_cpu_binding":true,"enable_force_eplb":true,"combine_quant_mode":4,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true,"multistream_engram_overlap":true,"engram_lookup_backend":"aicpu_urma_cube_hbm","enable_dsa_cp":false,"finegrained_tp_config":{"lmhead_tensor_parallel_size":8},"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"scheduler_config":{"recompute_scheduler_enable":true}}' \
    > "/workspace/pd/logs/d-r${dp_rank}.log" 2>&1 &
done

wait
```

四台 D 必须连续发起，不要等待 D0 单独 ready：

```bash
docker exec -d -e LOCAL_IP=197.33.47.1 -e DP_START=0  dsv41-d0 bash /workspace/pd/run_decode.sh
docker exec -d -e LOCAL_IP=197.33.48.1 -e DP_START=8  dsv41-d1 bash /workspace/pd/run_decode.sh
docker exec -d -e LOCAL_IP=197.33.49.1 -e DP_START=16 dsv41-d2 bash /workspace/pd/run_decode.sh
docker exec -d -e LOCAL_IP=197.33.50.1 -e DP_START=24 dsv41-d3 bash /workspace/pd/run_decode.sh
```

## 5. Proxy

48 个后端全部通过 `/health` 后，在 P0 启动 Proxy。

`run_proxy.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

P_HOSTS=()
P_PORTS=()
D_HOSTS=()
D_PORTS=()

for ip in 197.33.45.1 197.33.46.1; do
  for port in {8100..8107}; do
    P_HOSTS+=("$ip")
    P_PORTS+=("$port")
  done
done

for ip in 197.33.47.1 197.33.48.1 197.33.49.1 197.33.50.1; do
  for port in {8200..8207}; do
    D_HOSTS+=("$ip")
    D_PORTS+=("$port")
  done
done

python /workspace/pd/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 8300 \
  --workers 4 \
  --prefiller-hosts "${P_HOSTS[@]}" \
  --prefiller-ports "${P_PORTS[@]}" \
  --decoder-hosts "${D_HOSTS[@]}" \
  --decoder-ports "${D_PORTS[@]}"
```

```bash
docker exec -d dsv41-p0 bash /workspace/pd/run_proxy.sh
curl -f http://197.33.45.1:8300/healthcheck
```

健康状态应为 16 个 P、32 个 D：

```json
{"status":"ok","prefill_instances":16,"decode_instances":32,"request_num":0}
```

## 6. 实验结果

三档使用相同代码和服务拓扑。表中的吞吐是 32 个 D DP 各自在指定 `running` 下的单卡打屏结果，不是 32 卡求和。

| global BS | D 配置 | 实测状态 | 稳态单 D 卡吞吐 | Step 耗时 |
| --- | --- | --- | --- | --- |
| 384 | `max_num_seqs=16`，capture `[72,96]` | 32/32 DP 达到 `running=12` | 各 DP 峰值 **2852.3–2859.1 tokens/s** | P50 **21.672 ms**；[下载 profiling](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-compilefix-static-syn51-c384-dp0-running12-20261010/dsv41-a5-compilefix-static-syn51-c384-dp0-running12-ascend-output-20261010.tar.gz) |
| 1536 | `max_num_seqs=50`，capture `[288,300]` | 32/32 DP 达到 `running=48` | 各 DP 峰值 **5561.4–5588.6 tokens/s** | P50 **44.036 ms**；[下载 profiling](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-compilefix-static-syn51-c1536-dp0-running48-20261010/dsv41-a5-compilefix-static-syn51-c1536-dp0-running48-ascend-output-20261010.tar.gz) |
| 2560 | `max_num_seqs=84`，capture `[480,504]`，`HCCL_BUFFSIZE=2048` | 32/32 DP 达到 `running=80` | 各 DP 峰值 **6359.8–6406.3 tokens/s** | P50 **64.081 ms**；[下载 profiling](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-compilefix-static-syn51-c2560-dp0-running80-20261011/dsv41-a5-compilefix-static-syn51-c2560-dp0-running80-ascend-output-20261011.tar.gz) |

三档均为 synthetic acceptance length 5.1，仅用于性能比较，不代表真实接受率或准确率。Profiling 压缩包只包含 D0/DP0/卡0 解析后的 `ASCEND_PROFILER_OUTPUT/`。
