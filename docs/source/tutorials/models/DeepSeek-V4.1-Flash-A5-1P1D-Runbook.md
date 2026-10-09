# DeepSeek V4.1 A5：1009 镜像容器与 1P1D 完整启动手册

## 1. 版本与范围

整理日期：2026-10-09。容器参数已对照 133.106/108 的 docker inspect；服务脚本来自本次验证归档。

- 镜像：`vllm-ascend:dev-26.2.0.day20261009-A5-py311-openEuler24.03-lts-aarch64`。
- P：141.61.133.106，通信 IP 172.27.18.106，HTTP 8367。
- D：141.61.133.108，通信 IP 172.27.18.108，HTTP 8467。
- 代理：P 机器，HTTP 8967；客户端请求代理。
- 两台各用 8 张 A5，每台 DP8 / TP1，开启 EP；MRV2、DSpark 5 token、Engram CPU offload + DP shared memory。
- 这是纯文本配置：image=0。P eager、Engram 多流关闭；D FULL_DECODE_ONLY、Engram 多流开启。
- 下文 D 脚本是标准基线，不含后来 LMHead TP8 实验改动；不能据此认定现在正在运行的 D 就是基线配置。
- 本次仅整理脚本，没有重启服务或重新进行性能验证。500 GiB shm 是已用配置，并非测定的最低需求；memlock 不限也不代表同事报错根因已定位。

## 2. 部署前需要替换的内容

换机器时同时修改 P/D 脚本的 VLLM_HOST_IP、HCCL_IF_IP、代理目标 IP、engine_id；网卡 data0.3001 必须存在且可通信。检查模型、共享目录、CANN 头文件路径与实际环境一致。

以下使用共享目录 `/mnt/share/y00882530/dsv4_1/upstream_pr_1006`，已存在的文件会被写入命令覆盖；如需保留原脚本，请先复制目录并同步修改所有路径。两台每台需 8 张空闲 NPU 和足够的主机内存。不要在已有服务占卡时重复拉起。

## 3. 两台宿主机加载镜像

```bash
cd /mnt/share/y00882530
sha256sum -c vllm-ascend_dev-26.2.0.day20261009-A5-py311-openEuler24.03-lts-aarch64.tar.gz.sha256
docker load -i vllm-ascend_dev-26.2.0.day20261009-A5-py311-openEuler24.03-lts-aarch64.tar.gz
docker image inspect vllm-ascend:dev-26.2.0.day20261009-A5-py311-openEuler24.03-lts-aarch64
npu-smi info
free -h
```

## 4. 创建容器（宿主机执行）

下面参数对应已验证容器。宿主机必须配置 Ascend Docker runtime；检查 `docker info` 中的 Runtimes。目录挂载前先确认存在，尤其 `/etc/hccl_rootinfo.json` 应是文件。不要挂载宿主机整个 `/usr/lib64` 到容器。

在 P 宿主机设置：

```bash
CONTAINER=codex_dsv41_day1009_133106
```

在 D 宿主机设置：

```bash
CONTAINER=codex_dsv41_day1009_133108
```

然后分别执行下列创建命令。若同名容器已存在，先 inspect 确认归属和配置；不要自动删除。

```bash
docker run -d \
  --name "$CONTAINER" \
  --runtime ascend \
  --network host \
  --ipc shareable \
  --shm-size 500g \
  --privileged \
  --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --log-opt max-size=100m \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /mnt:/mnt \
  -v /etc/hixlep:/etc/hixlep:ro \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  --entrypoint sleep \
  vllm-ascend:dev-26.2.0.day20261009-A5-py311-openEuler24.03-lts-aarch64 infinity

docker exec "$CONTAINER" bash -c 'mkdir -p /workspace; df -h /dev/shm; ulimit -l; npu-smi info'
```

`shm-size` 控制共享内存文件系统容量；`memlock` 是锁页额度；`stack` 是栈上限。服务脚本另设 `ulimit -n 65536`（文件描述符数）。不要把这几个参数当成同一种限制。

## 5. 写入完整服务脚本

下面命令在任一能访问共享目录的 Linux 机器执行一次即可。保留原始验证参数，包括 P HCCL_BUFFSIZE=2300、D=1600；没有证据证明 P 的 2300 优于 1600，复现时先保持一致。

公共环境中的 HCCL_EXEC_TIMEOUT=204 会被 P/D 脚本覆盖为 1200。CPLUS_INCLUDE_PATH 是 cannbotdsl AICPU 编译头文件路径的现有兼容配置，仍是遗留项。自定义 TorchInductor 缓存目录用于隔离实验，不是模型功能必需项。

```bash
mkdir -p /mnt/share/y00882530/dsv4_1/upstream_pr_1006
```

### day1009-pd-env.sh

```bash
cat > /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-pd-env.sh <<'SCRIPT_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=3600000
export GLOO_SOCKET_IFNAME=data0.3001
export TP_SOCKET_IFNAME=data0.3001
export HCCL_SOCKET_IFNAME=data0.3001
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_DETERMINISTIC=true
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


aicpu_headers=/usr/local/Ascend/cann-9.2.0/tools/hcc/aarch64-target-linux-gnu/include/c++/14.3.0
export CPLUS_INCLUDE_PATH="$aicpu_headers:$aicpu_headers/aarch64-target-linux-gnu:$aicpu_headers/backward"
SCRIPT_EOF
```

### day1009-retry-pd-p.sh

```bash
cat > /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-p.sh <<'SCRIPT_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -n 65536
source /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-pd-env.sh
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HOST_IP=172.27.18.106
export HCCL_IF_IP=172.27.18.106
export HCCL_BUFFSIZE=2300
export HCCL_EXEC_TIMEOUT=1200
export TORCHINDUCTOR_CACHE_DIR=/mnt/share/y00882530/dsv4_1/upstream_pr_1006/cache/day1009_pd_p
mkdir -p /workspace
cd /workspace
exec vllm serve /mnt/share/weight/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v41-a5-1005-real-1p1d \
  --host 0.0.0.0 \
  --port 8367 \
  --api-server-count 1 \
  --data-parallel-size 8 \
  --data-parallel-rpc-port 16971 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --seed 1024 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 128 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --trust-remote-code \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 \
  --quantization deepseek_v4_fp8 \
  --async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":21860,"engine_id":"day1009-p133-106","kv_connector_extra_config":{"prefill":{"dp_size":8,"tp_size":1},"decode":{"dp_size":8,"tp_size":1}}}' \
  --additional-config '{"multistream_engram_overlap":false,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true,"enable_fused_mc2":0,"enable_force_eplb":false}' \
  >/mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-p-serve.log 2>&1
SCRIPT_EOF
```

### day1009-retry-pd-d.sh

```bash
cat > /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-d.sh <<'SCRIPT_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -n 65536
source /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-pd-env.sh
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HOST_IP=172.27.18.108
export HCCL_IF_IP=172.27.18.108
export HCCL_BUFFSIZE=1600
export HCCL_EXEC_TIMEOUT=1200
export TORCHINDUCTOR_CACHE_DIR=/mnt/share/y00882530/dsv4_1/upstream_pr_1006/cache/day1009_pd_d
mkdir -p /workspace
cd /workspace
exec vllm serve /mnt/share/weight/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v41-a5-1005-real-1p1d \
  --host 0.0.0.0 \
  --port 8467 \
  --api-server-count 1 \
  --data-parallel-size 8 \
  --data-parallel-rpc-port 16972 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --seed 1024 \
  --max-model-len 262144 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --trust-remote-code \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 \
  --quantization deepseek_v4_fp8 \
  --async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":21870,"engine_id":"day1009-d133-108","kv_connector_extra_config":{"prefill":{"dp_size":8,"tp_size":1},"decode":{"dp_size":8,"tp_size":1}}}' \
  --additional-config '{"multistream_engram_overlap":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true,"enable_fused_mc2":0,"enable_force_eplb":false,"scheduler_config":{"recompute_scheduler_enable":true}}' \
  >/mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-d-serve.log 2>&1
SCRIPT_EOF
```

### day1009-retry-pd-proxy.sh

```bash
cat > /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-proxy.sh <<'SCRIPT_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -n 65536
exec python /mnt/share/y00882530/dsv4_1/rebase_1005/server_dp32/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 8967 \
  --workers 1 \
  --prefiller-hosts 172.27.18.106 \
  --prefiller-ports 8367 \
  --decoder-hosts 172.27.18.108 \
  --decoder-ports 8467 \
  >/mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-proxy.log 2>&1
SCRIPT_EOF
```

代理脚本调用共享目录中的 `load_balance_proxy_server_example.py`，它是独立依赖，不是 vllm serve 自动提供的服务。部署前确认该文件存在；迁移到没有共享目录的环境，需要一起复制它并保留对应版本：

```bash
test -f /mnt/share/y00882530/dsv4_1/rebase_1005/server_dp32/load_balance_proxy_server_example.py
```

## 6. 启动顺序

P 宿主机执行：

```bash
docker exec -d codex_dsv41_day1009_133106 bash /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-p.sh
tail -f /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-p-serve.log
```

D 宿主机执行（可与 P 并行加载）：

```bash
docker exec -d codex_dsv41_day1009_133108 bash /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-d.sh
tail -f /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-d-serve.log
```

日志重定向会覆盖旧日志，重跑前自行归档。`docker exec -d` 返回只表示命令已提交，不代表模型就绪。等待两端加载/图捕获完成，并从 P 机器确认：

```bash
curl --fail http://172.27.18.106:8367/health
curl --fail http://172.27.18.108:8467/health
```

两端就绪后，在 P 宿主机启动代理：

```bash
docker exec -d codex_dsv41_day1009_133106 bash /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-proxy.sh
tail -f /mnt/share/y00882530/dsv4_1/upstream_pr_1006/day1009-retry-pd-proxy.log
```

## 7. 请求冒烟

在能访问 P 通信网 IP 的机器执行：

```bash
curl --fail --show-error --max-time 300 \
  http://172.27.18.106:8967/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v41-a5-1005-real-1p1d",
    "messages":[{"role":"user","content":"请简单介绍一下自己。"}],
    "max_tokens":256,
    "temperature":0,
    "stream":false
  }'
```

检查请求完成、返回结构正常、两端无异常；256 token 只是连通性冒烟上限，不能作为精度评测设置。Engram CPU offload 成功也不等于整网已启动，必须等 health 和实际请求通过。

## 8. 常见核对项

- 卡是否空闲、主机内存是否充足；500 GiB tmpfs 是容量上限，实际使用仍消耗主机内存。
- 容器内 `df -h /dev/shm` 的有效容量是否与 Docker 配置一致，有无额外覆盖挂载。
- 头文件路径是否真实存在，不能在不同镜像里盲目沿用 14.3.0 路径。
- 两端通信 IP/网卡、HTTP 和 Mooncake 端口是否匹配，engine_id 是否唯一。
- 确认使用镜像内已准备的运行依赖；这些脚本不执行 pip 升级或重编译。
- 容器使用 host network，无需额外 `-p` 映射。
- 标准 D 与 LMHead TP8 实验不要混用；做性能对照还需保持输入输出长度、并发、预热和客户端参数一致。

## 9. 停止本部署

仅对确认属于自己的专用容器执行（会停止容器内所有进程；P 容器中的代理也会停止），保留容器和共享日志：

P 宿主机：

```bash
docker stop -t 30 codex_dsv41_day1009_133106
```

D 宿主机：

```bash
docker stop -t 30 codex_dsv41_day1009_133108
```

之后 `docker start` 只会启动容器的 sleep 进程，需要重新执行第 6 节的服务命令。
