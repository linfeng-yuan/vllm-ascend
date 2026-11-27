#!/usr/bin/env bash
# Called once per DP rank by dp_online.py. Arguments match Aurora's template.
visible_devices=${1:?visible devices required}
engine_port=${2:?engine port required}
dp_size=${3:?DP size required}
dp_rank=${4:?DP rank required}
dp_address=${5:?DP address required}
local_ip=${NODE_DATA_IP:-$dp_address}
dp_rpc_port=${6:?DP RPC port required}
tp_size=${7:?TP size required}

set -euo pipefail
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY HCCL_INTRA_ROCE_ENABLE HCCL_ALGO

nic_name=${NPU_NIC_NAME:-data0.3001}
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FASTOKENS=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=2040 HCCL_CONNECT_TIMEOUT=1200
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics_dsv41_profile_$(basename "$PWD")_$(basename "$RUN_INSTANCE")_$dp_rank
if [[ ${AURORA_DRY_RUN:-0} != 1 ]]; then mkdir -p "$PROMETHEUS_MULTIPROC_DIR"; fi

unset aicpu_headers CPLUS_INCLUDE_PATH

mode=${TASK_MODE:?set TASK_MODE through start.sh}
kv_port=$((36010 + dp_rank))
prefill_dp_size=8
decode_dp_size=16
dp_external_args=()
if (( dp_size > 1 )); then
  dp_external_args=(
    --data-parallel-size-local 1
    --data-parallel-rank "$dp_rank"
    --data-parallel-address "$dp_address"
    --data-parallel-rpc-port "$dp_rpc_port"
  )
fi
common_args=(
  "${MODEL_PATH:?set MODEL_PATH}"
  --host 0.0.0.0 --port "$engine_port"
  --data-parallel-size "$dp_size" "${dp_external_args[@]}"
  --tensor-parallel-size "$tp_size"
  --max-model-len 1048576 --served-model-name dsv4_1
  --enable-expert-parallel --block-size 128
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}'
  --quantization deepseek_v4_fp8
  --tokenizer-mode deepseek_v41 --renderer-num-workers 4
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --reasoning-parser deepseek_v41 --trust-remote-code
)

case "$mode" in
  p-dp8|p-pcp8)
    export HCCL_BUFFSIZE=1024
    kv_role=kv_producer
    engine_id=$dp_rank
    if [[ "$mode" == p-pcp8 ]]; then
      prefill_dp_size=1
      role_args=(
        --max-num-batched-tokens 16384 --max-num-seqs 16
        --gpu-memory-utilization 0.90 --enforce-eager
        --prefill-context-parallel-size 8 --language-model-only
      )
      additional_config='{"dsv41_indexer_prologue":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"enable_pcp_o_proj_weight_sharding":true,"multistream_engram_overlap":false,"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true,"deepgemm_mega_moe":true,"enable_fused_mc2":3}'
    else
      role_args=(
        --max-num-batched-tokens 4096 --max-num-seqs 16
        --gpu-memory-utilization 0.90 --enforce-eager
      )
      additional_config='{"dsv41_indexer_prologue":true,"multistream_engram_overlap":false,"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true,"deepgemm_mega_moe":true,"enable_fused_mc2":3}'
    fi
    role_args+=(--speculative-config '{"num_speculative_tokens":5,"method":"dspark"}')
    if [[ ${P_PREFIX_CACHING:-1} == 1 ]]; then
      role_args+=(--enable-prefix-caching)
    else
      role_args+=(--no-enable-prefix-caching)
    fi
    ;;
  d-dp16|d-pcp8)
    export HCCL_BUFFSIZE=1600
    kv_role=kv_consumer
    engine_id=$((100 + dp_rank))
    if [[ "$mode" == d-pcp8 ]]; then
      prefill_dp_size=1
    fi
    role_args=(
      --max-num-batched-tokens 1024 --max-num-seqs 80
      --gpu-memory-utilization 0.92 --async-scheduling
      --no-enable-prefix-caching
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":480}'
      --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"max_iterations\":10}"
      --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51}'
    )
    additional_config='{"dsv41_indexer_prologue":true,"multistream_engram_overlap":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true,"enable_fused_mc2":3,"enable_force_eplb":true,"dsv41_mega_mhc":true,"combine_quant_mode":4,"scheduler_config":{"recompute_scheduler_enable":true},"deepgemm_mega_moe":true,"finegrained_tp_config":{"lmhead_tensor_parallel_size":8},"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}'
    ;;
  *) echo "Unknown TASK_MODE: $mode" >&2; exit 2 ;;
esac

kv_config=$(cat <<EOF
{"kv_connector":"MooncakeHybridConnector","kv_role":"$kv_role","kv_port":"$kv_port","engine_id":"$engine_id","kv_connector_extra_config":{"prefill":{"dp_size":$prefill_dp_size,"tp_size":$tp_size},"decode":{"dp_size":$decode_dp_size,"tp_size":$tp_size}}}
EOF
)
printf 'mode=%s rank=%s devices=%s port=%s kv_port=%s prefill_dp=%s decode_dp=%s\n' \
  "$mode" "$dp_rank" "$visible_devices" "$engine_port" "$kv_port" "$prefill_dp_size" "$decode_dp_size" >&2
command=(vllm serve "${common_args[@]}" "${role_args[@]}" \
  --kv-transfer-config "$kv_config" --additional-config "$additional_config")
if [[ ${AURORA_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
