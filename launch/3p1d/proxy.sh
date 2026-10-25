#!/usr/bin/env bash
set -euo pipefail
# Run on node 21 after all five backend APIs are healthy.
docker exec -d -e OMP_PROC_BIND=false -e OMP_NUM_THREADS=10 -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True -e TASK_QUEUE_ENABLE=1 ylf_pro0813_20261025 /usr/local/python3.12.13/bin/python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py --host 197.33.45.1 --port 18300 --workers 4 --prefiller-hosts 197.33.45.1 197.33.49.1 197.33.51.1 --prefiller-ports 18100 18100 18100 --decoder-hosts 197.33.47.1 197.33.48.1 --decoder-ports 18200 18200
