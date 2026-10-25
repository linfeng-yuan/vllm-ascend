#!/usr/bin/env bash
set -euo pipefail
# Run after the four backends are healthy and port 18300 is free.
docker exec -d ylf_pro0813_20261025 /usr/local/python3.12.13/bin/python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py --host 197.33.45.1 --port 18300 --workers 4 --prefiller-hosts 197.33.45.1 197.33.49.1 --prefiller-ports 18100 18100 --decoder-hosts 197.33.47.1 197.33.48.1 --decoder-ports 18200 18200
