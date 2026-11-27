#!/usr/bin/env bash
set -euo pipefail
repo=/vllm-workspace/dsv41-1124-unified-integration
vendor=${1:-/opt/dsv41-r148-profile/vendor}
test "$(git -C "$repo" rev-parse HEAD)" = 60ece1a4bc09480e014a28a58e6ca4ea7bd22abd
grep -Fx 'V41_FUSED_EPILOGUE_MAX_TOKENS = 384' "$repo/vllm_ascend/attention/dsa_v41.py"
test -s "$repo/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
test -s "$repo/vllm_ascend/libengram_hixl.so"
test -s "$repo/vllm_ascend/third_party/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so"
nm -D "$vendor/op_api/lib/libcust_opapi.so" | grep -w 'aclnnMegaMhcGetWorkspaceSize'
export PYTHONPATH="/opt/dsv41-profile-fastokens:$repo:/opt/dsv41-1124-nd-epilogue-wheels${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
import importlib.metadata
import json

for name, version in (("fastokens", "0.3.6"), ("cannbot_dsl", "0.8.2"), ("cannbot_arena_net", "0.3.0")):
    assert importlib.metadata.version(name) == version, name
    print(name, version)
with open("/opt/dsv41-1124-nd-epilogue-wheels/ops/_native/operators/attn_epilogue/operator.json") as source:
    assert len(json.load(source)["artifacts"]) == 856
PY
