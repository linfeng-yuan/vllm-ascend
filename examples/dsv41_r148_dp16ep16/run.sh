#!/usr/bin/env bash
# Inside the prepared runtime container, one invocation per role.
set -eo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
role=${1:?usage: run.sh p|d0|d1|proxy|bench}
: "${P_IP:?set P_IP}" "${D0_IP:?set D0_IP}" "${D1_IP:?set D1_IP}" "${RUN_DIR:?set RUN_DIR}"
source /opt/dsv41-r148-profile/env.sh
export PYTHONUNBUFFERED=1
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
ports=(8000 8001 8002 8003 8004 8005 8006 8007)
case "$role" in
  p)  ip=$P_IP;  master=$P_IP;  size=8;  rank=0; mode=p-dp8 ;;
  d0) ip=$D0_IP; master=$D0_IP; size=16; rank=0; mode=d-dp16 ;;
  d1) ip=$D1_IP; master=$D0_IP; size=16; rank=8; mode=d-dp16 ;;
  proxy)
      p_hosts=() d_hosts=()
      for port in "${ports[@]}"; do p_hosts+=("$P_IP"); d_hosts+=("$D0_IP"); done
      for port in "${ports[@]}"; do d_hosts+=("$D1_IP"); done
      exec python3 "$here/load_balance_proxy_server_example.py" --host "$D0_IP" --port 1999 --workers 4 \
          --prefiller-hosts "${p_hosts[@]}" --prefiller-ports "${ports[@]}" \
          --decoder-hosts "${d_hosts[@]}" --decoder-ports "${ports[@]}" "${ports[@]}"
      ;;
  bench)
      ulimit -Sn 8192
      : "${2:?pass the 128K JSONL path}"
      exec python3 "$here/measure_decode.py" "${@:2}"
      ;;
  *) printf 'Unknown role: %s\n' "$role" >&2; exit 2 ;;
esac
: "${MODEL_PATH:?set MODEL_PATH}" "${NPU_NIC_NAME:?set NPU_NIC_NAME}"
test -r "$MODEL_PATH/config.json"
mkdir -p "$RUN_DIR/profile" "$RUN_DIR/$role"
export NODE_DATA_IP="$ip" TASK_MODE="$mode" PROFILE_DIR="$RUN_DIR/profile"
export RUN_INSTANCE="$RUN_DIR/$role/$(date +%s)"
git -C "$repo" rev-parse HEAD > "$RUN_DIR/$role/source.sha"
test "$(cat "$RUN_DIR/$role/source.sha")" = 60ece1a4bc09480e014a28a58e6ca4ea7bd22abd
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_DIR/$role/launcher.exit"' EXIT
cd "$RUN_DIR/$role"
python3 "$here/launch/dp_online.py" --dp-size "$size" --tp-size 1 --pp-size 1 --pcp-size 1 --dcp-size 1 \
    --dp-size-local 8 --dp-rank-start "$rank" --dp-address "$master" --dp-rpc-port 12321 --vllm-start-port 8000 \
    2>&1 | tee "$RUN_DIR/$role/service.log"
