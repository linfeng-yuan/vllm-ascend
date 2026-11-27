#!/usr/bin/env bash
set -euo pipefail
: "${IMAGE:?set IMAGE}" "${CONTAINER:?set CONTAINER}" "${MODEL_PATH:?set MODEL_PATH}" "${RUN_DIR:?set RUN_DIR}"
: "${P_IP:?set P_IP}" "${D0_IP:?set D0_IP}" "${D1_IP:?set D1_IP}" "${NPU_NIC_NAME:?set NPU_NIC_NAME}"
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$RUN_DIR"
mounts=(-v "$here:/repro:ro" -v "$RUN_DIR:$RUN_DIR" -v "$MODEL_PATH:$MODEL_PATH:ro")
for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/dcmi /usr/local/sbin /root/host \
            /usr/local/bin/npu-smi /etc/hccl_rootinfo.json; do
    if [[ -e $path ]]; then mounts+=(-v "$path:$path"); fi
done
if [[ -n ${MHC_VENDOR:-} ]]; then
    test -s "$MHC_VENDOR/op_api/lib/libcust_opapi.so"
    mounts+=(-v "$MHC_VENDOR:/opt/dsv41-r148-profile/vendor:ro")
fi
devices=()
for device in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc /dev/ummu /dev/uburma; do
    test -e "$device"
    devices+=(--device "$device")
done
docker run -dit --name "$CONTAINER" --network host --privileged --ipc "${CONTAINER_IPC:-shareable}" --shm-size 500g \
    "${mounts[@]}" "${devices[@]}" \
    -e P_IP -e D0_IP -e D1_IP -e NPU_NIC_NAME -e MODEL_PATH -e RUN_DIR \
    --entrypoint /bin/bash "$IMAGE"
