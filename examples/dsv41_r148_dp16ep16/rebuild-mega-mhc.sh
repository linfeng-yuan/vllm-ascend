#!/usr/bin/env bash
# Run inside the ARM64 r148 container. The input directory is read-only.
set -eo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
inputs=$(realpath "${1:?usage: rebuild-mega-mhc.sh SOURCE_ARTIFACT_DIR BUILD_DIR}")
work=$(realpath -m "${2:?use a new build directory on a writable mount}")
mkdir -- "$work"
(cd "$inputs" && sha256sum -c "$here/mega-mhc-sources.sha256")
mkdir -p "$work/src" "$work/third_party/patch"
tar -xzf "$inputs/ops-transformer-224e4b136-lf.tar.gz" -C "$work/src"
tar -xzf "$inputs/native20-ops-tensor.tar.gz" -C "$work/src"
tar -xzf "$inputs/native20-opbase.tar.gz" -C "$work"
mv "$work/opbase" "$work/ops-base"
tar -xzf "$inputs/native20-catlass.tar.gz" -C "$work/third_party"
tar -xzf "$inputs/cann-cmake-master-059-lf.tar.gz" -C "$work/third_party"
cp "$inputs/"{abseil-cpp-20230802.1.tar.gz,eigen-5.0.0.tar.gz,json-3.12.0.tar.gz,makeself-release-2.5.0.tar.gz,protobuf-25.1.tar.gz} "$work/third_party/"
cp "$inputs/"{backport-CVE-2025-0838.patch,makeself-2.5.0.patch} "$work/third_party/patch/"

source /usr/local/Ascend/cann-9.3.0/set_env.sh
unset aicpu_headers CPLUS_INCLUDE_PATH
# r148 pip maps this env value to build_isolation=False for the graph stub wheel.
export PIP_NO_BUILD_ISOLATION=0
cd "$work/src/ops-transformer"
bash build.sh --pkg --soc=ascend950 --ops=mega_mhc --vendor_name=dsv41_mhc20 \
    "--cann_3rd_lib_path=$work/third_party" -j"${BUILD_JOBS:-18}" 2>&1 | tee "$work/build.log"
installer="$PWD/build_out/cann-ops-transformer-dsv41_mhc20_linux-aarch64.run"
test -s "$installer"
bash "$installer" --noexec --extract="$work/package"
vendor="$work/package/packages/vendors/dsv41_mhc20_transformer"
test -s "$vendor/op_api/lib/libcust_opapi.so"
nm -D "$vendor/op_api/lib/libcust_opapi.so" | grep -w 'aclnnMegaMhcGetWorkspaceSize'
printf '\nMegaMHC vendor directory: %s\n' "$vendor"
