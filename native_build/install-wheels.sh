#!/usr/bin/env bash
set -euo pipefail
python_bin=${1:?Usage: install-wheels.sh /path/to/python /path/to/wheels}
wheel_dir=${2:?Usage: install-wheels.sh /path/to/python /path/to/wheels}
abi=$("$python_bin" -c 'import sys,platform; assert platform.machine()=="aarch64", platform.machine(); assert sys.version_info[:2] in [(3,11),(3,12)],sys.version; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
compiler="$wheel_dir/cannbot_dsl-0.7.0-${abi}-${abi}-manylinux_2_28_aarch64.whl"
operator="$wheel_dir/cannbot_arena_net-0.3.0+ds41fix20261021-${abi}-${abi}-linux_aarch64.whl"
[[ -f "$compiler" && -f "$operator" ]] || { echo "Missing matching $abi wheels in $wheel_dir" >&2; exit 1; }
# Stop the model service before invoking this installer. Remove old distributions
# before installing the replacements because they share Python module names.
"$python_bin" -m pip uninstall -y cannbotdsl cannbot-arena-net-ops cannbot-arena-net cannbot-dsl
"$python_bin" -m pip install --no-index --no-deps "$compiler" "$operator"
"$python_bin" -c 'import sys,ops,cannbotdsl,importlib.metadata as m; print(sys.executable);print(ops.__file__);print(m.version("cannbot-arena-net"),m.version("cannbot-dsl"))'
