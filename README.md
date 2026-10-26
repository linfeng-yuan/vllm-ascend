# DSV4.1 Ascend950 ARM 算子修复与交接

本分支作为 DSV4.1 算子修复的独立交付分支，交付材料和构建说明位于仓库根目录。

本目录交付 2026-10-21 在 ARM64 环境构建的 Python 3.11 / 3.12 两组包所用的可重建源码、源码补丁和验证记录。2026-10-26 发布时，已重新核对交付包与原构建目录的 SHA256。

源码材料已上传；四个 wheel 经代理上传时，整包和单文件推送均发生连接中断，因此本分支先不包含 wheel。原始包保留在 45.107 的 `dsv41_workspace_verify_20261019` 容器和宿主机共享目录 `/mnt/share/y00882530/dsv41_release_20261021/deliverable/wheels/`。现场可按下文下载官方 DSL wheel，并分别构建 Python 3.11 / 3.12 的算子包。

## 解决的问题

| 问题 | 原因 | 本次修改 |
| --- | --- | --- |
| 首次请求仍然编译 QLI / QSLI | vLLM 未传 `seqused_q`，实际需要 `has_seqused_q=False`；原默认静态配置只有对应的 True 分支，binary key 不匹配 | 将 False / True 同时纳入默认静态导出；不改变 vLLM 调用语义 |
| 28 Cube 核机器无法直接沿用原算子 | QLI / QSLI 及其 metadata 存在固定 32 worker 的假设 | 纳入 cannbot-dsl PR181，从输入 NPU 读取实际 Cube 核数，调整任务、workspace 和 launch |
| AICPU metadata 构建依赖不存在的 HCC 目录 | 官方 DSL 0.7.0 使用 C++ 7.3.0 头文件路径，构建环境实际提供 14.3.0 | 构建器检测实际 HCC include 目录，并预编译、打包三个 AICPU metadata 动态库 |

这里的 28 / 32 核指 NPU Cube 核数。静态导出解决首次编译等待，改变输入长度仍须处于已覆盖的静态契约内；其他静态参数、源码或编译器版本变化可能需要重新导出。

## 具体改了什么

`native_build/` 提供完整构建输入，并与原始归档 `native_build.zip` 逐文件核对一致。以下路径均相对 `native_build/`：

| 文件 | 修改内容 |
| --- | --- |
| `samples/quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py` | 读取实际 Cube 核数；QLI 默认导出同时覆盖 False / True，去重后共 36 个静态配置 |
| `samples/quant_sparse_lightning_indexer_dsl/quant_sparse_lightning_indexer_dsl.py` | QSLI 按实际核数执行；默认导出同时覆盖 False / True，共 36 个配置；staging 保留与静态 TensorSpec 匹配的 36 worker 容量 |
| `samples/quant_lightning_indexer_metadata_dsl/quant_lightning_indexer_metadata_dsl.py` | metadata 使用实际 Cube 核数，保留固定 LI / LD 分区边界 36 |
| `samples/quant_sparse_lightning_indexer_metadata_dsl/quant_sparse_lightning_indexer_metadata_dsl.py` | 同步 QSLI metadata 的核数与分区契约 |
| `net/native_package/build.py` | 构建期自动发现 HCC C++ 头文件；导出、打包 AICore 与 AICPU 产物 |
| `net/native_package/pyproject.toml` | 固定算子 distribution 版本和 DSL 0.7.0 依赖，包含 AICPU 资源 |
| `install-wheels.sh` | 检查 ARM64 / Python ABI，清理提供同名模块的旧 distribution，再安装匹配的一组包 |
| `tests/` | 默认导出覆盖、实际核数、metadata 契约、静态命中和 NPU 参考数值检查 |

QSLI 的 staging 预留容量不会增加实际启动的 worker 数。相对按实际核数分配，32 核多占 272 KiB、28 核多占 544 KiB。不能只缩小运行时分配而保留当前固定 36 份的静态 TensorSpec。

编译器使用未修改的官方 `cannbot-dsl==0.7.0` wheel；修复在所附算子源码和 recipes 构建器中。`recipes_native_release.patch` 将完整构建目录加入指定版本的 cann-recipes-infer。它不修改 vLLM-Ascend 的运行时代码，也不自动更换现有镜像中的包。

## 文件与版本

- `native_build/`：可直接查看和构建的源码，含七个算子、构建器、安装脚本、测试和许可证。
- `native_build.zip`：相同源码的原始归档。
- `LICENSE`：源码所用的 CANN Open Software License Agreement Version 2.0。
- `patches/recipes_native_release.patch`：将构建输入加入 cann-recipes-infer 的完整补丁。
- `patches/operator_fixes.patch`：相对固定 cannbot-dsl 源码版本的修改，便于逐项审阅。
- `patches/pr181-source.patch`：PR181 原始补丁。
- `WHEEL_SHA256SUMS`：以下四个原始 wheel 的校验值；wheel 本身未上传，每组必须使用相同的 Python ABI。
- `evidence/`：上游版本、源文件 hash、wheel 来源、构建环境和原始验证摘要。
- `SHA256SUMS`：本目录交付文件校验值。

| Python / 架构 | 官方 DSL 编译器与运行时 | 本次构建的算子包 |
| --- | --- | --- |
| CPython 3.11 / aarch64 | `cannbot_dsl-0.7.0-cp311-cp311-manylinux_2_28_aarch64.whl` | `cannbot_arena_net-0.3.0+ds41fix20261021-cp311-cp311-linux_aarch64.whl` |
| CPython 3.12 / aarch64 | `cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl` | `cannbot_arena_net-0.3.0+ds41fix20261021-cp312-cp312-linux_aarch64.whl` |

当前打包器的 distribution 名为 `cannbot-arena-net`，导入名为 `ops`；沿用其 0.3.0 版本并追加修复标识，没有重新命名成旧的 `cannbot-arena-net-ops==0.1.0`。DSL 的 distribution 名为 `cannbot-dsl`，导入名为 `cannbotdsl`。

固定来源：

- [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)：`6d060311ae75f6105fe4216e20566875da5a8838`。
- [cannbot-dsl](https://gitcode.com/cann/cannbot-dsl)：samples / native builder 来自 `4519e5372a8b9306f7d823688c0cb8141ee34fb1`。
- [PR181](https://gitcode.com/cann/cannbot-dsl/pull/181)：`e9809cfca72d519c1d47c84fc3094cde11353af6`。

## 直接安装现有包

基础环境需提供匹配的 CANN、torch、torch_npu 等依赖。原构建环境为 CANN 9.2.0、torch 2.10.0、torch_npu 2.10.0.post7；算子 ELF 要求的最高 GLIBC 符号为 `GLIBC_2.34`。Python 3.11 和 3.12 分别构建，没有通过改 wheel 文件名复用另一版本。

在本目录执行：

```bash
sha256sum -c SHA256SUMS
# 从 45.107 取得原始 wheels/ 目录后再执行：
sha256sum -c WHEEL_SHA256SUMS
# 先停止将要更换包的服务；填写该服务实际使用的 Python 路径。
bash native_build/install-wheels.sh /path/to/service/python "$PWD/wheels"
```

安装器会卸载提供 `ops` / `cannbotdsl` 同名模块的旧 distribution，然后离线安装匹配的一组 wheel。应将安装步骤放在镜像中其他 vLLM / arena 依赖安装之后，避免随后被旧依赖覆盖。已有服务需要重启才能使用新包。

## 现场重建步骤

### 1. 准备源码和构建环境

先克隆交付分支，进入本目录并校验源码材料：

```bash
git clone --branch codex/dsv41-native-release-20261026 --single-branch \
  https://github.com/linfeng-yuan/vllm-ascend.git
cd vllm-ascend
sha256sum -c SHA256SUMS
```

完整源码已在 `native_build/`，克隆后可直接构建。若需将修改集成回 recipes 仓：

```bash
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
git checkout 6d060311ae75f6105fe4216e20566875da5a8838
git apply --check /path/to/patches/recipes_native_release.patch
git apply /path/to/patches/recipes_native_release.patch
cd models/deepseek_v4_1/native_build
```

构建必须使用 aarch64 的匹配 Python、DSL wheel 和 CANN 环境。两个 Python 环境均需已有匹配的 torch / torch_npu，以及 setuptools>=70.1、wheel、pytest。原构建使用 setuptools 81.0.0，详细依赖见 `evidence/build_environment_cp*.json`。

完整 `ds41` 组的 QW / K 导出会读取 NPU 设备属性，构建容器需要能看到 NPU。构建工作目录和输出目录应放在支持原子目录发布的本地文件系统。保留原来的七算子组：MQSMLA、QLI、QSLI、Indexer QW、Indexer K、Engram gate、Attention epilogue。

### 2. 下载并校验两个官方 DSL wheel

在本目录执行以下命令，按保存的官方地址下载，校验通过后才写入 `wheels/`。网络需要代理时，先在当前 shell 配置代理。此步骤下载官方 DSL；算子包使用下一步的源码现场构建。

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

destination = Path("wheels")
destination.mkdir(exist_ok=True)
for entry in json.loads(Path("evidence/dsl_wheel_provenance.json").read_text()):
    with urlopen(entry["url"], timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError("SHA256 mismatch: " + entry["file"])
    (destination / entry["file"]).write_bytes(data)
    print("Verified:", entry["file"])
PY
```

### 3. 分别构建 Python 3.11 / 3.12

从仓库根目录执行：

```bash
release_dir="$PWD"
cd native_build
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
unset CANNBOTDSL_DS41_PROFILES

export PYTHON=/path/to/python3.11
export CANNBOTDSL_WHEEL="$release_dir/wheels/cannbot_dsl-0.7.0-cp311-cp311-manylinux_2_28_aarch64.whl"
bash run-build.sh --group ds41 \
  --work-root /tmp/ds41-work-cp311 --output-root /tmp/ds41-output-cp311

export PYTHON=/path/to/python3.12
export CANNBOTDSL_WHEEL="$release_dir/wheels/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl"
bash run-build.sh --group ds41 \
  --work-root /tmp/ds41-work-cp312 --output-root /tmp/ds41-output-cp312
```

按实际路径修改 Python 和 `CANN_ENV`。构建结束会打印 `Final wheels:`。两个 ABI 使用独立 work / output 目录；不要再设置旧的临时 `CANNBOTDSL_DS41_PROFILES`，默认配置已包含 False 分支。

DSL wheel 也可按 `evidence/dsl_wheel_provenance.json` 中的官方地址和 SHA256 获取。源代码许可证保存在根目录和源码归档内，算子 wheel 同样包含许可证。

### 4. 安装后验证

使用服务对应的 Python，在 `native_build/` 目录运行：

```bash
python tests/test_release_contracts.py
python tests/verify_vllm_static.py --report /tmp/ds41-vllm-native.json
python tests/verify_all_static.py --report /tmp/ds41-all-native.json
python tests/run_kernel_precision.py --cores 32 --report /tmp/ds41-32.json
# 28 核实机使用 --cores 28；32 核卡上此参数只验证实际启动 28 worker。
```

验证脚本强制使用安装包内的静态产物，并禁用会掩盖遗漏的回退路径。预期每个 ABI 的默认组为 138 次静态命中、0 miss、0 后端编译；此前 vLLM 缺失的三组 False 配置应全部命中。

## 已完成验证及边界

原始验收记录见 `evidence/release_validation.json`：

- cp311 / cp312 各 138 个默认静态配置全部命中，0 miss、0 后端编译；各包含三个预编译 AICPU metadata 动态库。
- 两个 ABI 各完成 32 worker、28 worker 各 9 个 NPU 数值用例，合计 36 个，通过独立 MXFP4 / TopK 参考检查；覆盖 False 及长度变化。
- 实际硬件为 Ascend950DT_9582，32 Cube / 64 Vector 核。28 worker 在这台 32 核卡上真实执行；物理 28 核 SKU 尚未验收。
- 另有 cp311、32 核的两轮九场景性能 A/B：设备侧变化均在正负 1% 内，Python 纯算子调用耗时中位数约增加 1.2%。这评估整个 PR181，不能单独归因于 staging 预留；见 `evidence/performance_32core_20261021.txt`。
- 未完成整模型 DeepSWE 精度、vLLM 吞吐 / TTFT / TPOT 对照、物理 28 核性能 A/B，以及 cp312 性能 A/B。

本分支保存可重建的发布材料。后续镜像需要显式安装原始 wheel、现场构建的 wheel，或采用这份构建源码；其他官方镜像 tag 不保证包含这些修改。
