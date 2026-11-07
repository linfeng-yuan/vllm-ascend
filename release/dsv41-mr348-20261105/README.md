# DSV4.1 MR348 + MegaMoE 镜像源码与出包

本分支保存下面这个已交付镜像中的 vllm-ascend 源码：

`vllm-ascend-dev:0.32.0-dsl080-mr348-megamoe-metadata-fence-p8192-20261105`

镜像 ID：`sha256:9179fe2e4fcb4780f8ea0f1a9a9b7475ac94fbe6b36ff7e27553ee4ad4f36d5f`。
tar 所在目录：`/mnt/share/y00882530/releases/dsv41-mr348-megamoe-p8192-20261105/`。
文件名：`vllm-ascend-dev_0.32.0-dsl080-mr348-megamoe-metadata-fence-p8192-20261105.tar`。
tar 大小 26284340224 字节；SHA256：
`7542e5a6dcf1860f68feb982d8d875ca45b23bef4c807318f07d5bfda80f012c`。

## 与镜像的对应关系

镜像 Git HEAD 为 `80f1a251c30bf7057036677e855515c39f373815`，
只有两个未提交文件：`decoder_replay_state.py` 和 `engram/model_state.py`。
本分支在该 HEAD 上保留这两处 metadata 流等待修复，既有 DeepGEMM MegaMoE 集成也保留。
发布目录是额外的说明与审计文件；原 4739 个 tracked 文件的 SHA256 已全部与镜像逐项比对。
清单见 [image-source-manifest.json](image-source-manifest.json)。
这里保存 11 月 5 日的快照，没有混入 11 月 6 日的三项性能优化。

修复通过 `DecoderReplayInputs.metadata_producer_stream` 传递 metadata 生产流，
在 compact replay metadata 构建前用当前流 `wait_stream` 等待，避免两条 planner 路径交叠。
原始差异见 [image-metadata-fence.patch](image-metadata-fence.patch)。

## 固定依赖

| 组件 | 镜像版本或提交 |
| --- | --- |
| 平台 | Linux aarch64 / Ascend 950DT / CANN 9.2.0 / Python 3.12.13 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| vllm | 0.32.0，`5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vllm-ascend | 0.32.0，80f1a251 + 本分支两处等待修复 |
| cannbot-dsl | 0.8.0 |
| cannbot-arena-net | 0.3.0+ds41.local32pipe32.mr348.20261105 |
| deep-gemm / tilelang | 0.1.1+working45108 / 0.1.15 |

准确 wheel SHA 见 [provenance.json](provenance.json)。
MegaMoE 使用原交付的 DeepGEMM 工作包；普通公开仓库 HEAD 不能自动视为这个工作包的替代品。
arena 的完整源码、profile 和出包脚本在
[公开 cannbotdsl 分支](https://github.com/linfeng-yuan/cannbotdsl/tree/release/dsv41-mr348-local32-pipe32-20261105/release/dsv41-mr348-20261105)。
该分支的 `build_mr348.py --work-root /tmp/arena-build --jobs 2` 从当前源码全量编译 arena，
无需历史 arena base wheel；仍需 CANN 9.2.0 和独立的 cannbot-dsl 0.8.0 编译器依赖。
具体环境与命令见上面的公开分支说明。

## DeepGEMM 如何出包

当前交付镜像实际安装独立的工作包，现成产物是：

```text
/mnt/share/y00882530/clean-editable-megamoe-dsl080-20261104/wheels/deep_gemm-0.1.1+working45108-cp312-cp312-linux_aarch64.whl
```

该包 SHA256：`2a8c7c4749cf02f404d490a235fbd0152116b3d07e6fc256c4b95bd7c99aff7c`。
它保存成功运行环境的原生扩展、Python 模块和 JIT 头文件，并带两处 CANN 9.2 头文件兼容修改。
复刻本镜像时使用这个包，不需要重新手工编译：

```bash
python3 -m pip install --no-deps --force-reinstall \
  /mnt/share/y00882530/clean-editable-megamoe-dsl080-20261104/wheels/deep_gemm-0.1.1+working45108-cp312-cp312-linux_aarch64.whl
```

源码同时已有内置构建入口，条件为 A5 目标且 `COMPILE_CUSTOM_KERNELS=1`：

- `setup.py` 注册 `vllm_ascend.third_party.deep_gemm._C` 扩展。
- `build_ext` 调用 `tools/deepgemm/build.py`，固定 DeepGEMM-Ascend 提交
  `3a35326271b069b8b5bf1a754435acd6cc91575b` 及其递归依赖。
- 构建脚本先用上游 `setup.py` 生成临时 DeepGEMM wheel，再提取原生扩展、Python 模块、
  JIT 头文件和许可证到 `vllm_ascend/third_party/deep_gemm/`，并应用兼容修改。
- 这些内容随 vllm-ascend wheel 交付；临时的独立 DeepGEMM wheel 不另作发行产物。
- `pyproject.toml` 通过 setuptools 调用该构建入口，没有声明从 PyPI 安装 `deep-gemm`。

构建入口存在不等于重编产物已经与本镜像完成验收。
本次源码发布没有重新产出、验收内置 DeepGEMM 的完整 vllm-ascend wheel；
当前模型运行证据对应上面的独立工作包。
运行时若已安装符合 NPU API 的外部 `deep_gemm`，loader 优先使用外部包；否则使用内置包。
需要判断实际用了哪份，可查看 `Loaded NPU DeepGEMM ... from ...` 日志。

## 在原镜像中更新 Python 源码

最直接的使用方式是加载原 tar，再在独立容器中切换源码。
两处修复均为 Python 代码，`csrc` 与镜像相同；不能随意替换其他版本 `.so`。

```bash
docker load -i /path/to/vllm-ascend-dev_0.32.0-dsl080-mr348-megamoe-metadata-fence-p8192-20261105.tar
git clone --branch release/dsv41-mr348-megamoe-p8192-20261105 https://github.com/linfeng-yuan/vllm-ascend.git
cd vllm-ascend
git rev-parse HEAD:csrc
```

`csrc` tree 应为 `fd210ec80c1f969dc37674f6fa8d2a413bc08b5f`。
在原镜像的 `/vllm-workspace/vllm-ascend` 中更新上述两个文件即可；原有 editable 安装会使用它们。
完整替换源码时先保留镜像已有的编译产物，并核对 Python ABI、torch/torch_npu、CANN 和 csrc tree。
原 native binding SHA256 为
`be4bb9d47629020790bb9cab8da3d53d8bccb9c39559d5290fc594bd000416af`。

## 编译新的 vllm-ascend wheel

在匹配上表依赖的 arm64 构建容器中执行，`SOC_VERSION` 按目标 950DT 芯片填写。
这条路径重新编译 csrc，不依赖复用 `.so`。

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export SOC_VERSION='<实际 ascend950 芯片值>'
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8
export SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0
python3 -m pip wheel --no-deps --no-build-isolation . -w dist
sha256sum dist/*.whl
```

此处提供构建入口；本次发布做源码一致性核对，没有重新编译整套镜像或重复 NPU 精度测试。
重编后需要验证自定义算子装载、图模式和 DP8EP8 的实际模型请求。

## 从原镜像产出自己的派生镜像与 tar

下面的构建上下文放在本仓库根目录；依赖原镜像，保留原编译库与工作包。

```dockerfile
FROM vllm-ascend-dev:0.32.0-dsl080-mr348-megamoe-metadata-fence-p8192-20261105
COPY vllm_ascend/models/deepseek_v41/decoder_replay_state.py /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/decoder_replay_state.py
COPY vllm_ascend/models/deepseek_v41/engram/model_state.py /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/model_state.py
ENV CANNBOTDSL_NATIVE_BINARY_MODE=require
WORKDIR /vllm-workspace
CMD ["/bin/bash"]
```

```bash
docker build -f Dockerfile.release -t dsv41:mr348-source-published .
docker save -o dsv41-mr348-source-published.tar dsv41:mr348-source-published
sha256sum dsv41-mr348-source-published.tar > dsv41-mr348-source-published.tar.sha256
```

派生镜像的 image ID 与 tar SHA 会变化；不能把它标成原交付镜像的相同二进制。
这条路径依赖原镜像，并非 DeepGEMM、DSL 编译器和所有基础依赖从零重建。

## 原交付运行配置

P / D 均为 DP8 EP8 TP1，`deepgemm_mega_moe=true`。
P：`max-num-batched-tokens=8192`、`max-num-seqs=16`，prefix cache 和 bounded replay 开启。
D：`max-num-batched-tokens=1024`、`max-num-seqs=64`，两者关闭，`FULL_DECODE_ONLY`。
精确启动脚本仍在原镜像的 `/vllm-workspace/scripts/`：

```bash
bash /vllm-workspace/scripts/start-p.sh
bash /vllm-workspace/scripts/start-d.sh
```

需要按原发布目录的 `run-container.py` / `USAGE.txt` 挂载驱动、设备、模型与共享目录。
本次源码发布没有启动新模型服务，也没有占用 107/108 的 NPU 卡。
