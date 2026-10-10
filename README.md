# DSV4.1 A5 Profiling

Profiling 产物放在 GitHub Releases。本分支只保留本说明文件，不包含 vLLM / vLLM-Ascend 源码。

## 最新性能记录：compile 装饰器修复 + Static Kernel + synthetic 5.1（无 profiling，2026-10-10）

本条只记录无 profiling 性能，不包含 profiling 压缩包，也不作为新的 profiling 轮次。以下以第二轮热态结果为主；测试使用 synthetic acceptance，只用于性能比较，不代表真实接受率或准确率。

| 项目 | 配置 / 结果 |
| --- | --- |
| 代码 | 2200 TPS 基线运行代码 `5797a877`，仅叠加 compile 装饰器修复 `4058b6f`；运行时集成记录 `017852a94e0a1db8c8449fd91fb1d60e77f7a163`；未叠加 `4d82831`，未重编译自定义算子 SO |
| Wheels | 恢复旧版本：`cannbotdsl 0.4.dev28`、`cannbot_arena_net_ops 0.1.0` |
| 服务 | P：两组 node-local DP8/EP8；D：跨四节点 external DP32/EP32/TP1；Proxy workers=4 |
| D 关键配置 | MRV2、LMHead TP8、`max_num_seqs=16`、`max_num_batched_tokens=1024`、DSpark 5、synthetic 5.1、force EPLB、recompute scheduler、AICPU URMA Engram |
| 图配置 | NPUGraphEx 开启；Static Kernel 开启；Super Kernel 关闭；capture sizes `[72, 96]` |
| 负载 | 384 并发、384 请求；输入 129054 tokens；输出 4096 tokens；prefix repeat rate 100% |
| P 预热 | 两轮直连全部 16 个 P API；第二轮各 P 的 prefix cache 命中率均为 99.9768% |
| 第二轮请求校验 | 384 成功、0 失败；全部输出 4096 tokens；D 成功请求增量 384；生成 token 增量 1,572,864；32/32 D 均实测达到 running=12 |
| AISBench 整体吞吐 | **62,236.4926 tokens/s**；这里是端到端整体 output throughput，不与各 DP 的独立峰值相加 |
| AISBench TTFT | 平均 3130.4 ms；中位 3120.6 ms；P90 4278.7 ms；P99 4807.5 ms |
| AISBench TPOT | **平均 4.5 ms**；中位 4.5 ms；P90 **4.7 ms**；P99 4.7 ms |
| AISBench ITL | **平均 22.9 ms**；中位 21.5 ms；P90 **28.2 ms**；P99 77.6 ms |
| D 打屏，指定 running=12 | 32 个 DP 的 `Avg generation throughput` 峰值范围 **1820.4～1931.9 tokens/s**，均值 **1868.2188 tokens/s**；例如 DP6 为 **1931.9 tokens/s @ running=12, waiting=0** |
| D 打屏，本轮任意 active 窗口 | 各 DP 独立峰值范围 **2759.1～2818.4 tokens/s**，均值 **2788.6813 tokens/s**；全轮单 DP 最大值为 **DP30 2818.4 tokens/s @ running=9, waiting=0** |
| External KV | hits / queries = 49,556,352 / 49,556,736，命中率约 99.9992% |
| 结果目录 | `/mnt/shared/l00517252/ylf/dsv41-a5-compilefix-static-syn51-1010/run/2p1d/benchmarks/performance-20261010-c384-compilefix-static-syn51-r2-1806` |

AISBench TPOT / ITL 来自 `gsm8k.csv` 原始统计，不是由 vLLM 打屏吞吐反算。D 打屏值为服务端 10 秒统计窗口；每个 DP 的峰值发生时刻可能不同，因此不能把 32 个独立峰值相加作为整体峰值。

第一轮冷态结果为 AISBench output throughput 42,903.7745 tokens/s、TPOT 平均 4.5 ms / P90 4.8 ms、ITL 平均 23.2 ms / 中位 21.4 ms / P90 28.0 ms；由于 TTFT 平均 15,103.2 ms 且各 DP 进入 running=12 的时间不齐，本记录以第二轮热态结果为主。

与同机旧 wheels A/B 第二轮（51,882.1490 tokens/s、TPOT 5.7 ms、ITL 平均 23.5 ms）相比，本轮整体 output throughput 高约 19.96%，TPOT 平均低约 21.05%，ITL 平均低约 2.55%。但本轮同时改变了 Static Kernel、synthetic acceptance length 和 compile 装饰器，不能把差异归因于单一开关。

## 最新：主线四个 PR + AICPU URMA Engram，D0/DP0 running=12（第四次，2026-10-10）

[打开 Release](https://github.com/linfeng-yuan/vllm-ascend/releases/tag/dsv41-a5-main-prs4-urma-dp0-running12-20261010)

- [下载卡 0 解析结果 `ASCEND_PROFILER_OUTPUT`](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-main-prs4-urma-dp0-running12-20261010/dsv41-a5-main-prs4-urma-dp0-running12-ascend-output-20261010.tar.gz)
- [SHA256SUMS](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-main-prs4-urma-dp0-running12-20261010/SHA256SUMS)
- SHA256：`f975fb2e185ed1b50b48ee4b816424c44c83a19b9bb75acddaece6fbac0205e1`

本压缩包只包含 D0/DP0/卡 0 已解析的 `ASCEND_PROFILER_OUTPUT/`，不包含其他卡、原始 `PROF_*`、`FRAMEWORK/`、服务日志或源码。

| 项目 | 配置 / 结果 |
| --- | --- |
| Engram | P、D 均显式启用 `engram_lookup_backend=aicpu_urma_cube_hbm`；Profile 中出现 `EngramUrmaGather` |
| 无 profiling 性能 | 384/384 成功，全部输出 4096；TPOT 5.7 ms；AISBench output throughput 48,551.8969 tokens/s |
| 无 profiling 的 vLLM 打屏 | 32 个 D 都实测到 running=12；30 个稳定 running=12 DP 的峰值 2219.0～2225.2 tokens/s、均值 2221.73，对应 ITL 5.393～5.408 ms/token；DP0/1 的窗口峰值分别为 2213.6@running11、2216.0@running11 |
| Profiling 本轮性能 | 384/384 成功，全部输出 4096；TPOT 6.2 ms；AISBench output throughput 49,215.1912 tokens/s |
| Profiling 轮 vLLM 打屏 | 32 个 D 的峰值均对应 running=12、waiting=0；2210.7～2216.4 tokens/s，均值 2212.3125；换算 ITL 5.414～5.428 ms/token，均值 5.424 ms/token |
| 采集 | D0/DP0/卡 0；触发时 running=12、waiting=0；profiler 启停请求间隔 2.000 秒 |
| 解析 | 161,639 条 kernel；84 个 step 标记；step P50 22.706 ms、P90 24.798 ms；P50 / synthetic 4.15 ≈ 5.471 ms/token |
| 对比 | AISBench 整体 output throughput 比 UVA 低 1.96%；`EngramUrmaGather` 约 588.584 us/次，是 UVA gather 约 200.946 us/次的 2.93 倍，本负载下 URMA 不是更优开关 |

Release 正文包含 DP0～DP31 的逐 DP vLLM 打屏吞吐、每行对应的 running，以及 ITL/step interval 的换算关系。

## 历史：主线四个 PR + UVA Engram，D0/DP0 running=12（第三次，2026-10-09）

[打开 Release](https://github.com/linfeng-yuan/vllm-ascend/releases/tag/dsv41-a5-main-prs4-uva-dp0-running12-20261009)

- [下载卡 0 解析结果 `ASCEND_PROFILER_OUTPUT`](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-main-prs4-uva-dp0-running12-20261009/dsv41-a5-main-prs4-uva-dp0-running12-ascend-output-20261009.tar.gz)
- [SHA256SUMS](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-main-prs4-uva-dp0-running12-20261009/SHA256SUMS)
- SHA256：`37a5dc256c628be191bcae48158d35cb0c7a8828856d1976922dc3f118bae2fb`

本压缩包只包含 D0/DP0/卡 0 已解析的 `ASCEND_PROFILER_OUTPUT/`，不包含其他卡、原始 `PROF_*`、`FRAMEWORK/`、服务日志或源码。

| 项目 | 配置 / 结果 |
| --- | --- |
| 代码 | `linfeng-yuan/vllm-ascend:codex/dsv41-a5-main-prs-1009`，`3352c26c`；叠加 #18101、#18102、#18113、#18131 及既有 Gate/Router/LMHead TP 修改 |
| Engram | 本轮仍为默认 `engram_lookup_backend=uva`；代码虽包含 #18101，但本轮没有启用其 `aicpu_urma_cube_hbm` 后端 |
| 服务 | P：两组 node-local DP8/EP8；D：跨 4 节点 external DP32/EP32/TP1；Proxy workers=4 |
| D 关键配置 | MRV2、LMHead TP8、`max_num_seqs=16`、`max_num_batched_tokens=1024`、DSpark 5、synthetic 4.15、force EPLB、recompute scheduler |
| 图配置 | NPUGraphEx 开启；Static Kernel / Super Kernel 关闭；capture sizes `[72, 96]` |
| 负载 | 384 并发、384 请求、输入 129054 tokens、输出 4096 tokens |
| 无 profiling 性能 | 384 成功、0 失败；TPOT 5.7 ms；AISBench output throughput 49,520.4541 tokens/s；32 个 D 在 running=12、waiting=0 时的打屏峰值 2215.4～2222.6 tokens/s，均值 2217.87；换算 ITL 5.399～5.417 ms/token（各 DP 峰值不求和） |
| Profiling 本轮性能 | 384 成功、0 失败，全部输出 4096 tokens；TPOT 6.2 ms；AISBench output throughput 49,476.0217 tokens/s |
| 采集目标 | D0 / DP0 / 卡 0，触发时 running=12、waiting=0；profiler 启停请求间隔 2.000 秒 |
| 解析结果 | 158,058 条 kernel 记录；83 个 `_expand_idx_mapping_kernel` 标记、82 个相邻 step 间隔 |
| Step 间隔 | P50 22.690 ms，P90 26.594 ms；6 个 >=30 ms 的 profiling 扰动长尾 |
| Cast 观察 | 仍有 249 次 `[72,5120]` BF16→FP32 cast（合计约 884.414 us），所以不能表述为 #18102 已消除全部模型路径 FP32 cast |

解析结果包含 `trace_view.json`、`kernel_details.csv`、`operator_details.csv`、`task_time.csv`、`analysis.db` 等文件。Profile 中 `_engram_host_uva_gather_dequant_kernel` 共 166 次，确认本轮走 UVA；下一组单独切换 `aicpu_urma_cube_hbm`，避免和本轮混淆。

## 历史：主线 + LMHead TP8，D0/DP0 running=12（第二次，2026-10-09）

[打开 Release](https://github.com/linfeng-yuan/vllm-ascend/releases/tag/dsv41-a5-main-lmheadtp8-dp0-running12-20261009)

- [下载卡 0 解析结果 `ASCEND_PROFILER_OUTPUT`](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-main-lmheadtp8-dp0-running12-20261009/dsv41-a5-main-lmheadtp8-revert18081-dp0-running12-ascend-output-20261009.tar.gz)
- [SHA256SUMS](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-main-lmheadtp8-dp0-running12-20261009/SHA256SUMS)
- SHA256：`5a86160ca1a80f5a044a6a086c6cecb8c28733a0707245c3b3bc57eaa4766a24`

本压缩包只包含 D0/DP0/卡 0 的 `ASCEND_PROFILER_OUTPUT/`，不包含其他卡、原始 `PROF_*`、`FRAMEWORK/`、服务日志或源码。

| 项目 | 配置 / 结果 |
| --- | --- |
| 代码 | `linfeng-yuan/vllm-ascend:codex/dsv41-a5-main-perf-1009`，`63d01cd9`；基于上游主线并撤销 #18081 的相关装饰器改动 |
| 服务 | P：两组 node-local DP8/EP8；D：跨 4 节点 external DP32/EP32/TP1；Proxy workers=4 |
| D 关键配置 | MRV2、LMHead TP8、`max_num_seqs=16`、`max_num_batched_tokens=1024`、DSpark 5、synthetic 4.15、force EPLB、recompute scheduler |
| 图配置 | NPUGraphEx 开启；Static Kernel / Super Kernel 关闭；capture sizes `[72, 96]` |
| 负载 | 384 并发、384 请求、输入 129054 tokens、输出 4096 tokens |
| 采集目标 | D0 / DP0 / 卡 0，running=12、waiting=0 |
| 采集窗口 | profiler 启停请求之间 2.000 秒；42 次活动采样均为 running=12、waiting=0 |
| 请求结果 | 384 成功、0 失败，全部输出 4096 tokens |
| Profiling 本轮性能 | TPOT 6.6 ms；AISBench output throughput 47,555.65 tokens/s |
| 解析结果 | 177,857 条 kernel 记录；84 个 `_expand_idx_mapping_kernel` 标记、83 个相邻 step 间隔 |
| Step 间隔 | P50 23.251 ms，P90 23.309 ms；其中 6 个 >=30 ms 的 profiling 扰动长尾 |

解压后可直接查看 `ASCEND_PROFILER_OUTPUT/trace_view.json`、`kernel_details.csv`、`operator_details.csv`、`task_time.csv`、`analysis.db` 等离线解析结果。

## 历史：1008 镜像基线，D0/DP0 running=12（第一次，2026-10-09）

[打开第一次 Release](https://github.com/linfeng-yuan/vllm-ascend/releases/tag/dsv41-a5-dp0-running12-r2-20261009)

- [下载第一次 profiling](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-dp0-running12-r2-20261009/dsv41-a5-dp0-running12-r2-20261009-raw.tar.gz)
- SHA256：`4f162b79d5ceda4515324cf2cfa496cf38edfa0c8b4bd71bb83d841fbd9d4693`

第一次为 10 月 8 日晚镜像基线、NPUGraphEx 关闭；与上面的“主线 + LMHead TP8”第二次采集分开保留，避免混淆。

```bash
sha256sum -c SHA256SUMS
```

## 问题 profiling：新 cann-bot wheels 后 D 打屏吞吐下降（不计入性能刷新序列，2026-10-10）

[打开问题定位 Release](https://github.com/linfeng-yuan/vllm-ascend/releases/tag/dsv41-a5-whl060-issue-dp0-running12-20261010)

- [下载卡 0 解析结果 `ASCEND_PROFILER_OUTPUT`](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-whl060-issue-dp0-running12-20261010/dsv41-a5-whl060-issue-dp0-running12-ascend-output-20261010.tar.gz)
- [SHA256SUMS](https://github.com/linfeng-yuan/vllm-ascend/releases/download/dsv41-a5-whl060-issue-dp0-running12-20261010/SHA256SUMS)
- SHA256：`db086f9d297a64632b4d4691481c5a8150bb8b18a8a4d0722df4b96b2cb3d6a6`

本条仅用于记录问题，不作为“第五次”性能刷新。压缩包只包含 D0/DP0/卡 0 已解析的 `ASCEND_PROFILER_OUTPUT/`，不包含其他卡、原始 `PROF_*`、`FRAMEWORK/`、服务日志或源码。

| 项目 | 配置 / 结果 |
| --- | --- |
| 代码 | `linfeng-yuan/vllm-ascend:codex/dsv41-a5-2200tps-r4-1010`，运行时代码截止 `5797a877`；未带后续 NZ 试验 |
| Wheels | `cannbotdsl 0.6.0+g91e7c8f`，SHA256 `d77d5e7f…2c7cb7`；新版 `cannbot_arena_net_ops 0.1.0`，SHA256 `a7dd63fe…7c016` |
| 服务 | P：两组 node-local DP8/EP8；D：跨 4 节点 external DP32/EP32/TP1；Proxy workers=4 |
| D 关键配置 | MRV2、LMHead TP8、`max_num_seqs=16`、`max_num_batched_tokens=1024`、DSpark 5、synthetic 4.15、force EPLB、recompute scheduler |
| 图配置 | NPUGraphEx 开启；Static Kernel / Super Kernel 关闭；capture sizes `[72, 96]` |
| 无 profiling 第二轮 | 384 成功、0 失败，全部输出 4096；TPOT 6.0 ms、P90 6.1 ms；AISBench output throughput 49,381.0098 tokens/s |
| 无 profiling 的 D 打屏 | 32 个 D 都实测到 running=12；峰值 1970.3～2082.4 tokens/s、均值 2024.5313；同代码旧 wheels 基线均值约 2228.44 |
| Profiling 本轮性能 | 384 成功、0 失败，全部输出 4096；TPOT 6.6 ms、P90 6.8 ms；AISBench output throughput 45,858.6568 tokens/s |
| Profiling 轮 D 打屏 | 32 个 D 峰值均对应 running=12；2081.1～2087.4 tokens/s，均值 2085.375 |
| 采集 | D0/DP0/卡 0；触发时 running=12、waiting=0；profiler 启停请求间隔 2.004 秒 |
| 解析 | 156,156 条 kernel；83 个 step 标记、82 个相邻 step 间隔；P50 23.963 ms、P90 24.028 ms；5 个 >=30 ms profiling 长尾 |
| 对比 | 旧 wheels 的第四次 URMA profile step P50 为 22.706 ms；本轮增加 1.257 ms（+5.54%） |
| 恢复旧 wheels A/B 第一轮 | 384/384 成功；AISBench 52,105.2357 tokens/s；TPOT 5.7 ms、P90 5.9 ms；ITL 平均 23.5 ms、中位 22.4 ms；D running=12 均值 2219.3 tokens/s |
| 恢复旧 wheels A/B 第二轮 | 384/384 成功；AISBench 51,882.1490 tokens/s；TPOT 5.7 ms、P90 5.8 ms；ITL 平均 23.5 ms、中位 22.5 ms；D running=12 均值 2196.8594 tokens/s |

逐 kernel 对比显示，`EngramUrmaGather` 本轮约 582.99 us/次，未比旧轮约 588.58 us/次变慢。主要差异集中在新版 wheel 启用的 `mixed_quant_sparse_flash_mla`、`QuantLightningIndexer`、`QuantSparseLightningIndexer` 路径；其额外单步开销与约 1.26 ms 的 step P50 增量基本吻合。同机保持代码、脚本和配置不变，六个容器恢复旧 wheels 后连续两轮 D 打屏均值恢复到 2219.3 / 2196.9 tokens/s，强支持性能回退来自新版 wheel 运行路径，而非当前服务配置。
