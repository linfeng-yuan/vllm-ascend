# DSV4.1 A5 Profiling

Profiling 产物放在 GitHub Releases。本分支只保留本说明文件，不包含 vLLM / vLLM-Ascend 源码。

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
