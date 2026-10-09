# DSV4.1 A5 Profiling

Profiling 产物放在 GitHub Releases。本分支只保留本说明文件，不包含 vLLM / vLLM-Ascend 源码。

## 最新：主线 + LMHead TP8，D0/DP0 running=12（第二次，2026-10-09）

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
