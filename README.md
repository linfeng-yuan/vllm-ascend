# DeepSeek-V4-Pro-0813 PD 性能测试

最新完成 **2P1D，48 卡，AISBench 24 并发 / 256 请求**。D `max-num-seqs=16`，每 DP Running 峰值均为 12，KV 峰值为 **89.63%、89.59%**，没有抢占或返 P 重算。平均 TTFT **3306.2ms**、TPOT **12.6ms**，整体输出 **1443.3066 tokens/s**。配置见 [2P1D 工作点](profiles/2p1d-pcp8-128k1k-p90-kv95.json)，完整对比见本页最后一节。

2026-10-25，Ascend 950DT。保存 4K 输入 / 2K 输出的两种 D 并行配置复测，以及 128K 输入 / 1K 输出的 3P1D prefix 测试结果和启动条件。

两组使用相同 r35 镜像、原镜像内 vLLM / vLLM-Ascend 代码和算子、同一模型权重，没有叠加 MC2 容量修改。完整环境和参数见 [测试条件](CONDITIONS.md)，逐个进程的实际参数见 [启动记录](launch-records/)，按节点整理的命令见 [启动脚本](launch/)。

## 全程结果

TPOT、TTFT、QPS 和 token 吞吐均取 AISBench 全程汇总。每卡吞吐的分母为实际使用的 P+D 总卡数。端到端 token 吞吐为输入、输出 token 吞吐之和。

| 指标 | 2P1D：D DP32/EP32 | 1P1D：D DP2/PCP8/EP16 |
|---|---:|---:|
| P 并行配置，每个实例 | PP2/PCP8/EP8 | PP2/PCP8/EP8 |
| P 卡数 / D 卡数 / 总卡数 | 32 / 32 / 64 | 16 / 16 / 32 |
| 最大并发，客户端及 Proxy 实测均达到 | 960 | 128 |
| AISBench 平均并发 | 865.5199 | 111.6396 |
| 总请求数 | 5760 | 768 |
| 成功 / 失败 | 5760 / 0 | 768 / 0 |
| TTFT Avg，ms | 41082.5 | 2873.9 |
| TPOT Avg，ms | 10.9 | 15.3 |
| 单卡输入吞吐，tokens/s/卡 | 875.0681 | 418.2880 |
| 单卡输出吞吐，tokens/s/卡 | 437.5341 | 209.1440 |
| 单卡端到端吞吐，tokens/s/卡 | 1312.6022 | 627.4319 |
| QPS，整组 req/s | 13.6729 | 3.2679 |
| 整组输入吞吐，tokens/s | 56004.3589 | 13385.2149 |
| 整组输出吞吐，tokens/s | 28002.1795 | 6692.6074 |
| 整组端到端吞吐，tokens/s | 84006.5384 | 20077.8223 |
| 全程时长，s | 421.2701 | 235.0151 |

原始结果：[2P1D JSON](results/dp32-4k2k-c960-r3/result-summary.json)、[1P1D JSON](results/1p1d-pcp8-4k2k-c128-r2/result-summary.json)。目录中同时保存 AISBench 原始 JSON、CSV、数据集摘要、metrics 前后快照和 DP 监控汇总。

固定 4096/2048 长度时，整体输入吞吐等于 QPS×4096，输出吞吐等于 QPS×2048，因此输入吞吐随整组请求完成速度变化。它不是 P 独立满载时的预填充算力。两组 P 卡数、总卡数和并发不同，不能当作仅更换 D 并行方式的等资源对照。

## 稳态结果

使用镜像自带 AISBench `StablePerfMetricCalculator`，在独立分析进程将 `WAVE_OFFSET` 改为 0.1，通过 `--reuse` 重算原始打点，不重新发请求。全程结果保留。

| 指标 | 2P1D：D DP32/EP32 | 1P1D：D DP2/PCP8/EP16 |
|---|---:|---:|
| 总卡数 | 64 | 32 |
| 最大并发 | 960 | 128 |
| 稳态平均并发 | 938.4729 | 128 |
| 本窗口选中请求数 | 4800 | 640 |
| TTFT Avg，ms | 43128.0 | 1603.1 |
| TPOT Avg，ms | 11.0 | 15.4 |
| 单卡输入吞吐，tokens/s/卡 | 915.4965 | 495.9211 |
| 单卡输出吞吐，tokens/s/卡 | 457.7482 | 247.9605 |
| 单卡端到端吞吐，tokens/s/卡 | 1373.2447 | 743.8816 |
| QPS，整组 req/s | 14.3046 | 3.8744 |
| 整组输出吞吐，tokens/s | 29295.8876 | 7934.7371 |
| 稳态窗口，s | 335.5556 | 165.1876 |
| 同窗口 D 平均 Running 合计 / 每 DP | 325.66 / 10.18 | 122.15 / 61.08 |
| 全程 D 同时峰值 Running | 345 | 128 |

这两个稳态窗口均未满足 AISBench 文档要求的“窗口时长大于最大请求 E2EL 三倍”条件，存在窗口边界统计误差。稳态按客户端并发选取，不代表 D 必然满载。

稳态结果：[2P1D](results/dp32-4k2k-c960-r3/stable-wave01/result-summary.json)、[1P1D](results/1p1d-pcp8-4k2k-c128-r2/stable-wave01/result-summary.json)。对应的 `steady-window.json` 保存窗口和选中请求，`steady-monitor.json` 保存同窗口 Running。

DP32 组观察到 P 侧较长等待队列，D 稳态平均每 DP Running 为 10.18。PCP8 组的两个 D DP 稳态平均 Running 为 61.08/64，总计 122.15/128；客户端并发按测试约定封顶 128，没有继续增加并发去凑 50ms。

## 文件

- [CONDITIONS.md](CONDITIONS.md)：镜像、版本、拓扑、P/D 参数、HCCL buffer 和测试方法。
- `launch-records/`：原始启动参数与记录的环境变量，保留原标签。
- `launch/2p1d/`、`launch/1p1d/`：依据原始记录逐台整理的启动命令；Proxy 单独启动。
- `results/`：两组复测的全程与稳态结果、计数核验、DP Running 和数据集 SHA256。
- `tools/`：测试数据生成、AISBench 配置生成、监控、结果解析和稳态重算脚本；其中服务器绝对路径保留为当时值。

## 128K/1K：3P1D，64 并发 / 256 请求

本组 P 为三个 PP2/PCP8/EP8 实例，共 48 卡，prefix caching 开启；D 为 DP2/PCP8/EP16，共 16 卡，prefix caching 关闭。使用原镜像代码，D `max-num-seqs=8`、`HCCL_BUFFSIZE=272`，启动打印逻辑 KV 容量合计 2338177 tokens。全部单卡吞吐除以 **64 卡**。

下表为两次已完成的正式测试，TTFT、TPOT 和吞吐取 AISBench 全程统计。按用户确认，保留固定的文本前缀比例，并报告缓存块对齐后的实际命中率。

| 指标 | 90% 文本前缀 | 96% 文本前缀 |
|---|---:|---:|
| P 实测缓存命中率 | 87.5007% | 93.7507% |
| 最大并发，客户端及 Proxy 实测 | 64 | 64 |
| AISBench 平均并发 | 57.2306 | 57.1938 |
| 总请求数 | 256 | 256 |
| 成功 / 失败 | 256 / 0 | 256 / 0 |
| 实际输入 / 输出 tokens，每条 | 131072 / 1024 | 131072 / 1024 |
| TTFT Avg，ms | 28795.0 | 28064.7 |
| TPOT Avg，ms | 9.8 | 9.6 |
| 单卡输入吞吐，tokens/s/卡 | 3018.5060 | 3092.9874 |
| 单卡输出吞吐，tokens/s/卡 | 23.5821 | 24.1640 |
| 单卡端到端吞吐，tokens/s/卡 | 3042.0881 | 3117.1514 |
| QPS，整组 req/s | 1.4739 | 1.5102 |
| 整组输出吞吐，tokens/s | 1509.2530 | 1546.4937 |
| D0 / D1 同窗口平均 Running | 7.864 / 7.880 | 7.851 / 7.884 |
| D 同时峰值 Running，合计 | 16 | 16 |

两组逐条输入、输出长度以及 D 成功计数、生成 token 增量全部吻合。缓存命中率取正式测试前后的 P metrics 增量，预热流量已排除。输入 token 吞吐包含命中缓存的输入 tokens。

Running 使用 AISBench 稳态窗口内的同窗采样；上述延迟和 token 吞吐使用全程统计。两个稳态窗口分别为 125.59s、120.34s，均未达到最大 E2EL 三倍的时长判据，因此以全程 AISBench 数值作为性能主结果。

结果与证据：[90% 组](results/3p1d-pcp8-stock-128k1k-p90-c64-n256-r1/result-summary.json)、[96% 组](results/3p1d-pcp8-stock-128k1k-p96-c64-n256-r1/result-summary.json)、[90% 实际命中增量](results/3p1d-pcp8-stock-128k1k-p90-c64-n256-r1/observed-prefix.json)、[96% 实际命中增量](results/3p1d-pcp8-stock-128k1k-p96-c64-n256-r1/observed-prefix.json)、[测试后健康检查](launch-records/3p1d/after-c64-n256-health.json)。环境、数据和缓存处理见 [128K/1K 测试条件](CONDITIONS.md#128k1k3p1d64-并发--256-请求)。

此前的 16 并发 / 96 请求作为预跑单独保存在 `results/pilots/`。上述两组在 D `max-num-seqs=8` 时完成。

### D max-num-seqs=32，90% 文本前缀复测

按用户要求只重启两台 D，设置 `max-num-seqs=32`、`HCCL_BUFFSIZE=1088`（双 buffer 合计 2176MB）。P、模型代码和其他配置保持不变；AISBench 仍为 64 并发、256 请求、131072/1024 tokens。该档仅测试 90% 文本前缀。

| 指标 | 32 档实测 |
|---|---:|
| P 实测缓存命中率 | 87.5007% |
| 最大并发 / AISBench 平均并发 | 64 / 58.5489 |
| 总请求数 / 成功 / 失败 | 256 / 256 / 0 |
| TTFT Avg，ms | 17447.4 |
| TPOT Avg，ms | 11.0 |
| TPOT P50 / P90，ms | 10.2 / 13.7 |
| 单卡输入吞吐，tokens/s/卡 | 4174.6044 |
| 单卡输出吞吐，tokens/s/卡 | 32.6141 |
| 单卡端到端吞吐，tokens/s/卡 | 4207.2184 |
| QPS，整组 req/s | 2.0384 |
| 整组输出吞吐，tokens/s | 2087.3022 |
| D0 / D1 同窗口平均 Running | 12.422 / 12.278 |
| D0 / D1 Running 峰值 | 13 / 13 |
| D 同时峰值 Running，合计 | 26 |
| D0 / D1 成功请求数 | 128 / 128 |
| D 启动打印逻辑 KV tokens，合计 | 2018297 |

单卡指标仍除以 P+D 共 64 卡。输入、输出长度和 D 侧成功、生成 token 计数全部核验通过。整体输出吞吐比前面的 8 档 90% 组提高约 38.3%，但实际 Running 没有达到每 DP 32。

测试窗口内，Running=13 的 D 日志显示 KV 使用率约 97.8%～98.1%，同时仍有最多 19 条请求等待，表明当前 KV 容量限制了继续增加活跃请求。保留 [KV 压力证据](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c64-n256-r1/kv-pressure-summary.json) 和该目录下的 `decode-window-logs/` 原始日志。

为避免完整输入后缀被上一轮缓存，本轮 `request_index_start=2000`，前一轮为 1000。共同前缀、输入输出长度和词表分布保持一致，但不是逐条完全相同输入的消融实验。该轮稳态窗口为 89.38s，未达到最大 E2EL 三倍判据；性能表使用全程 AISBench 结果，Running 列使用同窗口采样。

原始结果：[32 档完整汇总](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c64-n256-r1/result-summary.json)、[实际命中率](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c64-n256-r1/observed-prefix.json)、[同窗口 Running](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c64-n256-r1/stable-wave01/steady-monitor.json)、[启动记录与健康检查](launch-records/3p1d-s32/)。

### 按 KV 低于 95% 调整客户端并发

服务配置保持 D `max-num-seqs=32`、`HCCL_BUFFSIZE=1088`，仅调整客户端并发。两组仍为 256 请求、131072/1024 tokens、90% 固定文本前缀，实测命中率均为 87.5007%。所有吞吐取 AISBench 全程值，单卡分母为 P+D 共 64 卡。

| 指标 | 20 并发 | 24 并发，当前采用 |
|---|---:|---:|
| 实测最大并发 / 平均并发 | 20 / 19.3924 | 24 / 22.9104 |
| 成功 / 失败 | 256 / 0 | 256 / 0 |
| D0 / D1 KV 采样峰值 | 75.5023% / 75.5193% | 90.6067% / 90.6164% |
| D0 / D1 Running 峰值 | 10 / 10 | 12 / 12 |
| D0 / D1 稳态平均 Running | 8.146 / 8.124 | 9.915 / 9.897 |
| TTFT Avg，ms | 2802.0 | 2863.2 |
| TPOT Avg，ms | 10.5 | 11.3 |
| 单卡输入吞吐，tokens/s/卡 | 2935.7236 | 3258.5208 |
| 单卡输出吞吐，tokens/s/卡 | 22.9353 | 25.4572 |
| 单卡端到端吞吐，tokens/s/卡 | 2958.6590 | 3283.9780 |
| QPS，整组 req/s | 1.4335 | 1.5911 |
| 整组输出吞吐，tokens/s | 1467.8618 | 1629.2604 |
| D 抢占计数增量，两个 DP | 0 / 0 | 0 / 0 |
| 每请求抢占次数累计增量，两个 DP | 0 / 0 | 0 / 0 |
| 返 P 重算日志匹配数 | 0 | 0 |

每 0.5 秒采集 KV 使用率、抢占计数及请求统计。KV 达到 95% 或检测到抢占时，停止本轮 AISBench 客户端进程组；这两组均未触发保护，且逐条 token 数与 D 侧计数一致。24 并发的实测 KV 峰值低于 95%，因此选用该档。两个稳态窗口均通过最大 E2EL 三倍的时长判据。

上一轮 64 并发的高 KV 占用不等于已经发生重算：本次 20 并发测试开始前，两台 D 自启动以来的 `num_preemptions_total` 和 `request_num_preemptions_sum` 均为 0，上一轮日志也没有返 P 重算消息。这里通过降低并发给 KV 留出余量。

两台 D 的启动参数均为 `--no-enable-prefix-caching`，运行时 `cache_config_info` 中 `enable_prefix_caching=False`，本地 prefix queries/hits 均为 0。`External prefix cache` 统计从 P 获取的 KV，和 D 本地 prefix caching 分开。见 [运行时核对](launch-records/3p1d-s32/3p1d-r18-prefix-caching-check.json)。

结果与验收证据：[20 并发结果](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c20-n256-kv95-r1/result-summary.json)、[24 并发结果](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c24-n256-kv95-r1/result-summary.json)、[24 并发 KV / 抢占 / 重算核验](results/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c24-n256-kv95-r1/kv-preemption-validation.json)、[测试后健康检查](launch-records/3p1d-s32/3p1d-after-kv95-health.json)。后缀 index 起点分别为 3000、4000，用于避免先前完整输入的缓存影响；共同前缀和输入输出长度保持一致。

## 128K/1K：2P1D，24 并发 / 256 请求

P 保留 21/22、25/26 两组，D 使用 23/24。27/28 的本实验 P 容器已停止，NPU 无运行进程。本组共 **48 卡**：32P + 16D。P prefix caching 开启，D 本地 prefix caching 关闭，Proxy workers=4。D 按要求设置 `max-num-seqs=16`，HCCL_BUFFSIZE 仍为 1088，其他参数沿用原镜像配置。

两列均为 131072 输入 / 1024 输出、90% 固定文本前缀，实际缓存命中率 87.5007%。延迟和吞吐以 AISBench 全程值为准；每卡吞吐除以各列实际使用的全部 P+D 卡数。

| 指标 | 前一轮 3P1D | 本轮 2P1D |
|---|---:|---:|
| 总卡数 | 64 | 48 |
| D max-num-seqs | 32 | 16 |
| 最大并发 / 平均并发 | 24 / 22.9104 | 24 / 22.8802 |
| 总请求数 / 成功 / 失败 | 256 / 256 / 0 | 256 / 256 / 0 |
| TTFT Avg，ms | 2863.2 | 3306.2 |
| TPOT Avg，ms | 11.3 | 12.6 |
| 单卡输入吞吐，tokens/s/卡 | 3258.5208 | 3848.8177 |
| 单卡输出吞吐，tokens/s/卡 | 25.4572 | 30.0689 |
| 单卡端到端吞吐，tokens/s/卡 | 3283.9780 | 3878.8866 |
| QPS，整组 req/s | 1.5911 | 1.4095 |
| 整组输出吞吐，tokens/s | 1629.2604 | 1443.3066 |
| D0 / D1 稳态平均 Running | 9.915 / 9.897 | 9.793 / 9.967 |
| D0 / D1 Running 峰值 | 12 / 12 | 12 / 12 |
| D0 / D1 KV 采样峰值 | 90.6067% / 90.6164% | 89.6277% / 89.5896% |
| D 抢占 / 返 P 重算 | 0 / 0 | 0 / 0 |

两个 P 能达到 D 每 DP Running=12，稳态平均 Running 合计 19.760，和前一轮的 19.813 接近。这里的 24 是峰值，不是全程维持值。本轮整体输出下降 11.4%，使用卡数减少 25%，按全部 P+D 卡数计算的单卡输出提高 18.1%。D 的 max-num-seqs 同时从 32 改为 16，因此不能把全部吞吐差异归因于少一组 P。

P 的正式窗口 metrics 记录如下，处理速率只计未命中的 query tokens，已排除共同前缀预热；它不是 P 独立饱和上限。

| P 实例 | 完成请求 | 未缓存输入 tokens/s | P 平均处理时间，ms | 其中排队，ms | 同窗口平均 Running / Waiting |
|---|---:|---:|---:|---:|---:|
| 21/22 | 130 | 11726.15 | 2199.3 | 443.1 | 0.839 / 0.211 |
| 25/26 | 126 | 11365.35 | 2128.6 | 388.5 | 0.789 / 0.155 |

两组 P 合计处理未缓存输入 23091.50 tokens/s；前一轮三个 P 合计为 26066.58 tokens/s。AISBench 的输入吞吐包含缓存命中的 tokens，和这一口径分开。P 平均处理时间取 P 自身 TTFT metrics，包含其队列时间，不等于客户端端到端 TTFT。

本轮 AISBench 稳态窗口 152.09 秒，满足最大 E2EL 三倍时长判据；该窗口 TTFT 为 2977.0ms、TPOT 为 12.3ms、整体输出为 1561.9999 tokens/s。全程主结果仍使用上表。D 启动打印 KV 容量合计 **2039522 tokens**，两 DP 分别为 1019998、1019524。

原始结果：[AISBench 全程结果](results/2p1d-pcp8-stock-s16-h1088-128k1k-p90-c24-n256-kv95-r1/result-summary.json)、[同窗口 D Running](results/2p1d-pcp8-stock-s16-h1088-128k1k-p90-c24-n256-kv95-r1/stable-wave01/steady-monitor.json)、[P 吞吐与排队](results/2p1d-pcp8-stock-s16-h1088-128k1k-p90-c24-n256-kv95-r1/prefill-throughput.json)、[KV / Running / 抢占验收](results/2p1d-pcp8-stock-s16-h1088-128k1k-p90-c24-n256-kv95-r1/kv-preemption-validation.json)、[启动记录与健康检查](launch-records/2p1d-pcp8/)、[启动脚本](launch/2p1d-pcp8/)。
