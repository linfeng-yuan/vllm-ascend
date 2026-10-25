# DeepSeek-V4-Pro-0813 PD 性能测试

2026-10-25，Ascend 950DT，4K 输入 / 2K 输出，无 prefix 命中。保存两种 D 并行配置各自最后一次复测的结果和启动条件。

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

后续 3P1D、128K/1K、90%/97% prefix 测试的配置和结果另行登记，不计入本页两组基线。
