# 测试条件

## 硬件和软件

| 项目 | 实际配置 |
|---|---|
| 硬件 | Ascend 950DT，8 卡/台，96GB/卡 |
| 2P1D 用卡 | 8 台，P 32 卡 + D 32 卡，共 64 卡 |
| 1P1D 用卡 | 4 台，P 16 卡 + D 16 卡，共 32 卡 |
| 模型 | DeepSeek-V4-Pro-0813，共 66 个 safetensors 分片 |
| 模型容器路径 | `/models/DeepSeek-V4-Pro-0813`，只读挂载共享 NFS 权重 |
| 镜像 | `swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:nightly-ci-releases-v0.30.0-a5-cann9.2.0_20261022115200-20261024-r35` |
| 镜像 ID | `sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06` |
| 镜像 digest | `sha256:09d468055583572a70e5f8937f441ef98710b06196df5eb9b277eb31afb551ba` |
| vLLM | 0.30.0，commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend commit | `ef47170f3bc433a069173b58f14930bab8ef1c19` |
| Python | `/usr/local/python3.12.13/bin/python` |
| AISBench | `3.1.20260609`，使用镜像自带工具 |
| fastokens | 0.2.0，`VLLM_USE_FASTOKENS=1` |
| 代码与算子 | 原镜像代码和算子；这两组未叠加后续 MC2 容量补丁 |

参考：[releases/v0.30.0 的 DeepSeek-V4-Pro 950DT PD 文档](https://github.com/vllm-project/vllm-ascend/blob/releases/v0.30.0/docs/source/tutorials/models/DeepSeek-V4-Pro.md#523-950dt-products-pd-separation-deployment)，读取文档时的分支 commit 为 `96c116003be65d9a54380a363c5f3c3ae6deb503`。

## 机器和并行配置

| 配置 | P | D |
|---|---|---|
| 2P1D | 21/22 为 P0，23/24 为 P1；每个 P 为 PP2/PCP8/EP8 | 25～28，每机 8 个 external DP engine，合计 DP32/EP32/TP1 |
| 1P1D | 21/22 为 P0，PP2/PCP8/EP8 | 23、24 各一个 external DP engine，合计 DP2/PCP8/EP16/TP1 |

节点 21～28 的数据 IP 依次为 `197.33.45.1`～`197.33.52.1`，使用数据网卡 `enp39s0f0`。P 的 head API 为 18100，D32 各机为 18200～18207，D2 两机各为 18200，Proxy 为 `197.33.45.1:18300`。

`1D` 指一个 D 实例；该实例内部可以有 32 个或 2 个 DP engine。P 的 PP 层数分配为 `32,29`。

## 服务参数

| 参数 | P，每个实例 | D DP32/EP32 | D DP2/PCP8/EP16 |
|---|---|---|---|
| max-num-seqs | 16 | 32 | 64 |
| max-num-batched-tokens | 8192 | 512 | 1024 |
| max-model-len | 1048576 | 1048576 | 256000 |
| gpu-memory-utilization | 0.94 | 0.92 | 0.95 |
| prefix caching | 关闭 | 关闭 | 关闭 |
| DSpark speculative tokens | 5 | 5 | 5 |
| DSpark enforce_eager | true | false | false |
| eager / 图模式 | eager | FULL_DECODE_ONLY | FULL_DECODE_ONLY |
| enable_npugraph_ex | — | true | true |
| enable_static_kernel | — | false | false |
| combine_quant_mode | — | 4 | 4 |
| recompute_scheduler_enable | — | true | true |
| HCCL_BUFFSIZE 参数 | 1024 | 1088 | 2176 |
| 对应双 buffer 合计 MB | 2048 | 2176 | 4352 |
| HCCL_OP_EXPANSION_MODE | CCU_SCHED | 未显式设置，使用镜像默认 | CCU_SCHED |
| PCP o_proj / embedding / lmhead weight sharding | 未显式开启 | — | true |

KV 和 indexer KV 均用 FP8，block-size=32；启用 hybrid KV cache manager、CPU binding、shared expert DP 和 shared expert overlap。DSpark 使用实际接受结果，没有 synthetic acceptance 配置。图捕获档位由镜像自动生成，未手工指定列表。

通信使用 `MooncakeConnectorV2`，P 为 `kv_producer`、D 为 `kv_consumer`。Proxy 沿用镜像内 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`，workers=4。启动 Proxy 前检查全部 P/D `/health`。

共同环境另有 `OMP_NUM_THREADS=10`、`OMP_PROC_BIND=false`、`TASK_QUEUE_ENABLE=1`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、`ASCEND_LOCAL_COMM_RES={"version":"1.3"}`、`RAYON_NUM_THREADS=16`、`TOKENIZERS_PARALLELISM=true`。HCCL socket、Gloo、TP socket 均使用数据网卡。完整记录见 `launch-records/`。

2P1D 的 D 启动打印逻辑 KV token 容量合计为 **120458635**。1P1D 的两个 D DP 分别打印 **763456、763388**，合计 **1526844**。这些是启动打印的逻辑容量，没有再乘 PCP 卡数。

## 容器

八机使用同名实验容器 `ylf_pro0813_20261025`，镜像 ID 相同。容器使用 host 网络、privileged、shm=512GB、memlock unlimited、nofile=1048576，挂载对应 Ascend 设备、驱动和存在的通信配置文件。共享权重只读，实验目录读写挂载到 `/workspace/experiment`。保存的逐台 `docker run` 参数见 [容器创建记录](container-records/)。

只有本实验进程使用所测 NPU。每次切换前核对进程、端口和任务归属，停止本实验容器内服务，不清理其他容器或使用宿主机范围的 pkill。

## AISBench

| 项目 | DP32/EP32 | DP2/PCP8/EP16 |
|---|---:|---:|
| case ID | dp32-4k2k-c960-r3 | 1p1d-pcp8-4k2k-c128-r2 |
| 并发上限 | 960 | 128 |
| data_num | 5760 | 768 |
| 输入 / 输出 tokens | 4096 / 2048 | 4096 / 2048 |
| 输入 API | /v1/completions | /v1/completions |
| stream | true | true |
| temperature | 0 | 0 |
| ignore_eos / add_special_tokens | true / false | true / false |
| AISBench retry | 1，表示总共尝试一次 | 1，表示总共尝试一次 |

请求数为并发的 6 倍。数据使用同一 tokenizer 和固定种子的单 token 混合词表生成；两组相同请求索引使用相同文本，tokenizer round-trip 检查长度。数据集 SHA256、首条 question SHA256、词表 SHA256 和生成方法保存在各组 `dataset.json`；请求数不同，因此整份数据集 SHA256 不同。

每组正式测试前单独预热 CANN 扩展和各 P API，预热流量不计入测试；本轮 P/D prefix caching 均关闭。正式测试前后保存 metrics，每秒采集全部 D DP Running/Waiting 和 Proxy 实际在途请求，按 D 日志字节偏移截取测试日志。测试结束后继续采集约 12 秒。

逐条核验所有请求成功，实际输入 4096、输出 2048 tokens；核对 D request_success 和 generation_tokens 增量与客户端一致。两个复测均通过核验。

## 差异与限制

两组总资源和并发不同。D 的 max-num-seqs、batch token 上限、上下文长度、利用率、buffer 和 PCP 开关也不同，均按上表记录。它们是各自部署下的实测工作点，不是只改变 D 并行方式的严格消融实验。

最初的 3P1D 在 27/28 遇到通信初始化异常，未产生性能结果。按测试要求改为四机 1P1D，D 移到 23/24 后完成复测。PCP8 的 batch token 上限随后明确设为 1024，D 上下文设为 256000，并补上 CCU_SCHED；本页记录的是这些修改后的成功配置。

稳态结果通过原始打点重算，并单独保存原始文件 SHA256、重算命令、窗口和 DP 监控。两个窗口都没有达到最大 E2EL 三倍的时长判据，因此同时提供全程结果。逐请求均值 `OutputTokenThroughput` 不是整组输出吞吐，本页没有用它替代 AISBench `Output Token Throughput`。

## 128K/1K：3P1D，64 并发 / 256 请求

本节是另一组测试条件，与前面的 4K/2K 四机 1P1D 分开记录。使用同一镜像、权重和原镜像代码，合计 **64 卡**。

| 项目 | 配置 |
|---|---|
| P | 21/22、25/26、27/28，共三个 PP2/PCP8/EP8 实例，48 卡 |
| D | 23/24，external DP2/PCP8/EP16，一个 D 实例，16 卡 |
| P / D prefix caching | 开启 / 关闭 |
| D max-num-seqs | 8 |
| D max-num-batched-tokens | 1024 |
| D max-model-len | 256000 |
| D gpu-memory-utilization | 0.95 |
| D HCCL_BUFFSIZE | 272，双 buffer 合计 544MB |
| D DSpark | 5，enforce_eager=false，实际接受结果 |
| D 图模式 | FULL_DECODE_ONLY，npugraph_ex=true，static_kernel=false |
| D combine_quant_mode / recompute scheduler | 4 / true |
| D 启动打印 KV tokens | DP0=1169326，DP1=1168851，合计 2338177 |
| 客户端并发 / 请求数 | 64 / 256，按本组明确指定值覆盖先前的 6 倍请求数规则 |
| 实际输入 / 输出 | 每条 131072 / 1024 tokens |
| 单卡吞吐分母 | 全部 P+D 共 64 卡 |

P 的 max-num-seqs=16、max-num-batched-tokens=8192、max-model-len=1048576、GPU 利用率 0.94、eager 保持不变。Proxy workers=4，登记三个 P API 和两个 D API。逐台命令见 [3P1D 启动脚本](launch/3p1d/)，实际参数与源码核对见 [3P1D 启动记录](launch-records/3p1d/)。

按用户确认，每条请求使用固定比例的文本前缀，并报告实际缓存命中率：

| 文本前缀目标 | 生成的共同前缀 tokens | 实际可命中的前缀 tokens/请求 | 预期实际命中率 |
|---|---:|---:|---:|
| 90% | 117952 | 114688 | 约 87.5% |
| 96% | 125824 | 122880 | 约 93.75% |

命中长度按模型缓存的 4096-token 边界向下对齐。实际比例使用 P metrics 的 `Δprefix_cache_hits_total / Δprefix_cache_queries_total`；queries 计数按该版本服务口径，每请求为 131071，因此小数结果略高于上述百分比。正式结果以各组 `observed-prefix.json` 为准。

每组先直连三个 P API，预热相应的共同前缀，max_tokens=1，预热不计入正式计数。镜像默认未注册开发模式的 `/reset_prefix_cache` HTTP 接口。为避免前一轮完整请求的后缀缓存影响本轮，正式 64/256 数据使用 `request_index_start=1000`，后缀种子为 `1000003+request_index_start+request_index`；共同前缀种子仍为 1。测试工具记录 P 进程标签，并检查相同完整输入没有在这些 P 进程中被正式测试过。96% 和 90% 的完整输入因前缀长度不同而不同，各自的数据集 SHA256、首条 question SHA256 和实际命中增量均归档。

输入 token 吞吐包含命中缓存的输入 tokens。仍按 AISBench 整体 Input/Output/Total Token Throughput 除以 64 计算三项单卡吞吐，不能把输入 token 吞吐直接当作 P 的独立计算吞吐。

16 并发 / 96 请求的两次运行作为预跑保存于 `results/pilots/`。正式对照使用 64 并发 / 256 请求。

## 128K/1K：D max-num-seqs=32 复测

拓扑仍为上述 3P1D、64 卡，三个 P 保持运行和 prefix caching 开启。只重启 23/24 的两个 D API，改动如下：

| 项目 | 前一轮 | 本轮 |
|---|---:|---:|
| D max-num-seqs | 8 | 32 |
| D HCCL_BUFFSIZE | 272 | 1088 |
| 双 buffer 合计 MB | 544 | 2176 |
| D0 启动打印 KV tokens | 1169326 | 1009352 |
| D1 启动打印 KV tokens | 1168851 | 1008945 |
| 请求后缀 index 起点 | 1000 | 2000 |

其他服务参数不变，继续使用原镜像源码和算子。客户端固定 64 并发、256 请求，逐条输入/输出为 131072/1024 tokens，90% 固定文本前缀对应实际 87.5007% 缓存命中。本轮未继续测试 96%。

重启前核对本实验容器归属与在途请求，向上一轮 D API 发送 SIGTERM 后再停止本实验 D 容器；不操作其他容器、驱动或网络。启动后确认五个后端健康、两台 D 图捕获完成、Proxy 健康和 PD 请求预热通过。逐进程命令、源码核对和健康检查保存在 `launch-records/3p1d-s32/`。
