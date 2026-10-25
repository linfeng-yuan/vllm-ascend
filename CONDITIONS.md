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
