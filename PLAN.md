# RTX 4090 单卡推理优化实验室

日期：2026-10-28。状态：项目设计；尚未下载模型、部署服务或测得性能提升。

## 目标与当前环境

用户选择 Windows + WSL2，重点是多人使用时保证响应速度、提高总吞吐。
本机只读检查确认 RTX 4090、24564 MiB 显存、Windows 驱动 610.88，以及 Ubuntu WSL2。
Ubuntu 与 docker-desktop 在检查时均停止；尚未验证 WSL CUDA、容器 GPU、主机内存及磁盘容量。

优化目标：在固定模型质量、工作负载与延迟 SLO 下，最大化 goodput，即每秒成功且达到延迟要求的请求数。
同时报告输出 tokens/s、SLO 达标率、错误率、GPU 能耗。GPU 利用率是诊断指标，不能单独代表效率。

第一版服务定位为中文通用对话 API；模型候选 Qwen/Qwen3-8B，固定 non-thinking 模式。
这是便于研究标准 dense Transformer 的实验模型选择，并非对所有业务最优或最新的声明。
先建立 BF16 基准，再在相同源模型上比较 AWQ/GPTQ INT4 或 FP8 的受支持实现。
模型仓库标注 Apache-2.0；商用交付保留实际下载版本的 LICENSE、NOTICE（如有）、模型 revision 和量化来源记录。

## 系统设计

```text
外部客户端 → TLS / API Key / 配额与长度限制 / 有界队列 → vLLM → RTX 4090
压测客户端 ──────────────────────────────────────────┘

Windows GPU exporter ───┐
vLLM /metrics ──────────┼→ Prometheus → Grafana
网关指标 ──────────────┘
压测逐请求记录 → JSONL + 实验配置 → 离线比较报告 / Grafana 注释
```

- 推理、Prometheus、Grafana 在 WSL2 中运行；优先使用可固定版本的容器或独立环境。
- Windows exporter 通过 NVML/nvidia-smi 采集支持的硬件指标。WSL2 的 NVML 查询不完整，不假定 DCGM 全部可用。
- 每个指标记录来源和采样时间；不支持的指标显示 N/A，失联显示 stale，不填 0。
- 硬件采样建议 1 秒，Prometheus 抓取建议 5 秒；短算子耗时用 CUDA events/profiler，而非采样曲线判断。
- 项目源码保存在本仓库；模型缓存、编译缓存和大量压测数据优先放 WSL 的 Linux 文件系统。
- 使用 Windows 主机 NVIDIA 驱动；不在 WSL 内另装 Linux 显示驱动。
- 对外服务提供流式接口、超时、客户端取消传播、请求体/输入输出 token 上限、过载拒绝、健康检查及重启恢复。
- 管理与指标端口限制在本地/私网；外部只开放服务入口。公网入口方式需依据实际网络决定。
- 不将用户 prompt、API Key 作为 Prometheus label；默认日志只记录性能元数据。

## 理论起点：先解释瓶颈

Roofline 模型：P_attainable <= min(P_compute_peak, BW_memory × arithmetic_intensity)。
arithmetic_intensity = FLOPs / 实际搬运字节数；必须匹配所用精度和存储层级。

Prefill 往往有较高并行度，decode 小 batch 往往受权重/KV 读取及启动开销限制；长上下文和大 batch 会改变瓶颈。
判断依据是 profiling，不预先断言所有推理都受带宽限制。

忽略激活和 KV 等流量时，dense 层 decode 每步约 2PB FLOPs、读取约 Ps 字节权重，算术强度约 2B/s。
P 是参数数、B 是同时 decode 的序列数、s 是每参数字节数。
这解释 batching 为何能摊薄权重读取；真实模型还包括 attention、KV、同步和调度成本，不能直接据此承诺 tokens/s。

Qwen3-8B 配置：36 层，8 个 KV heads，head_dim=128。
BF16 KV 每 token 字节数 = 2(K和V) × 36 × 8 × 128 × 2 = 147456 bytes = 144 KiB。
一条序列的总缓存长度达到 8192 tokens 时，未共享的 KV 约 1.125 GiB；4 条约 4.5 GiB。
总长度包含输入和已生成输出。8B BF16 权重粗估约 16 GB（14.9 GiB），还须预留激活、CUDA Graph、workspace、碎片及 Windows 桌面占用。
INT4 原始权重粗估约 4 GB，不含量化元数据及未量化层，也不等于服务总显存。

## 优化实验与依据

| 实验 | 理论与假设 | 操作与验证 | 可能代价 |
|---|---|---|---|
| 连续批处理 | iteration-level scheduling 及时补入新请求；权重读取在序列间复用 | 扫描并发和到达率；观察吞吐、排队、p95/p99 TTFT | 批次变大可能使单用户生成变慢 |
| Paged KV 管理 | OS 分页思想减少预留浪费、碎片与重复缓存 | 观察 KV 占用、抢占/重算、可承载活动 token 数 | 不改变每个 KV 元素的物理大小 |
| Chunked prefill | 将长 prefill 切片，降低对 decode 的阻塞 | 混合长短请求，扫描 token 调度预算 | 小块可能恶化长请求 TTFT |
| Prefix caching | 相同 token 前缀复用已有 KV，省掉重复 prefill | 重复前缀与无共享前缀两组分别测试，区分冷/热缓存 | 无共享时收益有限；占用缓存空间 |
| 权重量化 | 减少权重流量与显存，为 KV 和更大 batch 留空间 | BF16 对比受支持的 INT4/FP8 kernel，做质量回归 | 反量化开销、kernel 差异、精度损失 |
| KV 量化 | 减少随上下文增长的 KV 存储与流量 | 引擎支持时比较 BF16 KV 和 FP8 KV，重点测长上下文 | scale、kernel 支持和长文本质量需验证 |
| CUDA Graph | 减少 CPU kernel launch 开销 | 在其他配置一致时比较 eager/graph，检查 CPU/GPU 时间线 | 图捕获耗时、额外显存、shape 限制 |
| FlashAttention | tiling + online softmax 减少 attention 中间矩阵的显存读写 | 比较 reference 与支持的 fused attention；做正确性检查 | 不把 exact attention 计算复杂度降为线性 |
| Triton 融合算子 | 合并操作，减少中间张量读写和 kernel launch | 选择 profiling 热点，如 residual+RMSNorm、SwiGLU | 小算子提升不一定带来端到端提升 |
| 投机解码（后期可选） | 草稿生成 + 批量验证，利用低 batch 下的计算余量 | 测接受长度、草稿开销、低/高并发 goodput | 同卡草稿占显存，高并发可能退化 |

vLLM 已集成许多优化且可能默认启用。记录最终生效配置与后端，不把默认功能重复算作新增收益。
跨引擎比较只能说明系统差异，不能单独证明某个机制的收益。
对于无法在生产引擎中独立关闭的机制，用教学实现或专门 microbenchmark 做机制验证。
第一轮不以 CPU offload、单卡多模型副本、多卡并行或超频为主线。

## 实验协议

初始短对话 SLO 假设：输入约 512 tokens、输出约 128 tokens；p95 TTFT <= 1 秒，p95 请求级 TPOT <= 50 ms。
这是待基准校准的目标，不是性能承诺。长文本请求单独设定目标，不能混为同一延迟分布。
TTFT 从客户端发送请求到首个内容 token；TPOT=(最后 token 时间-首 token 时间)/(输出 token 数-1)，输出不足 2 tokens 时不计算。
流式 chunk 可能含多个 tokens；chunk 间隔不能直接当作 token 间隔。优先使用 tokenizer 计数和引擎 token 指标。
完整输出吞吐 = 测量窗口内输出 tokens / 窗口时长；同时单独报告输入 tokens/s。
goodput = 同时满足成功、TTFT、请求级 TPOT 约束的完成请求数 / 测量时长；拒绝、超时、错误均计入总体达标率的分母。

1. 固定模型 revision、量化来源、tokenizer/chat template、引擎/容器版本、驱动、采样参数、thinking 模式、随机种子和数据集 hash。
2. 区分启动/编译冷启动、服务预热和 prefix cache 冷/热条件。
3. 建立短对话、长输入短输出、混合长短、共享前缀、无共享前缀五组 workload；长度使用模型 tokenizer 校验。
4. 并发扫描从 1、2、4、8、16 起步；稳定后才尝试更大值。并发客户端数不等于活动 decode batch。
5. 同时做开放到达率压测（例如 Poisson arrivals），报告 offered load、完成率和超时，避免闭环压测掩盖排队问题。
6. 每个配置预热后至少三次重复；记录测量时长、样本量、波动，p99 样本不足时明确标记。
7. 每次只修改一个因素；在候选最佳组合上再验证交互效应。顺序交错或随机化，减少温度和系统背景负载影响。
8. 固定长度合成负载用于纯性能归因；真实 prompt 自然终止负载用于实际体验，两种结果分别呈现。
9. 量化校准集和评估集分开；在业务问答、结构化输出和长文本任务上与 BF16 对照，预先定义允许的质量差异。
10. 留存逐请求时间戳、错误和配置；profiler 单独运行，不能把重型 profiling 的性能当作正常服务结果。

## 页面与报告

第一版使用 Grafana，四组视图：

- 硬件：可用的 GPU busy、显存、功率、温度、时钟、限制/降频原因；注意主机指标也包含桌面等其他进程。
- 服务：输入/输出 tokens/s、goodput、TTFT/TPOT/E2E 分位数、请求完成率、排队、运行请求数、拒绝和超时。
- 引擎：KV 使用、抢占/重算、prefix 命中率及分母、引擎版本支持的 prefill/decode 时间。
- 实验：配置和模型版本、基准/候选时间段标记、吞吐—延迟曲线、并发—显存曲线、质量回归和能耗。

实际 DRAM GB/s、Tensor Core 活跃率、occupancy 等由可用的 Nsight 工具在专门 profiling 会话中获取；不假设基础 exporter 能提供。
GPU memory utilization 百分比通常不是显存容量使用率或实际带宽利用率，面板必须标明含义。
GPU busy=100% 不等于达到峰值 FLOPs。能耗通过功率随时间积分，再除以完成 token 数；GPU 能耗与整机能耗分开命名。
实验结果使用 JSONL/JSON 保存，run_id 用于关联配置和 Grafana 注释；避免把无限增长的实验 ID 放入长期高频指标标签。

## 与本地 CS336 衔接

- lectures/lecture_02.py：GPU、算术强度与 Roofline；用于预测 batching/量化何时可能有效。
- lectures/lecture_06.py：benchmark、profiling、Triton；用于热点定位与算子实验。
- lectures/lecture_10.py：inference；用于 prefill/decode、缓存和调度分析。
- cs336_a2/cs336_systems/flash_attention.py 与 flash_attention_triton.py：已有 attention 实现，先检查适用 shape、mask、GQA 支持及正确性，再作为教学实验基础。
- doc/2026-10-11-llm-inference-recap.md：接续 KV Cache、GQA、带宽的已有笔记。

生产服务使用 vLLM；教学代码独立验证原理。只有正确性、代表性 microbenchmark 和端到端验证均通过，才考虑接入自定义 kernel。
Amdahl 定律：若热点占总耗时 f、局部加速 s，整体加速上界约为 1/((1-f)+f/s)。
例如占比 5% 的算子加速两倍，端到端理论上只约 1.026 倍；据此选择有价值的热点。

## 里程碑与验收

1. 环境与最小闭环：验证 WSL CUDA/磁盘/RAM，固定依赖；下载固定 revision 模型；流式 API、真实监控、第一份压测记录可用。
2. 调度优化：完成并发、token budget、chunked prefill、prefix 实验；输出固定 SLO 的容量曲线。
3. 内存与量化：完成显存预算、量化质量回归；解释可承载请求数变化与实际吞吐差异。
4. CS336 算子实验：至少一个有理论推导、正确性检查、CUDA events 测量和 profiler 证据的实现。
5. 对外服务：接入 TLS/鉴权/有界队列/取消/恢复；进行至少一小时混合负载稳定性测试，明确单卡单机的可用性边界。

最终交付：可复现启动配置、Grafana 面板、压测工具、原始结果、质量回归和实验报告。
每个实验都回答：假设是什么、瓶颈证据是什么、修改了什么、收益与代价是什么、在哪些负载下失效。

## 依据

- 模型与许可证：https://huggingface.co/Qwen/Qwen3-8B
- 模型结构：https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- WSL CUDA 与监控限制：https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- vLLM 调优：https://docs.vllm.ai/en/stable/configuration/optimization/
- 量化硬件支持：https://docs.vllm.ai/en/stable/features/quantization/
- 服务指标：https://docs.vllm.ai/en/stable/usage/metrics/
- Roofline：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Orca：https://www.usenix.org/conference/osdi22/presentation/yu
- PagedAttention：https://arxiv.org/abs/2309.06180
- FlashAttention：https://arxiv.org/abs/2205.14135
- AWQ：https://arxiv.org/abs/2306.00978
- 投机解码：https://arxiv.org/abs/2302.01318
