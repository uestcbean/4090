# 从 CS336 到 MLE：逐步实验路线图

这是一份执行顺序，不是已经完成的实验清单。模型、指标和参考依据见 [PLAN](PLAN.md)。
所有目标数字均须在基准建立后确认；优化不要求必然提速，要求结论可信、能够复现。

## 完成一步的标准

每一步必须留下四项成果：能独立解释的知识、可检查的实现/配置、验收证据、一条或多条已推送的 Git commit。
实验报告需回答：预测是什么？什么证据支持瓶颈？改变了什么？收益和代价是什么？在哪些情况下不成立？

推荐单步提交顺序（不要求制造无意义提交）：

1. `docs(exp-XX): define hypothesis and acceptance criteria`：冻结假设、变量、工作负载和判定规则。
2. `feat/bench(exp-XX): implement ...`：完成代码与必要的正确性检查。
3. `results(exp-XX): report measurements and limitations`：结果、图表、结论、进度索引。

小实验可以合为一个有完整证据的提交。大型实验拆成上述三类，保留真实时间顺序；不倒填、不批量创建“已完成”空提交。
每个报告记录被测代码 commit SHA；结果提交记录在实验索引。结果提交本身无需包含自己的 SHA。
若测量时有未提交代码，记录 dirty 状态和 diff；正式结果尽量在干净、已提交的代码上运行。

## M0：建立科学实验习惯（步骤 00–01）

### 00 — 定义项目契约与证据链

- 学习：把“GPU 跑满”转成带质量和延迟约束的目标，区分 throughput 与 goodput。
- 动手：阅读 PLAN；用自己的话写服务场景、短/长输入分组、TTFT/TPOT 定义和初始 SLO；检查本仓库克隆、提交与推送链路。
- 产物：`docs/PROJECT_CHARTER.md`，填好第一份 `experiments/00-contract/report.md`。
- 验收：用一个假设例子说明为何“总吞吐升高，但 goodput 降低”；注明例子不是实测值。仓库状态和报告可对应。
- 提交：`docs(exp-00): define workload SLOs and experiment contract`。
- 面试自测：你的成功标准是什么？为什么 GPU busy 不是最终 KPI？

### 01 — 环境体检与显存预算

- 学习：driver / CUDA runtime / PyTorch 的关系；GB/GiB、权重、激活、workspace 与 KV Cache 的区别。
- 动手：验证 WSL 的 CUDA tensor 计算；记录 GPU、CPU、RAM、磁盘、WSL、驱动和软件版本；实现按模型 config 计算权重与 KV 预算的小脚本。
- 产物：`scripts/inspect_environment.py`、`scripts/memory_budget.py`、`experiments/01-environment/report.md`。
- 验收：给定模型 config，能手算与脚本交叉核对 KV/token；列出预留显存及尚未实测的假设。必要检查仅针对预算公式和设备可见性。
- 提交：`feat(exp-01): inspect WSL CUDA and estimate memory budget`。
- 面试自测：权重能放进 24GB，为什么服务仍会 OOM？

## M1：可调用、可观察、可测量（步骤 02–05）

### 02 — 跑通第一条真实流式请求

- 学习：tokenizer、chat template、prefill/decode、EOS、thinking 模式，以及模型 revision 与软件版本复现。
- 动手：固定模型 revision 和引擎版本；建立 BF16、有限上下文的启动配置；用流式客户端发出请求，保存首次与末次内容到达时间。
- 产物：`configs/serve/baseline.*`、`scripts/smoke_request.py`、模型清单与启动说明。
- 验收：一条请求正确结束；启动、停止、重启可重复；区分首次编译和预热后的请求；记录最终生效后端及默认优化。
- 提交：`feat(exp-02): serve pinned model with streaming smoke check`。
- 面试自测：为什么相同文字在不同模板下 token 数和性能可能不同？

### 03 — 建立质量基准

- 学习：性能约束必须建立在可接受输出质量之上；质量评估、校准集与测试集要分离。
- 动手：准备一组有明确评分标准、允许公开的小规模中文问答/抽取/JSON/长文定位样本；记录 BF16 的质量分数。固定模式和采样配置。
- 产物：`evals/fixtures/`、`evals/run_eval.py`、`evals/rubric.md`、BF16 结果摘要。
- 验收：自动评分与人工抽查一致；明确样本量和覆盖边界，不能把小型自建集称为全面能力评估；预先写量化允许退化阈值。
- 提交：`eval(exp-03): establish BF16 quality reference and rubric`。
- 面试自测：如何证明量化不是用输出质量换来表面提速？

### 04 — 第一版真实监控页面

- 学习：counter、gauge、histogram、rate、分位数和采样延迟；硬件 busy 与算力利用率的差别。
- 动手：Windows 采集支持的 GPU 指标，WSL 采集 vLLM 指标；接入 Prometheus/Grafana；加入源失联和 N/A 展示。
- 产物：`monitoring/` 中的 exporter、抓取配置、可导入 dashboard JSON 和一张真实截图。
- 验收：空闲→单请求→多请求时曲线有对应变化；停止采集器后显示失联而不是“GPU 空闲”；截图能追溯时间和配置。
- 提交：`feat(exp-04): add GPU and inference observability dashboard`。
- 面试自测：Prometheus 的 p95 和压测客户端的 p95 为什么可能不同？

### 05 — 可信压测与基准冻结

- 学习：TTFT、TPOT、E2E、吞吐、goodput；闭环并发与开放到达率；预热、样本量和尾延迟。
- 动手：实现或封装压测器，先测并发 1/2/4/8，再测到达率；覆盖短请求、长输入和混合负载；记录所有完成、拒绝、错误与超时。
- 产物：`benchmarks/`、固定 workload 文件及 hash、逐请求结果样例、`experiments/05-baseline/` 报告和吞吐—延迟曲线。
- 验收：用可控延时/错误的模拟服务验证计时、分母和超时统计；真实测量预热后至少三次重复；样本不足时不声称稳定 p99。
- 计数规则：流式 chunk 不等于 token；固定长度性能负载与自然结束的质量负载分开；定义负载发送窗口与排空窗口，防止边界请求漏算。
- 提交：`bench(exp-05): establish reproducible latency throughput baseline`。
- 面试自测：压测客户端自身成为瓶颈时，你怎么发现？为什么不只报告平均延迟？
- 里程碑：报告通过后给结果提交打 `baseline-v1` tag，并推送该 tag；之后不移动它。

## M2：用数据解释推理系统（步骤 06–10）

### 06 — 用 profiler 检验 Roofline 预测

- 学习：算术强度、带宽/计算/启动开销瓶颈、Amdahl 定律；对应 CS336 lecture 02、06。
- 动手：预测小 batch decode、较大 batch decode、长 prefill 的瓶颈；分别采集可用 profiler 时间线；必要时使用 CUDA events 做微基准。
- 产物：`experiments/06-profiling/report.md`，关键 trace 摘要、热点表及工具限制。
- 验收：至少一条“预测→证据→修正”；区分 profiler 实测与估计；不得用 GPU busy 推导 Tensor Core 利用率。
- 提交：`perf(exp-06): profile prefill decode and identify bottlenecks`。
- 面试自测：哪个热点最值得优化？它即使快两倍，端到端最多改善多少？

### 07 — 连续批处理、KV 容量与调度预算

- 学习：逐轮调度、权重复用、Paged KV、排队与服务容量；对应 lecture 10、Orca、PagedAttention。
- 动手：先固定 token budget 扫描活动序列上限，再固定序列上限扫描 token budget；联合观察排队、KV 压力、抢占与 SLO。
- 产物：`configs/sweeps/scheduling.*`、容量曲线、BF16 的候选配置。
- 验收：找到可接受 SLO 的负载范围与过载拐点，解释客户端并发和实际 batch 的差别。
- 归因限制：vLLM 内建机制若不能单独关闭，只能称为调度参数实验；可另写小型调度模拟器理解机制，不把模拟收益当作 GPU 实测。
- 提交：`perf(exp-07): characterize scheduler and KV capacity tradeoffs`。
- 面试自测：为什么 batch 更大后总吞吐上升，单用户却更慢？

### 08 — 长短请求混合与 Chunked Prefill

- 学习：prefill 与 decode 的资源竞争、队头阻塞和调度公平性。
- 动手：混合长输入短输出和短对话；只改变 chunked prefill 的可控设置或 token budget；按请求类型分别统计。
- 产物：长短请求各自的 TTFT/TPOT 分布、调度配置和混合负载报告。
- 验收：能指出哪组用户受益、哪组受损；记录引擎默认启用情况；不使用不可比的不同 workload 证明提升。
- 提交：`perf(exp-08): evaluate chunked prefill under mixed workloads`。
- 面试自测：为什么更小的 prefill 块可能改善生成连续性，却恶化首字延迟？

### 09 — Prefix Cache 的收益与反例

- 学习：token 前缀复用、缓存命中与时间局部性、冷启动和缓存容量。
- 动手：固定共享内容，测试无共享/部分共享/高共享三个组；分别控制冷缓存和预热缓存；记录命中统计及其分母。
- 产物：`benchmarks/workloads/prefix_*`、命中率—TTFT 图、一个失效案例。
- 验收：展示有收益与无收益的条件；能解释前缀前部变化为何破坏复用；不得把一次重复 prompt 的最好结果代表全部业务。
- 提交：`perf(exp-09): measure prefix cache benefit and failure cases`。
- 面试自测：你会怎样安排固定 system prompt 与动态内容的位置？

### 10 — 权重量化与质量—容量—速度权衡

- 学习：量化误差、scale、校准、weight-only 与激活量化、反量化成本、带宽节约。
- 动手：先选一种引擎明确支持的量化路径，与相同源 revision 的 BF16 比较；先匹配请求规模，再分别找各自的 SLO 容量上限；运行步骤 03 的质量回归。
- 产物：量化来源/校准 manifest、模型与运行时显存分解、质量/吞吐/延迟表。
- 验收：比较固定负载收益与最大容量收益；说明未量化层和元数据；质量超出阈值时拒绝该配置，不修改阈值掩盖退化。
- 提交：`perf(exp-10): evaluate quantization with quality and capacity gates`。
- 面试自测：INT4 体积更小却比 BF16 慢，有哪些可验证的原因？

## M3：从框架使用走向独立实现（步骤 11–12）

### 11 — CUDA Graph 与启动开销

- 学习：CPU 发射与 GPU 执行的关系、图捕获、shape 覆盖及额外显存。
- 动手：在受支持的配置下比较 graph 与 eager；记录编译/图捕获时间、预热后的吞吐与延迟、显存和 trace。
- 产物：`experiments/11-cuda-graph/`，启动成本和稳态效果分开呈现。
- 验收：核对实际生效后端；至少覆盖小 batch 和较高并发；若引擎同时改变编译策略，明确是组合差异而不是纯 Graph 因果证明。
- 提交：`perf(exp-11): evaluate graph execution overhead and memory cost`。
- 面试自测：为什么低并发下有效的优化，在高并发下可能不明显？

### 12 — 自己实现一个有依据的 Triton 优化

- 学习：访存、融合、tiling、正确性容差、代表性 shape 和 Amdahl；对应 lecture 06 与 A2。
- 动手：从步骤 06 的热点选择一个范围可控的算子，例如 residual + RMSNorm；写 PyTorch reference 和 Triton 实现，画出张量 shape 与读写量。
- 产物：`kernels/`、正确性测试、CUDA events benchmark、热点选择说明和性能曲线。
- 验收：覆盖真实 shape、边界长度和支持的 dtype，事先约定误差；正确预热和同步，避免只比较一次计时；对比 reference 与可用的成熟实现。
- 集成门槛：只在正确性通过、热点占比足够且接入范围可控时尝试服务集成，并再次跑端到端/质量测试。
- 提交：`feat(exp-12): implement and validate fused Triton kernel`，随后 `results(exp-12): report kernel gains and end-to-end limits`。
- 面试自测：你的代码减少了哪些读写？为什么没超过成熟 kernel？局部加速是否转化为服务收益？
- 贡献边界：已有 CS336 attention 代码是已有学习成果，不重新标成此次原创；改动需有清晰 diff。独立 kernel 实验未接入服务时明确写出。

## M4：可服务与可展示（步骤 13–14）

### 13 — 过载保护与对外服务可靠性

- 学习：有界队列、背压、admission control、取消传播、超时预算、健康检查与恢复。
- 动手：先在本地/私网入口实现 TLS/鉴权、输入输出长度限制、配额、队列限制、超时和客户端断连取消；再按实际网络配置公网入口。
- 产物：`gateway/`、部署配置、故障演练脚本和运维 runbook。
- 验收：未授权请求被拒；超限返回明确状态；过载下队列有界；取消释放后端工作；服务重启后恢复；指标端口不对外开放。
- 稳定性：至少一小时混合负载，观察成功率、尾延迟、显存趋势、重启恢复；明确测量期间的 offered load 与背景 GPU 活动。
- 提交：`feat(exp-13): add admission control cancellation and recovery`。
- 面试自测：为什么拒绝部分请求可能提升系统整体服务质量？单机单卡的可靠性边界是什么？

### 14 — 最终复现、消融与项目发布

- 学习：交互效应、消融实验、Pareto 权衡、结果可信度和技术表达。
- 动手：用冻结基准与最终组合，在相同 workload/SLO/质量规则下重复测量；逐项移除主要优化；根据部署文档在干净环境复现。
- 产物：`docs/FINAL_REPORT.md`、最终配置、结果索引、架构图、真实 dashboard 截图、3–5 分钟演示和实测简历描述。
- 验收：代码 SHA→配置→数据 hash→原始证据→图表→结论可追溯；同时展示收益、退化和限制；大型 artifact 提供校验和、位置与保留策略。
- 提交：`docs(exp-14): publish reproducible results and project case study`。
- 发布：验收后打并推送 `v1.0.0` tag；README 链接报告和复现入口。尚未完成时不要创建此 tag。
- 面试自测：如果换成另一种输入长度分布，你的“最佳配置”还成立吗？

## 可选扩展：主线验收后再做

- FP8 KV：以长上下文容量为目标，同时做长文本质量检查。
- 投机解码：分别测低/高并发、接受长度、草稿开销与同卡显存占用。
- 第二个推理引擎：作为系统对照，保持模型、模板和 workload 一致；不把跨引擎差异归因于单项技术。

扩展每项沿用报告模板并单独提交；它们不是完成 v1 的必要条件。

## 每次学习的操作习惯

开始前用 5 分钟回答当前实验的面试问题，并写下预期结果；实现一个可完成的小任务；结束时留下结果或阻塞记录。
不能解释的术语回到 lecture/论文补齐；可以使用 AI 协助查错和审查，但关键公式、指标与代码路径要能自己讲清楚。
提交前检查 diff 和暂存范围；推送后确认远程 SHA。不要 `git add .` 把无关数据和本机配置一并加入。
实验失败时记录复现条件、日志摘要和下一条假设；这也是可提交的真实进展。
