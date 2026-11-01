# 实验进度与证据

路线图提交不代表实验完成。完成实验后填写报告、被测代码 SHA 和 GitHub 结果提交链接。

| 步骤 | 实验 | 状态 | 报告 | 被测 SHA / 结果提交 |
|---|---|---|---|---|
| 00 | 项目契约 | completed；2026-10-29 引导式概念验收 | [报告](00-contract/report.md) | 文档步骤，无 GPU 测量；提交见文件历史 |
| 01 | 环境与显存预算 | 工程验证通过；2026-11-01，待学习验收 | [报告](01-environment/report.md) | `4eca0d1` / `a7f4ac5`；结果见报告历史 |
| 02 | 流式模型服务 | running；revision 已固定，代理路径已验证；未部署 | [报告](02-serving/report.md) | 提交见报告历史 |
| 03 | BF16 质量基准 | planned | — | — |
| 04 | 真实监控页面 | running；真实 GPU 采样及请求入口已验证，待引擎联调 | [说明](../monitoring/README.md) | 提交见文件历史 |
| 05 | 压测与基准冻结 | planned | — | — |
| 06 | Roofline 与 profiling | planned | — | — |
| 07 | 调度与 KV 容量 | planned | — | — |
| 08 | Chunked prefill | planned | — | — |
| 09 | Prefix cache | planned | — | — |
| 10 | 量化权衡 | planned | — | — |
| 11 | CUDA Graph | planned | — | — |
| 12 | Triton 算子 | planned | — | — |
| 13 | 服务可靠性 | planned | — | — |
| 14 | 最终复现与发布 | planned | — | — |
