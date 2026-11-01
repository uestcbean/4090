# EXP-02：流式服务准备

日期：2026-11-01。状态：running，尚未部署模型。

2026-11-01 更新：5 个权重分片下载及 SHA256 校验全部完成，tokenizer 等配套文件补齐。
完整性记录见 `download-verification.json`（非 LFS 配套文件检查大小）；观测页已实测显示 16.40 GB / 16.40 GB、100%、逐权重 SHA256 通过。
新增分段下载器 `scripts/download_ranges.py` 保留已有前缀、校验 HTTP Content-Range、合并后校验原始 SHA256；运行时不得与另一个下载器同时写同一模型目录。
下载监控去重与校验失效测试通过，项目共 12 项测试通过。模型启动和真实推理仍待执行。

## 已完成

- 复核 D 盘 Ubuntu 存储：Linux 文件系统仍有约 944 GiB 可用，WSL RAM 约 15 GiB。
- 固定模型 revision：`b968826d9c46dd6066d109eabc6255188de91218`，见 configs/models/qwen3-8b-manifest.json。
- 添加标准库流式客户端 scripts/smoke_request.py，明确关闭 thinking；检查非空内容、finish_reason 和 DONE。
- 在服务未启动时执行客户端，确认连接失败以非零退出码结束并保存失败状态，不伪造成功结果。
- 服务环境固定 vLLM 0.11.2、Transformers 4.57.1；隔离安装完成。vLLM 依赖 torch 2.9.0，不覆盖 EXP-01 的 torch 2.9.1 环境。
- 2026-11-01 15:50（UTC+8）：WSL 服务环境 CUDA 矩阵运算验证通过，实际 torch 2.9.0+cu128，识别 RTX 4090 / capability 8.9；见 `environment.json`。
- `vllm --version` 正常退出，返回 0.11.2；pip check 返回 No broken requirements found。完整依赖见 `requirements.freeze.txt`。这些仅证明环境检查通过，不等于模型加载/推理成功。

## 网络排查证据

WSL 和 Windows curl 直连 Hugging Face API 均在 20 秒后超时；WSL 访问 PyPI 成功。
Windows 系统代理配置为 127.0.0.1:10809，监听地址仅为回环地址。
Windows 显式代理命令成功获取模型 revision：

```powershell
curl.exe --proxy http://127.0.0.1:10809 -fsSL --max-time 25 https://huggingface.co/api/models/Qwen/Qwen3-8B
```

这说明代理路径可访问模型元数据；不等同于已验证所有权重下载 CDN。
本次没有修改全局代理、防火墙或 WSL 网络模式。后续可从 Windows 使用显式代理将权重下载到 D 盘，再由 WSL 加载本地路径。

### 权重传输源调整

Hugging Face 代理 CDN 长连接多次出现 curl 56 中断，续传单连接约几十至一百多 KiB/s。
ModelScope 同模型 16 MiB Range 测试返回 HTTP 206，平均约 738087 bytes/s。
通过两端文件清单核对，5 个 safetensors 的大小与 SHA256 全部相同；保存精简镜像清单 `configs/models/qwen3-8b-mirror.json`。
续传使用镜像文件固定 commit，其他元数据仍从原始 Hugging Face revision 获取。
最终模型身份和完整性验收仍以原始 Hugging Face manifest 为准；清单相同不代表本地文件已经下载完成。

```powershell
python scripts/download_parallel.py --modelscope-manifest configs/models/qwen3-8b-mirror.json
```

## 下一步

1. 服务引擎及依赖清单已固定；待模型加载时继续验证运行兼容性。
2. 按固定 revision 下载权重与 tokenizer，保存许可证并验证文件完整性。
3. 使用小上下文和有限并发在 localhost 启动 BF16 服务；记录真实显存与加载结果。
4. 运行流式客户端，再做停止/重启检查。真实请求未通过前，本步骤不能标记完成。

复现客户端：在服务启动后，在 WSL 仓库根目录执行：

```bash
python3 scripts/smoke_request.py --output experiments/02-serving/first-request.json
```

客户端的 first_content_chunk_seconds 是客户端观测值，不能直接把网络 chunk 当成单 token，也不能从一次请求推断服务容量。

## 真实观测演示（待执行）

```bash
python3 scripts/observe_serving.py --output experiments/02-serving/observations.json
```

脚本先发送一个本轮独有的长前缀，然后重复请求，再同时发送 4 个相同前缀请求。
每阶段保存请求内容、响应、usage、首内容等待时间、总耗时、引擎原始指标前后快照及区间统计。
运行时停止其他客户端请求，否则引擎全局计数包含额外流量。

学习任务：

1. 在 repeated_prefix 阶段计算 `Δprefix_cache_hits_total / Δprefix_cache_queries_total`，核对面板比例。
2. 解释为什么缓存复用主要节约已有前缀的 prefill 工作，而输出 token 仍需要 decode。
3. 在 concurrent_repeat 阶段观察运行请求数与显存；并发会共享部分前缀，但各请求继续生成的 KV 会产生额外占用。
4. 比较首内容等待和 TPOT，说明它们分别对应什么工作。一次演示受首次编译、预热和排队影响，不能证明优化提升幅度。

理论和口径依据：[固定版本 vLLM 指标源码](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/metrics/loggers.py)。
前缀命中统计单位是 token；KV 使用率 1 表示缓存池 100%；TPOT histogram 每个请求记录一个值。
监控区间均值为 `Δsum / Δcount`，不是直方图桶推算的 p95。脚本的三阶段演示不替代后续固定负载、多轮重复的基准测试。
