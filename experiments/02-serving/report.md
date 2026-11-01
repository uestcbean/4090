# EXP-02：流式服务准备

日期：2026-11-01。状态：running，尚未部署模型。

## 已完成

- 复核 D 盘 Ubuntu 存储：Linux 文件系统仍有约 944 GiB 可用，WSL RAM 约 15 GiB。
- 固定模型 revision：`b968826d9c46dd6066d109eabc6255188de91218`，见 configs/models/qwen3-8b-manifest.json。
- 添加标准库流式客户端 scripts/smoke_request.py，明确关闭 thinking；检查非空内容、finish_reason 和 DONE。
- 在服务未启动时执行客户端，确认连接失败以非零退出码结束并保存失败状态，不伪造成功结果。
- 当前 PyPI 返回 vLLM 0.32.0；尚未选择/安装服务引擎版本，不将最新版本等同于已验证的兼容版本。

## 网络排查证据

WSL 和 Windows curl 直连 Hugging Face API 均在 20 秒后超时；WSL 访问 PyPI 成功。
Windows 系统代理配置为 127.0.0.1:10809，监听地址仅为回环地址。
Windows 显式代理命令成功获取模型 revision：

```powershell
curl.exe --proxy http://127.0.0.1:10809 -fsSL --max-time 25 https://huggingface.co/api/models/Qwen/Qwen3-8B
```

这说明代理路径可访问模型元数据；不等同于已验证所有权重下载 CDN。
本次没有修改全局代理、防火墙或 WSL 网络模式。后续可从 Windows 使用显式代理将权重下载到 D 盘，再由 WSL 加载本地路径。

## 下一步

1. 固定服务引擎版本并在 D 盘独立环境安装，保存依赖清单。
2. 按固定 revision 下载权重与 tokenizer，保存许可证并验证文件完整性。
3. 使用小上下文和有限并发在 localhost 启动 BF16 服务；记录真实显存与加载结果。
4. 运行流式客户端，再做停止/重启检查。真实请求未通过前，本步骤不能标记完成。

复现客户端：在服务启动后，在 WSL 仓库根目录执行：

```bash
python3 scripts/smoke_request.py --output experiments/02-serving/first-request.json
```

客户端的 first_content_chunk_seconds 是客户端观测值，不能直接把网络 chunk 当成单 token，也不能从一次请求推断服务容量。
