# EXP-01：环境检查与显存预算

开始日期：2026-10-29。验证日期：2026-11-01。状态：工程验证完成，等待最后学习验收。

## 假设与学习记录

假设：模型权重能放进显存，并不保证多请求推理能运行；KV Cache 随未共享的缓存 token 总数线性增长。
作者解释：“因为推理过程中不同请求使用的KV Cache不是同一份。而且随着序列变长，KV Cache也会变大”。
讨论补充：相同前缀可以在引擎支持时共享对应缓存块；本次预算按不共享计算。
作者手算 4 个请求、每个 8192 tokens 的结果为 4.5，单位经纠正为 GiB。

## 实现与正确性证据

- 被测代码提交：`4eca0d1ef7217c7bb1b3d7e9db15a6998e930d48`。
- `scripts/memory_budget.py`：独立设置权重位宽、KV 元素字节数、各请求长度、总容量及运行时预留。
- `scripts/inspect_environment.py`：采集 WSL/CPU/RAM/磁盘/GPU/Python，显式记录缺失依赖；可选小型 CUDA 矩阵乘法。
- 5 项标准库单元测试通过，详见 [测试输出](tests.txt)。覆盖手算数值、不同请求长度、空缓存、量化边界、容量不足与非法参数。
- 配置为之前已核对的模型结构子集，参数量按 8B 粗估；不是已下载模型的精确参数计数，也不是冻结 revision 的部署 manifest。

## 预算结果（估计，不是实际显存分配）

公式：KV bytes = 2 × layers × KV heads × head dim × bytes/element × sum(cached lengths)。
BF16 KV：2 × 36 × 8 × 128 × 2 = 147456 bytes/token = 144 KiB/token。

| 配置 | 权重粗估 GiB | KV GiB | 运行时及其他预留 GiB | 合计 GiB |
|---|---:|---:|---:|---:|
| BF16 权重，BF16 KV，4×8192 | 14.901 | 4.5 | 3（假设） | 22.401 |
| INT4 权重，BF16 KV，4×8192 | 3.725 | 4.5 | 3（假设） | 11.225 |

JSON 结果：[BF16](budget-4x8192.json)、[理想化 INT4](budget-int4-4x8192.json)。
只改变权重精度，KV 不变。量化元数据、未量化层、分配器块取整和真实 workspace 不包含在精确项中。
预留值包含桌面等其他占用，不应再重复扣除相同部分；正余量不是成功运行保证。后续必须用实际服务峰值核对。

## 初次环境检查（实测快照）

详见 [安装前记录](environment-before.json)，其中 git_status 为空，明确对应上述代码提交。

- Ubuntu WSL2，kernel `6.18.54.1-microsoft-standard-WSL2`，Python 3.12.3，32 个逻辑 CPU。
- WSL 总内存约 15.52 GiB、可用约 14.71 GiB、swap 4 GiB；内存可用量随系统变化。
- RTX 4090，总显存 24564 MiB，可用 22855 MiB，Windows 驱动 610.88。
- 最初系统 Python 缺少 PyTorch；smoke check 正确返回失败，而非误报 GPU 可用即通过。
- Windows 注册表确认 Ubuntu VHD 位于 C 盘；检查时 C 盘可用 25397342208 bytes（约 23.65 GiB），D 盘约 1.18 TiB。
- WSL `df` 的约 1 TB 容量不能替代宿主机空间检查。下载模型前需要决定模型缓存/WSL 存储方案。

## 环境准备与边界

创建专用 `~/.venvs/4090-exp01`，使用官方 CUDA 12.8 wheel 源安装固定 PyTorch 2.9.1，避免修改系统 Python。
安装入口与复现命令见 [README](README.md)；依赖安装不代表已验证 CUDA 运算，需以 smoke 输出为准。
pip zipapp 初次下载超时，断点续传完成；不完整下载的文件无法执行，完整下载后才正常启动安装。
此次没有安装 Linux 显示驱动，没有下载模型，没有做性能压测，也没有更改 WSL 的磁盘或内存配置。

## CUDA 实际验证与解释器排查

2026-11-01 使用已有独立环境完成验证，本次没有重新安装 PyTorch。
最终被测代码：`a7f4ac5b9cad3b8867b994425cafc94c13d42360`。
[安装后快照](environment-after.json) 同时记录解释器路径、平台、全部已安装包版本和代码 SHA；采集时只有实验结果目录未跟踪，代码没有未提交修改。

- 解释器：`/home/bean/.venvs/4090-exp01/bin/python`，平台 `linux`，Python 3.12.3。
- PyTorch：`2.9.1+cu128`，构建 CUDA 12.8，`cuda_available=true`。
- 设备：RTX 4090，compute capability 8.9。
- 实测：4×4 float32 CUDA 矩阵乘法与 CPU 参考完全一致，`rtol=0, atol=0`，同步后检查通过。
- 预算单元测试再次运行，5 项全部通过。
- 环境没有 NumPy，PyTorch 提示 NumPy 初始化警告；本次纯 tensor 验证通过，不声称 NumPy 互操作可用。
- 此时宿主机 C 盘可用 24012181504 bytes（约 22.36 GiB），D 盘可用 1306752565248 bytes；模型存储方案仍需在步骤 02 处理。

用户指出机器已有 CUDA PyTorch，并提供在 WSL 终端执行 `python` 后出现 Anaconda / MSC / win32 的输出。
该输出说明那个命令启动了 Windows Python；它与 Ubuntu 系统 Python、此次 Linux 虚拟环境是不同解释器。
此前“缺少 PyTorch”的结论只适用于被检查的 Ubuntu 系统 Python，不能扩展成整台机器未安装。
Windows 环境的具体 PyTorch 版本本次未核验，也未修改。

## 下一步学习验收

环境和预算工具的工程验证已完成，但不代替作者理解：需要解释权重量化为什么不会自动缩小 KV Cache，以及估算余量为什么不保证服务成功。
模型下载、vLLM 部署和性能测试尚未开始。
