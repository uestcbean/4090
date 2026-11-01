# EXP-01：复现命令

在 Ubuntu WSL2 终端中进入仓库（本机路径为 `/mnt/d/bean/code/cs336/inference_lab`）。
预算和单元测试仅使用 Python 标准库，不依赖 GPU/PyTorch。

```bash
python3 -m unittest discover -s tests -v
python3 scripts/memory_budget.py \
  --config configs/models/qwen3-8b-budget.json \
  --lengths 8192 8192 8192 8192
python3 scripts/inspect_environment.py
```

`--lengths` 表示每个请求当前缓存的总 token 数；支持不同请求长度。
预算默认预留 3 GiB 给运行时及其他占用，此值是假设，不是测量值。
`--capacity-gib` 默认 24，可用实际总显存代替；不要同时减掉桌面占用、又将同一占用全部加入 reserve。

例：只减少权重精度，保持 BF16 KV：

```bash
python3 scripts/memory_budget.py \
  --config configs/models/qwen3-8b-budget.json \
  --lengths 8192 8192 8192 8192 --weight-bits 4
```

这是理想化 weight-only 存储估计，未计 scale、未量化层等，不能当作量化模型实测大小。

## 独立 CUDA 验证环境

环境位置：`~/.venvs/4090-exp01`。它只用于本步骤，不预先绑定未来 vLLM 的依赖。
使用官方提供的 PyTorch 2.9.1 / CUDA 12.8 wheel 组合，目的是固定一次小型验证环境，并非宣称它是最新版。
来源：[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)。

若系统缺少 pip，可先下载官方 pip zipapp 到本地完整文件，再执行；下载中断必须续传或重新下载，不能运行不完整文件。

```bash
python3 -m venv --without-pip ~/.venvs/4090-exp01
curl --fail --location --retry 2 https://bootstrap.pypa.io/pip/pip.pyz -o /tmp/4090-pip.pyz
python3 /tmp/4090-pip.pyz --python ~/.venvs/4090-exp01/bin/python install \
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
~/.venvs/4090-exp01/bin/python scripts/inspect_environment.py --cuda-smoke
```

检查脚本中的矩阵乘法只验证 CUDA 计算及结果正确性，不测吞吐、显存峰值或模型容量。
`nvidia-smi` 显示的驱动、PyTorch wheel 的 CUDA runtime、独立 CUDA toolkit 是不同组件；不需要为此在 WSL 另装 Linux 显示驱动。

## 环境事实的解释

WSL 的 `df` 报告虚拟文件系统容量，不能证明宿主机还可容纳同等大小的数据。
本机 Ubuntu 的虚拟磁盘位于 C 盘，而仓库位于 D 盘；在 Linux home 下载模型仍会占用 C 盘。
模型下载前需重新检查宿主机空间，再决定缓存位置或迁移方案。此次不迁移 WSL、不下载模型。
