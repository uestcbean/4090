# D 盘存储约定

2026-11-01，按用户要求将本项目运行环境移到 D 盘。

| 内容 | 位置 |
|---|---|
| 源码与实验报告 | `D:\bean\code\cs336\inference_lab` |
| Ubuntu 虚拟磁盘 | `D:\WSL\Ubuntu\ext4.vhdx` |
| WSL swap | `D:\WSL\swap\wsl-swap.vhdx` |
| 已验证 Python | `/home/bean/.venvs/4090-exp01/bin/python`，在上述 D 盘 VHD 中 |
| 项目模型、编译缓存与临时文件 | `/home/bean/inference-lab-data/`，在上述 D 盘 VHD 中 |

迁移使用本机支持的 `wsl --manage Ubuntu --move D:\WSL\Ubuntu`，执行前 Ubuntu 已停止，目标目录不存在。
迁移后注册表 BasePath 和 VHD 文件位置均验证为 D 盘，CUDA smoke check 通过。
没有注销或重装发行版。已有 Linux 路径及 Python 环境保持不变。

在 WSL Bash 终端启动本项目之前运行：

```bash
cd /mnt/d/bean/code/cs336/inference_lab
source scripts/project_env.sh
source ~/.venvs/4090-exp01/bin/activate
python scripts/inspect_environment.py --cuda-smoke
```

环境变量只作用于当前 shell 及其子进程；后续部署脚本/容器需要显式继承配置或映射存储，不能假设所有终端自动生效。
Windows 用户目录中的 `.wslconfig` 仅保存小型设置文件，指定 D 盘 swap；该设置对 WSL2 全局生效。
Docker Desktop 的独立虚拟磁盘仍在 C 盘，本项目暂不使用它存放镜像或模型；若后续采用 Docker Desktop，须先迁移其存储。
Windows Anaconda 和其他应用未移动。此约定保证本项目的大型运行数据位于 D 盘，不承诺 Windows 系统完全不写入 C 盘。

历史报告里 C 盘存储描述是迁移前快照，保留用于追溯。模型尚未下载，当前完成的是存储准备。
