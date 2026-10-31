# cluster/：NTU EEE GPU 集群脚本

完整教程（登录、存储、Slurm、本项目流程、常见问题）见 [../CLUSTER_GUIDE.md](../CLUSTER_GUIDE.md)。

| 文件 | 在哪运行 | 作用 |
|---|---|---|
| `sync.sh` | 本机 | `push` 上传代码和数据集；`pull` 取回 `runs/` 训练记录 |
| `env.sh` | 被各 sbatch 调用 | 缓存重定向到 SSD 项目目录 |
| `setup_env.sbatch` | CPU 任务 | 一次性建 conda 环境、下载预训练权重 |
| `backbones.sbatch` | GPU 任务（job array） | 7 个模型的骨干对比 |
| `train.sbatch` | GPU 任务 | 任意配置跑单个实验 |

所有 sbatch 都在 `deliverables_He/` 下提交，日志写到 `cluster/logs/`。
