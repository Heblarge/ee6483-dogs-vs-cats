# 在 NTU EEE GPU Cluster 02 上跑骨干对比

依据 [NTUEEECluster/docs](https://github.com/NTUEEECluster/docs)。流程：本机 ──ssh/rsync──> 登录节点（无 GPU）──sbatch──> GPU 计算节点。

| 文件 | 在哪运行 | 作用 |
|---|---|---|
| `sync.sh` | 本机 | `push` 上传代码和数据集；`pull` 取回训练记录 |
| `setup_env.sbatch` | 集群（CPU 任务） | 一次性：建 conda 环境、下载预训练权重 |
| `backbones.sbatch` | 集群（GPU 任务） | job array，7 个模型各占 1 块 GPU 并行 |
| `env.sh` | 被上面两个 source | 缓存目录全部指向 SSD 项目目录 |

## 0. 一次性准备（集群上手动做）

1. 连 NTUSECURE 或 NTU VPN，`ssh <用户名>@<登录节点IP>`。首次登录强制改密码（临时密码要输两次）。
2. 登录节点上运行 `storagemgr`，建一个 **SSD** 项目目录（如 `/projects/ee6483`；名字永久不可改）。
3. 可选：本机 `ssh-copy-id <用户名>@<登录节点IP>`，之后 rsync 不用反复输密码。

## 1. 上传（本机，在 deliverables_He/ 下）

```bash
export CLUSTER=<用户名>@<登录节点IP>  P=/projects/<项目目录>
bash cluster/sync.sh push
```

## 2. 建环境 + 提交训练（集群登录节点）

```bash
export P=/projects/<项目目录>
cd $P/ee6483/deliverables_He
sbatch cluster/setup_env.sbatch                                   # 约 10–20 分钟，CPU 任务不计费
sbatch --dependency=afterok:<上一步的 jobid> cluster/backbones.sbatch
squeue --me                                                       # 查看排队 / 运行状态
tail -f cluster/logs/ee6483-bb-<jobid>_6.out                      # 看 ViT 那个任务的输出
```

- 默认 `a40`（180 SU/GPU·h）。7 个任务全跑完约 7–15 GPU·h，即约 1,300–2,700 SU，远低于个人月额度。
  查看可用型号：`sinfo -h -o "%G" | tr ',' '\n' | sed -E 's/\(.*//; s/:[0-9]+$//' | sort -u`
- 换型号：`sbatch --gres=gpu:l40:1 cluster/backbones.sbatch`；补跑：`sbatch --array=6 ...`；加种子：`SEEDS="0 1 2" sbatch ...`。
- 已完成的 run 自动跳过，被中断后直接重新提交。

## 3. 取回并汇总（本机）

```bash
bash cluster/sync.sh pull
python scripts/summarize_backbones.py      # -> results/backbones.{csv,md}
```

## 集群规则（违反会被警告或封号）

- 登录节点只做编辑、传文件、提交任务；**不在登录节点装包或训练**（每人上限 3 核 16 GB，超了会杀掉你所有进程）。
- conda 环境和缓存不放 `$HOME`（50 GB）或 `/tmp`（4 GB）；`env.sh` 已全部重定向到 `$P/.tmp`、`$P/envs`。
- GPU 任务不要加 `--cpus-per-task` / `--mem`；每块 GPU 固定配 4 个 CPU。
- 不要用 `srun --pty` 交互任务长时间占着 GPU。
