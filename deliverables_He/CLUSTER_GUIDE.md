# NTU EEE GPU 集群使用教程（Cluster 02）

本教程整理自官方文档 [NTUEEECluster/docs](https://github.com/NTUEEECluster/docs)，并结合本项目的脚本（`cluster/`）给出完整流程。
集群规则会更新，遇到与本文不一致的情况以官方文档为准。

---

## 1. 集群是怎么组成的

```
你的电脑 ──SSH──> 登录节点 ──Slurm──> 计算节点
(需 NTUSECURE/VPN)   (没有 GPU)         (GPU 节点 / CPU 节点)
                        │                     │
                        └────── 共享存储 ──────┘
```

| 部分 | 用途 | 注意 |
|---|---|---|
| 登录节点 | 编辑代码、传文件、提交任务 | **没有 GPU**；每人最多 3 核、16 GB 内存，超出会杀掉你在该节点的**全部**进程 |
| 计算节点 | 真正跑训练 | 只能通过 Slurm 使用；只有在你有任务运行时才能 SSH 进去 |
| 共享存储 | 所有节点看到的文件完全一样 | 登录节点上传的文件，任务里直接可见；**没有备份** |

> 你断开最后一个 SSH 会话时，你在登录节点上的所有进程都会被杀掉，包括 `tmux` 和 `nohup`。所以长任务一律用 `sbatch` 提交。

---

## 2. 首次登录

**准备**：用户名（NTU 邮箱 `@` 前面的部分，小写）、审批邮件里的默认密码和登录节点 IP，并且连上 NTUSECURE 或 NTU VPN。

```bash
ssh <用户名>@<登录节点IP>
```

首次连接时会询问主机指纹，**只有**指纹是下面这个才输入 `yes`：

```
SHA256:AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM
```

首次登录会强制改密码，输入时屏幕上不显示任何字符。依次提示：

| 提示 | 输入 |
|---|---|
| `Password:` | 临时密码 |
| `Current Password:` | **再输一次**临时密码（正常现象） |
| `New password:` | 新密码 |
| `Retype new password:` | 再输一次新密码 |

新密码要求：至少 12 位；小写、大写、数字、符号四类中至少包含三类；不能含用户名；不能与最近 24 个密码重复。
改完可能被断开，用新密码重新登录即可。密码会定期过期，届时按同样方式修改。

- **连续输错 5 次锁定 15 分钟。**
- 先在终端里登录一次，再用 VS Code 等 IDE 连接（IDE 处理不了改密码流程）。
- 忘记密码：用学校邮箱给管理员发邮件。
- `Connection refused` 或超时：检查是否连上了 NTUSECURE 或 VPN。

**可选：免密登录。** 在**本机**运行一次 `ssh-copy-id <用户名>@<登录节点IP>`，之后 ssh 和 rsync 都不用再输密码。

---

## 3. 存储设置（必须先做）

| 路径 | 容量 | 用来放 |
|---|---|---|
| `/home/<用户名>` | 50 GB | 只放代码和配置 |
| `/projects/<名字>` | 按分配额度 | 数据、conda 环境、模型权重、缓存 |
| `/tmp`、`/var/tmp` | 每人 4 GB | 仅放小的临时文件；它会跟着你到每个节点，**从不自动清理** |

项目目录分两个层级：**SSD**（快，用于训练数据、环境、缓存）和 **HDD**（慢，大量小文件读取时尤其慢，只适合归档）。
训练用的数据和环境都要放在 SSD 上。

### 3.1 创建项目目录

在登录节点上运行：

```bash
storagemgr
```

这是一个交互式菜单：选 **SSD**，起一个名字（如 `ee6483`）。名字只能包含字母、数字和连字符，**创建后不能再改**。
目录默认只有你自己能访问。检查实际所在层级：`readlink -f /projects/<名字>`。

### 3.2 把缓存重定向到项目目录

装任何东西之前都要先做这一步。在集群的 `~/.bashrc` 末尾加上：

```bash
P=/projects/<名字>
export TMPDIR=$P/.tmp
export PIP_CACHE_DIR=$P/.tmp/pip CONDA_PKGS_DIRS=$P/.tmp/conda
export HF_HOME=$P/.tmp/hf TRITON_CACHE_DIR=$P/.tmp/triton XDG_CACHE_HOME=$P/.tmp/cache
mkdir -p $TMPDIR
```

本项目的任务脚本会通过 `cluster/env.sh` 自动完成同样的设置，并额外设置 `TORCH_HOME`，用于存放 torchvision 预训练权重。每个任务还会用独立的 `TMPDIR`，任务结束后自动清理。

### 3.3 空间不够时

```bash
storagemgr                       # 查看项目目录用量
du -sh ~ /tmp                    # 查看 home 和 /tmp
find /tmp -user $USER -mtime +30 -delete
```

**不要**对整个数据集目录或整个文件系统跑 `du`、`find`、`grep -r`，这会拖慢所有人的共享存储。

---

## 4. Slurm 基础

### 4.1 软件环境

CUDA、编译器和 Conda 都通过 Lmod 模块提供，**不要自己安装工具链**，也没有 `sudo`。

```bash
module avail                   # 列出所有模块
module spider CUDA             # 搜索
module load Miniforge3         # 加载 conda
module list / module purge
```

Python 包装在自己的 conda 环境里（base 环境只读）。本项目把环境放在 `$P/envs/ee6483`，不放 `$HOME`。

### 4.2 批量任务（sbatch）

最常用的方式。提交后即使退出登录，任务也会继续运行。

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:a40:1          # gpu:<型号>:<数量>，必须写明型号
#SBATCH --time=12:00:00           # 最长 3-00:00:00
#SBATCH --output=%x-%j.out        # %x=任务名 %j=任务号

module load Miniforge3
source activate <环境>
python train.py
```

```bash
sbatch job.sh            # 提交
squeue --me              # 查看自己的任务；REASON 列说明为什么在排队
scontrol show job <id>   # 任务详情
scancel <id>             # 取消
```

**不需要 GPU 的任务**（装环境、下载、预处理）：去掉 `--gres`，改用 `--cpus-per-task=N`（每个 CPU 配 3 GiB 内存）。CPU 任务**不计费**，而且可以联网下载。

### 4.3 交互式任务（调试用）

```bash
srun --gres=gpu:a40:1 --time=1:00:00 --pty bash
```

- 最多 **2 小时、1 块 GPU、1 个节点**（不写 `--time` 时只有 30 分钟）。
- SSH 一断，任务就结束；`salloc` 被禁用。
- 用完立即 `exit`：空闲的交互任务仍然占着 GPU，会被审计。

### 4.4 提交时会被检查的规则

| 规则 | 说明 |
|---|---|
| 必须指定一种 GPU 型号 | 只写 `--gres=gpu:2`（不带型号）或混用型号会被拒绝 |
| CPU 和内存随 GPU 自动分配 | 每块 GPU 配 4 个 CPU 和固定内存。GPU 任务**不要**写 `--cpus-per-task` 或 `--mem` |
| 时长 | 批量任务默认 1 h、最长 3 天；交互式任务最长 2 h |
| 禁止的选项 | `--exclusive`、`--cpus-per-gpu`、`--gpus-per-socket`、`--ntasks-per-gpu/core/socket`、`salloc` |
| 提交后不能修改 | 资源、时长等写错了，只能取消后重新提交 |

被拒绝时，错误信息会说明违反了哪条规则。**不要**换着参数反复尝试去绕过规则。

### 4.5 查看可用 GPU

节点和 GPU 型号会变化，以实时查询为准：

```bash
# 可以写进 --gres 的 GPU 型号
sinfo -h -o "%G" | tr ',' '\n' | sed -E 's/\(.*//; s/:[0-9]+$//' | sort -u
# 各节点 GPU 占用情况
sinfo -N -O "NodeList:18,Gres:22,GresUsed:34,StateCompact:8"
# 自己的账户和 QoS 限额
sacctmgr show assoc user=$USER format=Account,QOS%40,DefaultQOS
```

`pro6000` 和 `rtx5090` 是 Blackwell 架构，需要 CUDA 12.8 或以上、对应的 PyTorch（如 `cu128`），否则会报 `no kernel image is available`。本项目装的就是 `cu128` 版。

### 4.6 计费（SU）

GPU 时间按 Service Unit 计费：GPU 数 × 分钟数 × 型号权重。CPU 任务免费。

| 型号 | 每 GPU·小时的 SU |
|---|---:|
| pro6000 | 480 |
| rtx5090 | 360 |
| l40、6000ada | 240 |
| a40、a6000 | 180 |

个人账户每月有额度（`msc`、`ug-proj` 为 180,000 SU；`rose`、`phd` 为 806,400 SU），每月 1 日重置。查看本月已用额度：

```bash
sshare -U -nP -o GrpTRESMins,GrpTRESRaw | awk -F'|' '{match($1,/billing=([0-9]+)/,c); match($2,/billing=([0-9]+)/,u); if (c[1]>0) printf "%d / %d SU (%.1f%%)\n", u[1], c[1], 100*u[1]/c[1]; else print "No personal cap."}'
```

`--qos=override-limits-but-killable` 可以免费使用超出限额的空闲 GPU，但随时可能被抢占并重新排队，只适合会定期保存 checkpoint 的任务。

### 4.7 排队原因（`squeue --me` 的 REASON 列）

| REASON | 含义 |
|---|---|
| `Resources` / `Priority` | 在等 GPU 空出来 |
| `QOSMaxJobsPerUserLimit` | 你同时运行的任务数已达上限 |
| `QOSMaxGRESPerUser` | 你在该型号上的 GPU 数已达上限 |
| `AssocGrpBillingMinutes` | 本月额度用完 |
| `ReqNodeNotAvail` | 节点维护中，或 `--time` 与维护时间重叠 |

---

## 5. 本项目的完整流程

### 5.1 脚本一览（`deliverables_He/cluster/`）

| 文件 | 在哪运行 | 作用 |
|---|---|---|
| `sync.sh` | 本机 | `push`：上传代码和猫狗数据集；`pull`：取回 `runs/` 训练记录（不含 `.pt` 权重） |
| `env.sh` | 被各 sbatch 调用 | 缓存全部重定向到 `$P/.tmp`，并设置每个任务独立的 `TMPDIR` |
| `setup_env.sbatch` | CPU 任务 | 一次性：在 `$P/envs/ee6483` 建 conda 环境，装 PyTorch（cu128），下载 7 个模型的预训练权重 |
| `backbones.sbatch` | GPU 任务（job array） | 骨干对比：7 个模型各占 1 块 GPU 并行，每个跑 seed 0 和 1 |
| `train.sbatch` | GPU 任务 | 通用：用任意配置跑一个实验 |

集群上的目录结构和本仓库一致，所以配置里的 `../datasets/datasets` 不用改：

```
/projects/<名字>/
├── ee6483/
│   ├── deliverables_He/      代码（sync.sh push 上传）
│   └── datasets/datasets/    猫狗数据集（sync.sh push 上传，约 1.2 GB）
├── envs/ee6483/              conda 环境（setup_env.sbatch 创建）
└── .tmp/                     pip / conda / torch 等缓存
```

### 5.2 第一步：上传（本机，在 `deliverables_He/` 下）

```bash
export CLUSTER=<用户名>@<登录节点IP>
export P=/projects/<名字>
bash cluster/sync.sh push
```

之后改了代码再执行一次 `push` 就行，rsync 只传有变化的文件。

### 5.3 第二步：建环境（集群登录节点）

```bash
export P=/projects/<名字>          # 建议同时写进集群的 ~/.bashrc
cd $P/ee6483/deliverables_He
sbatch cluster/setup_env.sbatch    # 记下输出的 jobid，约 10–20 分钟
```

登录节点上不允许装包，所以装环境也作为 CPU 任务提交。完成后检查：`tail cluster/logs/ee6483-setup-<jobid>.out`，最后应打印 torch 版本。

### 5.4 第三步：提交骨干对比

```bash
# 等环境装好后自动开始
sbatch --dependency=afterok:<setup 的 jobid> cluster/backbones.sbatch
# 环境已经装好时直接提交
sbatch cluster/backbones.sbatch
```

| 需求 | 命令 |
|---|---|
| 换 GPU 型号 | `sbatch --gres=gpu:l40:1 cluster/backbones.sbatch` |
| 只补跑某几个模型 | `sbatch --array=2,6 cluster/backbones.sbatch`（下标：0 resnet18、1 resnet34、2 resnet50、3 efficientnet_b0、4 convnext_tiny、5 swin_t、6 vit_b_16） |
| 加种子 | `SEEDS="0 1 2" sbatch cluster/backbones.sbatch` |

已经跑完的 run 会自动跳过，任务中断后重新提交即可。预计 7 个模型并行各需 1–2 小时；按 a40 计算总共约 1,300–2,700 SU。

### 5.5 跑单个实验

```bash
sbatch cluster/train.sbatch configs/dogs_cats/d3_resnet18_finetune.yaml
sbatch cluster/train.sbatch configs/dogs_cats/d3_resnet18_finetune.yaml --set seed=1 optim.lr=3e-4
```

结果写在 `runs/<配置名>_s<seed>_<时间>/`。

### 5.6 查看进度

```bash
squeue --me
tail -f cluster/logs/ee6483-bb-<jobid>_6.out     # array 任务：<jobid>_<下标>
cat runs/backbones/b_vit_b_16_s0/history.csv    # 每个 epoch 的验证结果
```

### 5.7 取回结果并汇总（本机）

```bash
bash cluster/sync.sh pull
python scripts/summarize_backbones.py           # -> results/backbones.{csv,md}
```

---

## 6. 常见问题

| 现象 | 原因与解决 |
|---|---|
| `no kernel image is available for execution on the device` | 在 Blackwell GPU（rtx5090、pro6000）上用了旧版 PyTorch；本项目装的 cu128 版没有这个问题 |
| `cpus_per_task set by two different environment variables` | GPU 任务里写了 `--cpus-per-task`，删掉 |
| `oom_kill event` | 内存不够：申请更多 GPU（内存随 GPU 分配），或在 pro6000 上加 `-C highmem` |
| `Disk quota exceeded` | 见 3.3 节；常见原因是 conda 环境或 `~/.vscode-server` 放在了 home |
| 登录节点上的进程突然全没了 | 超过 16 GB 内存上限，或最后一个 SSH 会话断开了 |
| sbatch 报 `--output` 目录不存在 | Slurm 不会自动创建日志目录；`sync.sh push` 会创建 `cluster/logs/`，或手动 `mkdir -p cluster/logs` |
| `source activate` 报 `unbound variable` | 脚本开了 `set -u`；本项目的 sbatch 已在激活前后切换 `set +u` 和 `set -u` |

## 7. 必须遵守的规则

违反会被警告甚至封号：

- **不在登录节点**装包、预处理数据、训练或做全仓库搜索。
- 环境和缓存**不放** `$HOME` 或 `/tmp`。
- 不让空闲的交互任务占着 GPU。
- 不在计算节点上部署聊天机器人或推理服务。
- 不读取或列出别人的文件，也不把自己的文件设成所有人可读。
- 任务被拒绝时，不要换着参数反复尝试绕过规则；有问题用学校邮箱联系管理员。

### 使用 VS Code Remote SSH 时

VS Code 的文件监视会给共享存储造成压力。连接后按 `F1`，选择 **Preferences: Open Remote Settings (JSON)**，加入以下设置后重新连接：

```json
{
  "files.watcherExclude": { "**": true },
  "search.exclude": { "**": true },
  "search.followSymlinks": false
}
```

只打开代码目录，不要打开整个 home、数据集或 conda 环境目录。导致共享存储卡顿的 VS Code 进程会被管理员直接杀掉。
