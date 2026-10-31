#!/usr/bin/env bash
# 本机 <-> 集群同步（在本机运行）。需已连 NTUSECURE 或 NTU VPN。
#   export CLUSTER=<用户名>@<登录节点IP>  P=/projects/<你的 SSD 项目目录>
#   bash cluster/sync.sh push    # 上传代码 + 猫狗数据集（首次约 1.2 GB）
#   bash cluster/sync.sh pull    # 取回 runs/ 下的训练记录与日志（不含 .pt 权重）
# 集群上的布局与本仓库一致：$P/ee6483/{deliverables_He, datasets/datasets}，配置里的 ../datasets/datasets 无需修改。
set -euo pipefail
: "${CLUSTER:?请先 export CLUSTER=<用户名>@<登录节点IP>}"
: "${P:?请先 export P=/projects/<你的 SSD 项目目录>}"
HE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=$P/ee6483

case "${1:-}" in
  push)
    ssh "$CLUSTER" "mkdir -p $REMOTE/deliverables_He/cluster/logs $REMOTE/datasets"
    rsync -az --info=progress2 --exclude runs/ --exclude __pycache__/ --exclude '*.pt' \
      "$HE/" "$CLUSTER:$REMOTE/deliverables_He/"
    rsync -az --info=progress2 --exclude __MACOSX/ \
      "$HE/../datasets/datasets/" "$CLUSTER:$REMOTE/datasets/datasets/"
    ;;
  pull)
    mkdir -p "$HE/runs" "$HE/cluster/logs"
    rsync -az --info=progress2 --exclude '*.pt' \
      "$CLUSTER:$REMOTE/deliverables_He/runs/" "$HE/runs/"
    rsync -az "$CLUSTER:$REMOTE/deliverables_He/cluster/logs/" "$HE/cluster/logs/"
    ;;
  *) echo "用法: bash cluster/sync.sh push|pull"; exit 1 ;;
esac
