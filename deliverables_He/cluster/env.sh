# 集群公共环境：被 setup_env.sbatch / backbones.sbatch source。按 Storage Guide 要求把缓存全部放到 SSD 项目目录。
: "${P:?请先 export P=/projects/<你的 SSD 项目目录>}"
export TMPDIR=$P/.tmp/${SLURM_JOB_ID:-manual}
export PIP_CACHE_DIR=$P/.tmp/pip CONDA_PKGS_DIRS=$P/.tmp/conda
export HF_HOME=$P/.tmp/hf TRITON_CACHE_DIR=$P/.tmp/triton XDG_CACHE_HOME=$P/.tmp/cache
export TORCH_HOME=$P/.tmp/torch            # torchvision 预训练权重
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
ENV=$P/envs/ee6483                         # conda 环境放项目目录，不放 $HOME
