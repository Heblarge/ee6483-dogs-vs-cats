#!/bin/bash
# Non-training GPU/data smoke test for the isolated reproduction environment.
#SBATCH --job-name=ee6483-smoke
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

module load Miniforge3
set +u
source activate "$P/envs/ee6483"
source "$P/envs/ee6483-recheck/bin/activate"
set -u

python - <<'PY'
import torch
from torch.utils.data import DataLoader

from ee6483.src.data.cifar_parquet import get_cifar
from ee6483.src.data.transforms import build_cifar_eval_transform
from ee6483.src.models.backbones import build_cifar_resnet

assert torch.cuda.is_available(), "GPU is not visible inside the Slurm job"
dataset = get_cifar("data/cifar", True, build_cifar_eval_transform())
assert len(dataset) == 50_000
images, labels = next(iter(DataLoader(dataset, batch_size=8, num_workers=2)))
model = build_cifar_resnet("resnet18", 10).cuda().eval()
with torch.no_grad():
    logits = model(images.cuda())
assert logits.shape == (8, 10)
print("SMOKE PASS", torch.cuda.get_device_name(0), images.shape, labels.shape, logits.shape)
PY
