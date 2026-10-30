import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """固定 Python / NumPy / PyTorch 随机种子。deterministic=True 会更慢，但更可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """DataLoader worker 的种子，配合 generator 使用。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
