"""优化器与学习率调度。

参数分组：
- 分类头使用 lr；骨干使用 lr * backbone_lr_mult（微调预训练模型时常设 0.1）。
- bias 与 Norm 层参数不做 weight decay。
"""
import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def _param_groups(model, lr, weight_decay, backbone_lr_mult):
    head_name = getattr(model, "head_name", None)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = head_name is not None and name.startswith(head_name)
        no_decay = p.ndim <= 1  # bias、BN/LN 的 weight
        key = (is_head, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr if is_head or head_name is None else lr * backbone_lr_mult,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def build_optimizer(model, optim_cfg):
    name = optim_cfg.get("name", "adamw").lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    wd = float(optim_cfg.get("weight_decay", 0.0))
    groups = _param_groups(model, lr, wd, float(optim_cfg.get("backbone_lr_mult", 1.0)))
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=optim_cfg.get("momentum", 0.9),
                               nesterov=optim_cfg.get("nesterov", True))
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(optim_cfg.get("betas", (0.9, 0.999))))
    raise ValueError(f"未知优化器: {name}")


def build_scheduler(optimizer, sched_cfg, steps_per_epoch, epochs):
    """按 iteration 更新的调度器：cosine（含 warmup）| step | constant。"""
    name = (sched_cfg or {}).get("name", "cosine")
    total = steps_per_epoch * epochs
    warmup = int((sched_cfg or {}).get("warmup_epochs", 0) * steps_per_epoch)
    min_ratio = (sched_cfg or {}).get("min_lr_ratio", 0.01)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if name == "cosine":
            t = (step - warmup) / max(1, total - warmup)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        if name == "step":
            every = sched_cfg.get("step_epochs", 10) * steps_per_epoch
            return sched_cfg.get("gamma", 0.1) ** (step // every)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
