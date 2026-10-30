"""损失函数。模型输出 logits，不要预先做 softmax。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017)：降低易分样本权重。主要为后续不平衡实验预留。"""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        if target.dim() == 1:
            target = F.one_hot(target, logits.size(1)).float()
        p = logp.exp()
        loss = -(target * (1 - p) ** self.gamma * logp)
        if self.weight is not None:
            loss = loss * self.weight
        return loss.sum(1).mean()


def build_loss(loss_cfg):
    loss_cfg = dict(loss_cfg or {})
    name = loss_cfg.pop("name", "cross_entropy")
    weight = loss_cfg.pop("class_weight", None)
    weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
    if name == "cross_entropy":
        # label_smoothing 与 Mixup/CutMix 的软标签都由 CrossEntropyLoss 原生支持
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=loss_cfg.get("label_smoothing", 0.0))
    if name == "focal":
        return FocalLoss(gamma=loss_cfg.get("gamma", 2.0), weight=weight)
    raise ValueError(f"未知损失函数: {name}")
