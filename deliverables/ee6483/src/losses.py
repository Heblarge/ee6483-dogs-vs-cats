"""Loss functions used in the class-imbalance study (Part H)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLS(nn.Module):
    """Cross entropy with optional label smoothing (standard baseline)."""

    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.ls = label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.ls)


class FocalLoss(nn.Module):
    """Lin et al. 2017. Down-weights easy examples; helps when easy majority
    samples dominate the gradient."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean() if self.reduction == "mean" else loss


class ClassBalancedLoss(nn.Module):
    """Cui et al., CVPR 2019 -- effective number of samples.

        E_n = (1 - beta^n) / (1 - beta),  weight_i propto 1 / E_{n_i}

    beta -> 1 recovers inverse-frequency weighting; beta -> 0 recovers plain CE.
    """

    def __init__(self, class_counts, beta: float = 0.9999, label_smoothing: float = 0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        eff = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
        w = (1.0 / eff)
        w = w / w.sum() * len(counts)  # keep the mean weight at 1 -> lr stays comparable
        self.register_buffer("weight", w)
        self.ls = label_smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.ls)


class LDAMLoss(nn.Module):
    """Cao et al., NeurIPS 2019 -- label-distribution-aware margin.

    Margin per class m_i propto n_i^(-1/4); the margin shrinks as a class gets
    more samples, so rare classes are pushed further from the decision boundary.
    """

    def __init__(self, class_counts, max_m: float = 0.5, weight: torch.Tensor | None = None, s: float = 30.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        m_list = 1.0 / torch.sqrt(torch.sqrt(counts))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.s = s
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list[target]
        x_m = logits - batch_m[:, None] * index.float()
        output = torch.where(index, x_m, logits)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def logit_adjust(logits: torch.Tensor, class_counts, tau: float = 1.0) -> torch.Tensor:
    """Menon et al., ICLR 2021 -- post-hoc logit adjustment.

    A model trained with plain cross-entropy on an imbalanced set outputs
    scores that are shifted by the training prior. To obtain the Bayes-optimal
    predictor for a *balanced* test distribution you therefore SUBTRACT
    tau*log(prior) from the logits:

        argmax_c  f_c(x) - tau * log pi_c

    (Adding the term is the correct move only when the adjustment is folded
    into the *training* loss instead, which is a different estimator.)
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=logits.device)
    prior = counts / counts.sum()
    return logits - tau * torch.log(prior + 1e-12)


def balanced_softmax_loss(logits, target, class_counts, tau: float = 1.0):
    """Training-time variant: the prior term is added inside the loss."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=logits.device)
    prior = counts / counts.sum()
    return F.cross_entropy(logits + tau * torch.log(prior + 1e-12), target)


def make_sampler_weights(targets, q: float = 1.0):
    """Sampling probability p_j propto n_j^q.  q=0 -> class balanced, q=1 -> natural."""
    counts = torch.as_tensor(targets).bincount()
    per_sample = counts[torch.as_tensor(targets)].float() ** q
    return per_sample / per_sample.sum()
