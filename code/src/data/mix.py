"""Batch 级增强 Mixup / CutMix。输出软标签，配合 CrossEntropyLoss（支持概率目标）使用。"""
import numpy as np
import torch
import torch.nn.functional as F


class MixAugment:
    def __init__(self, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    @property
    def enabled(self):
        return self.mixup_alpha > 0 or self.cutmix_alpha > 0

    def __call__(self, images, targets):
        soft = F.one_hot(targets, self.num_classes).float()
        if not self.enabled or np.random.rand() > self.prob:
            return images, soft
        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or np.random.rand() < self.switch_prob)
        lam = np.random.beta(*(2 * [self.cutmix_alpha if use_cutmix else self.mixup_alpha]))
        perm = torch.randperm(images.size(0), device=images.device)
        if use_cutmix:
            h, w = images.shape[-2:]
            cut_h, cut_w = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
            cy, cx = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
            x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
            images = images.clone()
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
        else:
            images = lam * images + (1 - lam) * images[perm]
        return images, lam * soft + (1 - lam) * soft[perm]
