"""Grad-CAM (Selvaraju et al., ICCV 2017) for the error-analysis section.

Usage: python -m ee6483.src.gradcam --ckpt ... --split val --n 8 --out figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data.dvc_dataset import build_eval_loader
from .models.backbones import build_backbone
from .models.small_cnn import SmallCNN
from .utils import get_device
from .data.transforms import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx[:, None]).sum()
        score.backward(retain_graph=False)
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam, logits


def target_layer_of(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    return list(model.features.children())[-2]


def denormalise(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img, cam, alpha=0.45):
    import matplotlib.cm as cm

    heat = cm.jet(cam)[..., :3]
    return (1 - alpha) * img + alpha * heat


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = SmallCNN() if a["model"] == "smallcnn" else build_backbone(a["model"], 2, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data/cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="ee6483/figures")
    args = ap.parse_args()

    device = get_device()
    model, a = load(args.ckpt, device)
    loader, base = build_eval_loader(f"{args.data_root}/{args.split}", a["img_size"], 64, 2)
    cam_fn = GradCAM(model, target_layer_of(model))

    # Gather confident-correct and wrong samples, balanced across the two classes.
    # The loader visits the cat folder before the dog folder, so taking the first
    # n samples of each outcome would return only cats.
    half = max(1, args.n // 2)
    keep = half * 8
    pools = {"correct": {0: [], 1: []}, "wrong": {0: [], 1: []}}
    for imgs, targets in loader:
        imgs = imgs.to(device)
        cams, logits = cam_fn(imgs)
        pred = logits.argmax(1)
        conf = F.softmax(logits.detach(), 1).max(1).values
        for i in range(imgs.size(0)):
            t = int(targets[i])
            rec = (imgs[i].detach(), cams[i].detach(), t, int(pred[i]), float(conf[i]))
            key = "correct" if t == int(pred[i]) else "wrong"
            if len(pools[key][t]) < keep:      # keep more than we need, then rank
                pools[key][t].append(rec)
        if all(len(pools[k][c]) >= keep for k in pools for c in (0, 1)):
            break

    def top(pool):
        # most confident first, so the panels show the clearest examples
        return sorted(pool, key=lambda r: -r[4])[:half]

    correct = top(pools["correct"][0]) + top(pools["correct"][1])
    wrong = top(pools["wrong"][0]) + top(pools["wrong"][1])

    names = {0: "cat", 1: "dog"}
    for tag, pool in [("correct", correct[: args.n]), ("wrong", wrong[: args.n])]:
        if not pool:
            continue
        pairs = 2                      # image+CAM pairs per row
        rows = (len(pool) + pairs - 1) // pairs
        fig, axes = plt.subplots(rows, pairs * 2, figsize=(6.4, 1.95 * rows))
        axes = np.atleast_2d(axes)
        for k, (img, cam, t, p, c) in enumerate(pool):
            r, cc = divmod(k, pairs)
            base_img = denormalise(img)
            axes[r, cc * 2].imshow(base_img)
            axes[r, cc * 2].set_title(f"true={names[t]}  pred={names[p]}  p={c:.2f}", fontsize=7)
            axes[r, cc * 2].axis("off")
            axes[r, cc * 2 + 1].imshow(overlay(base_img, cam.cpu().numpy()))
            axes[r, cc * 2 + 1].set_title("Grad-CAM", fontsize=7)
            axes[r, cc * 2 + 1].axis("off")
        fig.tight_layout()
        out = Path(args.out) / f"gradcam_{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
