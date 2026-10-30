"""训练入口。

python scripts/train.py --config configs/dogs_cats/d0_simple_cnn.yaml
python scripts/train.py --config configs/dogs_cats/d2_resnet18_frozen.yaml --set optim.lr=3e-4 --set seed=1
"""
import argparse
import math
import time

import _bootstrap  # noqa: F401
import torch

from src.data import CLASSES, build_dataloaders
from src.policies import build_model
from src.training import Trainer, build_loss, build_optimizer, build_scheduler
from src.utils import load_config, save_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[], help="覆盖配置，如 optim.lr=1e-4")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 0), cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = args.run_id or f"{cfg.get('name', 'run')}_s{cfg.get('seed', 0)}_{time.strftime('%m%d-%H%M%S')}"
    out_dir = f"{cfg.get('output_dir', 'runs')}/{run_id}"
    save_config(cfg, f"{out_dir}/config.yaml")

    num_classes = len(CLASSES)
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} device={device}")

    model = build_model(cfg["model"], num_classes)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={cfg['model']['name']} trainable_params={n_train / 1e6:.2f}M")

    train_cfg = cfg.get("train", {})
    steps_per_epoch = math.ceil(len(train_loader) / train_cfg.get("grad_accum_steps", 1))
    optimizer = build_optimizer(model, cfg.get("optim", {}))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), steps_per_epoch, train_cfg.get("epochs", 10))
    trainer = Trainer(model, build_loss(cfg.get("loss")), optimizer, scheduler, train_cfg, out_dir, num_classes, device)
    summary = trainer.fit(train_loader, val_loader, cfg)
    print(f"输出目录: {out_dir}  {summary}")


if __name__ == "__main__":
    main()
