"""训练循环：forward -> loss -> backward -> 梯度裁剪 -> optimizer.step -> scheduler.step。

支持 AMP 混合精度、梯度累积、Mixup/CutMix、按验证指标保存最佳 checkpoint 与早停。
"""
import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from ..data.mix import MixAugment
from ..evaluation.metrics import evaluate_model


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_cfg, out_dir, num_classes, device):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_dir = Path(out_dir)
        self.epochs = train_cfg.get("epochs", 10)
        self.grad_clip = train_cfg.get("grad_clip_norm")          # None 表示不裁剪
        self.accum_steps = train_cfg.get("grad_accum_steps", 1)
        self.monitor = train_cfg.get("monitor", "accuracy")       # 选择 checkpoint 的验证指标
        self.patience = train_cfg.get("early_stop_patience")      # None 表示不早停
        self.use_amp = train_cfg.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        mix = train_cfg.get("mix") or {}
        self.mix = MixAugment(num_classes, **mix)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.mix.enabled:
                images, targets = self.mix(images, targets)

            # forward
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets) / self.accum_steps

            # backward（AMP 下先对 loss 缩放，防止 fp16 梯度下溢）
            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0 or step + 1 == len(loader):
                # 梯度裁剪：必须先 unscale，裁剪阈值才对应真实梯度范数
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            bs = images.size(0)
            total_loss += loss.item() * self.accum_steps * bs
            n += bs
            pbar.set_postfix(loss=total_loss / n, lr=self.optimizer.param_groups[0]["lr"])
        return total_loss / n

    def save_checkpoint(self, path, epoch, metrics, cfg):
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "metrics": metrics, "config": cfg}, path)

    def fit(self, train_loader, val_loader, cfg):
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.out_dir / "history.csv"
        best, best_epoch, bad_epochs = -float("inf"), -1, 0
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr", "seconds"])
            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                train_loss = self.train_one_epoch(train_loader, epoch)
                val = evaluate_model(self.model, val_loader, self.device, self.criterion, amp=self.use_amp)
                lr = self.optimizer.param_groups[0]["lr"]
                writer.writerow([epoch, f"{train_loss:.4f}", f"{val['loss']:.4f}", f"{val['accuracy']:.4f}",
                                 f"{val['macro_f1']:.4f}", f"{lr:.2e}", f"{time.time() - t0:.1f}"])
                f.flush()
                print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
                      f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f}")

                self.save_checkpoint(ckpt_dir / "last.pt", epoch, val, cfg)
                if val[self.monitor] > best:
                    best, best_epoch, bad_epochs = val[self.monitor], epoch, 0
                    self.save_checkpoint(ckpt_dir / "best.pt", epoch, val, cfg)
                else:
                    bad_epochs += 1
                    if self.patience and bad_epochs >= self.patience:
                        print(f"早停：{self.patience} 个 epoch 未提升")
                        break
        print(f"最佳 {self.monitor}={best:.4f} @ epoch {best_epoch}")
        return {"best_" + self.monitor: best, "best_epoch": best_epoch}
