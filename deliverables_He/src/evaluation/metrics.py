"""分类指标：accuracy、混淆矩阵、各类 precision/recall/F1、macro-F1。只依赖 numpy。"""
import numpy as np
import torch


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm  # 行：真实类别，列：预测类别


def classification_metrics(y_true, y_pred, num_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_logits(model, loader, device, amp=False, tta_hflip=False):
    """返回 (logits [N, C], targets 列表)。targets 对验证集是标签，对测试集是 id。"""
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
            if tta_hflip:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        all_logits.append(logits.float().cpu())
        all_targets += list(targets.tolist() if torch.is_tensor(targets) else targets)
    return torch.cat(all_logits), all_targets


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None, amp=False, tta_hflip=False):
    logits, targets = predict_logits(model, loader, device, amp, tta_hflip)
    y_true = torch.tensor(targets)
    metrics = classification_metrics(y_true.numpy(), logits.argmax(1).numpy(), logits.size(1))
    if criterion is not None:
        metrics["loss"] = float(torch.nn.functional.cross_entropy(logits, y_true))
    metrics["probs"] = torch.softmax(logits, 1)
    return metrics
