"""图像预处理与数据增强。随机增强只用于训练集；验证/测试使用固定的 Resize + CenterCrop。

配置示例（configs/*.yaml 中的 data 段）：
    img_size: 224
    normalize: imagenet          # imagenet | none
    augment:
      random_resized_crop: {scale: [0.6, 1.0]}
      hflip: 0.5
      color_jitter: {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.0}
      rotation: 10
      randaugment: {num_ops: 2, magnitude: 9}   # 与 trivialaugment 二选一
      trivialaugment: true
      random_erasing: 0.25
"""
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(data_cfg):
    if data_cfg.get("normalize", "imagenet") == "imagenet":
        return [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return []


def build_transforms(data_cfg, train):
    size = data_cfg.get("img_size", 224)
    aug = data_cfg.get("augment") or {}
    if not train:
        # 与 ImageNet 评估惯例一致：短边缩放到 size/0.875 后中心裁剪
        resize = int(round(size / data_cfg.get("crop_ratio", 0.875)))
        return T.Compose([T.Resize(resize), T.CenterCrop(size), T.ToTensor(), *_normalize(data_cfg)])

    ops = []
    if "random_resized_crop" in aug:
        rrc = aug["random_resized_crop"] or {}
        ops.append(T.RandomResizedCrop(size, scale=tuple(rrc.get("scale", (0.6, 1.0)))))
    else:
        ops += [T.Resize(size), T.CenterCrop(size)]
    if aug.get("hflip"):
        ops.append(T.RandomHorizontalFlip(aug["hflip"]))
    if aug.get("rotation"):
        ops.append(T.RandomRotation(aug["rotation"]))
    if aug.get("color_jitter"):
        ops.append(T.ColorJitter(**aug["color_jitter"]))
    if aug.get("randaugment"):
        ops.append(T.RandAugment(**(aug["randaugment"] if isinstance(aug["randaugment"], dict) else {})))
    elif aug.get("trivialaugment"):
        ops.append(T.TrivialAugmentWide())
    ops += [T.ToTensor(), *_normalize(data_cfg)]
    if aug.get("random_erasing"):
        ops.append(T.RandomErasing(p=aug["random_erasing"]))
    return T.Compose(ops)
