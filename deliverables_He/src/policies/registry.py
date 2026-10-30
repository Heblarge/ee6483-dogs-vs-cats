"""模型注册表。每个模型族在自己的子目录用 @register("name") 注册构建函数。

构建函数签名：fn(num_classes, pretrained, **kwargs) -> (model, head_name)
head_name 是分类头的参数名前缀，用于冻结骨干和给分类头设置单独学习率。
"""
MODEL_REGISTRY = {}


def register(name):
    def deco(fn):
        if name in MODEL_REGISTRY:
            raise KeyError(f"模型 {name} 重复注册")
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


def _apply_freeze(model, head_name, freeze, unfreeze_prefixes):
    """freeze: none | backbone。backbone 时只训练分类头和 unfreeze_prefixes 指定的层。"""
    if freeze in (None, "none", False):
        return
    if freeze != "backbone":
        raise ValueError(f"未知的 freeze 设置: {freeze}")
    trainable = (head_name, *unfreeze_prefixes)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(trainable)


def build_model(model_cfg, num_classes):
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}，可选: {sorted(MODEL_REGISTRY)}")
    freeze = cfg.pop("freeze", "none")
    unfreeze = tuple(cfg.pop("unfreeze_prefixes", ()))
    pretrained = cfg.pop("pretrained", False)
    model, head_name = MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **cfg)
    model.head_name = head_name
    _apply_freeze(model, head_name, freeze, unfreeze)
    return model
