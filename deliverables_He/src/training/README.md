# training/：训练流程

| 文件 | 内容 |
|---|---|
| `trainer.py` | `Trainer`：forward / backward / 梯度裁剪 / 参数更新 / 验证 / checkpoint / 早停 |
| `optim.py` | 优化器（SGD / Adam / AdamW）、参数分组、学习率调度（warmup + cosine / step） |
| `losses.py` | 交叉熵（支持 label smoothing、类别权重、软标签）、Focal Loss |

入口脚本：`python scripts/train.py --config configs/dogs_cats/<exp>.yaml [--set key=value ...]`

## 单步训练流程（`Trainer.train_one_epoch`）

```
images, targets ──(Mixup/CutMix, 可选)──▶ forward: logits = model(images)       [AMP autocast]
                                         loss = criterion(logits, targets) / accum_steps
backward:        scaler.scale(loss).backward()                                  [AMP 下先放大 loss，防止 fp16 下溢]
每 accum_steps 步：
  梯度裁剪:      scaler.unscale_(optimizer); clip_grad_norm_(params, max_norm)  [必须先 unscale]
  参数更新:      scaler.step(optimizer); scaler.update()
  清空梯度:      optimizer.zero_grad(set_to_none=True)
  学习率:        scheduler.step()                                               [按 iteration 更新]
```

每个 epoch 结束后在验证集评估，记录到 `runs/<run_id>/history.csv`，保存 `last.pt`；`train.monitor` 指标（默认 accuracy）创新高时另存为 `best.pt`。如果设置了 `early_stop_patience`，指标连续这么多个 epoch 没有提升就停止训练。

## 配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `optim.name` | `adamw` | `sgd`（momentum 0.9、nesterov）/ `adam` / `adamw` |
| `optim.lr` | 1e-3 | 分类头学习率 |
| `optim.backbone_lr_mult` | 1.0 | 骨干学习率倍数；微调预训练模型建议 0.1 |
| `optim.weight_decay` | 0 | bias 与 Norm 参数自动不做衰减 |
| `scheduler.name` | `cosine` | `cosine` / `step` / `constant` |
| `scheduler.warmup_epochs` | 0 | 线性 warmup，可为小数 |
| `scheduler.min_lr_ratio` | 0.01 | cosine 终点 = lr × 该值 |
| `train.epochs` | 10 | |
| `train.amp` | true | 混合精度，仅 CUDA 下生效 |
| `train.grad_clip_norm` | null | 全局梯度 L2 范数上限；常用 1.0~5.0 |
| `train.grad_accum_steps` | 1 | 梯度累积；等效 batch = batch_size × 该值 |
| `train.mix` | 无 | `{mixup_alpha, cutmix_alpha, prob, switch_prob}` |
| `train.monitor` | accuracy | `accuracy` 或 `macro_f1` |
| `loss.name` | cross_entropy | `cross_entropy` / `focal` |
| `loss.label_smoothing` | 0 | 常用 0.1 |
| `loss.class_weight` | null | 类别权重列表（为后续不平衡实验预留） |

## 各组件的选择依据

- **AdamW vs SGD**：AdamW 对学习率不敏感，收敛快，适合短周期实验和 Transformer。SGD+momentum 在 CNN 从零训练时最终精度常常更好，但需要更仔细地调学习率。AdamW 把 weight decay 与梯度更新解耦，比 Adam 加 L2 正则更合理。
- **Warmup + Cosine**：训练初期梯度统计不稳定（尤其是 Adam 的二阶矩估计和大 batch），warmup 可以避免早期发散。cosine 无需手动设定衰减节点。
- **梯度裁剪**：限制全局梯度范数，防止个别 batch 产生梯度爆炸。对 ViT 微调和较大学习率尤其有用；CNN 小学习率时影响不大，可作为一项消融对照。
- **差异化学习率**：预训练骨干已经接近好的解，学习率过大会破坏已有特征（灾难性遗忘）。新初始化的分类头需要较大学习率。
- **Label smoothing**：防止 logits 无限增大、模型过度自信，通常带来小幅精度提升，同时改善校准。
- **AMP**：显存约减半、速度提升明显；`GradScaler` 处理 fp16 的数值范围问题。
- **学习率选择**：先在小子集上用 2~3 个候选值（如 1e-3、3e-4、1e-4）各跑几个 epoch，比较验证曲线再决定。可用 LR range test 快速定位合理区间。

## 参考文献

**优化器**
- Robbins, H., Monro, S. *A Stochastic Approximation Method.* Annals of Mathematical Statistics, 22(3), 1951.
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. *On the importance of initialization and momentum in deep learning.* ICML 2013.
- Kingma, D. P., Ba, J. *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980
- Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101（AdamW）
- Wilson, A. C. et al. *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* NeurIPS 2017. arXiv:1705.08292（SGD 与自适应方法的泛化对比）

**学习率调度**
- Loshchilov, I., Hutter, F. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. arXiv:1608.03983（cosine 退火）
- Goyal, P. et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677, 2017.（线性缩放规则与 warmup）
- Smith, L. N. *Cyclical Learning Rates for Training Neural Networks.* WACV 2017. arXiv:1506.01186（LR range test）
- Smith, L. N., Topin, N. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.* 2019. arXiv:1708.07120（OneCycle）
- Howard, J., Ruder, S. *Universal Language Model Fine-tuning for Text Classification.* ACL 2018. arXiv:1801.06146（差异化学习率、逐层解冻）

**梯度裁剪与数值稳定**
- Pascanu, R., Mikolov, T., Bengio, Y. *On the difficulty of training recurrent neural networks.* ICML 2013. arXiv:1211.5063
- Zhang, J., He, T., Sra, S., Jadbabaie, A. *Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity.* ICLR 2020. arXiv:1905.11881
- Micikevicius, P. et al. *Mixed Precision Training.* ICLR 2018. arXiv:1710.03740

**损失与正则化**
- Szegedy, C. et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. arXiv:1512.00567（label smoothing）
- Müller, R., Kornblith, S., Hinton, G. *When Does Label Smoothing Help?* NeurIPS 2019. arXiv:1906.02629
- Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV 2017. arXiv:1708.02002
- Prechelt, L. *Early Stopping — But When?* In: Neural Networks: Tricks of the Trade, Springer, 1998.
- He, T. et al. *Bag of Tricks for Image Classification with Convolutional Neural Networks.* CVPR 2019. arXiv:1812.01187

**官方文档**
- [torch.optim](https://docs.pytorch.org/docs/stable/optim.html)、[Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)、[clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
