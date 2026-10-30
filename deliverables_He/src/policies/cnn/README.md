# cnn/：从零训练的卷积网络

## simple_cnn（D0/D1 基线）

```
输入 [B,3,224,224]
└─ Block×4：Conv3×3-BN-ReLU → Conv3×3-BN-ReLU → MaxPool2×2
   通道 32 → 64 → 128 → 256；空间尺寸 224 → 112 → 56 → 28 → 14
└─ AdaptiveAvgPool → [B,256]
└─ Dropout(0.3) → Linear(256, 2) → logits [B,2]
```

- 卷积层使用 Kaiming 初始化，与 ReLU 搭配。BN 能加快收敛，并降低对学习率的敏感度。
- 使用全局平均池化而不是大全连接层，参数量很小（约 1.2M），也能接受任意输入尺寸，之后迁移到 CIFAR-10 的 32×32 输入时可以直接复用。
- 配置项：`channels`（块数与宽度）、`dropout`。报告中的结构图按实际配置画出。

**可调方向**：数据较少时减少通道数，以减轻过拟合；增加块数后感受野变大，但下采样次数随之增加。迁移到 32×32 输入时一般只保留 2 到 3 次池化。

## vgg11_bn / vgg16_bn

torchvision 实现，只替换最后一层 `classifier[6]` 为 2 类。全连接层参数很多，从零训练容易过拟合，适合作为“经典深层 CNN + 预训练”的对照，不推荐作为主模型。注意：`classifier` 是 3 层全连接（约 1.2 亿参数），`freeze: backbone` 时会训练整个 MLP，而不只是最后一层。

## 参考文献

- LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 1998.（LeNet，CNN 基本范式）
- Krizhevsky, A., Sutskever, I., Hinton, G. E. *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.（AlexNet，ReLU、Dropout、数据增强）
- Simonyan, K., Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015. arXiv:1409.1556（VGG）
- Ioffe, S., Szegedy, C. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015. arXiv:1502.03167
- He, K., Zhang, X., Ren, S., Sun, J. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* ICCV 2015. arXiv:1502.01852（Kaiming 初始化）
- Lin, M., Chen, Q., Yan, S. *Network In Network.* ICLR 2014. arXiv:1312.4400（全局平均池化）
- Srivastava, N. et al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15, 2014.
