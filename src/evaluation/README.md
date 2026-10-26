# evaluation/：模型测试与结果导出

| 文件 | 内容 |
|---|---|
| `metrics.py` | accuracy、混淆矩阵、各类 precision / recall / F1、macro-F1；批量推理（可选水平翻转 TTA） |
| `predict.py` | 测试集预测，导出 `submission.csv`（检查 id 是否重复、数量是否为 500） |

入口脚本：

```bash
# 验证集评估：输出 val_metrics.json 与误分类清单 val_misclassified.csv
python scripts/evaluate.py --ckpt runs/<run_id>/checkpoints/best.pt [--tta]

# 测试集预测：输出 submission.csv 和带 p_dog 概率的 submission_probs.csv
python scripts/predict.py --ckpt runs/<run_id>/checkpoints/best.pt --out submissions/submission.csv [--tta]
```

checkpoint 内保存了完整配置，评估和预测会自动使用训练时的模型结构与预处理。

## 评估约定

- **验证集**用于选择超参数和 checkpoint，报告“验证准确率”。
- **测试集**没有标签，只用于生成提交文件，**不能**用来调参，也不能报告测试准确率。
- `submission.csv` 默认列名为 `id,label`（0=cat, 1=dog），按 id 升序排列。**提交前对照课程样例核对列名和格式**；如有不同，修改 `write_submission` 的 `columns` 参数。
- 比较实验时同时报告：最佳 epoch 的验证 accuracy、loss，以及训练/验证曲线（来自 `history.csv`），判断是否过拟合。
- 关键对照建议用 3 个种子重复，报告均值 ± 标准差。

## 样例分析（评分 10%）

1. 用 `val_misclassified.csv` 找出验证集上的错误样本，按 `p_dog` 排序：概率接近 0 或 1 的是“自信地错”，接近 0.5 的是“犹豫”。
2. 用 `submission_probs.csv` 找出测试集上置信度低的样本，人工查看图像并标注判断依据（如多只动物、主体过小、遮挡、卡通图、标签本身可疑等）。
3. 选 1~2 个案例深入分析，可借助 Grad-CAM 可视化模型关注的区域（`pip install grad-cam`，ResNet 的 target layer 为 `model.layer4[-1]`）。
4. 可选：计算 ECE / 画可靠性曲线，讨论模型是否过度自信，以及 label smoothing、Mixup 对校准的影响。

## 参考文献

- Sokolova, M., Lapalme, G. *A systematic analysis of performance measures for classification tasks.* Information Processing & Management, 45(4), 2009.
- Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017. arXiv:1610.02391
- Zeiler, M. D., Fergus, R. *Visualizing and Understanding Convolutional Networks.* ECCV 2014. arXiv:1311.2901
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. *On Calibration of Modern Neural Networks.* ICML 2017. arXiv:1706.04599
- Northcutt, C. G., Athalye, A., Mueller, J. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS 2021 Datasets and Benchmarks. arXiv:2103.14749（标签噪声，分析可疑样本时可引用）
- Gildenblat, J. *pytorch-grad-cam.* https://github.com/jacobgil/pytorch-grad-cam
