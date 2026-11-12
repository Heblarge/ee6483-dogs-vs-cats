# splits/：固定样本清单

存放 CSV 清单（`relpath,label`，路径相对于数据根目录），**只存清单，不存图像**，需要提交到 Git，保证组员使用同一批样本。

```bash
python scripts/make_split.py --split train --per-class 1000 --seed 0 --out splits/train_1k_s0.csv
python scripts/make_split.py --split val   --per-class 250  --seed 0 --out splits/val_250_s0.csv
```

在配置中引用：`data.train_split: splits/train_1k_s0.csv`、`data.val_split: splits/val_250_s0.csv`。不设置时使用目录下的全量数据。
