| exp | seed | model | img | aug | ft | n_train | epochs | val_acc | macro_f1 | test_acc | sec | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2_smallcnn | 42 | smallcnn | 128 | full | scratch | 20000 | 14 | 95.06 |  |  | 454 | E2_smallcnn\s42 |
| E3_r18_frozen | 42 | resnet18 | 224 | full | head | 20000 | 6 | 97.78 |  |  | 334 | E3_r18_frozen\s42 |
| E4_r18_ft | 42 | resnet18 | 224 | full | lpft | 20000 | 8 | 99.0 |  |  | 522 | E4_r18_ft\s42 |
| E4_r18_ft | 43 | resnet18 | 224 | full | lpft | 20000 | 8 | 98.9 |  |  | 498 | E4_r18_ft\s43 |
| E4_r18_ft | 44 | resnet18 | 224 | full | lpft | 20000 | 8 | 98.9 |  |  | 459 | E4_r18_ft\s44 |
| E5_r34_ft | 42 | resnet34 | 224 | full | lpft | 20000 | 4 | 97.54 |  |  | 287 | E5_r34_ft\s42 |
| E6_r50_ft | 42 | resnet50 | 160 | full | lpft | 20000 | 6 | 99.3 |  |  | 318 | E6_r50_ft\s42 |
| E7_aug_color | 42 | resnet18 | 160 | color | full | 20000 | 3 | 98.48 |  |  | 149 | E7_aug_color\s42 |
| E7_aug_flip | 42 | resnet18 | 160 | flip | full | 20000 | 3 | 98.62 |  |  | 134 | E7_aug_flip\s42 |
| E7_aug_full | 42 | resnet18 | 160 | full | full | 20000 | 3 | 98.12 |  |  | 177 | E7_aug_full\s42 |
| E7_aug_geo | 42 | resnet18 | 160 | geo | full | 20000 | 3 | 98.56 |  |  | 120 | E7_aug_geo\s42 |
| E7_aug_none | 42 | resnet18 | 160 | none | full | 20000 | 3 | 98.36 |  |  | 132 | E7_aug_none\s42 |
| E7_data_1000 | 42 | resnet18 | 160 | full | full | 2000 | 3 | 96.76 |  |  | 96 | E7_data_1000\s42 |
| E7_data_250 | 42 | resnet18 | 160 | full | full | 500 | 3 | 93.56 |  |  | 89 | E7_data_250\s42 |
| E7_data_2500 | 42 | resnet18 | 160 | full | full | 5000 | 3 | 97.9 |  |  | 105 | E7_data_2500\s42 |
| E7_data_5000 | 42 | resnet18 | 160 | full | full | 10000 | 3 | 98.2 |  |  | 117 | E7_data_5000\s42 |
| E7_ft_head | 42 | resnet18 | 160 | full | head | 20000 | 3 | 96.24 |  |  | 154 | E7_ft_head\s42 |
| E7_ft_last | 42 | resnet18 | 160 | full | last_stage | 20000 | 3 | 98.2 |  |  | 165 | E7_ft_last\s42 |
| E7_ft_lpft | 42 | resnet18 | 160 | full | lpft | 20000 | 3 | 98.32 |  |  | 174 | E7_ft_lpft\s42 |
| E7_lr_1e-3 | 42 | resnet18 | 160 | full | full | 20000 | 3 | 97.7 |  |  | 207 | E7_lr_1e-3\s42 |
| E7_lr_1e-4 | 42 | resnet18 | 160 | full | full | 20000 | 3 | 98.62 |  |  | 210 | E7_lr_1e-4\s42 |
| E7_lr_1e-5 | 42 | resnet18 | 160 | full | full | 20000 | 3 | 97.68 |  |  | 215 | E7_lr_1e-5\s42 |
| E7_lr_3e-3 | 42 | resnet18 | 160 | full | full | 20000 | 3 | 96.06 |  |  | 207 | E7_lr_3e-3\s42 |
| E7_lr_3e-4 | 42 | resnet18 | 160 | full | full | 20000 | 3 | 98.24 |  |  | 212 | E7_lr_3e-4\s42 |
| E7_res_128 | 42 | resnet18 | 128 | full | full | 20000 | 3 | 97.94 |  |  | 167 | E7_res_128\s42 |
| E7_res_224 | 42 | resnet18 | 224 | full | full | 20000 | 3 | 98.76 |  |  | 235 | E7_res_224\s42 |
| E7_res_96 | 42 | resnet18 | 96 | full | full | 20000 | 3 | 97.08 |  |  | 148 | E7_res_96\s42 |
| E7_scratch | 42 | resnet18 | 160 | full | scratch | 20000 | 6 | 90.66 |  |  | 341 | E7_scratch\s42 |
| E7b_aug_full_8ep | 42 | resnet18 | 160 | full | full | 20000 | 8 | 98.66 |  |  | 323 | E7b_aug_full_8ep\s42 |
| E7b_aug_none_8ep | 42 | resnet18 | 160 | none | full | 20000 | 8 | 98.42 |  |  | 241 | E7b_aug_none_8ep\s42 |
| E7c_aug_full_lowdata | 42 | resnet18 | 160 | full | full | 500 | 8 | 96.3 |  |  | 88 | E7c_aug_full_lowdata\s42 |
| E7c_aug_none_lowdata | 42 | resnet18 | 160 | none | full | 500 | 8 | 95.94 |  |  | 87 | E7c_aug_none_lowdata\s42 |
| G1_cifar_scratch | 42 | resnet18-cifar | 32 | - | ce | 45000 | 25 | 93.44 | 93.6 | 93.6 | 1540 | G1_cifar_scratch\s42 |
| G2_cifar_pretrained | 42 | resnet18-cifar | 160 | - | ce | 45000 | 7 | 95.58 | 95.19 | 95.2 | 579 | G2_cifar_pretrained\s42 |
| H_cb | 42 | resnet18-cifar | 32 | - | cb | 11908 | 18 | 54.02 | 53.91 | 54.04 | 658 | H_cb\s42 |
| H_cb_long | 42 | resnet18-cifar | 32 | - | cb | 11908 | 30 | 59.22 | 59.48 | 59.5 | 1069 | H_cb_long\s42 |
| H_ce | 42 | resnet18-cifar | 32 | - | ce | 11908 | 18 | 62.54 | 60.27 | 62.44 | 809 | H_ce\s42 |
| H_decouple | 42 | resnet18-cifar | 32 | - | decouple | 11908 | 18 | 71.06 | 71.14 | 71.22 | 679 | H_decouple\s42 |
| H_logit_adjust | 42 | resnet18-cifar | 32 | - | logit_adjust | 11908 | 18 | 73.72 | 72.96 | 72.99 | 0 | H_logit_adjust\s42 |
| H_resample | 42 | resnet18-cifar | 32 | - | resample | 11908 | 18 | 72.26 | 72.0 | 72.02 | 726 | H_resample\s42 |