# H_logit_adjust/s42

`metrics.json` is the result reported in Section 9.4 (test accuracy 72.99%, macro-F1 72.96%, tau = 1.0).
It is written by `ee6483/scripts/eval_logit_adjust.py`, which loads the `H_ce/s42` checkpoint, sweeps
tau in {0, 0.25, 0.5, 0.75, 1.0, 1.5} on the validation split, and applies the selected tau once to
the test set. There is no separate training run; the `history` field is copied from `H_ce`.

`s42/train.log` is **superseded** and kept only as a record; do not cite it. It comes from an earlier
in-training implementation (`train_cifar.py --method logit_adjust`, also logged in
`results/run_log.txt`). Its per-epoch history is identical to `H_ce`, but its test accuracy is 51.56%.
With the current code (logits minus tau·log prior, Section 9.3.5) the same recipe gives 72.99%, and the
2026-11-11 recheck gives 73.81% (`docs/RECHECK_2026-11-11.md`), so the 51.56% most likely reflects the
sign of the adjustment used at the time. The git history starts after that fix, so this cannot be
confirmed from the repository.
