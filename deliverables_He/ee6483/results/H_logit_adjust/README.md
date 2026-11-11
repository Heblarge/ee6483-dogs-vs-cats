# H_logit_adjust/s42

`metrics.json` is the result reported in Section 9.4 (test accuracy 72.99%, macro-F1 72.96%, tau = 1.0).
It is written by `ee6483/scripts/eval_logit_adjust.py`, which loads the `H_ce/s42` checkpoint, sweeps
tau in {0, 0.25, 0.5, 0.75, 1.0, 1.5} on the validation split, and applies the selected tau once to
the test set. There is no separate training run; the `history` field is copied from `H_ce`.

`train.log` in this folder is **superseded**. It comes from an earlier in-training implementation
(`train_cifar.py --method logit_adjust`), whose per-epoch history is identical to `H_ce` but whose
final test accuracy is 51.56%. With the current code (logits minus tau·log prior, the formula in
Section 9.3.5) the same checkpoint and tau = 1 give 72.99%, so the 51.56% almost certainly reflects the
sign of the adjustment used at the time. The git history starts after that fix, so this cannot be
confirmed from the repository. The file is kept only as a record and should not be cited.
