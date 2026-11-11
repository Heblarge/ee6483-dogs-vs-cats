# H_logit_adjust/s42

`metrics.json` is the result reported in Section 9.4 (test accuracy 72.99%, macro-F1 72.96%, tau = 1.0).
It is written by `ee6483/scripts/eval_logit_adjust.py`, which loads the `H_ce/s42` checkpoint, sweeps
tau in {0, 0.25, 0.5, 0.75, 1.0, 1.5} on the validation split, and applies the selected tau once to
the test set. There is no separate training run; the `history` field is copied from `H_ce`.

The independent 2026-11-11 recheck is documented in `docs/RECHECK_2026-11-11.md`.
There is no separate training log or checkpoint for this post-hoc method.
