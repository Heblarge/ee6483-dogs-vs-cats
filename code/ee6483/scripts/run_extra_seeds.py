"""Re-run existing experiment recipes with additional seeds (no other change).

    python -m ee6483.scripts.run_extra_seeds --phases ablation lrsearch augcheck --seeds 43 44
    python -m ee6483.scripts.run_extra_seeds --phases imbalance --only H_decouple H_cb H_cb_long --seeds 43 44

Commands are taken verbatim from run_experiments.PHASES; only ``--seed`` is replaced (or
appended), so the recipe has a single source of truth. Results land in ``<exp>/s<seed>/`` and
never overwrite the seed-42 runs. Post-hoc steps without ``--exp-id`` (logit adjustment) are
skipped; run eval_logit_adjust on the new CE checkpoints separately.
"""
from __future__ import annotations

import argparse

from .run_experiments import PHASES, already_done, job


def with_seed(cmd: list[str], seed: int) -> list[str]:
    cmd = list(cmd)
    if "--seed" in cmd:
        cmd[cmd.index("--seed") + 1] = str(seed)
    else:
        cmd += ["--seed", str(seed)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", choices=sorted(PHASES), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these exp ids")
    ap.add_argument("--skip", nargs="*", default=[], help="exp ids to leave out")
    args = ap.parse_args()
    rc_all = 0
    for seed in args.seeds:
        for ph in args.phases:
            for cmd, tag in PHASES[ph]:
                if "--exp-id" not in cmd:
                    continue
                exp = cmd[cmd.index("--exp-id") + 1]
                if exp in args.skip or (args.only is not None and exp not in args.only):
                    continue
                c = with_seed(cmd, seed)
                if already_done(c):
                    print(f"skip (done): {exp} seed {seed}", flush=True)
                    continue
                rc_all |= job(c, f"[{ph} seed{seed}] {tag}")
    print("\nALL DONE", rc_all)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
