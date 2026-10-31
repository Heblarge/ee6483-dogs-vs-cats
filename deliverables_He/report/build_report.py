"""Assemble report/sections/*.md into a single REPORT.md, in a fixed order.

Tokens of the form {{NAME}} are replaced from report/tokens.json so that numbers
quoted in the prose always match the latest metrics.json (no hand-copied digits).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ORDER = [
    "00_front.md",
    "01_survey.md",
    "02_data.md",
    "03_method.md",
    "04_hyperparams.md",
    "05_results.md",
    "06_error.md",
    "07_ablation.md",
    "08_cifar.md",
    "09_imbalance.md",
    "10_conclusion.md",
    "11_refs.md",
    "12_appendix.md",
]

ROOT = Path("report")
OUT = ROOT / "REPORT.md"


def main():
    tokens = {}
    tf = ROOT / "tokens.json"
    if tf.exists():
        tokens = json.loads(tf.read_text(encoding="utf-8"))

    parts = []
    for name in ORDER:
        p = ROOT / "sections" / name
        if not p.exists():
            print(f"  (missing, skipped) {name}")
            continue
        txt = p.read_text(encoding="utf-8")
        txt = re.sub(r"\{\{(\w+)\}\}", lambda m: str(tokens.get(m.group(1), m.group(0))), txt)
        parts.append(txt.strip())
        print(f"  + {name} ({len(txt)} chars)")

    body = "\n\n<div style=\"page-break-after: always;\"></div>\n\n".join(parts)

    # number the figures in order of appearance (idempotent: skips already numbered)
    import itertools

    counter = itertools.count(1)

    def _num(m):
        # always strip any existing number first, so adding or removing a figure
        # renumbers the rest instead of leaving stale captions behind
        alt = re.sub(r"^Figure \d+\.\s*", "", m.group(1))
        return f"![Figure {next(counter)}. {alt}]"

    body = re.sub(r"!\[([^\]]*)\]", _num, body)
    OUT.write_text(body, encoding="utf-8", newline="\n")
    left = re.findall(r"\{\{\w+\}\}", body)
    print(f"\nwrote {OUT}  ({len(body)} chars, ~{len(body.split())} words)")
    if left:
        print("UNRESOLVED TOKENS:", sorted(set(left)))


if __name__ == "__main__":
    main()
