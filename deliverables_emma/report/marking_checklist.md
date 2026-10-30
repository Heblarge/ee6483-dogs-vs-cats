# Marking Checklist — trace every mark to evidence

Use this before submission. Each row must be *demonstrably* answered in the report.

| Item | Marks | Question | Where answered | Evidence / artefact | Done |
|---|---|---|---|---|---|
| a-1 | 20% (shared) | Problem definition; supervised vs unsupervised; closed vs open-set; with/without domain shift | §1, §2.1 | — | ☐ |
| a-2 | ↑ | Recent developments, dozens of papers skimmed | §2.2 | reference list | ☐ |
| a-3 | ↑ | 1–2 leading groups, in-depth reading of selected papers | §2.3 | — | ☐ |
| a-4 | ↑ | Which approach is the baseline and what improvements are proposed | §2.4 | — | ☐ |
| b-1 | 5% | Amount of training/testing data, pre-processing, augmentation | §3 | Table 1 | ☐ |
| b-2 | 15% | Model architecture figure, input/output dims, structure, loss, training strategy, **code + run instructions** | §4 | arch figure, README.md | ☐ |
| c | 5% | How parameters were chosen **and why** | §5 | hp table | ☐ |
| d | 15% | Validation accuracy + apply to test + `submission.csv` | §6 | `submission.csv`, `runs/*/result.json` | ☐ |
| e | 10% | Analyse 1–2 correctly **and** incorrectly classified samples (test set) | §7 | wrong/right grids (val, explained) | ☐ |
| f | 10% | How model choice and data processing affect validation accuracy | §8 | Table 2, compare.png | ☐ |
| g | 10% | CIFAR-10: dataset description, adaptation, test results | §9 | `runs/cifar10/result.json`, CM | ☐ |
| h | 10% | ≥2 approaches for class imbalance, described and justified | §10 | Table 3, per-class bars | ☐ |
| — | — | Group members **and** individual contributions | title page | — | ☐ |
| — | — | References cited for all sources | §12 | — | ☐ |
| — | — | NTU Honour Code / no plagiarism | — | — | ☐ |

## Submission checklist

- [ ] `submission.csv` — 500 rows, header `id,label`, ids 1–500, labels ∈ {0,1}
- [ ] Report PDF, single file, English
- [ ] Code folder (or GitHub link) with `README.md` that actually works from a clean checkout
- [ ] Uploaded to NTULearn before **Fri 13 Nov 2026, 23:59**

## Defects that cost marks in this brief (self-check)

1. Using the CIFAR-10 **test set** to choose the best epoch → results not credible.
2. Reporting only overall accuracy for the imbalanced experiment → hides minority collapse.
3. No figure for item b (architecture) → guaranteed deduction.
4. Error analysis with no explanation of *why* the model failed.
5. No run instructions / code that does not run → 15% item b partially lost.
6. Inconsistent numbers between text, tables and `submission.csv`.
