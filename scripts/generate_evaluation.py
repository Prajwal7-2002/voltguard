"""Generate notebooks/04_evaluation.ipynb.

The notebook reads metrics from the served model's metrics.json rather than
hardcoding numbers, so it can't drift from what training actually measured.
"""

import json


def _cell(cell_type: str, source: str) -> dict:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


INTRO = """# Model Evaluation Report

All numbers below are read from the served model's `metrics.json`, written by
`scripts/train.py`. The test set is **whole drive profiles the model never saw**:
the data is sampled at 2 Hz, so a random row split would put near-identical rows in
both train and test and inflate every metric.

Read the results against the **always-Nominal baseline**. About 91% of rows are
Nominal, so accuracy alone says very little."""

LOAD = """import json
from pathlib import Path

import pandas as pd

models = Path('../models')
version = (models / 'latest.txt').read_text().strip()
m = json.loads((models / version / 'metrics.json').read_text())
print('Model version:', version)
print('Split:', m['benchmark_split'], '| test drives:', m['test_groups'], '| test rows:', m['test_rows'])"""

HEADLINE = """base = m['baseline_always_nominal']
pd.DataFrame({
    'Model': [m['test_accuracy'], m['test_f1_macro'], m['cv_f1_macro_mean']],
    'Always Nominal': [base['test_accuracy'], base['test_f1_macro'], None],
}, index=['Accuracy (held-out drives)', 'Macro F1 (held-out drives)', 'Macro F1 (drive-level CV mean)']).round(3)"""

PER_CLASS = """per_class = pd.DataFrame(m['per_class_report']).T[['precision', 'recall', 'f1-score', 'support']]
per_class['drives_with_class'] = [m['profiles_per_class'][str(i)] for i in range(len(per_class))]
per_class.round(3)"""

CONFUSION = """labels = m['confusion_matrix_labels']
pd.DataFrame(m['confusion_matrix'], index=[f'actual: {l}' for l in labels], columns=labels)"""

LIMITS = """## Limitations

- **The labels are proxies, not observed failures.** The PMSM dataset has no fault
  annotations. A "fault" means a hidden thermal channel (stator winding, tooth, yoke
  or magnet temperature) is in its top few percent. The model therefore gives early
  warning of thermal stress from electrical and control signals; it does not detect
  component failures.
- **Some classes can't be evaluated.** A class that occurs in fewer than two drives
  can't be both learned and tested on an unseen drive (see `not_evaluable_classes`),
  and it is left out of macro F1.
- **Scores vary from drive to drive.** With only about 13 test drives, scores change
  noticeably between splits. Treat the drive-level CV standard deviation as the error bar."""

NOTES = """print('Not evaluable:', m.get('not_evaluable_classes'))
print('CV macro F1: %.3f ± %.3f' % (m['cv_f1_macro_mean'], m['cv_f1_macro_std']))
for note in m['leakage_notes']:
    print('-', note)"""


def create_evaluation_notebook() -> None:
    cells = [
        _cell("markdown", INTRO),
        _cell("code", LOAD),
        _cell("markdown", "## 1. Headline metrics vs. baseline"),
        _cell("code", HEADLINE),
        _cell("markdown", "## 2. Per-class performance"),
        _cell("code", PER_CLASS),
        _cell("markdown", "## 3. Confusion matrix (held-out drives)"),
        _cell("code", CONFUSION),
        _cell("code", NOTES),
        _cell("markdown", LIMITS),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open("notebooks/04_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    create_evaluation_notebook()
    print("Wrote notebooks/04_evaluation.ipynb")
